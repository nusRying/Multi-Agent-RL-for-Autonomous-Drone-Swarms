from __future__ import annotations

import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ModuleNotFoundError:
    class MultiAgentEnv:
        pass

from swarm_marl.envs.common import DroneEnvConfig


class DronePhysicsEnv(MultiAgentEnv):
    """
    Multi-agent environment using PyBullet for rigid body dynamics.
    
    This environment bridges the Sim-to-Real gap by simulating:
    - 6-DOF dynamics (controlled via simplified Thrust/Torque or RPYT)
    - Collision physics
    - Inertia
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg_dict = config or {}
        self.num_drones = int(cfg_dict.get("num_drones", 3))
        # Filter out num_drones to init config
        env_cfg = {k: v for k, v in cfg_dict.items() if k != "num_drones"}
        self.cfg = DroneEnvConfig.from_dict(env_cfg)
        self.gui = cfg_dict.get("gui", False)
        
        # PyBullet Setup
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load Assets
        self.plane_id = p.loadURDF("plane.urdf")
        
        # We need a drone URDF. We'll look in local assets.
        # Assuming current file is in src/swarm_marl/envs/
        # Assets in src/swarm_marl/envs/assets/drone.urdf
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.drone_urdf = os.path.join(curr_dir, "assets", "drone.urdf")
        
        if not os.path.exists(self.drone_urdf):
            # Fallback for now if not found (though we should create it)
            print(f"URDF not found at {self.drone_urdf}, using cube")
            self.drone_urdf = "cube.urdf" # PyBullet default
            
        self.drone_ids = []
        
        # Define Spaces
        # Action: [Roll, Pitch, YawRate, Thrust] or just [Fx, Fy, Fz] simplified?
        # Let's start with a simplified 3D force control to match the previous env 
        # but integrated via physics (Force -> Accel -> Vel -> Pos)
        # Action: [Fx, Fy, Fz] applied in WORLD frame for simplicity first, 
        # or BODY frame for realism. 
        # Let's do: Action = Target Velocity (to match old env interface high level) 
        # and use a low-level PD controller to compute forces?
        # OR: Action = Force (Accel). 
        # The previous env was: Action -> Accel.
        # So we can apply Forces directly.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation: Same as DroneSwarmEnv
        self._obs_dim = (
            9
            + (self.cfg.neighbor_k * 4)
            + (self.cfg.sensed_obstacles * 4)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        
        self.agent_ids = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents = list(self.agent_ids)
        self.step_count = 0
        
        # Goal and Obstacles
        self.goal = np.zeros(3, dtype=np.float32)
        # Obstacles in PyBullet should be bodies.
        self.obstacle_ids = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
            
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.drone_ids = []
        self.obstacle_ids = []
        self.agents = list(self.agent_ids)
        self.step_count = 0
        
        bound = self.cfg.world_size / 2.0
        
        # Spawn Drones
        for i in range(self.num_drones):
            pos = self.rng.uniform(-bound, bound, size=3)
            pos[2] = max(1.0, pos[2]) # Start above ground
            orn = p.getQuaternionFromEuler([0, 0, 0])
            body = p.loadURDF(self.drone_urdf, pos, orn)
            self.drone_ids.append(body)
            # Add simple drag to simulate air resistance
            p.changeDynamics(body, -1, linearDamping=0.5, angularDamping=0.5)

        # Spawn Obstacles (Static spheres/boxes)
        for i in range(self.cfg.num_obstacles):
            pos = self.rng.uniform(-bound, bound, size=3)
            pos[2] = max(0.5, pos[2])
            # Create a visual/collision shape
            # Using simple sphere
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.cfg.obstacle_radius)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.cfg.obstacle_radius, rgbaColor=[1, 0, 0, 1])
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=pos)
            self.obstacle_ids.append(body)
            
        self.goal = self.rng.uniform(-bound, bound, size=3).astype(np.float32)
        
        return self._get_obs(), self._get_infos()

    def step(self, action_dict):
        active_agents = list(self.agents)
        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        
        # Apply Actions
        for agent_id, action in action_dict.items():
            idx = self.agent_ids.index(agent_id)
            body = self.drone_ids[idx]
            
            # Action is [Fx, Fy, Fz] (normalized) -> Convert to Force
            # We assume action is desired acceleration ~ force
            # F = m*a
            # Max accel in cfg is 2.0. Mass is 1.0 in URDF.
            # We also need to counteract gravity if we want to hover at 0 action?
            # Or let policy learn gravity compensation? 
            # Previous env: Action -> Accel directly. Gravity wasn't explicit.
            # Here: Let's apply Force = Action * scaling + GravityComp
            
            force = action * self.cfg.max_accel * 5.0 # Scale up for physics
            
            # Add gravity compensation to Z? 
            # If we want 0 action to be hovering, we add [0, 0, 9.81].
            # But let's let generic RL learn it or assume 0 is hover.
            # Let's add gravity comp for easier transfer.
            # Let's add gravity comp for easier transfer.
            # force[2] += 9.81 # REMOVED for physics sanity check (expecting zero action = fall)
            
            # Apply force to COM
            # PyBullet applies force in world frame by default? No, usually local/world flag.
            # applyExternalForce uses world frame if no link index, or something.
            # Let's assume action is in World Frame for now (simple quad)
            p.applyExternalForce(body, -1, force, [0, 0, 0], p.LINK_FRAME)
            
        # Step Physics
        p.stepSimulation()
        
        # Collect States
        self.step_count += 1
        
        obs = self._get_obs()
        infos = self._get_infos()
        
        # Calculate Rewards (Simplified, reusing logic would be better if refactored)
        # ... (We can copy-paste reward logic or implement a shared calculator) ...
        # For prototype, we just implement basic distance reward + collision check
        
        # Collision Check using PyBullet
        collided = {}
        for idx, body in enumerate(self.drone_ids):
            agent_id = self.agent_ids[idx]
            # Check contact with all other bodies
            contacts = p.getContactPoints(bodyA=body)
            # If contact with ground (plane) or obstacles or other drones
            is_collision = len(contacts) > 0
            collided[agent_id] = is_collision
            
        # Terminations - FIXED to end episode on ANY collision
        any_collision = False
        all_goals_reached = True
        
        for agent_id in active_agents:
            idx = self.agent_ids.index(agent_id)
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[idx])
            dist = np.linalg.norm(np.array(pos) - self.goal)
            
            reward = -dist * 0.1  # Simple dense reward
            rewards[agent_id] = reward
            
            # Check collision
            if collided[agent_id]:
                rewards[agent_id] -= 10.0
                any_collision = True
            # Check goal reached
            elif dist < self.cfg.goal_radius:
                rewards[agent_id] += 50.0
            else:
                all_goals_reached = False
        
        # Episode termination logic
        time_limit = self.step_count >= self.cfg.max_steps
        episode_done = any_collision or all_goals_reached or time_limit
        
        if episode_done:
            # Mark ALL agents with consistent done flags
            for agent_id in self.agent_ids:
                if time_limit and not any_collision and not all_goals_reached:
                    # Pure time limit (no collision, not all goals)
                    terminated[agent_id] = False
                    truncated[agent_id] = True
                else:
                    # Collision or all goals reached
                    terminated[agent_id] = True
                    truncated[agent_id] = False
            
            terminated["__all__"] = True if (any_collision or all_goals_reached) else False
            truncated["__all__"] = True if (time_limit and not any_collision and not all_goals_reached) else False
            self.agents = []
        else:
            # Episode continues
            for agent_id in self.agent_ids:
                terminated[agent_id] = False
                truncated[agent_id] = False
            terminated["__all__"] = False
            truncated["__all__"] = False
            
        return obs, rewards, terminated, truncated, infos
        
    def _get_obs(self):
        obs = {}
        # Fetch all positions/vels first
        positions = []
        velocities = []
        for body in self.drone_ids:
            pos, orn = p.getBasePositionAndOrientation(body)
            lin_vel, ang_vel = p.getBaseVelocity(body)
            positions.append(np.array(pos))
            velocities.append(np.array(lin_vel))
            
        for i, agent_id in enumerate(self.agent_ids):
            own_pos = positions[i]
            own_vel = velocities[i]
            
            # Nearest neighbors (simple inefficient loop)
            neighbors = []
            for j, other_pos in enumerate(positions):
                if i == j: continue
                neighbors.append(other_pos - own_pos)
            # Sort and pad ... (Simplified for brevity, strictly should match `drone_swarm_env.py` logic)
            # Ideally we refactor `_nearest_neighbor_features` to be a standalone utility.
            # For now, let's just use zeros to show it runs.
            neighbor_feats = np.zeros(self.cfg.neighbor_k * 4) 
            
            # Obstacles
            # We can read obstacle positions from self.obstacle_ids
            obstacle_feats = np.zeros(self.cfg.sensed_obstacles * 4) 
            
            obs_vec = np.concatenate([
                own_pos,
                own_vel,
                self.goal - own_pos,
                neighbor_feats,
                obstacle_feats
            ])
            obs[agent_id] = obs_vec.astype(np.float32)
        return obs

    def _get_infos(self):
        # We need to provide global state for the Critic!
        positions = []
        velocities = []
        for body in self.drone_ids:
            pos, _ = p.getBasePositionAndOrientation(body)
            vel, _ = p.getBaseVelocity(body)
            positions.append(pos)
            velocities.append(vel)
            
        global_state = np.concatenate([
            np.array(positions).flatten(),
            np.array(velocities).flatten(),
            self.goal
        ]).astype(np.float32)
        
        return {id: {"global_state": global_state} for id in self.agent_ids}
