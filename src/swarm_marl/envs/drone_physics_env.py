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
    Multi-Agent drone environment using PyBullet for rigid-body physics simulation.

    This environment wraps the `pybullet` physics engine to provide realistic 6-DOF
    drone dynamics, replacing the simplified kinematic model in `DroneSwarmEnv`.
    It is used for Phase 6+ training and the interactive 3D dashboard (Phase 10).

    ─────────────────────────────────────────────────────────────────────────────
    PHYSICS ALIGNMENT PROBLEM (CRITICAL FOR DASHBOARD USE):
    ─────────────────────────────────────────────────────────────────────────────
    The trained policy was learned in a KINEMATIC (zero-G) environment where:
        new_velocity = clip(old_velocity + action * max_accel * dt, max_speed)
    The policy's "action" directly represents desired net acceleration.

    In this PHYSICS environment, gravity exists (-9.81 m/s² in the Z axis).
    To make the policy work, we must:
        1. Convert action → force: F = action * max_accel * mass
        2. Add gravity compensation: Fz += mass * g_comp
        3. Apply force at the drone's Center of Mass position (not world origin!)
        4. Apply at EVERY physics sub-step (not just once per env.step call)

    The current gravity compensation factor is g_comp ≈ 9.5.
    Tuning guide:
        - g_comp = 9.81 → Perfect compensation but any non-zero action causes fast rise
        - g_comp = 9.5  → Slight under-compensation (current): drones sink slightly at action=0
        - g_comp = 9.0  → More aggressive sink if stable hover needed
        - g_comp = 0.0  → No compensation; drones fall to ground immediately

    ─────────────────────────────────────────────────────────────────────────────
    SUB-STEPPING:
    ─────────────────────────────────────────────────────────────────────────────
    PyBullet default timestep = 1/240s ≈ 4ms
    Our training timestep (dt) = 0.1s = 100ms
    Sub-steps per env.step() = int(0.1 * 240) = 24

    We call p.applyExternalForce() + p.stepSimulation() 24 times per env step.
    This ensures the physics integration produces accurate results for our dt.

    ─────────────────────────────────────────────────────────────────────────────
    DOMAIN RANDOMIZATION (Phase 8):
    ─────────────────────────────────────────────────────────────────────────────
    Each episode, drone mass and damping are randomized:
        mass ∈ [0.9, 1.1] kg   (±10% around nominal 1.0 kg)
        linear damping ∈ [0.4, 0.6]
    The gravity compensation reads the actual mass via p.getDynamicsInfo() to adapt.

    ─────────────────────────────────────────────────────────────────────────────
    OBSERVATION SPACE (matches DroneSwarmEnv exactly for policy transfer):
    ─────────────────────────────────────────────────────────────────────────────
    dim = 9 + (neighbor_k * 4) + (sensed_obstacles * 4)
    With defaults (neighbor_k=3, sensed_obstacles=4): dim = 9 + 12 + 16 = 37

    Layout: [own_pos(3) | own_vel(3) | goal_vec(3) | neighbor_feats(K*4) | obs_feats(M*4)]
    Note: own_vel is CLAMPED to max_speed before being placed in the observation.

    ─────────────────────────────────────────────────────────────────────────────
    ACTION SPACE:
    ─────────────────────────────────────────────────────────────────────────────
    Box(low=-1.0, high=1.0, shape=(3,))
    Represents normalized desired acceleration [ax, ay, az].

    ─────────────────────────────────────────────────────────────────────────────
    EPISODE TERMINATION:
    ─────────────────────────────────────────────────────────────────────────────
    terminated=True  → Collision OR all goals reached (early success)
    truncated=True   → Time limit exceeded (max_steps)

    Args:
        config (dict): Environment configuration dictionary. All keys are optional
                       and will use defaults from DroneEnvConfig if not provided.
            - num_drones (int): Number of agents in the swarm (default: 3)
            - gui (bool): Whether to open PyBullet GUI window (default: False)
            - world_size (float): World boundary ±world_size/2 (default: 20.0)
            - dt (float): Time step per env.step() call (default: 0.1)
            - max_steps (int): Maximum steps before truncation (default: 400)
            - max_speed (float): Hard velocity limit m/s (default: 4.0)
            - max_accel (float): Max acceleration m/s² maps action=1 (default: 2.0)
            - num_obstacles (int): Static obstacles per episode (default: 8)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the PyBullet environment.

        Connects to PyBullet (GUI or DIRECT), loads the drone URDF,
        sets gravity (-9.81 m/s²), and defines observation/action spaces.

        Note: The PyBullet client is NOT reset here; it is reset in each
        call to reset() via p.resetSimulation().
        """
        super().__init__()
        cfg_dict = config or {}
        self.num_drones = int(cfg_dict.get("num_drones", 3))
        # Filter out num_drones to init config (DroneEnvConfig doesn't accept it)
        env_cfg = {k: v for k, v in cfg_dict.items() if k != "num_drones"}
        self.cfg = DroneEnvConfig.from_dict(env_cfg)
        self.gui = cfg_dict.get("gui", False)

        # ── PyBullet Connection ──────────────────────────────────────────────
        # GUI mode opens a 3D window; DIRECT mode is headless (faster, for training).
        try:
            if self.gui:
                self.client = p.connect(p.GUI)
            else:
                self.client = p.connect(p.DIRECT)
        except p.error:
            # If GUI already connected (e.g., interactive dashboard spawning multiple envs),
            # gracefully fall back to the default client (ID = 0).
            print("PyBullet GUI already connected or failed, attempting to use existing client...")
            self.client = 0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)  # Earth gravity in the -Z direction

        # ── Asset Loading ─────────────────────────────────────────────────────
        self.plane_id = p.loadURDF("plane.urdf")

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.drone_urdf = os.path.join(curr_dir, "assets", "drone.urdf")
        if not os.path.exists(self.drone_urdf):
            print(f"URDF not found at {self.drone_urdf}, using cube")
            self.drone_urdf = "cube.urdf"

        self.drone_ids = []  # List of PyBullet body IDs, one per drone

        # ── Spaces ─────────────────────────────────────────────────────────
        # Action: 3D normalized force direction. Matches DroneSwarmEnv interface.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # Observation: identical layout to DroneSwarmEnv for policy transfer.
        self._obs_dim = (
            9
            + (self.cfg.neighbor_k * 4)
            + (self.cfg.sensed_obstacles * 4)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        # Agent bookkeeping
        self.agent_ids = [f"drone_{i}" for i in range(self.num_drones)]
        self.agent_id_to_index = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        self.agents = list(self.agent_ids)
        self.step_count = 0

        # Goal and obstacle state
        self.goal = np.zeros(3, dtype=np.float32)
        self.goal_id = -1          # PyBullet visual body ID for the green sphere
        self.obstacle_ids = []     # PyBullet body IDs for static obstacles

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment to a new episode.

        - Clears all physics bodies from last episode.
        - Spawns drones at random positions (z ≥ 1.0 m, above ground).
        - Applies domain randomization to mass and damping.
        - Spawns random static obstacles.
        - Places goal at a random location (height between 0.5 and 2.0 m).
        - Creates green visual sphere for goal in GUI mode.
        - Computes initial observations and info dict.

        Returns:
            obs (dict): {agent_id: obs_array(37,)} for each agent.
            infos (dict): {agent_id: {"distance_to_goal", "reached_goal", "collision"}}
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # ── Reset Physics World ───────────────────────────────────────────────
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

        self.drone_ids = []
        self.obstacle_ids = []
        self.agents = list(self.agent_ids)
        self.step_count = 0

        bound = self.cfg.world_size / 2.0

        # ── Spawn Drones ──────────────────────────────────────────────────────
        # Each drone is loaded from drone.urdf (mass=1.0, box shape 0.3×0.3×0.05).
        # Domain randomization is applied: mass ±10%, damping ±20%.
        for i in range(self.num_drones):
            pos = self.rng.uniform(-bound, bound, size=3)
            pos[2] = max(1.0, pos[2])  # Ensure drone starts above ground
            orn = p.getQuaternionFromEuler([0, 0, 0])
            body = p.loadURDF(self.drone_urdf, pos, orn)
            self.drone_ids.append(body)

            # Domain Randomization
            mass_noise = self.rng.uniform(0.9, 1.1)
            damp_noise = self.rng.uniform(0.8, 1.2)
            p.changeDynamics(body, -1,
                             mass=1.0 * mass_noise,
                             linearDamping=0.5 * damp_noise,
                             angularDamping=0.5 * damp_noise)
            # Debug: print actual mass for first drone
            actual_mass = p.getDynamicsInfo(body, -1)[0]
            if i == 0:
                print(f"DEBUG: Drone 0 Mass={actual_mass:.3f}")

        # ── Spawn Obstacles ───────────────────────────────────────────────────
        for i in range(self.cfg.num_obstacles):
            pos = self.rng.uniform(-bound, bound, size=3)
            pos[2] = max(0.5, pos[2])
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.cfg.obstacle_radius)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.cfg.obstacle_radius,
                                             rgbaColor=[1, 0, 0, 1])  # Red spheres
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape,
                                     baseVisualShapeIndex=vis_shape, basePosition=pos)
            self.obstacle_ids.append(body)

        # ── Goal ──────────────────────────────────────────────────────────────
        self.goal = self.rng.uniform(-bound, bound, size=3).astype(np.float32)
        self.goal[2] = self.rng.uniform(0.5, 2.0)  # Keep goal at reachable height

        # Visual goal sphere (green) for GUI mode
        if self.gui:
            goal_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.cfg.goal_radius,
                                            rgbaColor=[0, 1, 0, 0.5])
            self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_vis,
                                              basePosition=self.goal)

        # ── Initial Observations & Info ───────────────────────────────────────
        obs = self._get_obs()
        infos = {}
        for i, agent_id in enumerate(self.agent_ids):
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
            dist = float(np.linalg.norm(np.array(pos) - self.goal))
            infos[agent_id] = {
                "distance_to_goal": dist,
                "reached_goal": False,
                "collision": False
            }

        return obs, infos

    def set_goal(self, new_pos: np.ndarray | list[float]):
        """
        Dynamically reposition the goal sphere.

        Called by the interactive dashboard when the user clicks to drag
        the green sphere to a new location.

        Args:
            new_pos: 3D position [x, y, z] for the new goal location.
        """
        self.goal = np.array(new_pos, dtype=np.float32)
        if self.gui and self.goal_id != -1:
            p.resetBasePositionAndOrientation(self.goal_id, self.goal, [0, 0, 0, 1])

    def step(self, action_dict):
        """
        Advance the simulation by one timestep (dt = cfg.dt seconds).

        This method:
        1. Converts policy actions to physical forces (F = ma + gravity_comp).
        2. Runs PyBullet physics for 'steps_per_call' sub-steps.
        3. Applies velocity clamping to match kinematic training constraints.
        4. Computes observations, rewards, termination flags, and info.

        ─────────────────────────────────────────────────────────────────────
        FORCE MODEL (see class docstring for full explanation):
        ─────────────────────────────────────────────────────────────────────
            force = action * max_accel * mass        # Desired net acceleration
            force[Z] += mass * 9.5                   # Under-compensated gravity
            p.applyExternalForce(body, -1, force, pos, WORLD_FRAME)
                                          ^^^
                                      pos = COM, NOT [0,0,0] !!

        CRITICAL: Force MUST be applied at the drone's current position.
        Applying at [0,0,0] creates a torque arm → wildly spinning drones.

        ─────────────────────────────────────────────────────────────────────
        SUB-STEPPING:
        ─────────────────────────────────────────────────────────────────────
            steps_per_call = int(dt * 240) = 24  (for dt=0.1s)
            Force is applied at EVERY sub-step to ensure continuous thrust.

        Args:
            action_dict (dict): {agent_id: action_array(3,)} for each agent.

        Returns:
            obs (dict): Updated observations.
            rewards (dict): Per-agent reward scalars.
            terminated (dict): Per-agent booleans + "__all__".
            truncated (dict): Per-agent booleans + "__all__".
            infos (dict): Per-agent info dicts with "distance_to_goal", etc.
        """
        active_agents = list(self.agents)
        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}

        # ── Force Application (with sub-stepping) ────────────────────────────
        # Note: pyBullet's applyExternalForce is "one-shot" — it resets to zero
        # after each p.stepSimulation() call. So we must re-apply at every sub-step.
        steps_per_call = int(self.cfg.dt * 240)
        for _ in range(steps_per_call):
            for agent_id, action in action_dict.items():
                idx = self.agent_ids.index(agent_id)
                body = self.drone_ids[idx]

                # Read actual mass (randomized per episode)
                mass = p.getDynamicsInfo(body, -1)[0]
                # Current position (used for force application point = COM)
                pos, _ = p.getBasePositionAndOrientation(body)

                # Convert action [norm accel] → raw force [N]
                # F = m * a_desired
                force = action * self.cfg.max_accel * mass

                # Add under-compensated gravity to prevent drones from free-falling.
                # Using 9.5 instead of 9.81 creates a slight "heavy" bias.
                # This means at action=0, drones sink slightly (realistic behavior).
                # If drones still fly away: reduce this value toward 9.0.
                # If drones fall to ground: increase this value toward 9.81.
                force[2] += (mass * 9.5)

                # CRITICAL: Apply at COM (pos), NOT world origin [0,0,0].
                # Applying at [0,0,0] when drone is at [5,5,5] creates a torque
                # arm of length ~8.66m → catastrophic spin instability.
                p.applyExternalForce(body, -1, force, pos, p.WORLD_FRAME)

                # ── Velocity Clamping ──────────────────────────────────────────
                # The policy was trained in a kinematic env with hard speed cap.
                # We replicate this by resetting the rigid body velocity if exceeded.
                lin_vel, ang_vel = p.getBaseVelocity(body)
                v = np.array(lin_vel)
                speed = np.linalg.norm(v)
                if speed > self.cfg.max_speed:
                    v_clamped = (v / speed) * self.cfg.max_speed
                    p.resetBaseVelocity(body, v_clamped, ang_vel)

            p.stepSimulation()

        # ── Collect State ─────────────────────────────────────────────────────
        self.step_count += 1
        obs = self._get_obs()
        infos = self._get_infos()

        # ── Collision Detection ───────────────────────────────────────────────
        collided = {}
        for idx, body in enumerate(self.drone_ids):
            agent_id = self.agent_ids[idx]
            contacts = p.getContactPoints(bodyA=body)
            collided[agent_id] = len(contacts) > 0

        # ── Rewards & Terminations ─────────────────────────────────────────────
        any_collision = False
        all_goals_reached = True

        for agent_id in active_agents:
            idx = self.agent_ids.index(agent_id)
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[idx])
            dist = np.linalg.norm(np.array(pos) - self.goal)

            reward = -dist * 0.1  # Dense reward: negative distance to goal
            rewards[agent_id] = reward

            if collided[agent_id]:
                rewards[agent_id] -= 10.0
                any_collision = True
            elif dist < self.cfg.goal_radius:
                rewards[agent_id] += 50.0
            else:
                all_goals_reached = False

            # Update info with actual collision data (overrides placeholder in _get_infos)
            infos[agent_id]["collision"] = bool(collided[agent_id])

        # ── Episode Done Logic ─────────────────────────────────────────────────
        time_limit = self.step_count >= self.cfg.max_steps
        episode_done = any_collision or all_goals_reached or time_limit

        if episode_done:
            for agent_id in self.agent_ids:
                if time_limit and not any_collision and not all_goals_reached:
                    terminated[agent_id] = False
                    truncated[agent_id] = True
                else:
                    terminated[agent_id] = True
                    truncated[agent_id] = False
            terminated["__all__"] = True if (any_collision or all_goals_reached) else False
            truncated["__all__"] = True if (time_limit and not any_collision and not all_goals_reached) else False
            self.agents = []
        else:
            for agent_id in self.agent_ids:
                terminated[agent_id] = False
                truncated[agent_id] = False
            terminated["__all__"] = False
            truncated["__all__"] = False

        return obs, rewards, terminated, truncated, infos

    def _get_obs(self):
        """
        Compute observations for all agents.

        Fetches position and velocity from PyBullet, clamps velocity to max_speed
        (to match kinematic training conditions), then builds the observation vector:
            [own_pos(3) | own_vel(3) | goal_vec(3) | neighbor_feats(K*4) | obs_feats(M*4)]

        Returns:
            obs (dict): {agent_id: np.ndarray(obs_dim,)} for each agent.
        """
        obs = {}
        positions = []
        velocities = []
        for body in self.drone_ids:
            pos, orn = p.getBasePositionAndOrientation(body)
            lin_vel, ang_vel = p.getBaseVelocity(body)
            # Clamp to trained max_speed so obs matches kinematic env distribution
            v = np.array(lin_vel)
            v_norm = np.linalg.norm(v)
            if v_norm > self.cfg.max_speed:
                v = (v / v_norm) * self.cfg.max_speed
            positions.append(np.array(pos))
            velocities.append(v)

        for i, agent_id in enumerate(self.agent_ids):
            own_pos = positions[i]
            own_vel = velocities[i]
            goal_vec = self.goal - own_pos  # Vector pointing from drone to goal

            neighbor_feats = self._nearest_neighbor_features(i, positions)
            obstacle_feats = self._nearest_obstacle_features(own_pos)

            obs_vec = np.concatenate([
                own_pos,
                own_vel,
                goal_vec,
                neighbor_feats.flatten(),
                obstacle_feats.flatten()
            ])
            obs[agent_id] = obs_vec.astype(np.float32)
        return obs

    def _nearest_neighbor_features(self, index, positions):
        """
        Compute neighbor features for the K nearest drones.

        For each neighbor (sorted by distance), returns [rel_x, rel_y, rel_z, distance].
        If fewer than K neighbors exist, pads with zeros.

        Args:
            index (int): Index of the ego drone (self) in self.agent_ids.
            positions (list[np.ndarray]): List of all drone 3D positions.

        Returns:
            np.ndarray: Shape (neighbor_k * 4,). Relative positions + distances.
        """
        if self.num_drones <= 1 or self.cfg.neighbor_k <= 0:
            return np.zeros(self.cfg.neighbor_k * 4, dtype=np.float32)

        rel_vectors = []
        rel_distances = []
        own = positions[index]
        for j, other_pos in enumerate(positions):
            if index == j:
                continue
            vec = other_pos - own
            rel_vectors.append(vec)
            rel_distances.append(np.linalg.norm(vec))

        order = np.argsort(rel_distances)
        k = min(self.cfg.neighbor_k, len(order))

        features = []
        for idx in order[:k]:
            vec = rel_vectors[idx]
            features.extend([vec[0], vec[1], vec[2], rel_distances[idx]])

        needed = (self.cfg.neighbor_k * 4) - len(features)
        if needed > 0:
            features.extend([0.0] * needed)
        return np.array(features, dtype=np.float32)

    def _nearest_obstacle_features(self, own_pos):
        """
        Compute obstacle features for the M nearest obstacles.

        For each obstacle (sorted by distance), returns [rel_x, rel_y, rel_z, distance].
        Obstacle positions are fetched live from PyBullet's state.

        Args:
            own_pos (np.ndarray): Ego drone's 3D position.

        Returns:
            np.ndarray: Shape (sensed_obstacles * 4,). Relative positions + distances.
        """
        if self.cfg.num_obstacles == 0 or self.cfg.sensed_obstacles <= 0:
            return np.zeros(self.cfg.sensed_obstacles * 4, dtype=np.float32)

        obs_positions = []
        for b_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(b_id)
            obs_positions.append(pos)

        rel = np.array(obs_positions) - own_pos
        dist = np.linalg.norm(rel, axis=1)
        order = np.argsort(dist)
        k = min(self.cfg.sensed_obstacles, len(order))

        features = []
        for idx in order[:k]:
            vec = rel[idx]
            features.extend([vec[0], vec[1], vec[2], dist[idx]])

        needed = (self.cfg.sensed_obstacles * 4) - len(features)
        if needed > 0:
            features.extend([0.0] * needed)
        return np.array(features, dtype=np.float32)

    def _get_infos(self):
        """
        Build per-agent info dictionaries for one environment step.

        Returns a dict containing:
        - global_state: Concatenated positions + velocities + goal (used by centralized critic).
        - distance_to_goal: Euclidean distance from agent to goal.
        - reached_goal: Boolean, True if within cfg.goal_radius.
        - collision: Placeholder False (overwritten by step() with actual collision data).

        Note: step() calls this method FIRST, then overwrites infos[agent_id]["collision"]
        with the actual PyBullet collision detection result.

        Returns:
            infos (dict): {agent_id: {global_state, distance_to_goal, reached_goal, collision}}
        """
        infos = {}

        all_pos = []
        all_vel = []
        for body in self.drone_ids:
            pos, _ = p.getBasePositionAndOrientation(body)
            vel, _ = p.getBaseVelocity(body)
            all_pos.append(pos)
            all_vel.append(vel)

        # Global state for centralized critic: all positions + velocities + goal
        global_state = np.concatenate([
            np.array(all_pos).flatten(),  # Shape: N*3
            np.array(all_vel).flatten(),  # Shape: N*3
            self.goal                     # Shape: 3
        ]).astype(np.float32)

        for i, agent_id in enumerate(self.agent_ids):
            pos = all_pos[i]
            dist = float(np.linalg.norm(np.array(pos) - self.goal))
            infos[agent_id] = {
                "global_state": global_state,
                "distance_to_goal": dist,
                "reached_goal": dist < self.cfg.goal_radius,
                "collision": False  # Placeholder; step() writes the real value
            }

        return infos
