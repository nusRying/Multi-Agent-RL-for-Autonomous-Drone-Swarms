from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ModuleNotFoundError:  # pragma: no cover - allows local env smoke tests without ray
    class MultiAgentEnv:  # type: ignore[override]
        pass

from swarm_marl.envs.common import DroneEnvConfig


class DroneSwarmEnv(MultiAgentEnv):
    """
    Multi-agent 3D swarm environment.

    Each drone receives local observations only:
    - own position + velocity
    - goal vector
    - nearest-neighbor relative vectors + distances
    - nearest-obstacle relative vectors + distances
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()

        cfg_dict = config or {}
        self.num_drones = int(cfg_dict.get("num_drones", 3))
        env_cfg = {k: v for k, v in cfg_dict.items() if k != "num_drones"}
        self.cfg = DroneEnvConfig.from_dict(env_cfg)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.agent_ids = [f"drone_{i}" for i in range(self.num_drones)]
        self.agent_id_to_index = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        self.agents = list(self.agent_ids)

        self._obs_dim = (
            9
            + (self.cfg.neighbor_k * 4)
            + (self.cfg.sensed_obstacles * 4)
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.positions = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacles = np.zeros((self.cfg.num_obstacles, 3), dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = list(self.agent_ids)
        self.step_count = 0

        bound = self.cfg.world_size / 2.0
        self.positions = self.rng.uniform(-bound, bound, size=(self.num_drones, 3)).astype(np.float32)
        self.velocities = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.goal = self.rng.uniform(-bound, bound, size=3).astype(np.float32)
        self.obstacles = self.rng.uniform(
            -bound,
            bound,
            size=(self.cfg.num_obstacles, 3),
        ).astype(np.float32)

        observations = {agent_id: self._build_obs(i) for i, agent_id in enumerate(self.agent_ids)}
        infos = {
            agent_id: {
                "distance_to_goal": self._distance_to_goal(i),
                "global_state": self._global_state(),
            }
            for i, agent_id in enumerate(self.agent_ids)
        }
        return observations, infos

    def step(self, action_dict: dict[str, np.ndarray]):
        active_agent_ids = list(self.agents)
        if not active_agent_ids:
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

        active_indices = [self.agent_id_to_index[agent_id] for agent_id in active_agent_ids]
        prev_distances = {
            agent_id: float(np.linalg.norm(self.goal - self.positions[idx]))
            for agent_id, idx in zip(active_agent_ids, active_indices)
        }

        for agent_id, i in zip(active_agent_ids, active_indices):
            raw_action = action_dict.get(agent_id, np.zeros(3, dtype=np.float32))
            action = np.asarray(raw_action, dtype=np.float32).reshape(3)
            action = np.clip(action, -1.0, 1.0)
            accel = action * self.cfg.max_accel

            self.velocities[i] = self.velocities[i] + accel * self.cfg.dt
            self.velocities[i] = self._clip_speed(self.velocities[i])
            self.positions[i] = self.positions[i] + self.velocities[i] * self.cfg.dt

        self.positions = np.clip(
            self.positions,
            -self.cfg.world_size / 2.0,
            self.cfg.world_size / 2.0,
        )
        self.step_count += 1

        curr_distances = {
            agent_id: float(np.linalg.norm(self.goal - self.positions[idx]))
            for agent_id, idx in zip(active_agent_ids, active_indices)
        }
        reached = {
            agent_id: curr_distances[agent_id] <= self.cfg.goal_radius
            for agent_id in active_agent_ids
        }
        collided = self._collision_mask(active_indices)
        formation_penalties = self._formation_penalties(active_indices)

        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        obs: dict[str, np.ndarray] = {}

        any_collision = any(bool(collided[idx]) for idx in active_indices)
        time_limit = self.step_count >= self.cfg.max_steps
        next_active_agents: list[str] = []

        for agent_id, i in zip(active_agent_ids, active_indices):
            progress = (prev_distances[agent_id] - curr_distances[agent_id]) * self.cfg.reward_progress_scale
            reward = progress + formation_penalties.get(i, 0.0)
            if reached[agent_id]:
                reward += self.cfg.reward_goal
            if collided[i]:
                reward += self.cfg.reward_collision
            rewards[agent_id] = float(reward)

            done_agent = bool(reached[agent_id] or collided[i])
            terminated[agent_id] = done_agent
            truncated[agent_id] = bool(time_limit and not done_agent)

            if not done_agent and not time_limit and not any_collision:
                obs[agent_id] = self._build_obs(i)
                infos[agent_id] = {
                    "distance_to_goal": curr_distances[agent_id],
                    "reached_goal": bool(reached[agent_id]),
                    "collision": bool(collided[i]),
                    "global_state": self._global_state(),
                }
                next_active_agents.append(agent_id)

        all_reached = len(next_active_agents) == 0 and not any_collision and not time_limit
        episode_done = bool(all_reached or any_collision)
        terminated["__all__"] = episode_done
        truncated["__all__"] = bool(time_limit and not episode_done)

        if terminated["__all__"] or truncated["__all__"]:
            self.agents = []
        else:
            self.agents = next_active_agents

        return obs, rewards, terminated, truncated, infos

    def _distance_to_goal(self, index: int) -> float:
        return float(np.linalg.norm(self.goal - self.positions[index]))

    def _clip_speed(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed <= self.cfg.max_speed or speed < 1e-8:
            return velocity
        return (velocity / speed) * self.cfg.max_speed

    def _collision_mask(self, active_indices: list[int]) -> dict[int, bool]:
        collisions = {idx: False for idx in active_indices}
        if not active_indices:
            return collisions

        if self.cfg.num_obstacles > 0:
            active_pos = self.positions[active_indices]
            dist_to_obstacles = np.linalg.norm(
                active_pos[:, None, :] - self.obstacles[None, :, :],
                axis=2,
            )
            threshold = self.cfg.collision_radius + self.cfg.obstacle_radius
            obstacle_hits = np.any(dist_to_obstacles <= threshold, axis=1)
            for local_i, idx in enumerate(active_indices):
                if obstacle_hits[local_i]:
                    collisions[idx] = True

        if len(active_indices) > 1:
            for ai, i in enumerate(active_indices):
                for j in active_indices[ai + 1 :]:
                    if np.linalg.norm(self.positions[i] - self.positions[j]) <= (2.0 * self.cfg.collision_radius):
                        collisions[i] = True
                        collisions[j] = True
        return collisions

    def _formation_penalties(self, active_indices: list[int]) -> dict[int, float]:
        penalties = {idx: 0.0 for idx in active_indices}
        if len(active_indices) <= 1:
            return penalties

        for i in active_indices:
            dists = []
            for j in active_indices:
                if i == j:
                    continue
                dists.append(float(np.linalg.norm(self.positions[i] - self.positions[j])))
            if dists:
                spacing_error = float(np.mean(np.abs(np.asarray(dists) - self.cfg.desired_spacing)))
                penalties[i] = -self.cfg.reward_formation_scale * spacing_error
        return penalties

    def _build_obs(self, index: int) -> np.ndarray:
        own_position = self.positions[index]
        own_velocity = self.velocities[index]
        goal_vector = self.goal - own_position
        neighbor_features = self._nearest_neighbor_features(index)
        obstacle_features = self._nearest_obstacle_features(index)

        obs = np.concatenate(
            [
                own_position,
                own_velocity,
                goal_vector,
                neighbor_features,
                obstacle_features,
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _nearest_neighbor_features(self, index: int) -> np.ndarray:
        if self.num_drones <= 1 or self.cfg.neighbor_k <= 0:
            return np.zeros(self.cfg.neighbor_k * 4, dtype=np.float32)

        rel_vectors: list[np.ndarray] = []
        rel_distances: list[float] = []
        own = self.positions[index]
        for j in range(self.num_drones):
            if j == index:
                continue
            vec = self.positions[j] - own
            rel_vectors.append(vec)
            rel_distances.append(float(np.linalg.norm(vec)))

        order = np.argsort(np.asarray(rel_distances))
        k = min(self.cfg.neighbor_k, len(order))

        features: list[float] = []
        for idx in order[:k]:
            vec = rel_vectors[int(idx)]
            features.extend(
                [float(vec[0]), float(vec[1]), float(vec[2]), float(rel_distances[int(idx)])]
            )
        needed = (self.cfg.neighbor_k * 4) - len(features)
        if needed > 0:
            features.extend([0.0] * needed)
        return np.asarray(features, dtype=np.float32)

    def _nearest_obstacle_features(self, index: int) -> np.ndarray:
        if self.cfg.num_obstacles == 0 or self.cfg.sensed_obstacles <= 0:
            return np.zeros(self.cfg.sensed_obstacles * 4, dtype=np.float32)

        own = self.positions[index]
        rel = self.obstacles - own
        dist = np.linalg.norm(rel, axis=1)
        order = np.argsort(dist)
        k = min(self.cfg.sensed_obstacles, len(order))

        features: list[float] = []
        for idx in order[:k]:
            vec = rel[idx]
            features.extend([float(vec[0]), float(vec[1]), float(vec[2]), float(dist[idx])])

        needed = (self.cfg.sensed_obstacles * 4) - len(features)
        if needed > 0:
            features.extend([0.0] * needed)
        return np.asarray(features, dtype=np.float32)

    def _global_state(self) -> np.ndarray:
        # Useful for CTDE critic models during training.
        return np.concatenate(
            [
                self.positions.reshape(-1),
                self.velocities.reshape(-1),
                self.goal.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
