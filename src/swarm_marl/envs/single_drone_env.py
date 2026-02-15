from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from swarm_marl.envs.common import DroneEnvConfig


class SingleDroneEnv(gym.Env):
    """
    3D continuous-control single-drone environment.

    Observation:
    - Position (3)
    - Velocity (3)
    - Goal vector (3)
    - Nearest obstacle vectors + distances (sensed_obstacles * 4)

    Action:
    - Normalized acceleration command in xyz: [-1, 1]^3
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = DroneEnvConfig.from_dict(config)
        self.rng = np.random.default_rng(self.cfg.seed)

        self._obs_dim = 9 + (self.cfg.sensed_obstacles * 4)
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

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.obstacles = np.zeros((self.cfg.num_obstacles, 3), dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        bound = self.cfg.world_size / 2.0
        self.position = self.rng.uniform(-bound, bound, size=3).astype(np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.goal = self.rng.uniform(-bound, bound, size=3).astype(np.float32)
        self.obstacles = self.rng.uniform(
            -bound,
            bound,
            size=(self.cfg.num_obstacles, 3),
        ).astype(np.float32)
        self.step_count = 0

        obs = self._build_obs()
        info = {"distance_to_goal": self._distance_to_goal()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)

        previous_distance = self._distance_to_goal()

        accel = action * self.cfg.max_accel
        self.velocity = self.velocity + accel * self.cfg.dt
        self.velocity = self._clip_speed(self.velocity)
        self.position = self.position + self.velocity * self.cfg.dt
        self.position = np.clip(
            self.position,
            -self.cfg.world_size / 2.0,
            self.cfg.world_size / 2.0,
        )

        self.step_count += 1

        current_distance = self._distance_to_goal()
        progress_reward = (previous_distance - current_distance) * self.cfg.reward_progress_scale
        reached_goal = current_distance <= self.cfg.goal_radius
        collision = self._is_collision()

        reward = progress_reward
        if reached_goal:
            reward += self.cfg.reward_goal
        if collision:
            reward += self.cfg.reward_collision

        terminated = bool(reached_goal or collision)
        truncated = bool(self.step_count >= self.cfg.max_steps)

        obs = self._build_obs()
        info = {
            "distance_to_goal": current_distance,
            "reached_goal": reached_goal,
            "collision": collision,
        }
        return obs, float(reward), terminated, truncated, info

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def _clip_speed(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed <= self.cfg.max_speed or speed < 1e-8:
            return velocity
        return (velocity / speed) * self.cfg.max_speed

    def _is_collision(self) -> bool:
        if self.obstacles.size == 0:
            return False
        distances = np.linalg.norm(self.obstacles - self.position, axis=1)
        return bool(np.any(distances <= (self.cfg.obstacle_radius + self.cfg.collision_radius)))

    def _build_obs(self) -> np.ndarray:
        goal_vector = self.goal - self.position
        obstacle_features = self._nearest_obstacle_features()
        obs = np.concatenate(
            [
                self.position,
                self.velocity,
                goal_vector,
                obstacle_features,
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _nearest_obstacle_features(self) -> np.ndarray:
        if self.cfg.num_obstacles == 0:
            return np.zeros(self.cfg.sensed_obstacles * 4, dtype=np.float32)

        rel = self.obstacles - self.position
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

