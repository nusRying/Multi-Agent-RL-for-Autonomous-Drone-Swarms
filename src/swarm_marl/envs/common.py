from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DroneEnvConfig:
    world_size: float = 20.0
    dt: float = 0.1
    max_steps: int = 400
    max_speed: float = 4.0
    max_accel: float = 2.0
    collision_radius: float = 0.5
    goal_radius: float = 0.8
    num_obstacles: int = 8
    sensed_obstacles: int = 4
    neighbor_k: int = 3
    obstacle_radius: float = 0.8
    desired_spacing: float = 2.5
    reward_progress_scale: float = 2.0
    reward_goal: float = 25.0
    reward_collision: float = -25.0
    reward_formation_scale: float = 0.15
    seed: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "DroneEnvConfig":
        if not raw:
            return cls()
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

