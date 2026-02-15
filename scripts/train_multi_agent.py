from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs import DroneSwarmEnv

try:
    import ray
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires RLlib. Install with `python -m pip install -r requirements-rllib.txt` "
        "on Python 3.10-3.12."
    ) from exc

from swarm_marl.training import build_multi_agent_ppo_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-agent swarm PPO baseline.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of PPO training iterations.")
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers.")
    parser.add_argument("--num-drones", type=int, default=3, help="Number of drones in swarm.")
    parser.add_argument("--num-obstacles", type=int, default=8, help="Number of static obstacles.")
    parser.add_argument("--max-steps", type=int, default=400, help="Episode step limit.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/multi_agent"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_name = "drone_swarm_v0"

    env_config = {
        "num_drones": args.num_drones,
        "num_obstacles": args.num_obstacles,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    register_env(env_name, lambda cfg: DroneSwarmEnv(cfg))
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    config = build_multi_agent_ppo_config(
        env_name=env_name,
        env_config=env_config,
        num_workers=args.num_workers,
    )
    algo = config.build()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training multi-agent PPO for {args.iterations} iterations "
        f"(drones={args.num_drones}, obstacles={args.num_obstacles}, workers={args.num_workers})"
    )
    for i in range(1, args.iterations + 1):
        result = algo.train()
        reward_mean = result.get("episode_reward_mean", 0.0)
        length_mean = result.get("episode_len_mean", 0.0)
        print(f"iter={i:04d} reward_mean={reward_mean:9.3f} len_mean={length_mean:7.2f}")

    checkpoint = algo.save(str(args.checkpoint_dir))
    print(f"Saved checkpoint: {checkpoint}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
