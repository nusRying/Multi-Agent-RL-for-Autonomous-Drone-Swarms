from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("RLLIB_TEST_NO_TF_IMPORT", "1")
os.environ.setdefault("RLLIB_TEST_NO_JAX_IMPORT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs import DroneSwarmEnv
from swarm_marl.utils import extract_episode_stats

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
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("reports/metrics/train_multi_agent.csv"),
        help="CSV file to append per-iteration metrics.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=16384, help="PPO train batch size.")
    parser.add_argument("--minibatch-size", type=int, default=2048, help="PPO minibatch size.")
    parser.add_argument("--num-sgd-iter", type=int, default=10, help="PPO SGD epochs/iterations.")
    parser.add_argument(
        "--fast-debug",
        action="store_true",
        help="Use a much smaller PPO profile for faster per-iteration feedback.",
    )
    return parser.parse_args()


def _append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    if args.fast_debug:
        args.train_batch_size = 4096
        args.minibatch_size = 512
        args.num_sgd_iter = 2

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
        gamma=args.gamma,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_sgd_iter=args.num_sgd_iter,
    )
    algo = config.build()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Training multi-agent PPO for {args.iterations} iterations "
        f"(drones={args.num_drones}, obstacles={args.num_obstacles}, workers={args.num_workers}, "
        f"batch={args.train_batch_size}, minibatch={args.minibatch_size}, "
        f"sgd_iter={args.num_sgd_iter}, lr={args.lr}, gamma={args.gamma})"
    )
    csv_fields = [
        "iteration",
        "episode_reward_mean",
        "episode_len_mean",
        "timesteps_total",
        "time_total_s",
        "lr",
        "gamma",
        "train_batch_size",
        "minibatch_size",
        "num_sgd_iter",
        "num_drones",
        "num_obstacles",
        "max_steps",
        "seed",
    ]
    for i in range(1, args.iterations + 1):
        result = algo.train()
        reward_mean, length_mean = extract_episode_stats(result)
        print(f"iter={i:04d} reward_mean={reward_mean:9.3f} len_mean={length_mean:7.2f}")
        _append_csv_row(
            args.metrics_csv,
            {
                "iteration": i,
                "episode_reward_mean": f"{reward_mean:.6f}",
                "episode_len_mean": f"{length_mean:.6f}",
                "timesteps_total": int(_to_float(result.get("timesteps_total", 0), 0.0)),
                "time_total_s": f"{_to_float(result.get('time_total_s', 0.0)):.6f}",
                "lr": f"{args.lr:.8f}",
                "gamma": f"{args.gamma:.6f}",
                "train_batch_size": args.train_batch_size,
                "minibatch_size": args.minibatch_size,
                "num_sgd_iter": args.num_sgd_iter,
                "num_drones": args.num_drones,
                "num_obstacles": args.num_obstacles,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
            csv_fields,
        )

    checkpoint = algo.save(str(args.checkpoint_dir))
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved metrics CSV: {args.metrics_csv}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
