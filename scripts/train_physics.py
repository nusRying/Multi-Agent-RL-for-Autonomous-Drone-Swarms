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

import pandas as pd

from swarm_marl.envs.drone_physics_env import DronePhysicsEnv
from swarm_marl.utils import extract_episode_stats

try:
    import ray
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires RLlib. Install with `python -m pip install -r requirements-rllib.txt` "
        "on Python 3.10-3.12."
    ) from exc

from swarm_marl.training.config_builders import build_ctde_ppo_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-agent swarm in Physics Env.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of PPO training iterations.")
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers (0=local only, recommended for PyBullet).")
    parser.add_argument("--num-drones", type=int, default=3, help="Number of drones in swarm.")
    parser.add_argument("--num-obstacles", type=int, default=8, help="Number of static obstacles.")
    parser.add_argument("--max-steps", type=int, default=400, help="Episode step limit.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    try:
        import torch
        default_gpus = 1 if torch.cuda.is_available() else 0
    except ImportError:
        default_gpus = 0
    parser.add_argument("--num-gpus", type=float, default=default_gpus, help="Number of GPUs to use for training.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/physics_run"))
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("reports/metrics/train_physics.csv"),
        help="CSV file to append per-iteration metrics.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=4000, help="PPO train batch size (smaller for physics).")
    parser.add_argument("--minibatch-size", type=int, default=128, help="PPO minibatch size.")
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
    
    # GPU Diagnostics
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Requested GPUs: {args.num_gpus}")
        else:
            print("WARNING: CUDA not available, training on CPU")
            args.num_gpus = 0
    except ImportError:
        print("PyTorch not available for GPU check")
        args.num_gpus = 0
    print("="*60 + "\n")
    
    if args.fast_debug:
        args.train_batch_size = 1000
        args.minibatch_size = 64
        args.num_sgd_iter = 2

    env_name = "drone_physics_v0"

    env_config = {
        "num_drones": args.num_drones,
        "num_obstacles": args.num_obstacles,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "gui": False, # Always headless for training
    }

    # Register the Physics Env
    register_env(env_name, lambda cfg: DronePhysicsEnv(cfg))
    
    # Configure Ray runtime_env to ensure workers can import swarm_marl
    ray.init(
        ignore_reinit_error=True, 
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(SRC)}}
    )

    # Use CTDE config builder but point to our new env
    # Note: Physics env has same observation space structure so CTDE model works!
    config = build_ctde_ppo_config(
        env_name=env_name,
        env_config=env_config,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        gamma=args.gamma,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_sgd_iter=args.num_sgd_iter,
    )
    
    # We can also enable Attention here if we want!
    # Let's keep it simple (CTDE only) for the first physics run to isolate variables.

    algo = config.build()
    
    # Check for existing checkpoint and CSV to resume
    start_iteration = 0
    checkpoint_path = args.checkpoint_dir / "rllib_checkpoint.json"
    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint at {args.checkpoint_dir}, resuming...")
        try:
            # RLlib on Windows needs absolute paths to avoid Arrow URI errors
            abs_checkpoint_dir = str(args.checkpoint_dir.absolute())
            algo.restore(abs_checkpoint_dir)
            
            # Use CSV as source of truth for iteration count
            if args.metrics_csv.exists():
                try:
                    df = pd.read_csv(args.metrics_csv)
                    if not df.empty:
                        start_iteration = int(df["iteration"].max())
                        print(f"  Resuming from iteration {start_iteration} (found in CSV)")
                except Exception as csv_e:
                    print(f"  WARNING: Could not read iteration from CSV: {csv_e}")
                    # Fallback to internal counter if CSV fails
                    if hasattr(algo, "iteration"):
                        start_iteration = algo.iteration
            
        except Exception as e:
            print(f"  WARNING: Failed to restore checkpoint: {e}")
            print("  Starting from scratch instead.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.iterations <= start_iteration:
        print(f"\nTarget iterations ({args.iterations}) already reached (current: {start_iteration}).")
        print("Use a higher --iterations value to continue training.")
        return

    print(
        f"\nTraining Physics PPO from iteration {start_iteration + 1} to {args.iterations}"
    )
    print(f"  Drones: {args.num_drones}")
    print(f"  Obstacles: {args.num_obstacles}")
    print(f"  Workers: {args.num_workers} (0=local only)")
    print(f"  GPUs: {args.num_gpus}")
    print(f"  Batch: {args.train_batch_size}, Minibatch: {args.minibatch_size}")
    print()
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
    curr_iter = start_iteration
    while curr_iter < args.iterations:
        curr_iter += 1
        result = algo.train()
        reward_mean, length_mean = extract_episode_stats(result)
        print(f"iter={curr_iter:04d} reward_mean={reward_mean:9.3f} len_mean={length_mean:7.2f}")
        _append_csv_row(
            args.metrics_csv,
            {
                "iteration": curr_iter,
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
        
        # Save checkpoint more frequently for physics?
        if curr_iter % 10 == 0:
            checkpoint = algo.save(str(args.checkpoint_dir))
            print(f"  Saved checkpoint: {checkpoint}")

    checkpoint = algo.save(str(args.checkpoint_dir))
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved metrics CSV: {args.metrics_csv}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
