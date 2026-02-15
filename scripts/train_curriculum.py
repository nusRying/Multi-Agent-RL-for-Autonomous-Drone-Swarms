from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs import DroneSwarmEnv
from swarm_marl.training import build_multi_agent_ppo_config
from swarm_marl.utils import load_structured_config

try:
    import ray
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires RLlib. Install with `python -m pip install -r requirements-rllib.txt` "
        "on Python 3.10-3.12."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged multi-agent curriculum training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/curriculum_v1.yaml"),
        help="Path to curriculum config (YAML/JSON).",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/curriculum"),
        help="Root directory where stage checkpoints are saved.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("reports/metrics/curriculum_runs.csv"),
        help="CSV file where per-stage summary is appended.",
    )
    return parser.parse_args()


def _normalize_checkpoint_path(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "path"):
        return str(raw.path)
    if hasattr(raw, "checkpoint"):
        checkpoint = raw.checkpoint
        if isinstance(checkpoint, str):
            return checkpoint
        if hasattr(checkpoint, "path"):
            return str(checkpoint.path)
    return str(raw)


def _append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_structured_config(args.config)
    stages = cfg.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"No stages found in curriculum config: {args.config}")

    env_name = "drone_swarm_curriculum_v0"
    register_env(env_name, lambda env_cfg: DroneSwarmEnv(env_cfg))
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    prev_checkpoint_path: str | None = None
    csv_fields = [
        "curriculum_name",
        "stage_id",
        "stage_name",
        "iterations",
        "reward_mean_final",
        "episode_len_mean_final",
        "checkpoint_path",
    ]

    try:
        for stage_idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                raise ValueError(f"Stage index {stage_idx} is not a mapping/object.")

            stage_id = stage.get("stage_id", stage_idx + 1)
            stage_name = str(stage.get("stage_name", f"stage_{stage_id}"))
            stage_iterations = int(stage.get("train_iterations", 50))

            env_cfg = dict(stage.get("env_config", {}))
            env_cfg["seed"] = int(args.seed + stage_idx)

            print(
                f"\n=== Stage {stage_id}: {stage_name} ===\n"
                f"iterations={stage_iterations} env_config={env_cfg}"
            )

            algo_cfg = build_multi_agent_ppo_config(
                env_name=env_name,
                env_config=env_cfg,
                num_workers=args.num_workers,
            )
            algo = algo_cfg.build()

            if prev_checkpoint_path:
                print(f"Restoring previous stage weights: {prev_checkpoint_path}")
                algo.restore(prev_checkpoint_path)

            last_reward = 0.0
            last_len = 0.0
            for i in range(1, stage_iterations + 1):
                result = algo.train()
                last_reward = float(result.get("episode_reward_mean", 0.0))
                last_len = float(result.get("episode_len_mean", 0.0))
                print(
                    f"stage={stage_name} iter={i:04d} "
                    f"reward_mean={last_reward:9.3f} len_mean={last_len:7.2f}"
                )

            stage_ckpt_dir = args.checkpoint_root / f"stage_{stage_id}_{stage_name}"
            stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = algo.save(str(stage_ckpt_dir))
            prev_checkpoint_path = _normalize_checkpoint_path(checkpoint)
            print(f"Saved stage checkpoint: {prev_checkpoint_path}")

            _append_csv_row(
                args.metrics_csv,
                {
                    "curriculum_name": cfg.get("name", "curriculum"),
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "iterations": stage_iterations,
                    "reward_mean_final": f"{last_reward:.6f}",
                    "episode_len_mean_final": f"{last_len:.6f}",
                    "checkpoint_path": prev_checkpoint_path,
                },
                csv_fields,
            )
            algo.stop()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

