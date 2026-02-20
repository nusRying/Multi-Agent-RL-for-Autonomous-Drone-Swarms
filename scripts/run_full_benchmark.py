"""Run the full evaluation benchmark across all trained checkpoints.

Executes evaluate_protocol.py against each approach (CTDE, Attention, Physics)
and merges results into a single CSV for the final report.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full benchmark across all checkpoints.")
    p.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "eval_protocol_quick.yaml",
        help="Eval protocol YAML (default: quick for fast smoke-test).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "reports" / "metrics" / "final_benchmark.csv",
        help="Combined output CSV.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


# Map: label -> (checkpoint dir, mode, use_attention)
CHECKPOINTS = {
    "ctde_baseline": ("checkpoints/ctde_run", "ctde", False),
    "ctde_attention": ("checkpoints/attention_run", "ctde", True),
    "physics_ctde": ("checkpoints/physics_run", "physics", False),
}


def main() -> None:
    args = parse_args()
    eval_script = ROOT / "scripts" / "evaluate_protocol.py"
    python = sys.executable

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Remove old combined CSV so results are fresh
    if args.output_csv.exists():
        args.output_csv.unlink()

    for label, (ckpt_dir, mode, use_attention) in CHECKPOINTS.items():
        ckpt_path = ROOT / ckpt_dir
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {label}  (mode={mode}, attention={use_attention})")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}\n")

        cmd = [
            python,
            str(eval_script),
            "--protocol", str(args.protocol),
            "--checkpoint", str(ckpt_path),
            "--mode", mode,
            "--num-workers", str(args.num_workers),
            "--output-csv", str(args.output_csv),
        ]
        if use_attention:
            cmd.append("--attention")

        env = os.environ.copy()
        env["RLLIB_TEST_NO_TF_IMPORT"] = "1"
        env["RLLIB_TEST_NO_JAX_IMPORT"] = "1"

        result = subprocess.run(cmd, env=env, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"[WARN] {label} evaluation exited with code {result.returncode}")

    if args.output_csv.exists():
        print(f"\n✅ Combined benchmark saved: {args.output_csv}")
    else:
        print("\n⚠️  No benchmark results were produced.")


if __name__ == "__main__":
    main()
