"""Generate comparison plots across all training approaches.

Reads the training CSVs from CTDE, Attention, and Physics runs and produces
overlaid reward and episode-length curves for the final report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training comparison across approaches.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "plots",
        help="Directory to save plots.",
    )
    return p.parse_args()


# Map: label -> CSV path (relative to ROOT)
TRAINING_CSVS = {
    "CTDE Baseline": "reports/metrics/train_ctde.csv",
    "CTDE + Attention": "reports/metrics/train_attention.csv",
    "Physics (CTDE)": "reports/metrics/train_physics.csv",
}

COLORS = {
    "CTDE Baseline": "#2563eb",
    "CTDE + Attention": "#7c3aed",
    "Physics (CTDE)": "#dc2626",
}


def _to_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v != v:  # NaN check
            return None
        return v
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Return (iterations, rewards, ep_lengths) from a training CSV."""
    iters, rewards, lengths = [], [], []
    if not path.exists():
        return iters, rewards, lengths

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = _to_float(row.get("iteration"))
            rw = _to_float(row.get("episode_reward_mean"))
            el = _to_float(row.get("episode_len_mean"))
            if it is not None and rw is not None:
                iters.append(it)
                rewards.append(rw)
                lengths.append(el if el is not None else 0.0)
    return iters, rewards, lengths


def _drop_user_site_paths() -> None:
    """Remove broken user-site packages that shadow conda env."""
    cleaned = []
    for p in sys.path:
        lower = p.lower()
        if "appdata\\roaming\\python" in lower and "site-packages" in lower:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned


def main() -> None:
    args = parse_args()
    _drop_user_site_paths()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Install with `pip install matplotlib`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Reward Comparison ──
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False
    for label, csv_rel in TRAINING_CSVS.items():
        csv_path = ROOT / csv_rel
        iters, rewards, _ = _load_csv(csv_path)
        if not iters:
            print(f"[SKIP] No data for {label}: {csv_path}")
            continue
        has_data = True
        ax.plot(iters, rewards, label=label, color=COLORS[label], linewidth=2, alpha=0.85)

    if has_data:
        ax.set_xlabel("Training Iteration", fontsize=12)
        ax.set_ylabel("Mean Episode Reward", fontsize=12)
        ax.set_title("Training Reward Comparison: CTDE vs Attention vs Physics", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        reward_path = args.output_dir / "reward_comparison.png"
        fig.savefig(reward_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {reward_path}")

    # ── Episode Length Comparison ──
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    has_data2 = False
    for label, csv_rel in TRAINING_CSVS.items():
        csv_path = ROOT / csv_rel
        iters, _, lengths = _load_csv(csv_path)
        if not iters:
            continue
        has_data2 = True
        ax2.plot(iters, lengths, label=label, color=COLORS[label], linewidth=2, alpha=0.85)

    if has_data2:
        ax2.set_xlabel("Training Iteration", fontsize=12)
        ax2.set_ylabel("Mean Episode Length (steps)", fontsize=12)
        ax2.set_title("Episode Length Comparison: CTDE vs Attention vs Physics", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        length_path = args.output_dir / "episode_length_comparison.png"
        fig2.savefig(length_path, dpi=150)
        plt.close(fig2)
        print(f"Saved: {length_path}")

    if not has_data and not has_data2:
        print("No training CSVs found. Run training first.")


if __name__ == "__main__":
    main()
