from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input metrics CSV file produced by training scripts.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("reports/plots/training_metrics.png"),
        help="Output plot PNG path.",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default="iteration",
        help="Column name to use as x-axis.",
    )
    parser.add_argument(
        "--y1-col",
        type=str,
        default="episode_reward_mean",
        help="Primary y-axis column.",
    )
    parser.add_argument(
        "--y2-col",
        type=str,
        default="episode_len_mean",
        help="Secondary y-axis column.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training Progress",
        help="Plot title.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _drop_user_site_paths() -> None:
    # Some environments have broken user-site packages (for example in
    # AppData\Roaming) that shadow the active conda env packages.
    cleaned = []
    for p in sys.path:
        lower = p.lower()
        if "appdata\\roaming\\python" in lower and "site-packages" in lower:
            continue
        cleaned.append(p)
    sys.path[:] = cleaned


def _load_columns(path: Path, x_col: str, y1_col: str, y2_col: str) -> tuple[list[float], list[float], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    xs: list[float] = []
    y1s: list[float] = []
    y2s: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        missing = [c for c in [x_col, y1_col, y2_col] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns {missing} in {path}")

        for row in reader:
            x = _to_float(row.get(x_col))
            y1 = _to_float(row.get(y1_col))
            y2 = _to_float(row.get(y2_col))
            if x is None or y1 is None or y2 is None:
                continue
            xs.append(x)
            y1s.append(y1)
            y2s.append(y2)

    if not xs:
        raise ValueError(f"No numeric rows found in {path} for selected columns.")
    return xs, y1s, y2s


def main() -> None:
    args = parse_args()
    xs, y1s, y2s = _load_columns(args.input_csv, args.x_col, args.y1_col, args.y2_col)

    _drop_user_site_paths()
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install with `python -m pip install matplotlib`."
        ) from exc

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(xs, y1s, color="tab:blue", linewidth=2, label=args.y1_col)
    ax1.set_xlabel(args.x_col)
    ax1.set_ylabel(args.y1_col, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, y2s, color="tab:orange", linewidth=2, label=args.y2_col)
    ax2.set_ylabel(args.y2_col, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(args.title)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {args.output_png}")


if __name__ == "__main__":
    main()
