from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously refresh a training plot from CSV metrics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input metrics CSV to watch.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        required=True,
        help="Output PNG to refresh.",
    )
    parser.add_argument("--interval-sec", type=float, default=10.0, help="Refresh interval in seconds.")
    parser.add_argument("--x-col", type=str, default="iteration", help="X-axis column.")
    parser.add_argument("--y1-col", type=str, default="episode_reward_mean", help="Primary Y-axis column.")
    parser.add_argument("--y2-col", type=str, default="episode_len_mean", help="Secondary Y-axis column.")
    parser.add_argument("--title", type=str, default="Training Progress", help="Plot title.")
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Maximum number of refreshes. Use 0 for unlimited.",
    )
    return parser.parse_args()


def _run_plot_once(
    python_exe: str,
    script_path: Path,
    input_csv: Path,
    output_png: Path,
    x_col: str,
    y1_col: str,
    y2_col: str,
    title: str,
) -> int:
    cmd = [
        python_exe,
        str(script_path),
        "--input-csv",
        str(input_csv),
        "--output-png",
        str(output_png),
        "--x-col",
        x_col,
        "--y1-col",
        y1_col,
        "--y2-col",
        y2_col,
        "--title",
        title,
    ]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    plot_script = root / "scripts" / "plot_metrics.py"
    if not plot_script.exists():
        raise FileNotFoundError(f"Missing plotting script: {plot_script}")

    updates = 0
    print(
        f"Watching {args.input_csv} -> {args.output_png} every {args.interval_sec:.1f}s "
        f"(max_updates={args.max_updates or 'unlimited'})"
    )
    try:
        while True:
            updates += 1
            code = _run_plot_once(
                python_exe=sys.executable,
                script_path=plot_script,
                input_csv=args.input_csv,
                output_png=args.output_png,
                x_col=args.x_col,
                y1_col=args.y1_col,
                y2_col=args.y2_col,
                title=args.title,
            )

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            if code == 0:
                print(f"[{ts}] refresh #{updates}: ok")
            else:
                print(f"[{ts}] refresh #{updates}: plot command failed (exit={code})")

            if args.max_updates > 0 and updates >= args.max_updates:
                break
            time.sleep(max(0.2, args.interval_sec))
    except KeyboardInterrupt:
        print("\nStopped watcher.")


if __name__ == "__main__":
    main()

