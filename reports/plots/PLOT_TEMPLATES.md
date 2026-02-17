# Plot Placeholders and Naming Standard

Use these names so figures are easy to reference in weekly/final reports.

## Core plots

1. `reward_curve_<stage>.png`
2. `success_rate_<stage>.png`
3. `collision_rate_<stage>.png`
4. `formation_error_<stage>.png`
5. `path_efficiency_<stage>.png`

Example:
- `reward_curve_week04_n3.png`
- `success_rate_week05_n5_curriculum.png`

## Recommended figure metadata

Each figure should include:
- Title with scenario and algorithm.
- X-axis: training iterations or environment steps.
- Y-axis: metric value.
- Legend for each seed and a mean trend.

## Failure analysis figures

- `collision_heatmap_<stage>.png`
- `trajectory_overlay_<stage>.png`
- `neighbor_distance_hist_<stage>.png`

## Plot script

Use:
- `scripts/plot_metrics.py`

Example:
```powershell
python scripts/plot_metrics.py --input-csv reports/metrics/train_multi_agent.csv --output-png reports/plots/reward_curve_multi_agent.png --title "Multi-Agent PPO Training"
```

Live refresh example:
```powershell
python scripts/watch_plot_metrics.py --input-csv reports/metrics/train_multi_agent.csv --output-png reports/plots/reward_curve_multi_agent_live.png --interval-sec 8 --title "Multi-Agent PPO Live"
```

Note:
- `scripts/plot_metrics.py` strips broken user-site package paths before importing matplotlib (helpful on some Windows conda setups).

Expected training CSV columns:
- `iteration`
- `episode_reward_mean`
- `episode_len_mean`

Defaults:
- x-axis: `iteration`
- y1: `episode_reward_mean`
- y2: `episode_len_mean`
