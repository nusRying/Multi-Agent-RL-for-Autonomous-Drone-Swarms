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

## Placeholder script target

When you add plotting code later, store it in:
- `scripts/plot_metrics.py`

Expected inputs:
- `reports/metrics/run_log_template.csv`
- `reports/metrics/final_metrics_template.csv`

