# Reports Workspace

This directory is the experiment evidence store for your portfolio and final paper-style summary.

## Structure

- `reports/templates/`
  - Reusable markdown templates.
- `reports/metrics/`
  - CSV files for run logs and final benchmark summaries.
- `reports/plots/`
  - Plot naming standards and placeholder specs.
- `reports/weekly/`
  - One markdown file per week, based on the weekly template.

## How to use

1. Start each week by copying `reports/templates/WEEK_REPORT_TEMPLATE.md` to `reports/weekly/weekXX.md`.
2. Log every training/eval run in `reports/metrics/run_log_template.csv`.
3. Update `reports/metrics/final_metrics_template.csv` only when milestone-level results are stable.
4. For checkpoint benchmarking, run `scripts/evaluate_protocol.py` and merge output into final metrics.
5. Generate and store plots using names in `reports/plots/PLOT_TEMPLATES.md`.
6. At project end, copy `reports/templates/FINAL_REPORT_TEMPLATE.md` to `reports/final_report.md`.

## Minimum evidence checklist

- 5-seed metrics for key milestones.
- Learning curves (reward, success rate, collision rate).
- Failure analysis notes and example scenarios.
- Final summary table and reproducible config references.
