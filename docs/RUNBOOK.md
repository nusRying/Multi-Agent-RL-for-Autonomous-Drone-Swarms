# Runbook (Practical Commands)

This is the shortest reliable workflow for local training and evaluation on Windows.

## 1. Environment

```powershell
conda activate comp_vision
cd "C:\Users\umair\Videos\Multi-Agent RL for Autonomous Drone Swarms"
```

## 2. Install Dependencies

```powershell
python -m pip install -r requirements.txt -r requirements-rllib.txt -r requirements-dev.txt
```

## 3. Quick Multi-Agent Training (Fast Feedback)

```powershell
python scripts/train_multi_agent.py `
  --iterations 20 `
  --num-drones 3 `
  --num-obstacles 4 `
  --max-steps 120 `
  --fast-debug `
  --metrics-csv reports/metrics/multi_debug.csv `
  --checkpoint-dir checkpoints/multi_debug
```

## 4. Live Plot Refresh (Second Terminal)

```powershell
python scripts/watch_plot_metrics.py `
  --input-csv reports/metrics/multi_debug.csv `
  --output-png reports/plots/reward_curve_multi_debug_live.png `
  --interval-sec 8 `
  --title "Multi-Agent Live Progress"
```

## 5. Quick Evaluation (Recommended First)

Uses `configs/eval_protocol_quick.yaml` (`seeds=[0]`, `episodes_per_seed=10`).

```powershell
python scripts/evaluate_protocol.py `
  --checkpoint "checkpoints/multi_debug" `
  --protocol "configs/eval_protocol_quick.yaml" `
  --output-csv "reports/metrics/eval_quick.csv"
```

## 6. Full Evaluation (Slower)

```powershell
python scripts/evaluate_protocol.py `
  --checkpoint "checkpoints/multi_debug" `
  --protocol "configs/eval_protocol_v1.yaml" `
  --output-csv "reports/metrics/eval_full.csv"
```

## 7. Curriculum Training

```powershell
python scripts/train_curriculum.py `
  --config configs/curriculum_v1.yaml `
  --fast-debug `
  --checkpoint-root checkpoints/curriculum_v1 `
  --metrics-csv reports/metrics/curriculum_runs.csv
```

## Notes

- You can pass either a direct checkpoint folder or a parent folder. Scripts now auto-detect nested checkpoint directories.
- Ray warnings about metrics exporter agent on Windows are noisy but usually non-fatal.
- If you see errors, check `docs/TROUBLESHOOTING.md`.

