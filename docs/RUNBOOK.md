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

## 7. CTDE Training (Centralized Critic)

```powershell
python scripts/train_ctde.py `
  --iterations 100 `
  --num-drones 3 `
  --checkpoint-dir checkpoints/ctde_run `
  --metrics-csv reports/metrics/train_ctde.csv
```

## 8. Attention Training (CTDE + Attention)

```powershell
python scripts/train_attention.py `
  --iterations 200 `
  --num-workers 2 `
  --checkpoint-dir checkpoints/attention_run `
  --metrics-csv reports/metrics/train_attention.csv
```

## 9. Physics Training (PyBullet Environment)

```powershell
python scripts/train_physics.py `
  --iterations 200 `
  --num-workers 2 `
  --num-gpus 1 `
  --checkpoint-dir checkpoints/physics_run `
  --metrics-csv reports/metrics/train_physics.csv
```

## 10. Visualize Physics Simulation

```powershell
python scripts/run_physics_gui.py
```

## 11. Visualize Trained Swarm Behavior

```powershell
python scripts/visualize_swarm.py --checkpoint checkpoints/ctde_run
```

## 12. Curriculum Training

```powershell
python scripts/train_curriculum.py `
  --config configs/curriculum_v1.yaml `
  --fast-debug `
  --checkpoint-root checkpoints/curriculum_v1 `
  --metrics-csv reports/metrics/curriculum_runs.csv
```

## Notes

- **GPU Support**: All training scripts (`train_attention.py`, `train_physics.py`, etc.) automatically detect and use GPU if available.
- You can pass either a direct checkpoint folder or a parent folder. Scripts now auto-detect nested checkpoint directories.
- Ray warnings about metrics exporter agent on Windows are noisy but usually non-fatal.
- Physics simulation (`train_physics.py`) is slower than point-mass environments due to PyBullet overhead.
- If you see errors, check `docs/TROUBLESHOOTING.md`.
