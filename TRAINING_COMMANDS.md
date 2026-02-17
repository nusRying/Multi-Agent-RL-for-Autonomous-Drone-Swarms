# Multi-Agent RL Drone Swarm Training Commands

## GPU is Now Configured and Ready! ✅

All fixes have been applied:

- ✅ GPU logging added to training scripts
- ✅ Physics environment termination logic fixed
- ✅ Default workers set to 0 for PyBullet stability

---

## Commands to Run

### 1. **Physics Training** (Recommended - Most Stable)

```powershell
python scripts/train_physics.py --iterations 200
```

**What this does:**

- Trains in PyBullet physics environment
- Uses GPU automatically (you'll see GPU info at startup)
- 0 workers (single process, very stable)
- Saves to `checkpoints/physics_run/`
- Metrics to `reports/metrics/train_physics.csv`

---

### 2. **Attention Training** (Point-Mass Environment)

```powershell
python scripts/train_attention.py --iterations 200 --num-workers 2
```

**What this does:**

- Trains with CTDE + Attention mechanism
- Uses GPU automatically
- 2 workers for faster rollouts
- Saves to `checkpoints/attention_run/`
- Metrics to `reports/metrics/train_attention.csv`

---

### 3. **Visualize Physics Simulation** (Quick Preview)

```powershell
python scripts/run_physics_gui.py
```

**What this does:**

- Opens PyBullet GUI window
- Shows 3 drones with physics simulation
- Press close window or Ctrl+C to exit

---

## Monitoring GPU Usage

**While training is running**, open another terminal and run:

```powershell
nvidia-smi
```

You should see:

- Process: `python.exe`
- GPU Memory: ~1-2GB
- GPU Utilization: varies during training

---

## Expected Output

Both training scripts will now show:

```
============================================================
GPU CONFIGURATION
============================================================
PyTorch CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 3050 Laptop GPU
CUDA Version: 11.8
Requested GPUs: 1.0
============================================================

Training Physics PPO for 200 iterations
  Drones: 3
  Obstacles: 8
  Workers: 0 (0=local only)
  GPUs: 1.0
  Batch: 4000, Minibatch: 128
```

Then you'll see iteration progress:

```
iter=0001 reward_mean=  -45.123 len_mean=  85.50
iter=0002 reward_mean=  -42.891 len_mean=  92.30
...
```

---

## Troubleshooting

If you still see errors:

1. Check `nvidia-smi` shows your GPU
2. Verify PyTorch CUDA with: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try reducing batch size: `--train-batch-size 2000`

---

## My Recommendation

**Start with Physics Training** since that's what we've been working on:

```powershell
python scripts/train_physics.py --iterations 200
```

Let it run for at least 10-20 iterations to verify everything is stable!
