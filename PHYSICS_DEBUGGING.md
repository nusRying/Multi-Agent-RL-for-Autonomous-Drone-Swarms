# Physics Dashboard — Debugging Guide

> **Goal:** Get the `physics_run` checkpoint drones to hover stably and navigate to the green goal sphere in the interactive PyBullet dashboard.

---

## The Core Problem

The `physics_run` policy was **trained in a kinematic (zero-gravity) environment** (`DroneSwarmEnv`) and is now being **deployed in a physics-based world** (`DronePhysicsEnv`). The mismatch lies in how "action" is interpreted:

| Environment                     | How action works                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **DroneSwarmEnv** (training)    | `new_vel = clip(vel + action * max_accel * dt, max_speed)` — action is directly added as acceleration. No gravity. |
| **DronePhysicsEnv** (dashboard) | `applyExternalForce(mass * action * max_accel + gravity_comp)` — action must overcome gravity or it fails.         |

---

## Force Application Formula (Current Code)

```python
# In drone_physics_env.py → step() → sub-step loop:
mass = p.getDynamicsInfo(body, -1)[0]        # Actual mass (±10% randomized)
pos, _ = p.getBasePositionAndOrientation(body)

force = action * cfg.max_accel * mass        # Desired net force
force[2] += (mass * 9.5)                    # ← Gravity compensation

p.applyExternalForce(body, -1, force, pos, p.WORLD_FRAME)
#                                    ^^^
#                             COM position — NOT [0,0,0] !
```

---

## Symptom → Cause → Fix

### Symptom 1: Drones shoot upward uncontrollably

**Cause:** Gravity compensation (`g_comp`) is too high. When the policy
outputs an action designed to hover (e.g., action_z ≈ 0), the
`force[2] += mass * g_comp` still pushes them upward.

**Diagnostic:** What does the policy output for Z action?

```python
# Add this debug print in the loop in interactive_dashboard.py:
print(f"Action Z: {action_dict['drone_0'][2]:.3f}")
```

- If action_z ≈ 0 and drones still rise → `g_comp` is the culprit.
- If action_z ≈ 1.0, the policy is actively thrusting.

**Fix:** Reduce `g_comp` in `drone_physics_env.py`:

```python
force[2] += (mass * 9.0)   # Try 9.0, 8.5, 8.0 until hovering
```

---

### Symptom 2: Drones fall to the ground immediately

**Cause:** Gravity compensation is too low (or absent). The policy can't output enough force to overcome gravity.

**Fix:** Increase `g_comp`:

```python
force[2] += (mass * 9.8)   # Try 9.7, 9.75, 9.8
```

---

### Symptom 3: Drones spin/tumble wildly

**Cause:** Force is being applied at `[0, 0, 0]` (world origin) instead of the drone's position. This creates a torque arm equal to the drone's distance from the origin.

**Fix:** Verify this line in `drone_physics_env.py`:

```python
# WRONG:
p.applyExternalForce(body, -1, force, [0, 0, 0], p.WORLD_FRAME)

# CORRECT:
pos, _ = p.getBasePositionAndOrientation(body)
p.applyExternalForce(body, -1, force, pos, p.WORLD_FRAME)
```

---

### Symptom 4: "Not connected to physics server" crash

**Cause:** `p.getMouseEvents()` or `p.getKeyboardEvents()` is called without an active PyBullet window. This happens in `--headless` mode or after an earlier crash killed the window.

**Fix:** In `interactive_dashboard.py`, ensure these calls are guarded:

```python
mouse_events = []
if not args.headless:
    mouse_events = p.getMouseEvents()

if not args.headless:
    keys = p.getKeyboardEvents()
```

---

### Symptom 5: Drones move but don't reach the goal

**Cause A:** The policy never learned to navigate. Check training quality:

```bash
python -c "
import pandas as pd
df = pd.read_csv('reports/metrics/train_physics.csv')
print(df[['iteration','episode_reward_mean']].tail(20))
"
```

A well-trained policy should show `episode_reward_mean > 0` and trending upward.

**Cause B:** Observation mismatch. Verify both envs produce the same 37-dim obs:

```python
# In drone_physics_env.py → _get_obs():
# Layout must be: [pos(3) | vel(3) | goal_vec(3) | neighbor_K*4 | obstacle_M*4]
# In drone_swarm_env.py → step() → observation building: same layout
```

**Cause C:** Time-step mismatch. Verify `env_config["dt"] == 0.1` in the dashboard.

---

### Symptom 6: Animation too fast or too slow

**Cause:** The `time.sleep()` in the dashboard loop doesn't match the physics `dt`.

**Fix:**

```python
# dt = 0.1s per physics step → visualization must sleep 0.1s per frame
time.sleep(0.1)   # = 10 Hz real-time visualization
```

---

## Key File Locations

| File                                       | Role                                                      |
| ------------------------------------------ | --------------------------------------------------------- |
| `src/swarm_marl/envs/drone_physics_env.py` | **Physics simulation** — edit here for force/gravity      |
| `scripts/interactive_dashboard.py`         | **Dashboard** — edit here for timing, controls, inference |
| `src/swarm_marl/envs/common.py`            | **Config defaults** — `dt`, `max_accel`, `max_speed`      |
| `reports/metrics/train_physics.csv`        | **Training history** — check reward learning curves       |
| `checkpoints/physics_run/`                 | **Trained weights** — what the policy knows               |

---

## Step-by-Step Debugging Procedure

1. **Run headless stability test:**

   ```bash
   python scripts/interactive_dashboard.py --test --headless --checkpoint checkpoints/physics_run
   ```

   - Expected: `"Test completed successfully."` with no "WARNING: Drone rising" messages.

2. **If rising too high:** Reduce `g_comp` in `drone_physics_env.py → step()`:

   ```python
   force[2] += (mass * X.X)  # Decrease X.X by 0.2 each attempt
   ```

3. **If stable in headless but fails in GUI:**
   - Check if `p.getMouseEvents()` is guarded with `if not args.headless:`.
   - Try running with `--headless` always to eliminate GUI-related crashes.

4. **If drones don't move toward goal:**
   - Print policy actions: add `print(action_dict)` in the main loop.
   - If actions are all-zeros or random-looking, the checkpoint hasn't learned.
   - Check `reports/metrics/train_physics.csv`: does `episode_reward_mean` trend up?

5. **If unsatisfied with `physics_run` checkpoint:**
   - Retrain: `python scripts/train_physics.py --iterations 300`
   - Or test with the better-trained kinematic checkpoint: `--checkpoint checkpoints/ctde_run`
     (needs `--env-mode kinematic` flag in dashboard, or use kinematic fallback env).

---

## Gravity Compensation Tuning Table

| `g_comp` value | Effect at `action_z = 0`        | Stability                            |
| -------------- | ------------------------------- | ------------------------------------ |
| 9.81           | Perfect hover (no drift)        | Very sensitive to errors             |
| 9.5            | Slight sink (~0.03 m/s descent) | ✅ Recommended for training mismatch |
| 9.0            | Moderate sink (~0.8 m/s)        | Good if policy is strong flyer       |
| 8.0            | Rapid descent                   | Only if policy thrusts heavily       |
| 0.0            | Free fall (-9.81 m/s²)          | Drones crash immediately             |
