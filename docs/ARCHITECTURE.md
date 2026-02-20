# Project Architecture: Multi-Agent RL for Autonomous Drone Swarms

> **Version:** Phase 10 (Interactive Dashboard)  
> **Last Updated:** 2026-02-20  
> **Status:** Training Complete, Dashboard Under Debugging

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Environment Architecture](#3-environment-architecture)
   - [DroneSwarmEnv (Kinematic)](#31-droneswarmenv-kinematic)
   - [DronePhysicsEnv (PyBullet)](#32-dronephysicsenv-pybullet)
4. [Observation & Action Spaces](#4-observation--action-spaces)
5. [Training Pipeline](#5-training-pipeline)
   - [Model Architecture (CTDE)](#51-model-architecture-ctde)
   - [Configuration Builders](#52-configuration-builders)
   - [Callbacks](#53-callbacks)
6. [Interactive Dashboard](#6-interactive-dashboard)
   - [Physics Alignment Problem](#61-physics-alignment-problem)
   - [Known Issues & Debugging Guide](#62-known-issues--debugging-guide)
7. [Checkpoints](#7-checkpoints)
8. [Key Parameters Reference](#8-key-parameters-reference)
9. [Scripts Reference](#9-scripts-reference)

---

## 1. Project Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL)** system where a swarm of autonomous drones learns to navigate a 3D world, reach shared goals, and avoid obstacles. It progresses through 10 phases:

| Phase | Description                                          | Status    |
| ----- | ---------------------------------------------------- | --------- |
| 1     | Baseline Single-Agent PPO                            | ✅        |
| 2     | Multi-Agent Shared-Policy PPO                        | ✅        |
| 3     | CTDE (Centralized Training, Decentralized Execution) | ✅        |
| 4     | 3D Visualization                                     | ✅        |
| 5     | Communication & Attention Mechanism                  | ✅        |
| 6     | High-Fidelity PyBullet Physics Simulation            | ✅        |
| 7     | Training Optimization & Checkpoint Resumption        | ✅        |
| 8     | Domain Randomization & Robustness                    | ✅        |
| 9     | Evaluation & Final Report                            | ✅        |
| 10    | Interactive 3D Dashboard                             | 🔧 Active |

---

## 2. Repository Layout

```
Multi-Agent RL for Autonomous Drone Swarms/
│
├── src/swarm_marl/
│   ├── envs/
│   │   ├── common.py              # DroneEnvConfig dataclass (shared params)
│   │   ├── drone_swarm_env.py     # ← KINEMATIC env (used in training phases 1-5)
│   │   ├── drone_physics_env.py   # ← PHYSICS env (PyBullet, used in phases 6-10)
│   │   └── assets/
│   │       └── drone.urdf         # Drone rigid body definition (mass=1.0, inertia)
│   ├── training/
│   │   ├── models.py              # CentralizedCriticModel + AttentionBlock
│   │   ├── callbacks.py           # GlobalStateCallback (populates global state for CTDE critic)
│   │   └── config_builders.py     # build_ctde_ppo_config(), etc.
│   └── utils.py                   # extract_episode_stats()
│
├── scripts/
│   ├── train_physics.py           # Training script for physics_run checkpoint
│   ├── train_ctde.py              # Training script for ctde_run checkpoint
│   ├── interactive_dashboard.py   # ← MAIN DASHBOARD SCRIPT (Phase 10)
│   ├── verify_dashboard.py        # Headless dashboard stability test
│   ├── evaluate_protocol.py       # Evaluation framework
│   └── ...
│
├── checkpoints/
│   ├── physics_run/               # ← Main checkpoint used by dashboard
│   ├── ctde_run/
│   ├── attention_run/
│   └── ...
│
├── reports/
│   ├── final_report.md
│   └── metrics/
│       ├── train_physics.csv      # Physics training history (per-iteration rewards)
│       └── ...
│
└── docs/
    ├── ARCHITECTURE.md            # ← This file
    ├── EXPERIMENT_PLAN.md
    └── RUNBOOK.md
```

---

## 3. Environment Architecture

### 3.1 DroneSwarmEnv (Kinematic)

**File:** `src/swarm_marl/envs/drone_swarm_env.py`

This is the **original training environment**. It is purely kinematic — no rigid body physics, no gravity.

**Dynamics Model:**

```python
accel  = action * cfg.max_accel        # action ∈ [-1, 1]³
vel    = clip(vel + accel * dt, max_speed)
pos    = pos + vel * dt
```

**Key properties:**

- There is **no gravity**. The drones exist in a zero-G world.
- `dt = 0.1` seconds per step.
- `max_accel = 2.0 m/s²`
- `max_speed = 4.0 m/s` (hard clamp on velocity)
- Collisions are geometric distance checks (not physics).
- Episodes end on collision, goal-reach, or time limit.

**Why this matters for the dashboard:** The policy `physics_run` was trained in this environment. Its "mental model" is that action `a` produces `a * max_accel` net acceleration. There is **no concept of gravity** in the policy's experience.

---

### 3.2 DronePhysicsEnv (PyBullet)

**File:** `src/swarm_marl/envs/drone_physics_env.py`

This environment wraps PyBullet to add realistic rigid-body dynamics. It was designed to bridge the sim-to-real gap.

**Key challenges (Dashboard debugging):**

The fundamental problem is mapping a zero-G kinematic policy into a world with gravity. The approach:

```
Policy Action Output (a) ∈ [-1, 1]³
       ↓
Desired net acceleration = a * max_accel
       ↓
To achieve this in physics (with gravity):
F_applied = (a * max_accel * mass) + (mass * g_compensation)
```

**Current Implementation (as of Phase 10 debugging):**

```python
mass = p.getDynamicsInfo(body, -1)[0]
pos, _ = p.getBasePositionAndOrientation(body)
force = action * cfg.max_accel * mass
force[2] += (mass * 9.5)   # Under-compensate slightly to bias descent
p.applyExternalForce(body, -1, force, pos, p.WORLD_FRAME)  # Apply at COM!
```

**Sub-stepping:**
Because PyBullet's default time step is `1/240s (~4ms)`, and our `dt = 0.1s`, we call `stepSimulation()` 24 times per `env.step()` call:

```python
steps_per_call = int(cfg.dt * 240)  # = 24
for _ in range(steps_per_call):
    # apply force
    p.applyExternalForce(...)
    p.stepSimulation()
```

**Domain Randomization** (Phase 8):

```python
mass_noise = rng.uniform(0.9, 1.1)  # ±10% mass randomization
damp_noise = rng.uniform(0.8, 1.2)  # ±20% damping randomization
p.changeDynamics(body, -1, mass=1.0*mass_noise, linearDamping=0.5*damp_noise, ...)
```

---

## 4. Observation & Action Spaces

Both environments share identical observation/action spaces (this is intentional for policy transfer).

### Action Space

```
Box(low=-1.0, high=1.0, shape=(3,), dtype=float32)
# [Fx, Fy, Fz] normalized desired acceleration direction + magnitude
```

### Observation Space

```
Total dim = 9 + (neighbor_k * 4) + (sensed_obstacles * 4)
           = 9 + (3 * 4) + (4 * 4)
           = 9 + 12 + 16
           = 37  (default for 3 drones, 4 sensed obstacles)
```

**Observation breakdown:**
| Index | Content | Dim |
|-------|---------|-----|
| [0:3] | Own position `(x, y, z)` | 3 |
| [3:6] | Own velocity `(vx, vy, vz)` | 3 |
| [6:9] | Goal vector `(goal - pos)` | 3 |
| [9:21] | Nearest K neighbor features `[(rel_vec(3), dist(1)) × K]` | K×4 |
| [21:37] | Nearest M obstacle features `[(rel_vec(3), dist(1)) × M]` | M×4 |

---

## 5. Training Pipeline

### 5.1 Model Architecture (CTDE)

**File:** `src/swarm_marl/training/models.py`

**CentralizedCriticModel** (inherits `TorchModelV2`):

```
┌─────────────────────────────────────────┐
│              Actor (Policy)             │
│  Input: Local Observation (37 dims)     │
│  Architecture: FC[256, 256, ReLU]       │
│  Output: Action Logits (6 dims: mean+std)│
└─────────────────────────────────────────┘
                   ↓ (outputs action, for execution)

┌─────────────────────────────────────────┐
│              Critic (Value)             │
│  Input: Global State = concat[          │
│    All positions (N*3),                 │
│    All velocities (N*3),                │
│    Goal (3)                             │
│  ]  = (6N + 3) dims                    │
│  Architecture: FC[256, 256, ReLU]       │
│  Output: V(s) scalar                   │
└─────────────────────────────────────────┘
```

**Optional Attention (Phase 5):**

- When `use_attention=True`, the actor processes neighbor features through a `MultiheadAttention` block before the FC layers.
- Each neighbor is embedded to `32 dims`, attention is applied, and the context is concatenated with own-state features.

### 5.2 Configuration Builders

**File:** `src/swarm_marl/training/config_builders.py`

`build_ctde_ppo_config()` is the main config used by both `train_physics.py` and `interactive_dashboard.py`.

Key settings:

```python
PPOConfig()
  .environment(env=env_name, env_config=env_config)
  .framework("torch")
  .callbacks(GlobalStateCallback)           # Injects global state into infos
  .multi_agent(
      policies={"shared_policy": ...},      # All drones share one policy
      policy_mapping_fn=lambda id: "shared_policy",
  )
  .training(
      gamma=0.99,
      lr=3e-4,
      train_batch_size=4000,
      ...
  )
```

### 5.3 Callbacks

**File:** `src/swarm_marl/training/callbacks.py`

`GlobalStateCallback` is called by RLlib's `on_postprocess_trajectory()` hook. It copies `global_state` from `infos` into each agent's sample batch so the centralized critic can access shared information during training.

---

## 6. Interactive Dashboard

**File:** `scripts/interactive_dashboard.py`

### Architecture

```
main()
  │
  ├─ parse_args()
  │   ⟹ --checkpoint, --num-drones, --attention, --test, --headless
  │
  ├─ build_ctde_ppo_config(env_config)    # Build algorithm with same config as training
  ├─ algo.restore(checkpoint_dir)          # Load trained weights
  │
  ├─ env = DronePhysicsEnv(env_config)    # Create PyBullet env (GUI or Direct)
  │     OR env = algo.workers.local_worker().env
  │
  ├─ obs, info = env.reset()
  │
  └─ LOOP:
      ├─ Handle Keyboard Events (Q=quit, R=reset)
      ├─ Handle Mouse Events (click to move goal sphere)
      ├─ Compute Actions: algo.compute_single_action(obs, "shared_policy")
      ├─ obs, rewards, terminated, truncated, info = env.step(action_dict)
      ├─ Print telemetry every 60 frames
      ├─ Auto-reset on episode end
      └─ sleep(0.1)   # 10 Hz visualization loop
```

### 6.1 Physics Alignment Problem

**Root Cause:** The trained policy (`physics_run`) was trained in `DroneSwarmEnv` (kinematic, zero-G). The dashboard runs `DronePhysicsEnv` (physics, real gravity).

**Policy's learned behavior:** it outputs actions as if `action = net_acceleration`. In zero-G, this works perfectly.

**In physics world:** We must:

1. Convert `action → force = action * mass * max_accel`
2. Add gravity compensation: `force[2] += mass * g_comp`
3. Apply at Center of Mass (not world origin!)
4. Apply force at every physics sub-step, not just once

**The Gravity Comp Dilemma:**

| g_comp | Effect                                                                   |
| ------ | ------------------------------------------------------------------------ |
| 9.81   | Exact compensation, but any positive action → drones rise uncontrollably |
| 9.5    | Slight under-compensation ("heavy bias"), drones sink if action ≈ 0      |
| 0.0    | No compensation, drones fall like rocks (policy can't overcome gravity)  |

**Current status:** Using `g_comp = 9.5`. This is a compromise. If drones still rise, reduce to `9.0` or `8.5`. If they fall too fast, increase to `9.6` or `9.7`.

---

### 6.2 Known Issues & Debugging Guide

#### Issue A: "Drones flying away / shooting up"

**Cause:** Gravity compensation too high OR force applied at wrong position.

**Check:** In `drone_physics_env.py` → `step()`, verify:

```python
# This line must use 'pos', NOT [0,0,0]
p.applyExternalForce(body, -1, force, pos, p.WORLD_FRAME)
```

**Fix:** Reduce `force[2] += mass * g_comp` — lower `g_comp` value.

---

#### Issue B: "Drones falling / dropping"

**Cause:** Gravity compensation too low OR not applied at every sub-step.

**Check:** In the sub-step loop, ensure `applyExternalForce` is called **inside** the `for _ in range(steps_per_call)` loop.

**Fix:** Increase `g_comp` value toward 9.81.

---

#### Issue C: "Not connected to physics server" Crash

**Cause:** `p.getMouseEvents()` or `p.getKeyboardEvents()` called when no GUI window is active (headless mode or PyBullet crash).

**Fix:** These calls must be guarded by `if not args.headless:`.

---

#### Issue D: "KeyError: distance_to_goal"

**Cause:** `_get_infos()` not returning this key. It was previously only returning `global_state`.

**Fix:** `_get_infos()` must return:

```python
infos[agent_id] = {
    "global_state": global_state,
    "distance_to_goal": float(dist),
    "reached_goal": dist < cfg.goal_radius,
    "collision": False  # updated later in step()
}
```

---

#### Issue E: "Drones don't move toward green sphere"

**Cause A:** Observation mismatch — the `DronePhysicsEnv` may compute observations differently from `DroneSwarmEnv`.

**Check:** Verify `_build_obs()` in both files produces arrays in the same order:

```
[pos(3), vel(3), goal_vec(3), neighbor_feats(K*4), obstacle_feats(M*4)]
```

**Cause B:** Poor training — the `physics_run` checkpoint may not be well-trained enough to reach the goal in physics conditions. Check `reports/metrics/train_physics.csv` for reward curves. If `episode_reward_mean` is near 0 or negative, the policy hasn't learned.

**Cause C:** Time-step mismatch between training (`dt=0.1`) and inference. Ensure dashboard passes `"dt": 0.1` in `env_config`.

---

## 7. Checkpoints

| Checkpoint                       | Environment     | Description                                            |
| -------------------------------- | --------------- | ------------------------------------------------------ |
| `checkpoints/physics_run/`       | DronePhysicsEnv | Main checkpoint for dashboard. Trained with PyBullet.  |
| `checkpoints/ctde_run/`          | DroneSwarmEnv   | CTDE-PPO without attention.                            |
| `checkpoints/attention_run/`     | DroneSwarmEnv   | CTDE-PPO with attention mechanism.                     |
| `checkpoints/multi_baseline_n3/` | DroneSwarmEnv   | Shared-PPO baseline, N=3 drones, evaluated at N=3/5/8. |

**How to inspect a checkpoint:**

```bash
# See what the policy learned — run evaluation
python scripts/evaluate_protocol.py --mode ctde --checkpoint checkpoints/physics_run --env-mode physics
```

**How to resume training:**

```bash
python scripts/train_physics.py --iterations 200
# Automatically detects existing checkpoint and resumes
```

---

## 8. Key Parameters Reference

All default values are in `src/swarm_marl/envs/common.py`:

```python
@dataclass
class DroneEnvConfig:
    world_size: float = 20.0    # World bounds: [-10, 10] in each axis
    dt: float = 0.1             # Simulation time step (seconds)
    max_steps: int = 400        # Max steps per episode
    max_speed: float = 4.0      # Hard velocity cap (m/s)
    max_accel: float = 2.0      # Max acceleration (maps action=1 → 2 m/s²)
    collision_radius: float = 0.5  # Drone collision radius
    goal_radius: float = 0.8    # Distance to trigger goal-reached
    num_obstacles: int = 8      # Number of random obstacles per episode
    sensed_obstacles: int = 4   # Number of nearest obstacles in observation
    neighbor_k: int = 3         # Number of nearest drones in observation
    obstacle_radius: float = 0.8
    desired_spacing: float = 2.5
    reward_progress_scale: float = 2.0  # Progress reward multiplier
    reward_goal: float = 25.0           # Bonus for reaching goal
    reward_collision: float = -25.0     # Penalty for collision
    reward_formation_scale: float = 0.15
```

---

## 9. Scripts Reference

| Script                                       | Purpose                             | Usage                                                                                            |
| -------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| `train_physics.py`                           | Train in PyBullet env               | `python scripts/train_physics.py --iterations 100`                                               |
| `train_ctde.py`                              | Train CTDE in kinematic env         | `python scripts/train_ctde.py --iterations 200`                                                  |
| `interactive_dashboard.py`                   | Run 3D interactive dashboard        | `python scripts/interactive_dashboard.py --checkpoint checkpoints/physics_run`                   |
| `interactive_dashboard.py --test --headless` | Headless stability test (100 steps) | `python scripts/interactive_dashboard.py --test --headless --checkpoint checkpoints/physics_run` |
| `evaluate_protocol.py`                       | Multi-swarm evaluation              | `python scripts/evaluate_protocol.py --mode ctde --checkpoint checkpoints/ctde_run`              |
| `run_full_benchmark.py`                      | Run all benchmarks                  | `python scripts/run_full_benchmark.py`                                                           |
| `plot_comparison.py`                         | Generate comparison plots           | `python scripts/plot_comparison.py`                                                              |
| `export_onnx.py`                             | Export policy to ONNX               | `python scripts/export_onnx.py --checkpoint checkpoints/ctde_run`                                |
| `visualize_swarm.py`                         | Matplotlib 3D visualization         | `python scripts/visualize_swarm.py --checkpoint checkpoints/ctde_run`                            |
| `verify_dashboard.py`                        | Headless physics test (no RLlib)    | `python scripts/verify_dashboard.py`                                                             |

---

## 10. Quick Debugging Checklist

If the dashboard drones do not behave correctly:

- [ ] **Headless test passes:** `python scripts/interactive_dashboard.py --test --headless ...` → "Test completed successfully."
- [ ] **dt matches training:** `env_config["dt"] == 0.1` in both training script and dashboard.
- [ ] **Force applied at COM:** `p.applyExternalForce(..., pos, p.WORLD_FRAME)` — not `[0,0,0]`.
- [ ] **Force applied every sub-step:** `applyExternalForce` is **inside** the `for _ in range(steps_per_call)` loop.
- [ ] **Training was successful:** Check `reports/metrics/train_physics.csv` — reward should be consistently positive (>0) and trending up.
- [ ] **Observation dimension matches:** Both envs produce `37`-dim observations for default config.
- [ ] **Velocity is clamped:** After `applyExternalForce`, call `p.resetBaseVelocity` if speed exceeds `max_speed`.
- [ ] **Gravity compensation:** Value in `force[2] += mass * g_comp` — adjust between `9.0` and `9.81` for stable hover.
