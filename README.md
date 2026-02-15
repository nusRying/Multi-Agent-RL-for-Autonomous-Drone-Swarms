# Multi-Agent Reinforcement Learning for Autonomous Drone Swarms

This repository is a full starter pipeline for a CV-grade MARL project:
- Phase A: single-agent control sanity check.
- Phase B: multi-agent swarm coordination with local observations.
- Phase C: upgrade to CTDE (centralized training, decentralized execution).
- Phase D: migration to high-fidelity simulation (Isaac Sim / Isaac Lab).
- Phase E: sim-to-real and edge deployment preparation.

The code currently implements a mathematically clean 3D point-mass swarm environment and RLlib-ready training scripts. This README explains the complete pipeline, core equations, and rationale for each design choice.

Detailed execution tracker:
- `docs/EXPERIMENT_PLAN.md` (10-week milestones, metric gates, experiment matrix, risk register)

Experiment templates:
- `reports/templates/WEEK_REPORT_TEMPLATE.md`
- `reports/templates/FINAL_REPORT_TEMPLATE.md`
- `reports/metrics/run_log_template.csv`
- `reports/metrics/final_metrics_template.csv`
- `reports/plots/PLOT_TEMPLATES.md`
- `configs/curriculum_v1.yaml`
- `configs/domain_randomization_v1.yaml`
- `configs/eval_protocol_v1.yaml`

## 1. Repository Scope and Design Philosophy

This starter intentionally separates concerns:
- Environment dynamics are simple and explicit so reward shaping and policy behavior are easy to debug.
- Training plumbing is isolated in config builders/scripts so algorithms can be swapped later.
- CTDE hooks are included now (`global_state`) to avoid refactoring later.

Reason for this structure:
- In swarm RL, failed projects often come from hidden coupling between environment logic and training code.
- Strict interfaces (`obs`, `action`, `reward`, `done`, `info`) make migration to Isaac Sim significantly easier.

## 2. End-to-End Pipeline (Complete Roadmap)

### Phase 0: Setup and Validation
1. Create environment and install only lightweight dependencies first.
2. Run smoke tests to validate API contracts (observation shapes, action handling, done flags).
3. Confirm deterministic behavior with fixed seeds.

Why:
- Early shape/API bugs cause long training runs to fail late. Smoke tests protect iteration speed.

### Phase 1: Single-Agent Baseline
1. Train one drone with PPO to reach a target with obstacle avoidance.
2. Verify stable learning curves (reward up, collisions down).
3. Freeze baseline as a control for future multi-agent regressions.

Why:
- If one drone cannot learn in your dynamics loop, multi-agent training will hide root issues.

### Phase 2: Multi-Agent Shared Policy Baseline
1. Enable `N` drones in the same world.
2. Use one shared policy across all agents (parameter sharing).
3. Add cooperative formation regularization and collision penalties.

Why:
- Parameter sharing improves sample efficiency and enforces behavioral consistency among homogeneous drones.

### Phase 3: CTDE Upgrade
1. Keep actor observations local (already true in this code).
2. Use global state only in critic during training.
3. Move from shared PPO to centralized-critic methods (for example MADDPG-style critics).

Why:
- CTDE helps resolve partial observability and MARL non-stationarity while preserving decentralized inference.

### Phase 4: Curriculum and Robustness
1. Start with low drone count and sparse obstacles.
2. Gradually increase swarm size and obstacle complexity.
3. Add domain randomization (mass, drag, thrust scaling, sensor noise).

Why:
- Curriculum reduces optimization instability.
- Domain randomization reduces overfitting to simulator-specific dynamics.

### Phase 5: Deployment and CV Packaging
1. Export trained policy to ONNX/TensorRT.
2. Measure latency and throughput on edge-like hardware.
3. Report reproducible metrics with confidence intervals.

Why:
- Deployment evidence is a major CV differentiator compared to simulation-only work.

## 3. Mathematical Formulation (MDP / Markov Game)

For multi-agent swarms, the problem is a cooperative partially observable Markov game.

### 3.1 State Variables

At time step `t`:
- Drone `i` position: `p_i^t in R^3`
- Drone `i` velocity: `v_i^t in R^3`
- Goal position: `g in R^3`
- Obstacle centers: `o_m in R^3`, `m = 1..M`

Global state:
`s^t = {p_1^t,...,p_N^t, v_1^t,...,v_N^t, g}`

Code mapping:
- `SingleDroneEnv`: `position`, `velocity`, `goal`, `obstacles`
- `DroneSwarmEnv`: `positions`, `velocities`, `goal`, `obstacles`

### 3.2 Action Space

Each agent outputs normalized acceleration:
`u_i^t in [-1,1]^3`

Physical acceleration:
`a_i^t = a_max * u_i^t`

Reason:
- Continuous control is closer to real drone control than discrete actions.
- Normalized actions simplify policy optimization and transfer across scales.

### 3.3 Transition Dynamics

Given timestep `dt`:

1. Velocity update:
`v_i^(t+1) = v_i^t + a_i^t * dt`

2. Speed clipping:
`v_i^(t+1) <- min(1, v_max / ||v_i^(t+1)||_2) * v_i^(t+1)`

3. Position update:
`p_i^(t+1) = p_i^t + v_i^(t+1) * dt`

4. World boundary clipping:
`p_i^(t+1) <- clip(p_i^(t+1), -W/2, W/2)`

Reason:
- Point-mass dynamics are intentionally simple for fast experimentation.
- Speed clipping prevents policy exploitation via unbounded acceleration.
- Boundary clipping avoids undefined out-of-bounds states.

### 3.4 Observation Model

Actors are decentralized: each drone receives local observation `obs_i^t`.

#### Single-Agent Observation
`obs^t = [p^t, v^t, (g - p^t), obstacle_features]`

Obstacle features use nearest `K_obs` obstacles:
- For each nearest obstacle `m`: `[delta_x, delta_y, delta_z, distance]`
- Zero-pad if fewer are available.

Dimension:
`dim(obs_single) = 9 + 4*K_obs`

#### Multi-Agent Observation
For agent `i`:
`obs_i^t = [p_i^t, v_i^t, (g - p_i^t), neighbor_features_i, obstacle_features_i]`

Neighbor features use nearest `K_n` neighbors:
- For each nearest neighbor `j`: `[p_j - p_i, ||p_j - p_i||_2]`

Obstacle features are identical in form to single-agent.

Dimension:
`dim(obs_multi) = 9 + 4*K_n + 4*K_obs`

Reason:
- Nearest-neighbor encoding scales better than full all-to-all concatenation.
- Relative vectors provide translation invariance.
- Local observations enforce realistic decentralized inference.

### 3.5 Reward Function

For agent `i`, define:
- Goal distance: `d_i^t = ||g - p_i^t||_2`
- Progress reward:
`r_progress_i^t = k_p * (d_i^(t-1) - d_i^t)`

If drone reaches goal:
`I_goal_i^t = 1[d_i^t <= r_goal]`

If collision occurs:
`I_col_i^t = 1[collision_i^t]`

Formation error:
- Pairwise distances from `i` to others: `D_ij^t = ||p_i^t - p_j^t||_2`
- Desired spacing: `d_star`
- Mean absolute spacing deviation:
`e_form_i^t = mean_{j!=i} |D_ij^t - d_star|`
- Formation reward:
`r_form_i^t = -k_f * e_form_i^t`

Total reward:
`r_i^t = r_progress_i^t + r_form_i^t + I_goal_i^t * R_goal + I_col_i^t * R_collision`

Default constants (`DroneEnvConfig`):
- `k_p = 2.0`
- `R_goal = +25.0`
- `R_collision = -25.0`
- `k_f = 0.15`

Reason:
- Progress shaping reduces sparse-reward failures.
- Strong collision penalty enforces safety as a first-class objective.
- Formation term regularizes swarm coherence without dominating the mission objective.

### 3.6 Collision and Termination Logic

Obstacle collision if:
`||p_i - o_m||_2 <= (r_collision_drone + r_obstacle)`

Drone-drone collision if:
`||p_i - p_j||_2 <= 2*r_collision_drone`

Per-agent terminal:
`done_i = reached_goal_i OR collided_i`

Episode terminal (`__all__` in multi-agent):
`done_episode = all_agents_reached_goal OR any_collision`

Time truncation:
`truncated = (t >= max_steps) AND NOT done_episode`

Reason:
- Ending episode on any collision enforces strict cooperative safety.
- Splitting `terminated` and `truncated` preserves RL semantics.

### 3.7 CTDE Signal Already Present

`info["global_state"] = concat(positions, velocities, goal)`

Reason:
- Enables centralized critics later without changing actor observation schema.

## 4. Why PPO Baseline First (Before MADDPG)

Current training scripts use PPO:
- `scripts/train_single_agent.py`
- `scripts/train_multi_agent.py`

PPO clipped objective:
`L_clip(theta) = E_t[min(r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t)]`

Policy ratio:
`r_t(theta) = pi_theta(a_t | o_t) / pi_theta_old(a_t | o_t)`

Advantage (GAE form):
`A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}`
`delta_t = r_t + gamma * V(o_{t+1}) - V(o_t)`

Reason for PPO baseline:
- Higher stability during early reward/dynamics tuning.
- Lower sensitivity to deterministic-policy exploration noise.
- Faster debugging before CTDE-specific complexity.

## 5. CTDE and MADDPG Upgrade Path (Math + Architecture)

When upgrading to MADDPG-style CTDE:

### 5.1 Actor-Critic Structure
- Actor per agent (or shared actor for homogeneous drones):
`mu_i(o_i) -> a_i`
- Centralized critic:
`Q_i(s, a_1,...,a_N)`

### 5.2 Critic Target
`y_i = r_i + gamma * Q_i'(s', a_1',...,a_N')`
where `a_j' = mu_j'(o_j')`

### 5.3 Critic Loss
`L_i(phi_i) = E[(Q_i(s, a_1,...,a_N) - y_i)^2]`

### 5.4 Actor Gradient
`grad_{theta_i} J ~= E[grad_{a_i} Q_i(s,a_1,...,a_N) * grad_{theta_i} mu_i(o_i)]`

Reason:
- Centralized critic handles non-stationary teammate policies.
- Decentralized actors keep inference deployment-friendly.

## 6. Hyperparameters and Rationale

### Environment Defaults (`DroneEnvConfig`)
- `world_size=20.0`: enough room for maneuvering, avoids trivial near-contact starts.
- `dt=0.1`: stable integration at manageable control frequency.
- `max_speed=4.0`, `max_accel=2.0`: bounded kinematics for stable optimization.
- `max_steps=400`: enough horizon for coordination plus obstacle avoidance.
- `collision_radius=0.5`, `obstacle_radius=0.8`: conservative safety geometry.
- `num_obstacles=8`: non-trivial but learnable obstacle density.
- `neighbor_k=3`: scalable local coordination signal.
- `sensed_obstacles=4`: relevant obstacle context without exploding observation size.
- `desired_spacing=2.5`: practical separation for collision margin.

### PPO Defaults (`config_builders.py`)
- `gamma=0.99`: long-horizon credit assignment.
- `lr=3e-4`: robust default for PPO in continuous control.
- `fcnet_hiddens=[256,256]`: enough capacity for nonlinear navigation behavior.
- Single-agent batch: `8192` and minibatch `1024`.
- Multi-agent batch: `16384` and minibatch `2048`.

Reason for larger multi-agent batch:
- Joint dynamics increase gradient variance; larger batches stabilize updates.

## 7. Data Flow Through the Codebase

1. Training script registers environment in RLlib.
2. RLlib calls `reset()` and receives per-agent observations.
3. Policy outputs continuous actions.
4. Environment updates dynamics, rewards, and done flags.
5. RLlib collects trajectories and optimizes policy.
6. Checkpoints are written under `checkpoints/`.

Core files:
- `scripts/train_single_agent.py`
- `scripts/train_multi_agent.py`
- `scripts/train_curriculum.py`
- `scripts/evaluate_protocol.py`
- `src/swarm_marl/envs/single_drone_env.py`
- `src/swarm_marl/envs/drone_swarm_env.py`
- `src/swarm_marl/training/config_builders.py`

## 8. Evaluation Protocol and Metrics

Run evaluation with fixed seeds and deterministic policy inference.

Primary metrics:
- Success rate:
`SR = episodes(all goals reached, no collisions) / total episodes`
- Collision-free rate:
`CFR = episodes(zero collisions) / total episodes`
- Mean time-to-goal:
`TTG = mean(first goal-reach step over successful episodes)`
- Mean formation error:
`FE = mean_t mean_i e_form_i^t`
- Path efficiency:
`PE_i = ||start_i - goal||_2 / traveled_distance_i`

Reporting standard:
- Mean plus standard deviation across at least 5 seeds.
- Curves for reward, success, and collision count.

## 9. Curriculum Strategy (Recommended)

Example schedule:
1. Stage 1: `N=3`, `num_obstacles=0`
2. Stage 2: `N=3`, `num_obstacles=4`
3. Stage 3: `N=5`, `num_obstacles=8`
4. Stage 4: `N=8`, dynamic obstacles

Transition criterion:
- Advance when rolling success rate exceeds threshold (for example 85 percent).

Reason:
- Reduces early optimization collapse from combinatorial task difficulty.

## 10. Isaac Sim / Isaac Lab Migration Plan

Keep interfaces invariant:
- Same action semantics.
- Same local observation schema.
- Same reward decomposition.

Migration sequence:
1. Implement Isaac task exposing equivalent `obs/reward/done/info`.
2. Compare observation and action distributions between toy and Isaac envs.
3. Re-tune reward scales under realistic rigid-body dynamics.
4. Resume curriculum and domain randomization.

Reason:
- Interface consistency lets you keep training code and experiment logic unchanged.

## 11. Sim-to-Real Robustness Objective

Domain randomization samples simulator parameters:
`xi ~ P(xi)`

Optimize expected return over randomized dynamics:
`max_pi E_{xi~P}[E_{tau~pi,xi}[sum_t gamma^t r_t]]`

Reason:
- Policy learns invariances instead of overfitting to one simulator instance.

## 12. Installation and Usage (Deferred-Friendly)

You can keep this file-only setup now and install later when ready.

Base dependencies:
```powershell
python -m pip install -r requirements.txt
```

Dev tools:
```powershell
python -m pip install -r requirements-dev.txt
```

RLlib later:
```powershell
python -m pip install -r requirements-rllib.txt
```

Run smoke tests:
```powershell
pytest
```

Train single-agent baseline:
```powershell
python scripts/train_single_agent.py --iterations 50
```

Train multi-agent baseline:
```powershell
python scripts/train_multi_agent.py --iterations 100 --num-drones 5
```

Train with curriculum config:
```powershell
python scripts/train_curriculum.py --config configs/curriculum_v1.yaml
```

Evaluate a checkpoint against protocol scenarios:
```powershell
python scripts/evaluate_protocol.py --checkpoint checkpoints/multi_agent/<checkpoint_dir> --protocol configs/eval_protocol_v1.yaml
```

## 13. Project Structure

```text
.
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
|-- requirements-rllib.txt
|-- scripts/
|   |-- train_single_agent.py
|   |-- train_multi_agent.py
|   |-- train_curriculum.py
|   `-- evaluate_protocol.py
|-- docs/
|   `-- EXPERIMENT_PLAN.md
|-- configs/
|   |-- curriculum_v1.yaml
|   |-- domain_randomization_v1.yaml
|   `-- eval_protocol_v1.yaml
|-- reports/
|   |-- templates/
|   |   |-- WEEK_REPORT_TEMPLATE.md
|   |   `-- FINAL_REPORT_TEMPLATE.md
|   |-- metrics/
|   |   |-- run_log_template.csv
|   |   `-- final_metrics_template.csv
|   `-- plots/
|       `-- PLOT_TEMPLATES.md
|-- src/
|   `-- swarm_marl/
|       |-- envs/
|       |   |-- common.py
|       |   |-- single_drone_env.py
|       |   `-- drone_swarm_env.py
|       |-- utils/
|       |   `-- config_io.py
|       `-- training/
|           `-- config_builders.py
`-- tests/
    |-- conftest.py
    `-- test_env_smoke.py
```

## 14. Current Limitations

- Dynamics are point-mass, not full rigid-body quadrotor physics.
- No explicit inter-agent communication module yet.
- PPO baseline does not yet include a centralized critic implementation.
- Evaluation script assumes shared-policy setup for multi-agent PPO checkpoints.

These limitations are intentional for rapid iteration before Isaac Sim migration.

## 15. CV Framing Template

Use action-result statements with hard metrics:

`Developed a decentralized multi-agent drone swarm controller with local-observation policies and CTDE-ready architecture, achieving X percent collision-free navigation and Y percent goal attainment in 3D obstacle-rich simulation; exported inference-ready policy artifacts for edge deployment benchmarking.`
