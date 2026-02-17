# MARL Drone Swarm Experiment Plan

This document is an execution plan to move from local prototype to CV-ready results with measurable milestones.

## 1. Target Outcomes

Primary project outcome:

- Train decentralized swarm policies with collision-aware coordination in 3D.

CV outcome:

- Report hard metrics, reproducible runs, and deployment artifacts (checkpoint + ONNX/TensorRT export later).

Minimum acceptance targets:

- Single-agent success rate >= 90 percent in static-obstacle scenes.
- Multi-agent (N=5) collision-free episode rate >= 80 percent in static-obstacle scenes.
- Multi-agent (N=8) collision-free episode rate >= 65 percent in harder scenes.
- Reproducibility: no more than 10 percent relative variation across 5 seeds for key metrics.

## 2. Metric Definitions

Use these exact metrics for all stages.

- Success Rate (SR):
  `SR = successful_episodes / total_episodes`

- Collision-Free Rate (CFR):
  `CFR = episodes_with_zero_collisions / total_episodes`

- Mean Time to Goal (TTG):
  `TTG = mean(first_goal_step over successful episodes)`

- Mean Formation Error (FE):
  `FE = mean_t mean_i |mean_{j!=i}(||p_i - p_j||) - d_star|`

- Path Efficiency (PE):
  `PE_i = ||start_i - goal||_2 / traveled_distance_i`
  `PE = mean_i(PE_i)`

- Reward Stability:
  Rolling standard deviation of episode reward over last K evaluations.

## 3. Experiment Logging Standard

Track every run with:

- `run_id`
- `date`
- `git_commit` (when git enabled)
- `seed`
- `env_config` (N, obstacles, world, dt, reward constants)
- `algo_config` (lr, batch sizes, gamma, net size)
- `train_iterations`
- `eval_episodes`
- `SR`, `CFR`, `TTG`, `FE`, `PE`
- notes (failures, anomalies, hypothesis)

Store outputs under:

- `checkpoints/`
- `reports/metrics/` (CSV or JSON summary)
- `reports/plots/` (learning curves)

## 4. Weekly Execution Plan (10 Weeks)

## Week 1: Environment Validation and Baseline Sanity

Goals:

- Verify dynamics, reward signs, observation dimensions, and termination behavior.

Tasks:

- Run smoke tests and add any missing edge-case tests.
- Manually inspect random rollouts for obvious reward bugs.
- Confirm seeds produce consistent resets/transitions.

Milestone gate:

- Zero API/test failures.
- Reward components behave as expected (progress positive when moving toward goal).

Deliverables:

- Test pass report.
- Short environment sanity note in `reports/week1.md`.

## Week 2: Single-Agent PPO Baseline

Goals:

- Train one drone to reach goal with obstacle avoidance.

Tasks:

- Run at least 5 seeds with same config.
- Plot reward trend and success trend.
- Tune only high-impact parameters first (`lr`, reward scales, max steps).

Milestone gate:

- SR >= 70 percent by end of week on held-out eval seeds.
- Collision rate trending downward.

Deliverables:

- Best checkpoint in `checkpoints/single_agent/`.
- Metrics table in `reports/metrics/week2_single_agent.csv`.

## Week 3: Single-Agent Ablations

Goals:

- Quantify what matters before moving to swarm training.

Tasks:

- Ablation A: remove progress reward.
- Ablation B: vary collision penalty magnitude.
- Ablation C: reduce obstacle sensing count.

Milestone gate:

- Clear ranking of reward terms and sensing features by impact on SR/CFR.

Deliverables:

- Ablation summary in `reports/week3_ablation.md`.
- One recommended single-agent config to freeze for Week 4+.

## Week 4: Multi-Agent Shared Policy (N=3)

Goals:

- Achieve stable cooperative behavior with parameter sharing.

Tasks:

- Train `N=3` with current multi-agent PPO setup.
- Validate neighbor feature utility by comparing `neighbor_k=0` vs default.
- Inspect failure modes: early collisions, clustering, deadlock.

Milestone gate:

- CFR >= 70 percent and SR >= 65 percent on evaluation suite.

Deliverables:

- Best multi-agent checkpoint (`N=3`).
- Failure taxonomy note in `reports/week4_multi_agent_n3.md`.

## Week 5: Curriculum Expansion (N=5, More Obstacles)

Goals:

- Scale policy without collapse.

Tasks:

- Introduce staged curriculum:

1. Start from easier scenario.
2. Increase obstacle density.
3. Increase drone count to `N=5`.

- Track per-stage metrics independently.

Milestone gate:

- CFR >= 80 percent for `N=5` static obstacles.
- No catastrophic forgetting when revisiting easier stage.

Deliverables:

- Curriculum schedule file `configs/curriculum_v1.yaml`.
- Metrics sheet with stage-wise results.

## Week 6: CTDE Prototype (Centralized Critic)

Goals:

- Add centralized-critic training while preserving decentralized actor observations.

Tasks:

- Use `info["global_state"]` as critic input.
- Implement and train CTDE variant (MADDPG-style or centralized-value PPO variant).
- Compare CTDE vs shared-PPO baseline on same seeds.

Milestone gate:

- CTDE improves at least one key metric (`SR` or `CFR`) by >= 5 absolute points.

Deliverables:

- CTDE training note `reports/week6_ctde.md`.
- Comparison table baseline vs CTDE.

## Week 7: Communication/Attention Enhancement

**Status: ✅ COMPLETED**

Goals:

- Improve scaling beyond local nearest-neighbor static encoding.

Tasks:

- ✅ Added multi-head attention mechanism to aggregate neighbor observations
- ✅ Integrated attention into `CentralizedCriticModel` actor network
- ✅ Created `train_attention.py` with GPU support

Milestone gate:

- ✅ Attention mechanism implemented and ready for training
- 🔄 Quantitative comparison pending (requires training run)

Deliverables:

- ✅ Model implementation in `src/swarm_marl/training/models.py`
- ✅ Training script `scripts/train_attention.py`
- 🔄 Performance comparison (pending)

## Week 8: High-Fidelity Physics Simulation

**Status: ✅ COMPLETED**

Goals:

- Bridge sim-to-real gap with physics-based dynamics.

Tasks:

- ✅ Implemented `DronePhysicsEnv` with PyBullet
- ✅ Added rigid body dynamics (gravity, drag, collisions)
- ✅ Created `train_physics.py` and `run_physics_gui.py`
- ✅ Verified physics simulation (gravity test passed)
- ✅ Added GPU support to all training scripts

Milestone gate:

- ✅ Physics environment verified and functional
- ✅ Visualization working
- 🔄 Training comparison pending

Deliverables:

- ✅ `src/swarm_marl/envs/drone_physics_env.py`
- ✅ `scripts/train_physics.py`
- ✅ `scripts/run_physics_gui.py`
- ✅ GPU optimization across all training scripts

## Week 9: Domain Randomization and Robustness

**Status: 🔄 IN PROGRESS**

Goals:

- Improve scaling beyond local nearest-neighbor static encoding.

Tasks:

- Add one communication mechanism:

1. Mean neighbor embedding, or
2. Attention-weighted neighbor aggregation.

- Visualize neighbor importance if attention is used.

Milestone gate:

- FE decreases and collision hot spots reduce in dense scenes.

Deliverables:

- Model diagram in `reports/figures/communication_block.png`.
- Quantitative delta report vs Week 6.

## Week 8: Domain Randomization and Robustness

Goals:

- Increase policy robustness for sim-to-real transfer.

Tasks:

- Randomize at least: mass scale, drag, thrust scale, sensor noise.
- Train with randomized parameters.
- Evaluate on unseen randomization ranges.

Milestone gate:

- Performance drop on unseen randomization <= 15 percent relative.

Deliverables:

- Randomization spec `configs/domain_randomization_v1.yaml`.
- Robustness report in `reports/week8_robustness.md`.

## Week 9: Isaac Sim / Isaac Lab Migration

Goals:

- Move from lightweight prototype env to high-fidelity physics.

Tasks:

- Implement interface-compatible task in Isaac stack.
- Match observation/action/reward semantics with this repo.
- Re-run key baseline and one CTDE model.

Milestone gate:

- Policy trains without API mismatch.
- At least one model reaches non-trivial SR (>40 percent) in Isaac scenes.

Deliverables:

- Migration note `reports/week9_isaac_migration.md`.
- Interface parity checklist.

## Week 10: Edge Export and Portfolio Packaging

Goals:

- Produce deployment and portfolio artifacts.

Tasks:

- Export best policy to ONNX.
- Benchmark inference latency on target edge profile (or emulated profile).
- Prepare final project report with method, math, experiments, and limitations.

Milestone gate:

- End-to-end demo clip + final metrics table complete.

Deliverables:

- `artifacts/model.onnx`
- `reports/final_report.md`
- `reports/final_metrics.csv`
- `media/demo.mp4`

## 5. Standard Experiment Matrix

Run this matrix at minimum:

| Stage             | N Drones | Obstacles | Seeds | Episodes Eval | Required Report |
| ----------------- | -------: | --------: | ----: | ------------: | --------------- |
| Single baseline   |        1 |         8 |     5 |           100 | Week 2          |
| Multi baseline    |        3 |         8 |     5 |           100 | Week 4          |
| Curriculum target |        5 |         8 |     5 |           100 | Week 5          |
| Hard setting      |        8 |        12 |     5 |           100 | Week 8+         |

## 6. Risk Register and Mitigation

Risk: reward hacking (agents exploit shaping).

- Mitigation: add metric-level checks (SR/CFR/TTG/FE), not reward alone.

Risk: MARL instability from non-stationarity.

- Mitigation: parameter sharing early, then CTDE.

Risk: scaling collapse at higher `N`.

- Mitigation: curriculum and communication module.

Risk: simulation gap to Isaac/realistic dynamics.

- Mitigation: preserve interfaces, apply domain randomization, then migrate.

## 7. Weekly Checklist Template

Use this checklist each week:

- [ ] Define hypothesis and success threshold before training.
- [ ] Run at least 3 seeds during development, 5 seeds for milestone claim.
- [ ] Save config and commit hash with every run.
- [ ] Evaluate on fixed holdout seeds.
- [ ] Document failures and next hypothesis.
- [ ] Update cumulative metrics table.

## 8. CV-Ready Evidence Bundle

To claim strong project impact, keep these artifacts:

- Reproducible config files.
- Multi-seed metrics CSV.
- Learning curves and collision analysis plots.
- One architecture diagram.
- One demo video.
- One deployment benchmark summary.

Use action-result phrasing in resume:

- "Developed..."
- "Implemented..."
- "Achieved X percent..."
- "Reduced collision rate by Y percent..."
