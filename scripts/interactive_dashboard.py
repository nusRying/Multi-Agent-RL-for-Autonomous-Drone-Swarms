"""
Interactive 3D Drone Swarm Dashboard
=====================================
Launches a PyBullet GUI window and runs the trained RL policy in real-time.

Usage:
    # Standard GUI run (requires trained checkpoint)
    python scripts/interactive_dashboard.py --checkpoint checkpoints/physics_run

    # Headless stability test run (100 steps, no GUI)
    python scripts/interactive_dashboard.py --test --headless --checkpoint checkpoints/physics_run

    # Run with attention-based policy
    python scripts/interactive_dashboard.py --checkpoint checkpoints/attention_run --attention

Controls (GUI mode):
    Q key       → Quit simulation
    R key       → Reset episode (new random positions + goal)
    Left Click  → Drag green sphere to new goal location

Dependencies:
    - Ray / RLlib (registered for policy inference)
    - PyBullet (simulation + GUI)
    - DronePhysicsEnv (physics environment)
    - DroneEnvConfig (default configuration)

Architecture:
    main()
      ├─ parse_args()                        # CLI argument parsing
      ├─ register_env(DronePhysicsEnv)        # Register with Ray Tune
      ├─ build_ctde_ppo_config(env_config)    # Build algo config matching training
      ├─ algo.restore(checkpoint)             # Load trained policy weights
      ├─ env = DronePhysicsEnv(env_config)    # Create physics simulation
      └─ LOOP (10 Hz):
          ├─ Handle keyboard events (Q/R)
          ├─ Handle mouse click to drag green goal sphere
          ├─ algo.compute_single_action(obs, "shared_policy")
          ├─ env.step(action_dict)
          ├─ Log telemetry every 60 frames
          ├─ Auto-reset on episode end
          └─ sleep(0.1s)   # Match dt=0.1s for real-time speed

Physics Alignment:
    Training used DroneSwarmEnv (kinematic, zero-G).
    Dashboard uses DronePhysicsEnv (PyBullet, gravity = -9.81 m/s²).
    Key mapping: force = action * max_accel * mass + (mass * g_comp)
    Applied at drone's Center of Mass (NOT world origin [0,0,0]).

Common Issues:
    "Drones flying away"     → g_comp too high in drone_physics_env.py (reduce from 9.5 toward 9.0)
    "Drones on ground"       → g_comp too low (increase toward 9.81)
    "Not connected to server"→ p.getMouseEvents() called without GUI window
    "KeyError: distance_to_goal" → _get_infos() missing the key in DronePhysicsEnv
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

# ── Environment Variables ─────────────────────────────────────────────────────
# Prevent Ray from importing TensorFlow or JAX (we use PyTorch only)
os.environ.setdefault("RLLIB_TEST_NO_TF_IMPORT", "1")
os.environ.setdefault("RLLIB_TEST_NO_JAX_IMPORT", "1")

# ── Python Path Setup ──────────────────────────────────────────────────────────
# Makes `src/` importable without installing as a package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs.drone_physics_env import DronePhysicsEnv
from swarm_marl.training.config_builders import build_ctde_ppo_config

try:
    import ray
    from ray.tune.registry import register_env
except ImportError:
    print("Ray/RLlib not found. Run: pip install -r requirements-rllib.txt")
    sys.exit(1)


def parse_args():
    """
    Parse command-line arguments for the dashboard.

    Returns:
        argparse.Namespace with the following fields:
            checkpoint (str): Path to checkpoint directory.
            num_drones (int): Number of drones in the swarm (default: 3).
            attention (bool): Enable attention-based actor network if True.
            mode (str): Algorithm mode; "ctde" or "physics" (default: "physics").
            test (bool): If True, run for 100 steps then exit cleanly.
            headless (bool): If True, skip GUI; run in DIRECT mode (faster, no display).
    """
    parser = argparse.ArgumentParser(description="Interactive 3D Drone Swarm Dashboard")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/physics_run",
                        help="Path to checkpoint directory (e.g. checkpoints/physics_run)")
    parser.add_argument("--num-drones", type=int, default=3,
                        help="Number of drones in the swarm (default: 3)")
    parser.add_argument("--attention", action="store_true",
                        help="Enable Attention architecture if checkpoint trained with it")
    parser.add_argument("--mode", choices=["ctde", "physics"], default="physics",
                        help="Algorithm mode (default: physics)")
    parser.add_argument("--test", action="store_true",
                        help="Run a short 100-step stability test and exit")
    parser.add_argument("--headless", action="store_true",
                        help="Run without PyBullet GUI (faster, no display)")
    return parser.parse_args()


def _get_mouse_ray(client_id):
    """
    (Unused) Cast a ray from camera through mouse cursor for 3D picking.

    Placeholder for future advanced mouse interaction.
    Currently, goal dragging uses a simpler screen-to-world plane intersection.
    """
    events = p.getMouseEvents(client_id)
    if not events:
        return None, None
    for event in events:
        if event[0] == 2:  # Mouse drag/move event
            pass
    return None, None


def main():
    """
    Main entry point for the interactive dashboard.

    Flow:
        1. Parse CLI arguments.
        2. Set up env_config matching the training parameters EXACTLY.
           ⚠️  dt=0.1 is critical — matches train_physics.py default.
        3. Initialize Ray and build the PPO algorithm.
        4. Restore trained checkpoint weights.
        5. Get (or create) a DronePhysicsEnv instance.
        6. Enter the main interaction loop.

    Exit conditions:
        - User presses 'Q' (GUI mode)
        - args.test=True and 100 frames elapsed
        - Exception in simulation

    After exit, algo.stop(), ray.shutdown(), p.disconnect() are called.
    """
    args = parse_args()

    env_name = "drone_physics_interactive_v0"

    # ── Environment Config ────────────────────────────────────────────────────
    # ⚠️  IMPORTANT: dt=0.1 MUST match train_physics.py's default (which doesn't
    # explicitly set dt → uses DroneEnvConfig default of 0.1).
    # If dt mismatches, the policy's temporal reasoning is wrong.
    env_config = {
        "num_drones": args.num_drones,
        "gui": not args.headless,
        "world_size": 10.0,
        "max_steps": 10000,  # Long episode for interactive exploration
        "dt": 0.1,           # Must match training config!
    }

    register_env(env_name, lambda cfg: DronePhysicsEnv(cfg))

    # ── Ray Init ──────────────────────────────────────────────────────────────
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # ── Build Algorithm ───────────────────────────────────────────────────────
    # Must use the SAME config structure as train_physics.py for compatible checkpoint loading.
    config = build_ctde_ppo_config(
        env_name=env_name,
        env_config=env_config,
        num_workers=0,  # 0 workers = local rollouts only (required for PyBullet)
    )

    # Optional: Re-enable attention if checkpoint was trained with it
    if args.attention:
        policy_spec = config.policies["shared_policy"]
        policy_spec.config["model"]["custom_model_config"]["use_attention"] = True
        config.multi_agent(policies={"shared_policy": policy_spec})

    algo = config.build()

    # ── Checkpoint Loading ────────────────────────────────────────────────────
    # algo.restore() loads the policy network weights from disk.
    # If the checkpoint doesn't exist, we run with random (untrained) weights.
    checkpoint_dir = Path(args.checkpoint).absolute()
    if checkpoint_dir.exists():
        print(f"Restoring checkpoint from {checkpoint_dir}")
        algo.restore(str(checkpoint_dir))
    else:
        print(f"WARNING: Checkpoint {checkpoint_dir} not found. Running with random weights.")

    # ── Env Access ────────────────────────────────────────────────────────────
    # Attempt to get the environment from the local worker (avoids double-creating).
    # Falls back to creating a fresh standalone env if worker access fails.
    env = None
    try:
        if hasattr(algo, "workers"):
            try:
                env = algo.workers.local_worker().env
            except TypeError:
                env = algo.workers().local_worker().env
        elif hasattr(algo, "env_runner"):
            env = algo.env_runner.env
    except Exception as e:
        print(f"Internal worker access failed ({e}), creating fallback env...")

    if env is None:
        print("Using fallback standalone environment...")
        env = DronePhysicsEnv(env_config)

    # ── Episode Start ─────────────────────────────────────────────────────────
    obs, info = env.reset()

    print("\n" + "="*40)
    print("  INTERACTIVE DASHBOARD READY")
    print("  - Drag the GREEN sphere to move the goal.")
    print("  - Keyboard 'R': Reset Environment")
    print("  - Keyboard 'Q': Quit")
    print("="*40 + "\n")

    # ── Main Loop ─────────────────────────────────────────────────────────────
    iter_count = 0
    try:
        while True:
            iter_count += 1

            # ── Test Mode Exit ─────────────────────────────────────────────────
            if args.test and iter_count > 100:
                print("Test completed successfully.")
                break

            # ── Keyboard Events ────────────────────────────────────────────────
            # p.getKeyboardEvents() REQUIRES an active GUI window.
            # Do NOT call in headless mode — crashes with "Not connected to physics server".
            if not args.headless:
                keys = p.getKeyboardEvents()
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    print("Q pressed — exiting.")
                    break
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    obs, info = env.reset()
                    continue

            # ── Mouse Events (Goal Dragging) ───────────────────────────────────
            # p.getMouseEvents() also REQUIRES an active GUI window.
            # We build an empty list as default to avoid conditional indentation issues.
            mouse_events = []
            if not args.headless:
                mouse_events = p.getMouseEvents()

            for event in mouse_events:
                # event[0] = type, event[1] = x, event[2] = y, event[3] = button state
                if event[3] == 1:  # Left button pressed
                    # Cast a ray from the camera through the clicked pixel.
                    # Then intersect with a horizontal plane at height z=1.0.
                    cam_info = p.getDebugVisualizerCamera()
                    width = cam_info[0]
                    height = cam_info[1]
                    ray_start, ray_end = p.getRayFromScreenPoint(event[1], event[2])

                    # Ray-plane intersection: find 3D point at z=target_z
                    target_z = 1.0  # Goal height (meters)
                    if abs(ray_end[2] - ray_start[2]) > 1e-6:
                        t = (target_z - ray_start[2]) / (ray_end[2] - ray_start[2])
                        new_goal = np.array(ray_start) + t * (np.array(ray_end) - np.array(ray_start))
                        new_goal = np.clip(new_goal, -5, 5)  # Clamp to world bounds
                        new_goal[2] = target_z
                        env.set_goal(new_goal)

            # ── Policy Inference ───────────────────────────────────────────────
            # compute_single_action() is ~1ms per agent; fast enough for real-time.
            # explore=False uses the mean action (deterministic) for smooth behavior.
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                action_dict[agent_id] = algo.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy",
                    explore=False
                )

            # ── Physics Step ───────────────────────────────────────────────────
            obs, rewards, terminated, truncated, info = env.step(action_dict)

            # ── Telemetry Logging ──────────────────────────────────────────────
            # Print drone position, velocity, and goal distance once per second (60 frames @ 10Hz).
            if iter_count % 60 == 0:
                d0_id = env.agent_ids[0]
                if d0_id in obs:
                    pos = obs[d0_id][:3]
                    vel = obs[d0_id][3:6]
                    dist = info[d0_id]["distance_to_goal"]
                    dist = info[d0_id]["distance_to_goal"]
                    print(f"Frame {iter_count}: Drone 0 Pos={pos.round(2)} Vel={vel.round(2)} Dist={dist:.2f}")

                    if pos[2] > 5.0:
                        print("WARNING: Drone rising too high! Gravity compensation might be excessive.")
                    if pos[2] < 0.1:
                        print("WARNING: Drone on ground. Thrust might be insufficient.")

            # ── Auto-Reset ─────────────────────────────────────────────────────
            # When an episode ends (collision, goal reached, or timeout),
            # automatically reset for continuous interactive experience.
            if terminated["__all__"] or truncated["__all__"]:
                obs, info = env.reset()

            # ── Timing ────────────────────────────────────────────────────────
            # Sleep 0.1s per loop iteration to match the physics dt=0.1s.
            # This ensures visualization runs at 1x real-time speed.
            # If you want faster-than-real-time: reduce sleep (but visualization will blur).
            # If visualization feels jerky: increase sleep slightly (0.12s).
            time.sleep(0.1)

    finally:
        # ── Cleanup ────────────────────────────────────────────────────────────
        algo.stop()
        ray.shutdown()
        p.disconnect()

if __name__ == "__main__":
    main()
