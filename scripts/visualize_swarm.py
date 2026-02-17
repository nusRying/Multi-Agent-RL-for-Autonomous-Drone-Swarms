import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs import DroneSwarmEnv
from swarm_marl.utils.visualizer import SwarmVisualizer
from swarm_marl.training.config_builders import build_ctde_ppo_config

try:
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env
except ImportError:
    print("Ray/RLlib not installed. Visualization requiring checkpoints will fail.")

def main():
    parser = argparse.ArgumentParser(description="Visualize drone swarm policy.")
    parser.add_argument("--checkpoint", type=str, help="Path to RLlib checkpoint directory.")
    parser.add_argument("--num-drones", type=int, default=3, help="Number of drones.")
    parser.add_argument("--num-obstacles", type=int, default=8, help="Number of obstacles.")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode.")
    parser.add_argument("--save", action="store_true", help="Save animation to video.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Setup Environment
    env_config = {
        "num_drones": args.num_drones,
        "num_obstacles": args.num_obstacles,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    env = DroneSwarmEnv(env_config)
    
    # Load Policy if Checkpoint Provided
    algo = None
    if args.checkpoint:
        # Convert relative path to absolute path to avoid pyarrow URI errors
        args.checkpoint = os.path.abspath(args.checkpoint)
        
        ray.init(ignore_reinit_error=True)
        register_env("drone_swarm_v0", lambda cfg: DroneSwarmEnv(cfg))
        print(f"Loading checkpoint from {args.checkpoint}...")
        
        # Reconstruct config (assuming CTDE for now, but works for shared PPO too)
        # Note: We need to use the exact same config builder used for training to match model shapes
        # For robustness, we try to load the config from the checkpoint metadata if possible, 
        # or reconstruct it. RLlib's Algorithm.from_checkpoint is usually best.
        try:
            algo = Algorithm.from_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Failed to load checkpoint via Algorithm.from_checkpoint: {e}")
            print("Attempting to rebuild config and restore...")
            config = build_ctde_ppo_config(
                env_name="drone_swarm_v0",
                env_config=env_config,
                num_workers=0
            )
            algo = config.build()
            algo.restore(args.checkpoint)

    # Run Episode
    print("Running episode...")
    obs, info = env.reset(seed=args.seed)
    history = []
    
    # Store initial state
    history.append({
        "positions": env.positions.copy(),
        "velocities": env.velocities.copy(),
        "obstacles": env.obstacles.copy(),
        "goal": env.goal.copy()
    })

    terminated = {"__all__": False}
    truncated = {"__all__": False}
    step = 0
    while not terminated["__all__"] and not truncated["__all__"]:
        if algo:
            # Get actions from policy
            # We need to construct a proper input dict for compute_actions
            # Or use compute_single_action for each agent
            actions = {}
            for agent_id, agent_obs in obs.items():
                # We assume shared_policy
                actions[agent_id] = algo.compute_single_action(
                    observation=agent_obs,
                    policy_id="shared_policy",
                    explore=False # Deterministic for viz
                )
        else:
            # Random actions
            actions = {agent_id: env.action_space.sample() for agent_id in env.agents}
            
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        history.append({
            "positions": env.positions.copy(),
            "velocities": env.velocities.copy(),
            "obstacles": env.obstacles.copy(),
            "goal": env.goal.copy()
        })
        step += 1
        if step % 10 == 0:
            print(f"Step {step}...", end='\r')

    print(f"\nEpisode finished in {step} steps.")

    # Visualize
    viz = SwarmVisualizer(
        world_size=env.cfg.world_size,
        num_drones=args.num_drones,
        num_obstacles=args.num_obstacles
    )
    
    if args.save:
        save_path = "swarm_animation.mp4"
        viz.animate(history, save_path=save_path)
    else:
        viz.animate(history)

    if algo:
        ray.shutdown()

if __name__ == "__main__":
    main()
