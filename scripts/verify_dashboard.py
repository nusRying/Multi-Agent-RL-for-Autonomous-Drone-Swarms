
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from swarm_marl.envs.drone_physics_env import DronePhysicsEnv
from swarm_marl.training.config_builders import build_ctde_ppo_config

def main():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    checkpoint_path = ROOT / "checkpoints" / "physics_run"
    print(f"Loading checkpoint from {checkpoint_path}")

    env_name = "drone_physics_headless"
    # Use config matching dashboard
    env_config = {
        "num_drones": 3,
        "gui": False,
        "world_size": 10.0,
        "max_steps": 1000,
        "dt": 0.1
    }
    
    register_env(env_name, lambda cfg: DronePhysicsEnv(cfg))
    
    # Build Algo (dummy config to load checkpoint)
    config = build_ctde_ppo_config(env_name=env_name, env_config=env_config)
    algo = config.build()
    
    try:
        if checkpoint_path.exists():
             algo.restore(str(checkpoint_path))
             print("Checkpoint restored.")
        else:
             print("Checkpoint NOT found. Running with random weights.")
    except Exception as e:
        print(f"Restore failed: {e}")

    print("Creating verification environment...")
    env = DronePhysicsEnv(env_config)
    obs, info = env.reset()
    
    print("Running 200 steps...")
    
    for i in range(200):
        # Compute actions
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            action_dict[agent_id] = algo.compute_single_action(
                agent_obs, policy_id="shared_policy", explore=False
            )
            
        obs, rewards, term, trunc, info = env.step(action_dict)
        
        # Telemetry
        if i % 10 == 0:
            d0_pos = obs["drone_0"][:3]
            d0_vel = obs["drone_0"][3:6]
            print(f"Step {i}: Pos={d0_pos} Vel={d0_vel}")
            
            # Check stability
            if d0_pos[2] > 10.0:
                print("FAILURE: Drone flew away!")
                sys.exit(1)
            if d0_pos[2] < 0.0:
                print("FAILURE: Drone crashed through floor!")
                sys.exit(1)
                
    print("SUCCESS: 200 steps stable.")
    ray.shutdown()

if __name__ == "__main__":
    main()
