import sys
import time
import argparse
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from swarm_marl.envs.drone_physics_env import DronePhysicsEnv

def main():
    parser = argparse.ArgumentParser(description="Run Physics Env with GUI.")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    print("Initializing DronePhysicsEnv with GUI enabled...")
    env = DronePhysicsEnv(config={"gui": True, "num_drones": args.num_drones})
    obs, info = env.reset()
    
    print(f"Running for {args.steps} steps...")
    print("Close the PyBullet window to exit early.")
    
    try:
        for i in range(args.steps):
            # Random action for now, or just zero to see gravity
            # Let's apply a slight upward force to some to see if they hover?
            # Action range [-1, 1]. Action 0 -> Force 0.
            # Gravity comp was removed, so they fall at 0.
            # Need positive Z action to fight gravity.
            # Force = Action * 5.0 * mass. Gravity = 9.81 * mass.
            # Hover Action = 9.81 / 5.0 = 1.96... -> Out of bounds!
            # Wait, max_accel is 2.0. in config.
            # Force = Action * max_accel * 5.0 = Action * 10.0.
            # Hover Action = 9.81 / 10.0 = 0.981.
            # So Action ~0.98 should hover.
            
            actions = {}
            for agent_id in env.agents:
                # Oscillate around hover thrust
                thrust = 0.98 + 0.05 * np.sin(i * 0.1) 
                actions[agent_id] = np.array([0, 0, thrust])
            
            obs, rewards, terminated, truncated, _ = env.step(actions)
            
            if terminated["__all__"] or truncated["__all__"]:
                env.reset()
                
            time.sleep(1/240.0) # Real-time speed
            
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        print("Done.")

if __name__ == "__main__":
    main()
