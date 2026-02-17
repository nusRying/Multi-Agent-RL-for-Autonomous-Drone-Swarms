import sys
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from swarm_marl.envs.drone_physics_env import DronePhysicsEnv

def main():
    print("Initializing DronePhysicsEnv with GUI...")
    # Enable GUI to see it, or set to False for headless verification
    env = DronePhysicsEnv(config={"gui": False, "num_drones": 1})
    obs, info = env.reset()
    
    initial_z = obs["drone_0"][2]
    print("Initial Z position:", initial_z)
    
    # Step with zero action -> should fall due to gravity
    # PyBullet step is 1/240s. 100 steps = 0.4s.
    # d = 0.5 * 9.8 * 0.4^2 = 0.784m
    steps = 100
    print(f"Stepping with zero action for {steps} steps (expecting fall)...")
    for _ in range(steps):
        action = {"drone_0": np.zeros(3)}
        obs, _, _, _, _ = env.step(action)
        # Add a small delay if GUI is on to see it? No, script is --fast
        
    final_z = obs["drone_0"][2]
    print("Final Z position:", final_z)
    
    drop = initial_z - final_z
    print(f"Dropped: {drop:.4f} meters")
    
    if drop > 0.5:
        print("SUCCESS: Drone fell as expected under physics!")
    elif drop > 0.05:
         print("SUCCESS: Drone fell (but maybe hit ground?)")
    else:
        print("FAILURE: Drone did not fall meaningfully?")

if __name__ == "__main__":
    main()
