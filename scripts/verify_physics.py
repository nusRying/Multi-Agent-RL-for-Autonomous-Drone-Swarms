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
    
    print("Initial Z position:", obs["drone_0"][2])
    
    # Step with zero action -> should fall due to gravity
    print("Stepping with zero action (expecting fall)...")
    for _ in range(10):
        action = {"drone_0": np.zeros(3)}
        obs, _, _, _, _ = env.step(action)
        
    final_z = obs["drone_0"][2]
    print("Final Z position:", final_z)
    
    if final_z < 0.5: # Started at >1.0
        print("SUCCESS: Drone fell as expected under physics!")
    else:
        print("FAILURE: Drone did not fall?")

if __name__ == "__main__":
    main()
