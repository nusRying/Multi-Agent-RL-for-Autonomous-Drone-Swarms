"""Quick test to verify GPU is accessible and RLlib can use it."""
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig

print("=" * 60)
print("GPU DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: PyTorch CUDA
print("\n1. PyTorch CUDA Availability:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("   WARNING: CUDA not available!")

# Test 2: Create a simple tensor on GPU
if torch.cuda.is_available():
    print("\n2. GPU Tensor Test:")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x)
        print(f"   ✓ Successfully created and computed tensors on GPU")
        print(f"   Device: {y.device}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

# Test 3: RLlib Configuration
print("\n3. RLlib GPU Configuration Test:")
try:
    config = (
        PPOConfig()
        .framework("torch")
        .resources(num_gpus=1)
    )
    print(f"   ✓ PPOConfig created successfully")
    print(f"   Framework: torch")
    print(f"   num_gpus: 1")
    
    # Try to extract the actual config dict
    config_dict = config.to_dict()
    print(f"   Resources in config: {config_dict.get('num_gpus', 'NOT FOUND')}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test 4: Ray init with GPU
print("\n4. Ray GPU Resources:")
try:
    ray.init(ignore_reinit_error=True, num_gpus=1)
    print(f"   ✓ Ray initialized with GPU support")
    print(f"   Available resources: {ray.available_resources()}")
    ray.shutdown()
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
