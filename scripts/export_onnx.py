"""Export the best trained policy to ONNX format for edge deployment.

Loads a checkpoint, extracts the PyTorch actor network, and traces it
with a dummy observation to produce an ONNX file.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RLLIB_TEST_NO_TF_IMPORT", "1")
os.environ.setdefault("RLLIB_TEST_NO_JAX_IMPORT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained RLlib policy to ONNX.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory to load.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "policy.onnx",
        help="Output ONNX file path.",
    )
    p.add_argument(
        "--mode",
        choices=["multi", "ctde", "physics"],
        default="ctde",
        help="Config mode used to build the algorithm (must match checkpoint).",
    )
    p.add_argument("--num-drones", type=int, default=3)
    p.add_argument("--num-obstacles", type=int, default=8)
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    import numpy as np

    try:
        import ray
        from ray.tune.registry import register_env
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RLlib is required for export.") from exc

    from swarm_marl.envs import DroneSwarmEnv
    from swarm_marl.training import build_multi_agent_ppo_config, build_ctde_ppo_config

    env_config = {
        "num_drones": args.num_drones,
        "num_obstacles": args.num_obstacles,
        "max_steps": 400,
        "seed": 0,
    }

    env_name = "drone_export_v0"
    register_env(env_name, lambda cfg: DroneSwarmEnv(cfg))

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    if args.mode in ("ctde", "physics"):
        config = build_ctde_ppo_config(env_name=env_name, env_config=env_config, num_workers=0)
    else:
        config = build_multi_agent_ppo_config(env_name=env_name, env_config=env_config, num_workers=0)

    algo = config.build()

    # Restore checkpoint (use absolute path for Windows compatibility)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Find the actual checkpoint dir with algorithm_state.pkl
    if checkpoint_path.is_dir() and not (checkpoint_path / "algorithm_state.pkl").exists():
        candidates = sorted(
            checkpoint_path.rglob("algorithm_state.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            checkpoint_path = candidates[0].parent

    algo.restore(str(checkpoint_path))

    # Extract the policy's PyTorch model
    policy = algo.get_policy("shared_policy")
    model = policy.model

    # Build dummy observation matching the env's observation space
    obs_space = policy.observation_space
    dummy_obs = torch.from_numpy(
        np.zeros(obs_space.shape, dtype=np.float32)
    ).unsqueeze(0)

    # RLlib models expect (obs_dict, state, seq_lens) — we wrap accordingly
    dummy_state = []
    dummy_seq_lens = torch.tensor([1])

    # Export the actor (forward pass only)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Use a wrapper that exposes only the actor forward pass
    class ActorWrapper(torch.nn.Module):
        def __init__(self, rllib_model):
            super().__init__()
            self.rllib_model = rllib_model

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            input_dict = {"obs": obs}
            logits, _ = self.rllib_model(input_dict, [], torch.tensor([1]))
            return logits

    wrapper = ActorWrapper(model)
    wrapper.eval()

    try:
        torch.onnx.export(
            wrapper,
            dummy_obs,
            str(args.output),
            opset_version=args.opset,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
        )
        print(f"✅ Exported ONNX model: {args.output}")
        print(f"   Input shape:  {list(dummy_obs.shape)}")
        print(f"   Opset:        {args.opset}")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")
        print("   The model may contain operations not supported by ONNX.")
        print("   Falling back to TorchScript (.pt) export...")

        # Fallback: save as TorchScript
        pt_path = args.output.with_suffix(".pt")
        try:
            traced = torch.jit.trace(wrapper, dummy_obs)
            traced.save(str(pt_path))
            print(f"✅ Exported TorchScript model: {pt_path}")
        except Exception as e2:
            print(f"❌ TorchScript export also failed: {e2}")
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
