from __future__ import annotations

from typing import Any

try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
except ModuleNotFoundError as exc:  # pragma: no cover - requires optional dependency
    raise ModuleNotFoundError(
        "RLlib is not installed. Install with `python -m pip install -r requirements-rllib.txt` "
        "on a supported Python version."
    ) from exc


def build_single_agent_ppo_config(
    env_name: str,
    env_config: dict[str, Any] | None = None,
    num_workers: int = 0,
) -> PPOConfig:
    return (
        PPOConfig()
        .environment(env=env_name, env_config=env_config or {})
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=8192,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
    )


def build_multi_agent_ppo_config(
    env_name: str,
    env_config: dict[str, Any] | None = None,
    num_workers: int = 0,
) -> PPOConfig:
    shared_policy = PolicySpec()

    return (
        PPOConfig()
        .environment(env=env_name, env_config=env_config or {})
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers)
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=16384,
            sgd_minibatch_size=2048,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
        )
        .multi_agent(
            policies={"shared_policy": shared_policy},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    )
