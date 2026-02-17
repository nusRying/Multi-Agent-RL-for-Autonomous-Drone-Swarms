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

from ray.rllib.models import ModelCatalog
from swarm_marl.training.models import CentralizedCriticModel
from swarm_marl.training.callbacks import GlobalStateCallback


def _set_worker_config(config: PPOConfig, num_workers: int) -> PPOConfig:
    # Ray >=2.5x replaced `rollouts` with `env_runners`.
    if hasattr(config, "env_runners"):
        return config.env_runners(num_env_runners=num_workers)
    return config.rollouts(num_rollout_workers=num_workers)


def _disable_new_api_stack_if_available(config: PPOConfig) -> PPOConfig:
    # Ray >=2.5x enables a new API stack by default. Our env/policy setup is
    # currently built for the classic API stack.
    if hasattr(config, "api_stack"):
        try:
            return config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
        except TypeError:
            return config
    return config


def _set_training_config(
    config: PPOConfig,
    *,
    gamma: float,
    lr: float,
    train_batch_size: int,
    minibatch_size: int,
    num_sgd_iter: int,
) -> PPOConfig:
    base_kwargs: dict[str, Any] = {
        "gamma": gamma,
        "lr": lr,
        "train_batch_size": train_batch_size,
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
    }

    # Ray API naming differs across versions:
    # - older: sgd_minibatch_size, num_sgd_iter
    # - newer: minibatch_size, num_epochs
    old_kwargs = dict(base_kwargs)
    old_kwargs.update(
        {
            "sgd_minibatch_size": minibatch_size,
            "num_sgd_iter": num_sgd_iter,
        }
    )
    try:
        return config.training(**old_kwargs)
    except TypeError:
        new_kwargs = dict(base_kwargs)
        new_kwargs.update(
            {
                "minibatch_size": minibatch_size,
                "num_epochs": num_sgd_iter,
            }
        )
        return config.training(**new_kwargs)


def build_single_agent_ppo_config(
    env_name: str,
    env_config: dict[str, Any] | None = None,
    num_workers: int = 0,
    gamma: float = 0.99,
    lr: float = 3e-4,
    train_batch_size: int = 8192,
    minibatch_size: int = 1024,
    num_sgd_iter: int = 10,
) -> PPOConfig:
    cfg = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config or {})
        .framework("torch")
    )
    cfg = _disable_new_api_stack_if_available(cfg)
    cfg = _set_training_config(
        cfg,
        gamma=gamma,
        lr=lr,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_sgd_iter=num_sgd_iter,
    )
    return _set_worker_config(cfg, num_workers)


def build_multi_agent_ppo_config(
    env_name: str,
    env_config: dict[str, Any] | None = None,
    num_workers: int = 0,
    gamma: float = 0.99,
    lr: float = 3e-4,
    train_batch_size: int = 16384,
    minibatch_size: int = 2048,
    num_sgd_iter: int = 10,
) -> PPOConfig:
    shared_policy = PolicySpec()

    cfg = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config or {})
        .framework("torch")
        .multi_agent(
            policies={"shared_policy": shared_policy},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    )
    cfg = _disable_new_api_stack_if_available(cfg)
    cfg = _set_training_config(
        cfg,
        gamma=gamma,
        lr=lr,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_sgd_iter=num_sgd_iter,
    )
    return _set_worker_config(cfg, num_workers)


def build_ctde_ppo_config(
    env_name: str,
    env_config: dict[str, Any] | None = None,
    num_workers: int = 0,
    gamma: float = 0.99,
    lr: float = 3e-4,
    train_batch_size: int = 16384,
    minibatch_size: int = 2048,
    num_sgd_iter: int = 10,
) -> PPOConfig:
    # Register model (safe to call multiple times)
    ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

    # Calculate global state dim based on env config
    # global_state = [positions(N*3), velocities(N*3), goal(3)]
    env_cfg_dict = env_config or {}
    n_drones = int(env_cfg_dict.get("num_drones", 3))
    neighbor_k = int(env_cfg_dict.get("neighbor_k", 3))
    sensed_obstacles = int(env_cfg_dict.get("sensed_obstacles", 4))
    
    global_state_dim = (6 * n_drones) + 3

    shared_policy = PolicySpec(
        config={
            "model": {
                "custom_model": "centralized_critic",
                "custom_model_config": {
                    "global_state_dim": global_state_dim,
                    "neighbor_k": neighbor_k,
                    "sensed_obstacles": sensed_obstacles,
                },
            },
        }
    )

    cfg = (
        PPOConfig()
        .environment(env=env_name, env_config=env_config or {})
        .framework("torch")
        .callbacks(GlobalStateCallback)
        .multi_agent(
            policies={"shared_policy": shared_policy},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
    )
    cfg = _disable_new_api_stack_if_available(cfg)
    cfg = _set_training_config(
        cfg,
        gamma=gamma,
        lr=lr,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        num_sgd_iter=num_sgd_iter,
    )
    return _set_worker_config(cfg, num_workers)
