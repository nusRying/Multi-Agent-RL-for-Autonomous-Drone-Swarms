from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_marl.envs import DroneSwarmEnv, SingleDroneEnv
from swarm_marl.training import build_multi_agent_ppo_config, build_single_agent_ppo_config
from swarm_marl.utils import load_structured_config

try:
    import ray
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script requires RLlib. Install with `python -m pip install -r requirements-rllib.txt` "
        "on Python 3.10-3.12."
    ) from exc


@dataclass
class EpisodeSummary:
    success: int
    collision_free: int
    time_to_goal: float
    formation_error: float
    path_efficiency: float
    episode_reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against protocol scenarios.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/eval_protocol_v1.yaml"),
        help="Evaluation protocol config (YAML/JSON).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path to load.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "multi"],
        default="auto",
        help="Policy/environment mode. 'auto' infers from scenario num_drones.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="RLlib rollout workers for restore config.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/metrics/eval_results.csv"),
        help="Path to output aggregate evaluation CSV.",
    )
    return parser.parse_args()


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _formation_error_from_positions(positions: dict[str, np.ndarray], desired_spacing: float) -> float:
    if len(positions) <= 1:
        return 0.0
    keys = list(positions.keys())
    errors: list[float] = []
    for i, aid in enumerate(keys):
        dists: list[float] = []
        for j, bid in enumerate(keys):
            if i == j:
                continue
            dists.append(_distance(positions[aid], positions[bid]))
        if dists:
            errors.append(float(np.mean(np.abs(np.asarray(dists) - desired_spacing))))
    return float(np.mean(errors)) if errors else 0.0


def _extract_checkpoint_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return str(path)


def _build_algo(mode: str, env_name: str, env_config: dict[str, Any], num_workers: int):
    if mode == "single":
        cfg = build_single_agent_ppo_config(env_name, env_config, num_workers=num_workers)
    else:
        cfg = build_multi_agent_ppo_config(env_name, env_config, num_workers=num_workers)
    return cfg.build()


def _run_single_episode_single_agent(algo, env: SingleDroneEnv) -> EpisodeSummary:
    obs, _ = env.reset()
    start_pos = obs[0:3].astype(np.float32)
    goal_vec = obs[6:9].astype(np.float32)
    goal = start_pos + goal_vec
    last_pos = start_pos.copy()
    traveled = 0.0
    episode_reward = 0.0
    step = 0
    reached_step: int | None = None
    collision = False

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        episode_reward += float(reward)

        pos = obs[0:3].astype(np.float32)
        traveled += _distance(last_pos, pos)
        last_pos = pos

        if bool(info.get("collision", False)):
            collision = True
        if bool(info.get("reached_goal", False)) and reached_step is None:
            reached_step = step

    success = int((not collision) and (reached_step is not None))
    collision_free = int(not collision)
    straight = _distance(start_pos, goal)
    path_eff = straight / traveled if traveled > 1e-8 else 0.0

    return EpisodeSummary(
        success=success,
        collision_free=collision_free,
        time_to_goal=float(reached_step) if reached_step is not None else math.nan,
        formation_error=0.0,
        path_efficiency=float(path_eff),
        episode_reward=float(episode_reward),
    )


def _run_single_episode_multi_agent(algo, env: DroneSwarmEnv) -> EpisodeSummary:
    obs, _ = env.reset()
    desired_spacing = float(env.cfg.desired_spacing)
    agent_ids = list(obs.keys())

    starts: dict[str, np.ndarray] = {}
    goals: dict[str, np.ndarray] = {}
    last_pos: dict[str, np.ndarray] = {}
    traveled: dict[str, float] = {}
    for aid, aobs in obs.items():
        pos = aobs[0:3].astype(np.float32)
        goal = pos + aobs[6:9].astype(np.float32)
        starts[aid] = pos.copy()
        goals[aid] = goal.copy()
        last_pos[aid] = pos.copy()
        traveled[aid] = 0.0

    episode_reward = 0.0
    step = 0
    any_collision = False
    all_reached_step: int | None = None
    formation_errors: list[float] = []

    terminated_all = False
    truncated_all = False
    while not (terminated_all or truncated_all):
        actions: dict[str, Any] = {}
        for aid, aobs in obs.items():
            try:
                act = algo.compute_single_action(aobs, policy_id="shared_policy", explore=False)
            except Exception:
                act = algo.compute_single_action(aobs, explore=False)
            actions[aid] = act

        obs, rewards, terminated, truncated, infos = env.step(actions)
        step += 1
        episode_reward += float(np.mean(list(rewards.values()))) if rewards else 0.0

        positions_now: dict[str, np.ndarray] = {}
        reached_all_flag = True
        for aid, aobs in obs.items():
            pos = aobs[0:3].astype(np.float32)
            traveled[aid] += _distance(last_pos[aid], pos)
            last_pos[aid] = pos
            positions_now[aid] = pos

            info = infos.get(aid, {})
            if bool(info.get("collision", False)):
                any_collision = True
            if not bool(info.get("reached_goal", False)):
                reached_all_flag = False

        formation_errors.append(_formation_error_from_positions(positions_now, desired_spacing))
        if reached_all_flag and all_reached_step is None:
            all_reached_step = step

        terminated_all = bool(terminated.get("__all__", False))
        truncated_all = bool(truncated.get("__all__", False))

    success = int((not any_collision) and (all_reached_step is not None))
    collision_free = int(not any_collision)

    path_eff_values: list[float] = []
    for aid in agent_ids:
        straight = _distance(starts[aid], goals[aid])
        dist = traveled[aid]
        path_eff_values.append(straight / dist if dist > 1e-8 else 0.0)

    return EpisodeSummary(
        success=success,
        collision_free=collision_free,
        time_to_goal=float(all_reached_step) if all_reached_step is not None else math.nan,
        formation_error=float(np.mean(formation_errors)) if formation_errors else 0.0,
        path_efficiency=float(np.mean(path_eff_values)) if path_eff_values else 0.0,
        episode_reward=float(episode_reward),
    )


def _aggregate(episodes: list[EpisodeSummary]) -> dict[str, float]:
    successes = [e.success for e in episodes]
    collision_free = [e.collision_free for e in episodes]
    ttg_values = [e.time_to_goal for e in episodes if not math.isnan(e.time_to_goal)]
    fe_values = [e.formation_error for e in episodes]
    pe_values = [e.path_efficiency for e in episodes]
    rew_values = [e.episode_reward for e in episodes]

    return {
        "success_rate": float(mean(successes)) if successes else 0.0,
        "collision_free_rate": float(mean(collision_free)) if collision_free else 0.0,
        "mean_time_to_goal": float(mean(ttg_values)) if ttg_values else math.nan,
        "formation_error": float(mean(fe_values)) if fe_values else 0.0,
        "path_efficiency": float(mean(pe_values)) if pe_values else 0.0,
        "episode_reward_mean": float(mean(rew_values)) if rew_values else 0.0,
        "episode_reward_std": _safe_std(rew_values),
    }


def _append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    protocol = load_structured_config(args.protocol)
    scenarios = protocol.get("scenarios", [])
    seeds = protocol.get("seeds", [0])
    episodes_per_seed = int(protocol.get("episodes_per_seed", 100))
    thresholds = protocol.get("pass_fail_thresholds", {})

    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"No scenarios found in protocol: {args.protocol}")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"No seeds found in protocol: {args.protocol}")

    env_single = "single_drone_eval_v0"
    env_multi = "drone_swarm_eval_v0"
    register_env(env_single, lambda cfg: SingleDroneEnv(cfg))
    register_env(env_multi, lambda cfg: DroneSwarmEnv(cfg))

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    checkpoint_path = _extract_checkpoint_path(args.checkpoint)

    all_rows: list[dict[str, Any]] = []
    fieldnames = [
        "scenario_id",
        "mode",
        "episodes",
        "success_rate",
        "collision_free_rate",
        "threshold_success_rate",
        "threshold_collision_free_rate",
        "pass_fail",
        "mean_time_to_goal",
        "formation_error",
        "path_efficiency",
        "episode_reward_mean",
        "episode_reward_std",
        "checkpoint_path",
    ]

    try:
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                raise ValueError(f"Scenario is not a mapping/object: {scenario}")

            scenario_id = str(scenario.get("scenario_id", "scenario"))
            env_cfg = dict(scenario.get("env_config", {}))
            num_drones = int(env_cfg.get("num_drones", 1))

            if args.mode == "auto":
                mode = "single" if num_drones <= 1 else "multi"
            else:
                mode = args.mode

            env_name = env_single if mode == "single" else env_multi
            algo_env_cfg = dict(env_cfg)
            algo_env_cfg["seed"] = int(seeds[0])

            algo = _build_algo(mode, env_name, algo_env_cfg, args.num_workers)
            algo.restore(checkpoint_path)

            episodes: list[EpisodeSummary] = []
            for seed in seeds:
                for ep in range(episodes_per_seed):
                    ep_seed = int(seed) + ep
                    if mode == "single":
                        env = SingleDroneEnv({**env_cfg, "seed": ep_seed})
                        summary = _run_single_episode_single_agent(algo, env)
                    else:
                        env = DroneSwarmEnv({**env_cfg, "seed": ep_seed})
                        summary = _run_single_episode_multi_agent(algo, env)
                    episodes.append(summary)

            aggregated = _aggregate(episodes)
            scenario_thresholds = {}
            if isinstance(thresholds, dict):
                raw = thresholds.get(scenario_id, {})
                if isinstance(raw, dict):
                    scenario_thresholds = raw
            min_sr = float(scenario_thresholds.get("min_success_rate", 0.0))
            min_cfr = float(scenario_thresholds.get("min_collision_free_rate", 0.0))
            is_pass = (
                aggregated["success_rate"] >= min_sr
                and aggregated["collision_free_rate"] >= min_cfr
            )

            row = {
                "scenario_id": scenario_id,
                "mode": mode,
                "episodes": len(episodes),
                "success_rate": f"{aggregated['success_rate']:.6f}",
                "collision_free_rate": f"{aggregated['collision_free_rate']:.6f}",
                "threshold_success_rate": f"{min_sr:.6f}",
                "threshold_collision_free_rate": f"{min_cfr:.6f}",
                "pass_fail": "PASS" if is_pass else "FAIL",
                "mean_time_to_goal": (
                    f"{aggregated['mean_time_to_goal']:.6f}"
                    if not math.isnan(aggregated["mean_time_to_goal"])
                    else ""
                ),
                "formation_error": f"{aggregated['formation_error']:.6f}",
                "path_efficiency": f"{aggregated['path_efficiency']:.6f}",
                "episode_reward_mean": f"{aggregated['episode_reward_mean']:.6f}",
                "episode_reward_std": f"{aggregated['episode_reward_std']:.6f}",
                "checkpoint_path": checkpoint_path,
            }
            all_rows.append(row)
            print(
                f"scenario={scenario_id} mode={mode} episodes={len(episodes)} "
                f"SR={row['success_rate']} CFR={row['collision_free_rate']} "
                f"TTG={row['mean_time_to_goal'] or 'nan'} status={row['pass_fail']}"
            )
            algo.stop()
    finally:
        ray.shutdown()

    _append_rows(args.output_csv, all_rows, fieldnames)
    print(f"Saved evaluation summary: {args.output_csv}")


if __name__ == "__main__":
    main()
