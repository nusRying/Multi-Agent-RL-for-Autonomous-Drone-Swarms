from __future__ import annotations

from typing import Any


def _get_nested(mapping: dict[str, Any], path: str) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_from_list(value: Any) -> float | None:
    if not isinstance(value, list) or not value:
        return None
    numeric = [_as_float(v) for v in value]
    filtered = [v for v in numeric if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _first_float(mapping: dict[str, Any], paths: list[str]) -> float | None:
    for path in paths:
        value = _as_float(_get_nested(mapping, path))
        if value is not None:
            return value
    return None


def extract_episode_stats(result: dict[str, Any]) -> tuple[float, float]:
    """
    Return (episode_reward_mean, episode_len_mean) across Ray RLlib result formats.

    Ray has shifted where these metrics live between versions/stacks.
    """
    reward_paths = [
        "episode_reward_mean",
        "env_runners.episode_reward_mean",
        "sampler_results.episode_reward_mean",
        "evaluation.episode_reward_mean",
    ]
    len_paths = [
        "episode_len_mean",
        "env_runners.episode_len_mean",
        "sampler_results.episode_len_mean",
        "evaluation.episode_len_mean",
    ]

    reward_mean = _first_float(result, reward_paths)
    len_mean = _first_float(result, len_paths)

    if reward_mean is None:
        reward_mean = _mean_from_list(_get_nested(result, "hist_stats.episode_reward"))
    if reward_mean is None:
        reward_mean = _mean_from_list(_get_nested(result, "env_runners.hist_stats.episode_reward"))

    if len_mean is None:
        len_mean = _mean_from_list(_get_nested(result, "hist_stats.episode_lengths"))
    if len_mean is None:
        len_mean = _mean_from_list(_get_nested(result, "env_runners.hist_stats.episode_lengths"))

    return float(reward_mean or 0.0), float(len_mean or 0.0)
