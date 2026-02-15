import numpy as np

from swarm_marl.envs import DroneSwarmEnv, SingleDroneEnv


def test_single_drone_env_smoke():
    env = SingleDroneEnv({"seed": 123, "max_steps": 10})
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "distance_to_goal" in info

    action = np.zeros(3, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance_to_goal" in info


def test_multi_agent_env_smoke():
    env = DroneSwarmEnv({"num_drones": 3, "seed": 123, "max_steps": 10})
    obs, infos = env.reset()
    assert len(obs) == 3
    assert len(infos) == 3

    actions = {agent_id: np.zeros(3, dtype=np.float32) for agent_id in obs}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)

    assert len(next_obs) == 3
    assert len(rewards) == 3
    assert "__all__" in terminated
    assert "__all__" in truncated
    assert all(isinstance(v, float) for v in rewards.values())
    assert all("global_state" in infos[agent_id] for agent_id in infos)

