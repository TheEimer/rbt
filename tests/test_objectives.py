
from arlbench.autorl.objectives import track_emissions, track_reward, track_runtime
import jax
import time
import numpy as np

from arlbench.algorithms import DQN

from arlbench.environments import make_env


DQN_OPTIONS = {
    "n_total_timesteps": 1e6,
    "n_envs": 10,
    "n_env_steps": 500,
    "n_eval_episodes": 10,
    "track_metrics": False,
    "track_traj": False,
}

def test_reward():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = track_reward(train_func, objectives, agent, DQN_OPTIONS["n_eval_episodes"])

    (runner_state, _), _ = train_func(runner_state, buffer_state)
    rewards = agent.eval(runner_state, DQN_OPTIONS["n_eval_episodes"])
    reward = np.mean(rewards)

    assert np.abs(reward - objectives["reward_mean"]) < 0.01

def test_runtime():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = track_runtime(train_func, objectives)

    start = time.time()
    train_func(runner_state, buffer_state)
    runtime = time.time() - start

    assert np.abs(runtime - objectives["runtime"]) < 0.05

def test_emissions():
    env = make_env("gymnax", "CartPole-v1", seed=42)
    rng = jax.random.PRNGKey(42)

    config = DQN.get_default_hpo_config()
    agent = DQN(config, DQN_OPTIONS, env)
    runner_state, buffer_state = agent.init(rng)

    objectives = {}
    train_func = agent.train
    train_func = track_emissions(train_func, objectives)
    train_func(runner_state, buffer_state)
    assert objectives["emissions"] > 0
    assert objectives["emissions"] < 1
