import time
import warnings

import jax
import numpy as np

from arlbench.core.algorithms import PPO
from arlbench.core.environments import make_env

N_UPDATES = 1e5
EVAL_STEPS = 1
EVAL_EPISODES = 10
N_ENVS = 1


def test_default_ppo_discrete(n_envs=10):
    env = make_env("gymnax", "CartPole-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > 450    


def test_default_ppo_continuous(n_envs=N_ENVS):
    env = make_env("gymnax", "Pendulum-v1", seed=42, n_envs=n_envs)
    rng = jax.random.PRNGKey(42)

    config = PPO.get_default_hpo_config()
    agent = PPO(config, env)
    algorithm_state = agent.init(rng)
    
    start = time.time()
    algorithm_state, results = agent.train(
        *algorithm_state,
        n_total_timesteps=N_UPDATES * n_envs,
        n_eval_steps=EVAL_STEPS,
        n_eval_episodes=EVAL_EPISODES
    )
    training_time = time.time() - start
    reward = results.eval_rewards[-1].mean()

    print(f"n_envs = {n_envs}, time = {training_time:.2f}, env_steps = {n_envs * algorithm_state.runner_state.global_step}, updates = {algorithm_state.runner_state.global_step}, reward = {reward:.2f}")
    assert reward > -1200    


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_default_ppo_discrete(n_envs=1)
        test_default_ppo_discrete(n_envs=10)
        # test_default_ppo_continuous()