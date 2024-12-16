from typing import Callable
import jax.numpy as jnp
import numpy as np
from flashbax.buffers.train_val_trajectory_buffer import TrainValTrajectoryBufferState

import jax
import functools
from arlbench import AutoRLEnv
from arlbench.core.algorithms import ResetDQN, DQNRunnerState
from sklearn.linear_model import LinearRegression
from chex import PRNGKey


def get_q_next(env: AutoRLEnv, rng: PRNGKey):
    assert env._algorithm_state is not None
    assert env._algorithm_state.runner_state is not None
    assert isinstance(env._algorithm_state.runner_state, DQNRunnerState)
    assert isinstance(env._algorithm, ResetDQN)
    assert isinstance(env._algorithm_state.buffer_state, TrainValTrajectoryBufferState)

    batch = env._algorithm.buffer.sample_train(env._algorithm_state.buffer_state, rng)

    rewards = batch.experience.reward
    dones = batch.experience.done
    obs = batch.experience.last_obs
    next_obs = batch.experience.obs
    actions = batch.experience.action

    if env.hpo_config["use_target_network"]:
        q_next_target = env._algorithm.network.apply(
            env._algorithm_state.runner_state.train_state.target_params, next_obs
        )  # (batch_size, num_actions)
    else:
        q_next_target = env._algorithm.network.apply(
            env._algorithm_state.runner_state.train_state.params, next_obs
        )  # (batch_size, num_actions)
    q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
    next_q_value = rewards + (1 - dones) * env.hpo_config["gamma"] * q_next_target

    return obs, actions, next_q_value


def get_q_values(env: AutoRLEnv, rng: PRNGKey):
    assert env._algorithm_state is not None
    assert env._algorithm_state.runner_state is not None
    assert isinstance(env._algorithm_state.runner_state, DQNRunnerState)
    assert isinstance(env._algorithm, ResetDQN)
    assert isinstance(env._algorithm_state.buffer_state, TrainValTrajectoryBufferState)

    batch = env._algorithm.buffer.sample_val(env._algorithm_state.buffer_state, rng)

    obs = batch.experience.last_obs

    if env.hpo_config["use_target_network"]:
        q_next_target = env._algorithm.network.apply(
            env._algorithm_state.runner_state.train_state.target_params, obs
        )  # (batch_size, num_actions)
    else:
        q_next_target = env._algorithm.network.apply(
            env._algorithm_state.runner_state.train_state.params, obs
        )  # (batch_size, num_actions)

    # We one-hot encode actions for the linear regression
    distinct_actions = jnp.arange(q_next_target.shape[1])

    # Now we have to bring everything to the right dimensions
    obs_tiled = jnp.repeat(obs, distinct_actions.shape[0], axis=0)
    actions_tiled = jnp.tile(distinct_actions, q_next_target.shape[0])
    q_values_flat = q_next_target.flatten()

    return obs_tiled, actions_tiled, q_values_flat


def get_train_data(env: AutoRLEnv) -> tuple[np.ndarray, np.ndarray]:
    assert env._algorithm_state is not None
    assert isinstance(env._algorithm_state.buffer_state, TrainValTrajectoryBufferState)
    assert isinstance(env._algorithm, ResetDQN)
    assert isinstance(env._algorithm_state.runner_state, DQNRunnerState)

    compiled_get_q_next = jax.jit(get_q_next, static_argnums=(0,))

    n_samples = len(env._algorithm_state.buffer_state.train_indices)
    rng = jax.random.PRNGKey(0)

    all_obs = []
    all_actions = []
    all_next_q_values = []

    while n_samples > 0:
        sample_rng, rng = jax.random.split(rng)
        obs, actions, next_q_value = compiled_get_q_next(env, sample_rng)

        obs = np.array(obs)
        actions = np.array(actions)
        next_q_value = np.array(next_q_value)

        all_obs.append(obs)
        all_actions.append(actions)
        all_next_q_values.append(next_q_value)

        n_samples -= obs.shape[0]

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_next_q_values = np.concatenate(all_next_q_values, axis=0)

    X = np.concatenate([all_obs, all_actions[:, None]], axis=1)
    y = all_next_q_values

    return X, y


def get_val_data(env: AutoRLEnv) -> tuple[np.ndarray, np.ndarray]:
    assert env._algorithm_state is not None
    assert isinstance(env._algorithm_state.buffer_state, TrainValTrajectoryBufferState)
    assert isinstance(env._algorithm, ResetDQN)
    assert isinstance(env._algorithm_state.runner_state, DQNRunnerState)

    compiled_get_q_values = jax.jit(get_q_values, static_argnums=(0,))

    n_samples = len(env._algorithm_state.buffer_state.val_indices)
    rng = jax.random.PRNGKey(0)

    all_obs = []
    all_actions = []
    all_next_q_values = []

    while n_samples > 0:
        sample_rng, rng = jax.random.split(rng)
        obs, actions, next_q_value = compiled_get_q_values(env, sample_rng)

        obs = np.array(obs)
        actions = np.array(actions)
        next_q_value = np.array(next_q_value)

        all_obs.append(obs)
        all_actions.append(actions)
        all_next_q_values.append(next_q_value)

        n_samples -= obs.shape[0]

    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_next_q_values = np.concatenate(all_next_q_values, axis=0)

    X = np.concatenate([all_obs, all_actions[:, None]], axis=1)
    y = all_next_q_values

    return X, y


def compute_msbe(env: AutoRLEnv) -> float:
    X_train, y_train = get_train_data(env)

    model = LinearRegression().fit(X_train, y_train)

    del X_train, y_train

    X_val, y_val = get_val_data(env)

    msbe = float(np.mean((model.predict(X_val) - y_val) ** 2))

    return msbe
