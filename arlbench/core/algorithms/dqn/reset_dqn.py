# Parts of this code are based on PureJaxRL (https://github.com/luchris429/purejaxrl).
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""DQN algorithm."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
)
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arlbench.core import running_statistics
from arlbench.core.algorithms.algorithm import Algorithm
from arlbench.core.algorithms.buffers import uniform_sample
from arlbench.core.algorithms.common import TimeStep
from arlbench.core.algorithms.neuron_recycler import (
    LayerReset,
    NeuronRecycler,
    NeuronRecyclerScheduled,
)
from arlbench.core.algorithms.prioritised_item_buffer import (
    make_prioritised_item_buffer,
)

from .dqn import (
    DQN,
    DQNMetrics,
    DQNRunnerState,
    DQNState,
    DQNTrainingResult,
    DQNTrainReturnT,
    DQNTrainState,
    Transition,
)
from .models import CNNQ, MLPQ

if TYPE_CHECKING:
    import chex
    from flashbax.buffers.prioritised_trajectory_buffer import (
        PrioritisedTrajectoryBufferState,
    )
    from flax.core.frozen_dict import FrozenDict

    from arlbench.core.environments import Environment
    from arlbench.core.running_statistics import RunningStatisticsState
    from arlbench.core.wrappers import Wrapper


RECYCLERS = {
    "dormancy": NeuronRecycler,
    "dormancy_schedule": NeuronRecyclerScheduled,
    "primacy": LayerReset,
}


class ResetDQN(Algorithm):
    """JAX-based implementation of ReDo Deep Q-Network (DQN)."""

    name: str = "redo_dqn"

    def __init__( # noqa: PLR0913
        self,
        hpo_config: Configuration,
        env: Environment | Wrapper,
        eval_env: Environment | Wrapper | None = None,
        deterministic_eval: bool = True,
        eval_eps: float = 0.05,
        cnn_policy: bool = False,
        nas_config: Configuration | None = None,
        track_trajectories: bool = False,
        track_metrics: bool = False,
        weight_recycler_config: Configuration | None = None,
        manual_epsilon: bool = False,
        epsilon: float = 0.05,
        offline_update_fraction: float = 0.1,
        manual_offline_updates: bool = False,
        manual_recycling: bool = False,
    ) -> None:
        """Creates a ReDo DQN algorithm instance.

        Args:
            hpo_config (Configuration): Hyperparameter configuration.
            env (Environment | AutoRLWrapper): Training environment.
            eval_env (Environment | AutoRLWrapper | None, optional): Evaluation environment (otherwise training environment is used for evaluation). Defaults to None.
            deterministic_eval (bool, optional): Use deterministic evaluation. Defaults to True.
            eval_eps (float, optional): Epsilon value for non-deterministic evaluation. Defaults to 0.05.
            cnn_policy (bool, optional): Use CNN network architecture. Defaults to False.
            nas_config (Configuration | None, optional): Neural architecture configuration. Defaults to None.
            track_trajectories (bool, optional):  Track metrics such as loss and gradients during training. Defaults to False.
            track_metrics (bool, optional): Track trajectories during training. Defaults to False.
            weight_recycler_config (Configuration | None, optional): Configuration for neuron recycling. Defaults to None.
        """
        if nas_config is None:
            nas_config = DQN.get_default_nas_config()

        super().__init__(
            hpo_config,
            nas_config,
            env,
            eval_env=eval_env,
            deterministic_eval=deterministic_eval,
            track_trajectories=track_trajectories,
            track_metrics=track_metrics,
        )

        self.eval_eps = eval_eps
        self.manual_epsilon = manual_epsilon
        self.epsilon = jnp.array(OmegaConf.to_container(epsilon, resolve=True))
        self.offline_update_fraction = offline_update_fraction
        self.manual_offline_updates = manual_offline_updates
        self.manual_recycling = manual_recycling

        # For the network, we need the properties of the action space
        action_size, discrete = self.action_type
        network_cls = CNNQ if cnn_policy else MLPQ
        self.network = network_cls(
            action_size,
            discrete=discrete,
            activation=self.nas_config["activation"],
            hidden_size=self.nas_config["hidden_size"],
        )

        self.buffer = make_prioritised_item_buffer(
            max_length=self.hpo_config["buffer_size"],
            min_length=self.hpo_config["buffer_batch_size"],
            sample_batch_size=self.hpo_config["buffer_batch_size"],
            add_batches=True,
            add_sequences=False,
            priority_exponent=self.hpo_config["buffer_alpha"],
            device=jax.default_backend(),
        )

        # This is how we can turn the prioritized sampling on/off for dynamic HPO
        # We always use the prioritized replay buffer, but if "buffer_prio_sampling"
        # is disabled, we replace the sampling function by the uniform sampling
        if self.hpo_config["buffer_prio_sampling"] is False:
            sample_fn = functools.partial(
                uniform_sample,
                batch_size=self.hpo_config["buffer_batch_size"],
                sequence_length=1,
                period=1,
            )
            self.buffer = self.buffer.replace(sample=sample_fn)

        self.weight_recycler_config = weight_recycler_config
        if self.weight_recycler_config is None:
            self.weight_recycler_config = {}
        elif isinstance(self.weight_recycler_config, DictConfig):
            self.weight_recycler_config = OmegaConf.to_container(self.weight_recycler_config, resolve=True)

        if "weight_recycler" in self.weight_recycler_config:
            self.recycler_cls = RECYCLERS[self.weight_recycler_config["weight_recycler"]]
            del self.weight_recycler_config["weight_recycler"]
        else:
            self.recycler_cls = NeuronRecycler

    @staticmethod
    def get_hpo_config_space(seed: int | None = None) -> ConfigurationSpace:
        """Returns the hyperparameter optimization (HPO) configuration space for DQN."""
        # defaults from https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        cs = ConfigurationSpace(
            name="DQNConfigSpace",
            seed=seed,
            space={
                "buffer_size": Integer(
                    "buffer_size", (1024, int(1e7)), default=1000000
                ),
                "buffer_batch_size": Categorical(
                    "buffer_batch_size", [4, 8, 16, 32, 64], default=16
                ),
                "buffer_prio_sampling": Categorical(
                    "buffer_prio_sampling", [True, False], default=False
                ),
                "buffer_alpha": Float("buffer_alpha", (0.01, 1.0), default=0.9),
                "buffer_beta": Float("buffer_beta", (0.01, 1.0), default=0.9),
                "buffer_epsilon": Float("buffer_epsilon", (1e-7, 1e-3), default=1e-6),
                "learning_rate": Float(
                    "learning_rate", (1e-6, 0.1), default=3e-4, log=True
                ),
                "gamma": Float("gamma", (0.8, 1.0), default=0.99),
                "tau": Float("tau", (0.01, 1.0), default=1.0),
                "initial_epsilon": Float("initial_epsilon", (0.5, 1.0), default=1.0),
                "target_epsilon": Float("target_epsilon", (0.001, 0.2), default=0.05),
                "exploration_fraction": Float("exploration_fraction", (0.005, 0.5), default=0.1),
                "use_target_network": Categorical(
                    "use_target_network", [True, False], default=True
                ),
                "train_freq": Integer("train_freq", (1, 256), default=4),
                "gradient steps": Integer("gradient_steps", (1, 256), default=1),
                "learning_starts": Integer("learning_starts", (0, 32768), default=1024),
                "target_update_interval": Integer(
                    "target_update_interval", (1, 2000), default=1000
                ),
                "normalize_observations": Categorical(
                    "normalize_observations", [True, False], default=False
                ),
            },
        )
        cs.add_conditions(
            [
                EqualsCondition(
                    cs["target_update_interval"], cs["use_target_network"], True
                ),
                EqualsCondition(cs["tau"], cs["use_target_network"], True),
            ]
        )

        return cs

    @staticmethod
    def get_default_hpo_config() -> Configuration:
        """Returns the default hyperparameter configuration for DQN."""
        return DQN.get_hpo_config_space().get_default_configuration()

    @staticmethod
    def get_nas_config_space(seed=None) -> ConfigurationSpace:
        """Returns the neural architecture search (NAS) configuration space for DQN."""
        return ConfigurationSpace(
            name="DQNNASConfigSpace",
            seed=seed,
            space={
                "activation": Categorical(
                    "activation", ["tanh", "relu"], default="tanh"
                ),
                "hidden_size": Integer("hidden_size", (1, 1024), default=64),
            },
        )

    @staticmethod
    def get_default_nas_config() -> Configuration:
        """Returns the default NAS configuration for DQN."""
        return DQN.get_nas_config_space().get_default_configuration()

    @staticmethod
    def get_checkpoint_factory(
        runner_state: DQNRunnerState,
        train_result: DQNTrainingResult | None,
    ) -> dict[str, Callable]:
        """Creates a factory dictionary of all possible checkpointing options for DQN.

        Args:
            runner_state (DQNRunnerState): Algorithm runner state.
            train_result (DQNTrainingResult | None): Training result.

        Returns:
            dict[str, Callable]: Dictionary of factory functions containing [opt_state, params, target_params, loss, trajectories].
        """
        train_state = runner_state.train_state

        def get_trajectories():
            if train_result is None or train_result.trajectories is None:
                return None

            traj = train_result.trajectories

            trajectories = {}
            trajectories["states"] = jnp.concatenate(traj.obs, axis=0)
            trajectories["action"] = jnp.concatenate(traj.action, axis=0)
            trajectories["reward"] = jnp.concatenate(traj.reward, axis=0)
            trajectories["dones"] = jnp.concatenate(traj.done, axis=0)

            return trajectories

        return {
            "opt_state": lambda: train_state.opt_state,
            "params": lambda: train_state.params,
            "target_params": lambda: train_state.target_params,
            "loss": lambda: train_result.metrics.loss
            if train_result and train_result.metrics
            else None,
            "trajectories": get_trajectories,
        }

    def init(
        self,
        rng: chex.PRNGKey,
        buffer_state: PrioritisedTrajectoryBufferState | None = None,
        network_params: FrozenDict | dict | None = None,
        target_params: FrozenDict | dict | None = None,
        opt_state: optax.OptState | None = None,
    ) -> DQNState:
        """Initializes DQN state. Passed parameters are not initialized and included in the final state.

        Args:
            rng (chex.PRNGKey): Random generator key.
            buffer_state (PrioritisedTrajectoryBufferState | None, optional): Buffer state. Defaults to None.
            network_params (FrozenDict | dict | None, optional): Networks parameters. Defaults to None.
            target_params (FrozenDict | dict | None, optional): Target network parameters. Defaults to None.
            opt_state (optax.OptState | None, optional): Optimizer state. Defaults to None.

        Returns:
            DQNState: DQN state.
        """
        rng, reset_rng = jax.random.split(rng)
        env_state, obs = self.env.reset(reset_rng)

        # If any of these if not defined, we need a dummy environment transition
        # to initialize them
        if buffer_state is None or network_params is None or target_params is None:
            dummy_rng = jax.random.PRNGKey(0)
            _action = self.env.sample_actions(dummy_rng)
            _, (_obs, _reward, _done, _) = self.env.step(env_state, _action, dummy_rng)

        if buffer_state is None:
            # This is how transitions will look like during training, so we need to pass one
            # once to the buffer to estimate and allocate the required buffer size
            _timestep = TimeStep(
                last_obs=_obs[0],
                obs=_obs[0],
                action=_action[0],
                reward=_reward[0],
                done=_done[0],
            )
            buffer_state = self.buffer.init(_timestep)

        rng, init_rng = jax.random.split(rng)
        if network_params is None:
            network_params = self.network.init(init_rng, _obs)
        if target_params is None:
            target_params = self.network.init(init_rng, _obs)

        train_state_kwargs = {
            "apply_fn": self.network.apply,
            "params": network_params,
            "target_params": target_params,
            "tx": optax.adam(self.hpo_config["learning_rate"]),
            "opt_state": opt_state,
        }
        train_state = DQNTrainState.create_with_opt_state(**train_state_kwargs)
        global_step = 0

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=running_statistics.init_state(obs[0]),
            env_state=env_state,
            obs=obs,
            global_step=global_step,
        )

        assert buffer_state is not None

        return DQNState(runner_state=runner_state, buffer_state=buffer_state)

    @functools.partial(jax.jit, static_argnums=0)
    def predict(
        self,
        runner_state: DQNRunnerState,
        obs: jnp.ndarray,
        rng: chex.PRNGKey,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Predict action(s) based on the current observation(s).

        Args:
            runner_state (DQNRunnerState): Algorithm runner state.
            obs (jnp.ndarray): Observation(s).
            rng (chex.PRNGKey | None, optional): Not used in DQN. Random generator key in other algorithms. Defaults to None.
            deterministic (bool): Return deterministic action. Defaults to True.

        Returns:
            jnp.ndarray: Action(s).
        """

        def random_action(rng: chex.PRNGKey, _) -> jnp.ndarray:
            _rngs = jax.random.split(rng, obs.shape[0])
            return jnp.array(
                [self.env.action_space.sample(_rngs[i]) for i in range(obs.shape[0])]
            )

        def greedy_action(_: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
            if self.hpo_config["normalize_observations"]:
                obs = running_statistics.normalize(obs, runner_state.normalizer_state)
            q_values = self.network.apply(runner_state.train_state.params, obs)
            return q_values.argmax(axis=-1)

        def sample_action(rng: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
            rnd_action = random_action(rng, obs)
            grd_action = greedy_action(rng, obs)
            return jax.lax.select(
                jax.random.uniform(rng, obs.shape[:1]) < self.eval_eps, rnd_action, grd_action
            )

        return jax.lax.cond(
            deterministic,
            greedy_action,
            sample_action,
            rng,
            obs,
        )

    @functools.partial(jax.jit, static_argnums=(0, 3, 4, 5), donate_argnums=(2,))
    def train(
        self,
        runner_state: DQNRunnerState,
        buffer_state: PrioritisedTrajectoryBufferState,
        n_total_timesteps: int = 1000000,
        n_eval_steps: int = 100,
        n_eval_episodes: int = 10,
    ) -> DQNTrainReturnT:
        """Performs one full training.

        Args:
            runner_state (DQNRunnerState): DQN runner state.
            buffer_state (PrioritisedTrajectoryBufferState): Buffer state.
            n_total_timesteps (int, optional): Total number of training timesteps. Update steps = n_total_timesteps // n_envs. Defaults to 1000000.
            n_eval_steps (int, optional): Number of evaluation steps during training. Defaults to 100.
            n_eval_episodes (int, optional): Number of evaluation episodes per evaluation during training. Defaults to 10.

        Returns:
            DQNTrainReturnT: Tuple of DQN algorithm state and training result.
        """
        self.weight_recycler = self.recycler_cls(list(runner_state.train_state.params.keys()), **self.weight_recycler_config)
        n_update_steps = int(
            np.ceil(
                n_total_timesteps
                / self.env.n_envs
                / self.hpo_config["train_freq"]
                / n_eval_steps
            )
        )

        def train_eval_step(
            carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], _: None
        ) -> tuple[
            tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], DQNTrainingResult
        ]:
            """Performs one iteration of training and evaluation.

            Args:
                carry (tuple[DQNRunnerState, PrioritisedTrajectoryBufferState]): DQN runner state and buffer state.
                _ (None): Unused parameter (required for jax.lax.scan).

            Returns:
                tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], DQNTrainingResult]: Tuple of (DQN runner state, buffer state) and training result.
            """
            runner_state, buffer_state = carry
            (runner_state, buffer_state), (metrics, trajectories) = jax.lax.scan(
                self._update_step,
                (runner_state, buffer_state),
                jnp.array([n_total_timesteps] * n_update_steps),
                n_update_steps,
            )
            eval_returns = self.eval(runner_state, n_eval_episodes)

            return (runner_state, buffer_state), DQNTrainingResult(
                eval_rewards=eval_returns, trajectories=trajectories, metrics=metrics
            )

        (runner_state, buffer_state), result = jax.lax.scan(
            train_eval_step,
            (runner_state, buffer_state),
            None,
            n_eval_steps,
        )
        return DQNState(runner_state=runner_state, buffer_state=buffer_state), result

    def update(
        self,
        train_state: DQNTrainState,
        observations: jnp.ndarray,
        is_weights: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> tuple[DQNTrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Update the Q-network.

        Args:
            train_state (DQNTrainState): DQN training state.
            observations (jnp.ndarray): Batch of observations.
            actions (jnp.ndarray): Batch of actions.
            next_observations (jnp.ndarray): Batch of next observations.
            rewards (jnp.ndarray): Batch of rewards.
            dones (jnp.ndarray): Batch of dones.

        Returns:
            tuple[DQNTrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Tuple of (train_state, loss, td_error, grads).
        """
        if self.hpo_config["use_target_network"]:
            q_next_target = self.network.apply(
                train_state.target_params, next_observations
            )  # (batch_size, num_actions)
        else:
            q_next_target = self.network.apply(
                train_state.params, next_observations
            )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * self.hpo_config["gamma"] * q_next_target

        def mse_loss(params: FrozenDict | dict) -> tuple[jnp.ndarray, jnp.ndarray]:
            q_pred = self.network.apply(
                params, observations
            )  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(q_pred.shape[0]), actions.squeeze().astype(int)
            ]  # (batch_size,)
            td_error = q_pred - next_q_value

            loss = jnp.mean(is_weights * optax.l2_loss(q_pred, next_q_value))
            return loss, td_error

        (loss_value, td_error), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss_value, td_error, grads


    def _update_step(
        self,
        carry: tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
        n_total_timesteps: int,
    ) -> tuple[
        tuple[DQNRunnerState, PrioritisedTrajectoryBufferState],
        tuple[DQNMetrics | None, Transition | None]
    ]:
        """Performs one iteration of updating including environment steps.

        Args:
            carry (tuple[DQNRunnerState, PrioritisedTrajectoryBufferState]): _description_
            n_total_timesteps (int): Number of environment steps to take.

        Returns:
            tuple[tuple[DQNRunnerState, PrioritisedTrajectoryBufferState], tuple[DQNMetrics | None, Transition | None]]: Updated runner state, recorded transitionsa and metrics if specified.
        """
        runner_state, buffer_state = carry
        (rng, train_state, normalizer_state, env_state, last_obs, global_step) = (
            runner_state
        )

        def take_step(
            carry: tuple[
                chex.PRNGKey,
                DQNTrainState,
                RunningStatisticsState,
                jnp.ndarray,
                Any,
                int,
                PrioritisedTrajectoryBufferState,
            ],
            _: None,
        ) -> tuple[
            tuple[
                chex.PRNGKey,
                DQNTrainState,
                RunningStatisticsState,
                jnp.ndarray,
                Any,
                int,
                PrioritisedTrajectoryBufferState,
            ],
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict],
        ]:
            """Takes one environment step (n_envs many steps).

            Args:
                carry (tuple[chex.PRNGKey, DQNTrainState, RunningStatisticsState, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState]): Carry for jax.lax.scan().
                _ (None): Unused parameter.

            Returns:
                tuple[ tuple[ chex.PRNGKey, DQNTrainState, RunningStatisticsState, jnp.ndarray, Any, int, PrioritisedTrajectoryBufferState, ], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict], ]: _description_
            """
            (
                rng,
                train_state,
                normalizer_state,
                last_obs,
                env_state,
                global_step,
                buffer_state,
            ) = carry

            def random_action(rng: chex.PRNGKey, _: jnp.ndarray) -> jnp.ndarray:
                """Takes a random action.

                Args:
                    rng (chex.PRNGKey): Random number generator key.
                    _ (jnp.ndarray): Observation(s), not used here.

                Returns:
                    jnp.ndarray: Action(s).
                """
                return self.env.sample_actions(rng)

            def greedy_action(_: chex.PRNGKey, obs: jnp.ndarray) -> jnp.ndarray:
                """Takes a greedy action.

                Args:
                    _ (chex.PRNGKey): Random number generator key, not used here.
                    obs (jnp.ndarray): Observation(s).

                Returns:
                    jnp.ndarray: Action(s).
                """
                if self.hpo_config["normalize_observations"]:
                    q_values = self.network.apply(
                        train_state.params,
                        running_statistics.normalize(obs, normalizer_state),
                    )
                else:
                    q_values = self.network.apply(train_state.params, obs)

                return q_values.argmax(axis=-1)

            rng, sample_rng, action_rng = jax.random.split(rng, 3)
            if self.manual_epsilon:
                epsilon = self.epsilon
            else:
                training_fraction = jnp.min(
                    jnp.array([global_step * self.env.n_envs / n_total_timesteps, self.hpo_config["exploration_fraction"]])
                )
                epsilon = self.hpo_config["initial_epsilon"] - training_fraction * (
                    (self.hpo_config["initial_epsilon"] - self.hpo_config["target_epsilon"])
                    / self.hpo_config["exploration_fraction"]
                )

            rand_action = random_action(sample_rng, last_obs)
            greedy_action = greedy_action(action_rng, last_obs)
            action = jax.lax.select(
                jax.random.uniform(sample_rng, shape=last_obs.shape[:1]) < epsilon,
                rand_action,
                greedy_action,
            )

            rng, step_rng = jax.random.split(rng)
            env_state, (obsv, reward, done, info) = self.env.step(
                env_state, action, step_rng
            )

            timestep = TimeStep(
                last_obs=last_obs, obs=obsv, action=action, reward=reward, done=done
            )
            buffer_state = self.buffer.add(buffer_state, timestep)

            global_step += 1

            def target_update(train_state: DQNTrainState) -> DQNTrainState:
                """Update the target network.

                Args:
                    train_state (DQNTrainState): DQN train state.

                Returns:
                    DQNTrainState:  DQN train state with updated target network.
                """
                return train_state.replace(
                    target_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_params,
                        self.hpo_config["tau"],
                    )
                )

            def dont_target_update(train_state: DQNTrainState) -> DQNTrainState:
                """Dummy function for jax.lax.cond(). Same interface as target_update()
                but this function does not change the network.

                Args:
                    train_state (DQNTrainState): DQN train state.

                Returns:
                    DQNTrainState: DQN train state (same as input).
                """
                return train_state

            if self.hpo_config["use_target_network"]:
                train_state = jax.lax.cond(
                    (
                        global_step
                        > self.hpo_config["learning_starts"] // self.env.n_envs
                    )
                    & (
                        global_step
                        % np.ceil(
                            self.hpo_config["target_update_interval"] / self.env.n_envs
                        )
                        == 0
                    ),
                    target_update,
                    dont_target_update,
                    train_state,
                )
            return (
                rng,
                train_state,
                normalizer_state,
                obsv,
                env_state,
                global_step,
                buffer_state,
            ), (
                obsv,
                action,
                reward,
                done,
                info,
            )


        (
            (
                rng,
                train_state,
                normalizer_state,
                last_obs,
                env_state,
                global_step,
                buffer_state,
            ),
            (
                observations,
                action,
                reward,
                done,
                info,
            ),
        ) = jax.lax.scan(
            take_step,
            (
                rng,
                train_state,
                normalizer_state,
                last_obs,
                env_state,
                global_step,
                buffer_state,
            ),
            None,
            self.hpo_config["train_freq"],
        )
        if self.hpo_config["normalize_observations"]:
            normalizer_state = running_statistics.update(normalizer_state, observations)

        def do_update(
            rng: chex.PRNGKey,
            train_state: DQNTrainState,
            normalizer_state: RunningStatisticsState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics
        ]:
            """Update the training state, i.e., the network parameters..

            Args:
                rng (chex.PRNGKey): Random number generator key.
                train_state (DQNTrainState): DQN training state.
                buffer_state (PrioritisedTrajectoryBufferState): Buffer state.

            Returns:
                tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics]:
                Random number generator key, training state, buffer state, and metrics.
            """

            def gradient_step(
                carry: tuple[
                    chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState
                ],
                _: None,
            ) -> tuple[
                tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState],
                DQNMetrics,
            ]:
                """Perform one gradient step and return the metrics..

                Args:
                    carry (tuple[ chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState ]): Carry for jax.lax.scan().
                    _ (None): Unused parameter.

                Returns:
                    tuple[ tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState], DQNMetrics, ]: Updated carry and metrics.
                """
                rng, train_state, buffer_state = carry
                rng, batch_sample_rng = jax.random.split(rng)
                batch = self.buffer.sample(buffer_state, batch_sample_rng)
                experience = batch.experience
                if self.hpo_config["normalize_observations"]:
                    last_obs = running_statistics.normalize(
                        experience.last_obs, normalizer_state
                    )
                    obs = running_statistics.normalize(experience.obs, normalizer_state)
                else:
                    last_obs = experience.last_obs
                    obs = experience.obs
                if self.hpo_config["buffer_prio_sampling"]:
                    is_weights = jnp.power(
                        (1.0 / batch.priorities), self.hpo_config["buffer_beta"]
                    )
                    is_weights = is_weights / jnp.max(is_weights)
                else:
                    is_weights = jnp.ones_like(batch.priorities)
                train_state, loss, td_error, grads = self.update(
                    train_state,
                    last_obs,
                    is_weights,
                    experience.action,
                    obs,
                    experience.reward,
                    experience.done,
                )
                new_priorities = jnp.abs(td_error) + self.hpo_config["buffer_epsilon"]
                buffer_state = self.buffer.set_priorities(
                    buffer_state, batch.indices, new_priorities
                )

                if not self.track_metrics:
                    loss = None
                    td_error = None
                    grads = None

                return (
                    rng,
                    train_state,
                    buffer_state,
                ), DQNMetrics(loss=loss, td_error=td_error, grads=grads)

            (rng, train_state, buffer_state), metrics = jax.lax.scan(
                gradient_step,
                (rng, train_state, buffer_state),
                None,
                self.hpo_config["gradient_steps"],
            )

            return rng, train_state, buffer_state, metrics

        def dont_update(
            rng: chex.PRNGKey,
            train_state: DQNTrainState,
            normalizer_state: RunningStatisticsState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey,
            DQNTrainState,
            RunningStatisticsState,
            PrioritisedTrajectoryBufferState,
            DQNMetrics,
        ]:
            """Dummy function for jax.lax.cond(). Does not update the network.

            Args:
                rng (chex.PRNGKey): Random number generator key.
                train_state (DQNTrainState): DQN train state.
                normalizer_state (RunningStatisticsState): Normalizer state for observation normalization.
                buffer_state (PrioritisedTrajectoryBufferState): Buffer state.

            Returns:
                tuple[chex.PRNGKey, DQNTrainState, RunningStatisticsState, PrioritisedTrajectoryBufferState, DQNMetrics]: Input parameters and dummy metrics.
            """
            if self.track_metrics:
                loss = jnp.array(
                    [((jnp.array([0]) - jnp.array([0])) ** 2).mean()]
                    * self.hpo_config["gradient_steps"]
                )
                td_error = jnp.array(
                    [
                        [[1] * self.hpo_config["buffer_batch_size"]]
                        * self.hpo_config["gradient_steps"]
                    ]
                ).mean(axis=0)
                grads = jax.tree_map(
                    lambda x: jnp.stack([x] * self.hpo_config["gradient_steps"]),
                    train_state.params,
                )
            else:
                loss = None
                td_error = None
                grads = None
            return (
                rng,
                train_state,
                buffer_state,
                DQNMetrics(loss=loss, td_error=td_error, grads=grads),
            )

        rng, train_state, buffer_state, metrics = jax.lax.cond(
            global_step
            > np.ceil(self.hpo_config["learning_starts"] // self.env.n_envs),
            do_update,
            dont_update,
            rng,
            train_state,
            normalizer_state,
            buffer_state,
        )

        # Neuron Recycling
        if not self.manual_recycling:
            train_state, recycled = self.recycle_neurons(train_state, buffer_state, global_step, rng)

        if not self.manual_offline_updates:
            offline_steps = int(self.weight_recycler.reset_period * self.offline_update_fraction * self.hpo_config["gradient_steps"])
            if offline_steps > 0:
                rng, train_state, buffer_state, metrics = self.fit_offline(rng, buffer_state, train_state, normalizer_state, global_step, recycled, offline_steps)

        runner_state = DQNRunnerState(
            rng=rng,
            train_state=train_state,
            normalizer_state=normalizer_state,
            env_state=env_state,
            obs=last_obs,
            global_step=global_step,
        )
        trajectories = None
        if self.track_trajectories:
            trajectories = Transition(
                obs=observations,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
        return (runner_state, buffer_state), (metrics, trajectories)
    
    def recycle_neurons(self, train_state, buffer_state, global_step, rng, force=None):
        intermediates = self.get_intermediates(rng, train_state.params, buffer_state)
        rng, key = jax.random.split(rng)
        params, opt_state, recycled = self.weight_recycler.maybe_update_weights(
                    global_step, intermediates, train_state.params, key, train_state.opt_state, is_reset=force
                )
        train_state = train_state.replace(params=params, opt_state=opt_state)
        return train_state, recycled

    def fit_offline(self, rng, buffer_state, train_state, normalizer_state, global_step, recycled, steps):
        def do_update(
            rng: chex.PRNGKey,
            train_state: DQNTrainState,
            normalizer_state: RunningStatisticsState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics
        ]:
            """Update the training state, i.e., the network parameters..

            Args:
                rng (chex.PRNGKey): Random number generator key.
                train_state (DQNTrainState): DQN training state.
                buffer_state (PrioritisedTrajectoryBufferState): Buffer state.

            Returns:
                tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState, DQNMetrics]:
                Random number generator key, training state, buffer state, and metrics.
            """

            def gradient_step(
                carry: tuple[
                    chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState
                ],
                _: None,
            ) -> tuple[
                tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState],
                DQNMetrics,
            ]:
                """Perform one gradient step and return the metrics..

                Args:
                    carry (tuple[ chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState ]): Carry for jax.lax.scan().
                    _ (None): Unused parameter.

                Returns:
                    tuple[ tuple[chex.PRNGKey, DQNTrainState, PrioritisedTrajectoryBufferState], DQNMetrics, ]: Updated carry and metrics.
                """
                rng, train_state, buffer_state = carry
                rng, batch_sample_rng = jax.random.split(rng)
                batch = self.buffer.sample(buffer_state, batch_sample_rng)
                experience = batch.experience
                if self.hpo_config["normalize_observations"]:
                    last_obs = running_statistics.normalize(
                        experience.last_obs, normalizer_state
                    )
                    obs = running_statistics.normalize(experience.obs, normalizer_state)
                else:
                    last_obs = experience.last_obs
                    obs = experience.obs
                if self.hpo_config["buffer_prio_sampling"]:
                    is_weights = jnp.power(
                        (1.0 / batch.priorities), self.hpo_config["buffer_beta"]
                    )
                    is_weights = is_weights / jnp.max(is_weights)
                else:
                    is_weights = jnp.ones_like(batch.priorities)
                train_state, loss, td_error, grads = self.update(
                    train_state,
                    last_obs,
                    is_weights,
                    experience.action,
                    obs,
                    experience.reward,
                    experience.done,
                )
                new_priorities = jnp.abs(td_error) + self.hpo_config["buffer_epsilon"]
                buffer_state = self.buffer.set_priorities(
                    buffer_state, batch.indices, new_priorities
                )

                return (
                    rng,
                    train_state,
                    buffer_state,
                ), DQNMetrics(loss=loss, td_error=td_error, grads=grads)

            (rng, train_state, buffer_state), metrics = jax.lax.scan(
                gradient_step,
                (rng, train_state, buffer_state),
                None,
                steps,
            )

            return rng, train_state, buffer_state, metrics

        def dont_update(
            rng: chex.PRNGKey,
            train_state: DQNTrainState,
            normalizer_state: RunningStatisticsState,
            buffer_state: PrioritisedTrajectoryBufferState,
        ) -> tuple[
            chex.PRNGKey,
            DQNTrainState,
            RunningStatisticsState,
            PrioritisedTrajectoryBufferState,
            DQNMetrics,
        ]:
            """Dummy function for jax.lax.cond(). Does not update the network.

            Args:
                rng (chex.PRNGKey): Random number generator key.
                train_state (DQNTrainState): DQN train state.
                normalizer_state (RunningStatisticsState): Normalizer state for observation normalization.
                buffer_state (PrioritisedTrajectoryBufferState): Buffer state.

            Returns:
                tuple[chex.PRNGKey, DQNTrainState, RunningStatisticsState, PrioritisedTrajectoryBufferState, DQNMetrics]: Input parameters and dummy metrics.
            """
            loss = jnp.array(
                    [((jnp.array([0]) - jnp.array([0])) ** 2).mean()]
                    * steps
                )
            td_error = jnp.array(
                    [
                        [[1] * self.hpo_config["buffer_batch_size"]]
                        * steps
                    ]
                ).mean(axis=0)
            grads = jax.tree_map(
                    lambda x: jnp.stack([x] * steps),
                    train_state.params,
                )
            return (
                rng,
                train_state,
                buffer_state,
                DQNMetrics(loss=loss, td_error=td_error, grads=grads),
            )

        loss = jnp.array(
                    [((jnp.array([0]) - jnp.array([0])) ** 2).mean()]
                    * steps
                )
        td_error = jnp.array(
                    [
                        [[1] * self.hpo_config["buffer_batch_size"]]
                        * steps
                    ]
                ).mean(axis=0)
        grads = jax.tree_map(
                    lambda x: jnp.stack([x] * steps),
                    train_state.params,
                )
        metrics = DQNMetrics(loss=loss, td_error=td_error, grads=grads)
        stepped_far = global_step > np.ceil(self.hpo_config["learning_starts"] // self.env.n_envs)
        for _ in range(steps):
                rng, train_state, buffer_state, metrics = jax.lax.cond(
                stepped_far & recycled,
                do_update,
                dont_update,
                rng,
                train_state,
                normalizer_state,
                buffer_state,
            )
        return rng, train_state, buffer_state, metrics

    def _sample_batch_for_statistics(self, rng, buffer_state):
        rng, batch_sample_rng = jax.random.split(rng)
        batch = self.buffer.sample(buffer_state, batch_sample_rng)
        experience = batch.experience
        return experience.obs

    def get_intermediates(self, rng, online_params, buffer_state):
        batch = self._sample_batch_for_statistics(rng, buffer_state)

        def apply_data(x):
            filter_rep = lambda l, _: l.name is not None and "act" in l.name
            return self.network.apply(
                online_params,
                x,
                capture_intermediates=filter_rep,
                mutable=["intermediates"],
            )

        _, state = jax.vmap(apply_data)(batch)
        return state["intermediates"]