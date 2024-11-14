"""Console script for arlbench."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import logging
import sys
import traceback
from typing import TYPE_CHECKING

import hydra
import jax
from arlbench.autorl import AutoRLEnv
from arlbench.core.environments import make_env
from arlbench.core.algorithms import DQN
from arlbench.utils.dict_helpers import to_dict

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
from flax.training import checkpoints
from flax.training import orbax_utils
import orbax
from pathlib import Path
import shutil
from hydra.utils import get_original_cwd
import jax.numpy as jnp
from smac.runhistory.dataclasses import TrialValue
from omegaconf import OmegaConf
import numpy as np


OmegaConf.register_new_resolver("eval", eval)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(cfg: DictConfig, logger: logging.Logger):
    # Initialize environment with general config
    autorl_cfg = OmegaConf.to_container(cfg.autorl, resolve=True)
    assert isinstance(autorl_cfg, dict)

    env = AutoRLEnv(config=autorl_cfg)

    # Reset environment and run for 10 steps
    _ = env.reset()
    done = False

    train_rewards = []
    incumbent_performances = []
    full_evals = {}
    td_errors = {}
    iteration = 0

    hp_config = to_dict(cfg.hp_config)

    while iteration < cfg.n_iterations and not done:
        _, objectives, te, tr, _ = env.step(hp_config)
        train_rewards.append(objectives)
        done = te or tr

        save_path = env._save(tag=f"rbt_iteration_{iteration}")
        rng = jax.random.key(cfg.autorl.seed)

        # TODO: what happens if I keep asking after that? Do I need a reset?
        scenario = Scenario(
            DQN.get_hpo_config_space(),
            n_trials=cfg.n_configs_per_iteration,
            min_budget=10,  # At least one offline fitting step
            max_budget=250,  # At most 25 offline fitting steps
        )

        # Create our intensifier
        intensifier = Hyperband(scenario, eta=cfg.hb_eta)

        def dummy(config, budget, seed):
            return 0

        # Create our SMAC object and pass the scenario and the train method
        smac = MFFacade(
            scenario,
            dummy,
            intensifier=intensifier,
            overwrite=True,
        )

        incumbent_path = None
        incumbent_performance = None
        full_evals[iteration] = []
        td_errors[iteration] = []

        current_budget = 10000
        n_configs = 0
        while n_configs < cfg.n_configs_per_iteration:
            env._load(save_path, seed=cfg.autorl.seed)
            config = smac.ask()
            if current_budget == 10000:
                current_budget = config.budget
            budget = config.budget
            hp_config = to_dict(config.config)

            env._hpo_config = hp_config

            train_state, _ = env._algorithm.recycle_neurons(env._algorithm_state.runner_state.train_state, env._algorithm_state.buffer_state, env._algorithm_state.runner_state.global_step, rng, True)
            rng, train_state, _, metrics = env._algorithm.fit_offline(
                rng,
                env._algorithm_state.buffer_state,
                train_state,
                env._algorithm_state.runner_state.normalizer_state,
                env._algorithm_state.runner_state.global_step,
                True,
                int(budget),
            )
            runner_state = env._algorithm_state.runner_state._replace(train_state=train_state)
            env.algorithm_state = env._algorithm_state._replace(runner_state=runner_state)
            n_configs += 1
            
            eval = -env.eval(cfg.n_eval_episodes).mean()
            full_evals[iteration].append(eval)
            if cfg.full_eval:
                performance = eval
            else:
                performance = np.abs(metrics.td_error.mean())

            td_errors[iteration].append(metrics.td_error.mean())

            smac_return = TrialValue(cost=performance, time=0.5)
            smac.tell(config, smac_return)
            if incumbent_performance is None or performance < incumbent_performance:
                incumbent_performance = performance
                incumbent_config = to_dict(config.config)

                shutil.rmtree(incumbent_path, ignore_errors=True)
                incumbent_path = env._save(tag=f"rbt_incumbent_iteration_{iteration}")

        if incumbent_path is not None:
            env._load(incumbent_path, seed=cfg.autorl.seed)
        hp_config = incumbent_config
        incumbent_performances.append(incumbent_performance)
        iteration += 1

    with open("results.csv", "w") as f:
        f.write("iteration,incumbent_performance\n")
        for i, p in enumerate(incumbent_performances):
            f.write(f"{i},{p}\n")

    with open("full_evals.csv", "w") as f:
        f.write("iteration,config_id,full_eval_performance\n")
        for i, evals in full_evals.items():
            for j, p in enumerate(evals):
                f.write(f"{i},{j},{p}\n")

    with open("td_errors.csv", "w") as f:
        f.write("iteration,config_id,td_error\n")
        for i, errors in td_errors.items():
            for j, e in enumerate(errors):
                f.write(f"{i},{j},{e}\n")


@hydra.main(version_base=None, config_path="examples/configs", config_name="rbt")
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(execute())  # pragma: no cover
