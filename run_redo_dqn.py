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
from arlbench.core.algorithms import ResetDQN
from arlbench.utils.dict_helpers import to_dict
from arlbench.utils.sbv import get_train_data, get_val_data
import functools

from hydra_plugins.hypersweeper.search_space_encoding import \
    search_space_to_config_space
from smac import MultiFidelityFacade as MFFacade, HyperparameterOptimizationFacade as HPOFacade, RandomFacade
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
import shutil
from smac.runhistory.dataclasses import TrialValue
from omegaconf import OmegaConf
import numpy as np
from collections import defaultdict
from ConfigSpace import Configuration, ConfigurationSpace, UniformIntegerHyperparameter
import pandas as pd

OmegaConf.register_new_resolver("eval", eval)


import logging


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
    train_info_dfs = []

    full_evals = defaultdict(list)
    iteration = 0

    hp_config = dict(ResetDQN.get_default_hpo_config())
    offline_steps = int(env._algorithm.weight_recycler.reset_period * env._algorithm.offline_update_fraction * hp_config["gradient_steps"])

    while iteration < cfg.n_iterations and not done:
        logger.info(f"Starting iteration {iteration}")

        logger.info("Running algorithm for one step...")
        _, objectives, te, tr, info = env.step(hp_config)
        train_info_dfs.append(info["train_info_df"])
        logger.info("Done.")

        train_rewards.append(objectives)
        done = te or tr

        rng = jax.random.key(cfg.autorl.seed)

        logger.info("Recycling neurons...")
        train_state, _ = env._algorithm.recycle_neurons(env._algorithm_state.runner_state.train_state, env._algorithm_state.buffer_state, env._algorithm_state.runner_state.global_step, rng, True)
        logger.info("Done.")

        logger.info("Fitting offline...")
        rng, train_state, _, metrics = env._algorithm.fit_offline(
            offline_steps,
            rng,
            env._algorithm_state.buffer_state,
            train_state,
            env._algorithm_state.runner_state.normalizer_state,
            env._algorithm_state.runner_state.global_step,
            True,
        )
        runner_state = env._algorithm_state.runner_state._replace(train_state=train_state)
        env.algorithm_state = env._algorithm_state._replace(runner_state=runner_state)
        logger.info("Done.")

    with open("full_evals.csv", "w") as f:
        f.write("iteration,config_id,full_eval_performance\n")
        for i, evals in full_evals.items():
            for j, p in enumerate(evals):
                f.write(f"{i},{j},{p}\n")

    train_info_dfs = pd.concat(train_info_dfs)
    train_info_dfs.to_csv("train_info.csv", index=False)

    if cfg.remove_checkpoints is True:
        shutil.rmtree("./checkpoints", ignore_errors=True)

@hydra.main(version_base=None, config_path="examples/configs", config_name="redo_dqn")
def execute(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.getLogger('absl').setLevel(logging.ERROR)

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
