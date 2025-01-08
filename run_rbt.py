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
from arlbench.core.algorithms import DQN, ResetDQN
from arlbench.utils.dict_helpers import to_dict
from arlbench.utils.sbv import compute_msbe, get_train_data, get_val_data

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
from collections import defaultdict
from ConfigSpace import ConfigurationSpace, Configuration
import pandas as pd

OmegaConf.register_new_resolver("eval", eval)


import absl.logging
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

    prev_incumbent_config = None
    incumbent_performances = []
    incumbent_eval_performances = []
    full_evals = defaultdict(list)
    td_errors = defaultdict(list)
    msbes = defaultdict(list)
    iteration = 0

    hp_config = to_dict(cfg.hp_config)

    while iteration < cfg.n_iterations and not done:
        logger.info(f"Starting iteration {iteration}")

        logger.info("Running algorithm for one step...")
        _, objectives, te, tr, info = env.step(hp_config)
        train_info_dfs.append(info["train_info_df"])
        logger.info("Done.")

        train_rewards.append(objectives)
        done = te or tr

        tag = f"rbt_iteration_{iteration}"

        save_path = env._save(tag=tag)
        assert env._algorithm_state is not None

        # We need to backup the buffer state to assign it again
        # after loading the checkpoint. As the buffer has a custom
        # state containing the train and validation splits, we 
        # cannot use the default state that is used when loading
        # a checkpoint
        buffer_state = env._algorithm_state.buffer_state

        rng = jax.random.key(cfg.autorl.seed)

        logger.info("Setting up SMAC...")

        hyperparameters = {}
        for name, hp in DQN.get_hpo_config_space().items():
            if name in cfg.hyperparameters:
                hyperparameters[name] = hp

        search_space = ConfigurationSpace(hyperparameters)

        # TODO: what happens if I keep asking after that? Do I need a reset?
        scenario = Scenario(
            search_space,
            n_trials=cfg.n_configs_per_iteration,
            min_budget=cfg.hb_min_budget,  # At least one offline fitting step
            max_budget=cfg.hb_max_budget,  # At most 25 offline fitting steps
            name=tag
        )

        # Create our intensifier
        intensifier = Hyperband(scenario, eta=cfg.hb_eta)

        def dummy(config, budget, seed):
            return 0

        # We add the incumbent of the last iteration to the initial design
        if prev_incumbent_config is not None:
            additional_configs = [Configuration(search_space, values=prev_incumbent_config)]
        else:
            additional_configs = None
        
        # Create our SMAC object and pass the scenario and the train method
        initial_design = MFFacade.get_initial_design(scenario=scenario, additional_configs=additional_configs)
        smac = MFFacade(
            scenario,
            dummy,
            intensifier=intensifier,
            overwrite=True,
            logging_level=False,
            initial_design=initial_design
        )
        logger.info("Done.")

        incumbent_path = None
        incumbent_performance = None

        if cfg.eval_criterion == "msbe":
            logger.info("Getting training data...")
            from sklearn.linear_model import LinearRegression

            X_train, y_train = get_train_data(env)
            model = LinearRegression().fit(X_train, y_train)
            logger.info("Done.")
        else:
            model = None

        n_configs = 0
        while n_configs < cfg.n_configs_per_iteration:
            logger.info(f"Starting config {n_configs} for iteration {iteration}")
            env._load(save_path, seed=cfg.autorl.seed, buffer_state=buffer_state)
            config = smac.ask()
            
            budget = config.budget
            hp_config = to_dict(config.config)

            new_hp_config = dict(ResetDQN.get_default_hpo_config())
            for k, v in hp_config.items():
                new_hp_config[k] = v

            env._hpo_config = new_hp_config

            logger.info("Recycling neurons...")
            train_state, _ = env._algorithm.recycle_neurons(env._algorithm_state.runner_state.train_state, env._algorithm_state.buffer_state, env._algorithm_state.runner_state.global_step, rng, True)
            logger.info("Done.")
            
            logger.info("Fitting offline...")
            rng, train_state, _, metrics = env._algorithm.fit_offline(
                int(budget),
                rng,
                env._algorithm_state.buffer_state,
                train_state,
                env._algorithm_state.runner_state.normalizer_state,
                env._algorithm_state.runner_state.global_step,
                True,
            )
            runner_state = env._algorithm_state.runner_state._replace(train_state=train_state)
            env.algorithm_state = env._algorithm_state._replace(runner_state=runner_state)
            n_configs += 1
            logger.info("Done.")
            
            logger.info("Evaluating config...")
            eval = -env.eval(cfg.n_eval_episodes).mean()
            logger.info("Done.")

            full_evals[iteration].append(eval)
            if cfg.eval_criterion == "eval_return":
                performance = eval
            elif cfg.eval_criterion == "td_error":
                performance = np.abs(metrics.td_error.mean())
            elif cfg.eval_criterion == "msbe":
                assert model is not None
                logger.info("Computing MSBE...")
                X_val, y_val = get_val_data(env)

                performance = float(np.mean((model.predict(X_val) - y_val) ** 2))
                msbes[iteration].append(performance)
                logger.info("Done.")

            td_errors[iteration].append(metrics.td_error.mean())

            logger.info(f"Config {n_configs} for iteration {iteration} finished with performance {performance}")

            smac_return = TrialValue(cost=performance, time=0.5)
            smac.tell(config, smac_return)
            if incumbent_performance is None or performance < incumbent_performance:
                incumbent_performance = performance
                incumbent_eval_performance = eval
                incumbent_config = env._hpo_config

                shutil.rmtree(incumbent_path, ignore_errors=True)
                incumbent_path = env._save(tag=f"rbt_incumbent_iteration_{iteration}")

        if incumbent_path is not None:
            env._load(incumbent_path, seed=cfg.autorl.seed)
        hp_config = incumbent_config
        prev_incumbent_config = config.config
        incumbent_performances.append(incumbent_performance)
        incumbent_eval_performances.append(incumbent_eval_performance)
        iteration += 1

    with open("incumbent_performances.csv", "w") as f:
        f.write("iteration,incumbent_performance\n")
        for i, p in enumerate(incumbent_performances):
            f.write(f"{i},{p}\n")

    with open("incumbent_eval_performances.csv", "w") as f:
        f.write("iteration,incumbent_performance\n")
        for i, p in enumerate(incumbent_eval_performances):
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

    if len(msbes) > 0:
        with open("msbes.csv", "w") as f:
            f.write("iteration,config_id,msbe\n")
            for i, errors in msbes.items():
                for j, e in enumerate(errors):
                    f.write(f"{i},{j},{e}\n")

    train_info_dfs = pd.concat(train_info_dfs)
    train_info_dfs.to_csv("train_info.csv", index=False)

@hydra.main(version_base=None, config_path="examples/configs", config_name="rbt")
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
