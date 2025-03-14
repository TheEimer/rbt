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
from arlbench.autorl.checkpointing import Checkpointer
from arlbench.utils.dict_helpers import to_dict
from arlbench.utils.sbv import get_train_data, get_val_data

from hydra_plugins.hypersweeper.search_space_encoding import \
    search_space_to_config_space
from smac import MultiFidelityFacade as MFFacade, HyperparameterOptimizationFacade as HPOFacade, RandomFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac import Scenario
from smac.intensifier.hyperband import Hyperband
import shutil
from smac.runhistory.dataclasses import TrialValue
from omegaconf import OmegaConf
import numpy as np
from collections import defaultdict
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
import pandas as pd

OmegaConf.register_new_resolver("eval", eval)


import logging


if TYPE_CHECKING:
    from omegaconf import DictConfig


def get_smac(
        cfg: DictConfig,
        configspace: ConfigurationSpace,
        tag: str,
        prev_incumbent_config: Configuration | None = None
    ):      
    """Instantiate a SMAC instance given the configuration.""" 
    # We add the incumbent of the last iteration to the initial design
    if prev_incumbent_config is not None:
        additional_configs = [Configuration(configspace, values=prev_incumbent_config)]
    else:
        additional_configs = []

    # This is just because SMAC expects a function,
    # it is not used at all
    def dummy(config, budget, seed):
        return 0
        
    if cfg.optimizer == "smac":   
        if cfg.optimize_replay_ratio:
            # We set n_trials abritrarily high, as we will stop the optimization
            # after the budget is exhausted
            n_trials = 100000

            # For the initial design we use 25% of the expected budget
            n_initial_configs = int(cfg.budget_per_iteration / cfg.replay_ratio * 0.25)
        else:
            if cfg.continuous_smac:
                n_trials = cfg.n_configs_per_iteration * cfg.n_iterations
            else:
                n_trials = cfg.n_configs_per_iteration

            # We always want to spend only 25% of the budget within the 
            # first iteration for the initial design
            n_initial_configs = int(cfg.n_configs_per_iteration * 0.25)

        scenario = Scenario(
            configspace=configspace,
            n_trials=n_trials,
            name=tag,
            use_default_config=True,
            seed=cfg.autorl.seed,
        )

        initial_design = HPOFacade.get_initial_design(
            scenario=scenario,
            n_configs=n_initial_configs,
            additional_configs=additional_configs,
        )
        return HPOFacade(
            scenario=scenario,
            target_function=dummy,
            overwrite=True,
            intensifier=HPOFacade.get_intensifier(scenario, max_config_calls=1),
            logging_level=False,
            initial_design=initial_design,
            config_selector=HPOFacade.get_config_selector(scenario, retrain_after=1),
        )
    elif cfg.optimizer == "rs":
        if cfg.optimize_replay_ratio:
            # We set n_trials abritrarily high, as we will stop the optimization
            # after the budget is exhausted
            n_trials = 100000
        else:
            if cfg.continuous_smac:
                n_trials = cfg.n_configs_per_iteration * cfg.n_iterations
            else:
                n_trials = cfg.n_configs_per_iteration

        scenario = Scenario(
            configspace=configspace,
            n_trials=n_trials,
            name=tag,
            seed=cfg.autorl.seed
        )
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=n_trials,
            additional_configs=additional_configs
        )

        return HPOFacade(
            scenario=scenario,
            target_function=dummy,
            overwrite=True,
            logging_level=False,
            intensifier=HPOFacade.get_intensifier(scenario, max_config_calls=1),
            initial_design=initial_design,
            model=RandomFacade.get_model(scenario),
            acquisition_function=RandomFacade.get_acquisition_function(scenario),
            acquisition_maximizer=RandomFacade.get_acquisition_maximizer(scenario),
        )
    elif cfg.optimizer == "smac_mf":
        if cfg.continuous_smac:
            n_trials = cfg.n_hb_configs_per_iteration * cfg.n_iterations
        else:
            n_trials = cfg.n_hb_configs_per_iteration

        # We always want to spend only 25% of the budget within the 
        # first iteration for the initial design
        n_initial_configs = int(cfg.n_hb_configs_per_iteration * 0.25)

        scenario = Scenario(
            configspace=configspace,
            n_trials=n_trials,
            name=tag,
            min_budget=cfg.hb_min_budget,
            max_budget=cfg.hb_max_budget,
            use_default_config=True,
            seed=cfg.autorl.seed
        )

        initial_design = MFFacade.get_initial_design(
            scenario=scenario,
            n_configs=n_initial_configs,
            additional_configs=additional_configs
        )
        intensifier = Hyperband(scenario, eta=cfg.hb_eta, incumbent_selection="any_budget")
        
        return MFFacade(
            scenario=scenario,
            target_function=dummy,
            intensifier=intensifier,
            overwrite=True,
            logging_level=False,
            initial_design=initial_design
        )
    elif cfg.optimizer == "rs_mf":
        if cfg.continuous_smac:
            n_trials = cfg.n_hb_configs_per_iteration * cfg.n_iterations
        else:
            n_trials = cfg.n_hb_configs_per_iteration
    
        scenario = Scenario(
            configspace=configspace,
            n_trials=n_trials,
            name=tag,
            min_budget=cfg.hb_min_budget,
            max_budget=cfg.hb_max_budget,
            use_default_config=True,
            seed=cfg.autorl.seed,
        )
        initial_design = SobolInitialDesign(
            scenario=scenario,
            n_configs=n_trials,
            additional_configs=additional_configs
        )
        intensifier = Hyperband(scenario, eta=cfg.hb_eta, incumbent_selection="any_budget")

        return MFFacade(
            scenario=scenario,
            target_function=dummy,
            intensifier=intensifier,
            model=RandomFacade.get_model(scenario),
            acquisition_function=RandomFacade.get_acquisition_function(scenario),
            acquisition_maximizer=RandomFacade.get_acquisition_maximizer(scenario),
            initial_design=initial_design,
            overwrite=True,
            logging_level=False,
            config_selector=RandomFacade.get_config_selector(scenario, retrain_after=1),
        )
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def run(cfg: DictConfig, logger: logging.Logger):
    # Initialize environment with general config
    autorl_cfg = OmegaConf.to_container(cfg.autorl, resolve=True)
    assert isinstance(autorl_cfg, dict)

    train_rewards = []
    train_info_dfs = []

    incumbent_ckpt_path = None
    prev_incumbent_config = None
    incumbent_performances = []
    incumbent_eval_performances = []
    full_evals = defaultdict(list)
    td_errors = defaultdict(list)
    msbes = defaultdict(list)
    iteration = 0

    configspace = search_space_to_config_space(search_space=cfg.search_space)    
    if cfg.optimize_replay_ratio and (cfg.optimizer == "smac" or cfg.optimizer == "rs"):
        # We add the replay ratio as a hyperparameter to the config space.
        # to let SMAC directly optimize it. This is not the case
        # for hyperband-based approaches that use the gradient steps
        # as budget
        configspace.add(
                UniformFloatHyperparameter(
                    name="replay_ratio",
                    lower=cfg.hb_max_budget / 50,
                    upper=cfg.hb_max_budget * 2,
                    default_value=cfg.hb_max_budget
                )
            )   
        
    hp_config = dict(cfg.hp_config)

    env = AutoRLEnv(config=autorl_cfg)
    _ = env.reset()

    while iteration < cfg.n_iterations:
        logger.info(f"Starting iteration {iteration}")

        logger.info("Running algorithm for one step...")
        # If selected, we use the default configuration to collect 
        # the rollouts
        if cfg.use_default_for_rollouts:
            hp_config = dict(cfg.hp_config)

        _, objectives, _, _, info = env.step(hp_config, checkpoint_path=incumbent_ckpt_path)
        train_info_dfs.append(info["train_info_df"])
        logger.info("Done.")

        train_rewards.append(objectives)

        tag = f"rbt_iteration_{iteration}"

        if not cfg.continuous_smac or (cfg.continuous_smac and iteration == 0):
            # By default, we re-instantiate SMAC in each iteration. This is not
            # the case for continuous SMAC
            logger.info("Setting up SMAC...")
            smac = get_smac(
                cfg=cfg,
                configspace=configspace,
                tag=tag,
                prev_incumbent_config=prev_incumbent_config
            )
            logger.info("Done.")

        save_path = env._save(tag=tag)
        assert env._algorithm_state is not None

        # DEBUG: This is for checking how the buffer quality affects the performance
        if "load_buffer_state" in cfg and cfg.load_buffer_state is not None:
            logger.info("Loading buffer state from checkpoint...")
            _, algorithm_kw_args = Checkpointer.load(cfg.load_buffer_state, env._algorithm_state)
            env._algorithm_state = env._algorithm_state._replace(buffer_state=algorithm_kw_args["buffer_state"])
            logger.info("Done.")
 
        # We need to backup the buffer state to assign it again
        # after loading the checkpoint. As the buffer has a custom
        # state containing the train and validation splits, we 
        # cannot use the default state that is used when loading
        # a checkpoint
        buffer_state = env._algorithm_state.buffer_state

        rng = jax.random.key(cfg.autorl.seed)
        incumbent_performance = None

        if cfg.eval_criterion == "msbe":
            # We compute the train and test splits in advance
            # to avoid recomputing them in each iteration
            logger.info("Getting training data...")
            from sklearn.linear_model import LinearRegression

            X_train, y_train = get_train_data(env)
            model = LinearRegression().fit(X_train, y_train)
            logger.info("Done.")
        else:
            model = None

        total_budget = 0
        n_configs = 0
        while total_budget < cfg.budget_per_iteration:
            config = smac.ask()
            hp_config = to_dict(config.config)

            if cfg.optimizer == "smac" or cfg.optimizer == "rs":
                if "replay_ratio" in hp_config:
                    replay_ratio = hp_config.pop("replay_ratio")
                else:
                    replay_ratio = cfg.replay_ratio
            else:
                assert config.budget is not None
                replay_ratio = config.budget

            # We only update the hyperparameters that are in the config space,
            # everything else is set to default
            new_hp_config = dict(cfg.hp_config)
            for k, v in hp_config.items():
                new_hp_config[k] = v
            
            logger.info(f"Starting config {n_configs} for iteration {iteration} with replay ratio {replay_ratio}")

            # Due to JAX caching, we need to re-instantiate the algorithm
            # with the new hyperparameters as this forces JAX to recompile
            env._hpo_config = new_hp_config
            env._algorithm = env._make_algorithm()
            env._algorithm_state = env._load(save_path, seed=cfg.autorl.seed, buffer_state=buffer_state)
            
            logger.info("Recycling neurons...")
            train_state, _ = env._algorithm.recycle_neurons(env._algorithm_state.runner_state.train_state, env._algorithm_state.buffer_state, env._algorithm_state.runner_state.global_step, rng, True)
            runner_state = env._algorithm_state.runner_state._replace(train_state=train_state)
            env._algorithm_state = env._algorithm_state._replace(runner_state=runner_state)
            logger.info("Done.")

            offline_steps = max(1, int(env._algorithm.weight_recycler.reset_period / env._algorithm.hpo_config["buffer_batch_size"] * replay_ratio))
            logger.info(f"Fitting offline for {offline_steps} steps...")
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
            env._algorithm_state = env._algorithm_state._replace(runner_state=runner_state)
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

            logger.info(f"Config {n_configs} for iteration {iteration} finished with cost {performance}")

            smac_return = TrialValue(cost=performance, time=0.5)
            smac.tell(config, smac_return)

            if incumbent_performance is None or performance < incumbent_performance:
                incumbent_performance = performance
                incumbent_eval_performance = eval
                incumbent_config = env._hpo_config
                
                if incumbent_ckpt_path is not None:
                    shutil.rmtree(incumbent_ckpt_path, ignore_errors=True)
                incumbent_ckpt_path = env._save(tag=f"rbt_incumbent_iteration_{iteration}")

            n_configs += 1
            total_budget += replay_ratio
        
        hp_config = incumbent_config
        prev_incumbent_config = config.config

        logging.info("#" * 80)
        logging.info(f"Best performance for iteration {iteration}: {incumbent_performance}")
        logging.info(f"Best eval performance for iteration {iteration}: {incumbent_eval_performance}")
        logging.info("#" * 80)

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

    if cfg.remove_checkpoints is True:
        shutil.rmtree("./checkpoints", ignore_errors=True)

@hydra.main(version_base=None, config_path="examples/configs", config_name="rbt_light_reset")
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
