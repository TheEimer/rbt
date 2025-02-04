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
from omegaconf import OmegaConf
from collections import defaultdict
import pandas as pd
import os
from functools import partial

OmegaConf.register_new_resolver("eval", eval)


import logging


if TYPE_CHECKING:
    from omegaconf import DictConfig


def run(cfg: DictConfig, logger: logging.Logger):
    # We check if we need to load a checkpoint for HyperPBT
    # If so, we load the first episode and first step of ARLBench since we always run only
    # one iteration
    if "load" in cfg and cfg.load:
        checkpoint_path = os.path.join(
            cfg.load,
            cfg.autorl.checkpoint_name,
            "default_checkpoint_c_episode_1_step_1",
        )
    else:
        checkpoint_path = None

    # We check if we need to save a checkpoint for HyperPBT
    # If so, we need to adapt the autorl config accordingly
    if "save" in cfg and cfg.save:
        cfg.autorl.checkpoint_dir = str(cfg.save).replace(".pt", "")
        if cfg.algorithm == "PPO":
            cfg.autorl.checkpoint = ["opt_state", "params"]
        else:
            cfg.autorl.checkpoint = ["opt_state", "params", "buffer"]

    # Initialize environment with general config
    autorl_cfg = OmegaConf.to_container(cfg.autorl, resolve=True)
    assert isinstance(autorl_cfg, dict)

    env = AutoRLEnv(config=autorl_cfg)
    _ = env.reset()

    logger.info("Running algorithm for one step...")
    _, _, _, _, info = env.step(cfg.hp_config, checkpoint_path=checkpoint_path)
    logger.info("Done.")

    rng = jax.random.key(cfg.autorl.seed)

    logger.info("Recycling neurons...")
    train_state, _ = env._algorithm.recycle_neurons(env._algorithm_state.runner_state.train_state, env._algorithm_state.buffer_state, env._algorithm_state.runner_state.global_step, rng, True)
    logger.info("Done.")

    logger.info("Fitting offline...")
    offline_steps = int(env._algorithm.weight_recycler.reset_period * cfg.replay_ratio)
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

    # Additionally, we store the evaluation rewards we had during training
    info["train_info_df"].to_csv("evaluation.csv", index=False)

    return env.eval(cfg.n_eval_episodes).mean()
    

@hydra.main(version_base=None, config_path="examples/configs", config_name="redo_dqn_pbt")
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
