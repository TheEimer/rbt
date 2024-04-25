from __future__ import annotations

import argparse
import logging
import os
import time

import jax
import pandas as pd
from arlbench.core.algorithms import SAC
from arlbench.core.environments import make_env


def test_sac(
    dir_name, log, framework, env_name, config, training_kw_args, seed, cnn_policy
):
    env = make_env(framework, env_name, n_envs=config["n_envs"], seed=seed)
    rng = jax.random.PRNGKey(seed)

    hpo_config = SAC.get_default_hpo_config()
    hpo_config["tau"] = 0.01
    hpo_config["learning_starts"] = 10000
    hpo_config["buffer_batch_size"] = 256
    hpo_config["buffer_alpha"] = 0.0
    hpo_config["buffer_beta"] = 0.0
    nas_config = SAC.get_default_nas_config()
    nas_config["activation"] = "relu"
    nas_config["hidden_size"] = 350

    agent = SAC(hpo_config, env, nas_config=nas_config, cnn_policy=cnn_policy)
    algorithm_state = agent.init(rng)

    start = time.time()
    log.info("training started")
    algorithm_state, result = agent.train(*algorithm_state, **training_kw_args)
    log.info("training finished")
    training_time = time.time() - start

    mean_return = result.eval_returns.mean(axis=1)
    std_return = result.eval_returns.std(axis=1)
    str_results = [
        f"{mean:.2f}+-{std:.2f}"
        for mean, std in zip(mean_return, std_return, strict=False)
    ]
    log.info(f"{training_time}, {str_results}")

    train_info_df = pd.DataFrame()
    for i in range(len(mean_return)):
        train_info_df[f"return_{i}"] = result.eval_returns[i]

    os.makedirs(
        os.path.join("sac_results", f"{framework}_{env_name}", dir_name), exist_ok=True
    )
    train_info_df.to_csv(
        os.path.join(
            "sac_results", f"{framework}_{env_name}", dir_name, f"{seed}_results.csv"
        )
    )
    with open(
        os.path.join(
            "sac_results", f"{framework}_{env_name}", dir_name, f"{seed}_info"
        ),
        "w",
    ) as f:
        f.write(f"sac_config: {config}\n")
        f.write(f"hpo_config: {hpo_config}\n")
        f.write(f"nas_config: {nas_config}\n")
        f.write(f"time: {training_time}\n")
        f.write(f"returns: {str_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-name", type=str)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--n-eval-steps", type=int)
    parser.add_argument("--n-eval-episodes", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env-framework", type=str)
    parser.add_argument("--env", type=str)
    parser.add_argument("--n-env-steps", type=int)
    parser.add_argument("--cnn-policy", type=bool, default=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    training_kw_args = {
        "n_total_timesteps": args.training_steps,
        "n_eval_steps": args.n_eval_steps,
        "n_eval_episodes": args.n_eval_episodes,
    }

    config = {
        "n_envs": args.n_envs,
        "track_metrics": False,
        "track_traj": False,
    }

    with jax.disable_jit(disable=False):
        test_sac(
            dir_name=args.dir_name,
            log=logger,
            framework=args.env_framework,
            env_name=args.env,
            config=config,
            training_kw_args=training_kw_args,
            seed=args.seed,
            cnn_policy=args.cnn_policy,
        )