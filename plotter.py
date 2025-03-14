import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


OPTIMIZERS = {
    "smac": "SMAC",
    "rs": "RS",
    "smac_mf": "SMAC + HB",
    "rs_mf": "RS + HB",
}

APPROACHES = {
    "dqn": "DQN",
    "redo_dqn": "ReDo DQN",
    "reset_dqn": "Reset DQN",
    "pbt": "PBT",
    "pbt_redo": "PBT Redo",
    # "rbt": "RBT-LightReset",
    # "rbt-fullreset": "RBT-FullReset",
    "rbt-mediumreset": "RBT-MediumReset",
    "rbt-mediumreset-cont": "RBT-MediumReset-Cont",
    "rbt-mediumreset-optbudget": "RBT-MediumReset-OB",
    "rbt-mediumreset-optbudget-cont": "RBT-MediumReset-OB-Cont",
    # "rbt-buf": "RBT-GoodBuffer",
    # "rbt-dr": "RBT-DefaultRollout-M-RST",
}

# RBT_REPLAY_RATIO = [0.01, 0.02]
RBT_REPLAY_RATIO = [0.01]
# RBT_REPLAY_RATIO = [0.01, 0.05, 0.1, 0.25]
RBT_METRICS = {
    "eval_return": "",
    # "td_error": "TD Error",
    # "msbe": "MSBE",
}

SEEDS = range(10)
PBT_POPULATION_SIZE = 8

class Plotter:
    def __init__(self):
        self.results_dir = Path("./results")

    def load_baseline_results(self, approach: str, env: str, replay_ratio: float | None = None):
        evals = []
        for seed in SEEDS:
            if replay_ratio:
                eval_path = self.results_dir / env / f"{approach}_{replay_ratio}" / str(seed) / "evaluation.csv"
            else:
                eval_path = self.results_dir / env / f"{approach}" / str(seed) / "evaluation.csv"
            if not eval_path.exists():
                print(f"Skipping {eval_path}")
                continue
            eval = pd.read_csv(eval_path)
            eval['seed'] = seed
            eval['env'] = env
            evals.append(eval)

        if len(evals) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(evals)
    
    def load_rbt_results(self, approach: str, optimizer: str, env: str, metric: str, replay_ratio: float, smooth_rbt: bool):
        all_inc_eval = []
        all_metrics = []

        base_dir = self.results_dir / env / f"{approach}_{optimizer}_{replay_ratio}_{metric}"
        
        for seed in SEEDS:
            train_info_path = base_dir / str(seed) / "train_info.csv"
            if not train_info_path.exists():
                print(f"Skipping {train_info_path}")
                continue
        
            train_info = pd.read_csv(train_info_path)

            if smooth_rbt:
                inc_eval_path = base_dir / str(seed) / "incumbent_eval_performances.csv"
                inc_eval = pd.read_csv(inc_eval_path)
                inc_eval['seed'] = seed
                inc_eval['env'] = env   
                inc_eval.loc[:, 'steps'] = (inc_eval['iteration'] + 1) * max(train_info['steps'])
                inc_eval.loc[:, 'returns'] = inc_eval['incumbent_performance'] * -1
            else:
                step_size = train_info['steps'].min()
                train_info['steps'] = np.arange(1, len(train_info) + 1) * step_size
                inc_eval = train_info[['steps', 'returns']].copy()

            all_inc_eval.append(inc_eval)


            evals_path = base_dir / str(seed) / "full_evals.csv"
            eval = pd.read_csv(evals_path)

            td_path = base_dir / str(seed) / "td_errors.csv"
            td = pd.read_csv(td_path)
            metrics = pd.merge(eval, td, on=['iteration', 'config_id'])

            msbe_path = base_dir / str(seed) / "msbes.csv"
            if msbe_path.exists():
                msbe = pd.read_csv(msbe_path)
                metrics = pd.merge(metrics, msbe, on=['iteration', 'config_id'])

            metrics['seed'] = seed
            metrics['approach'] =  f"{APPROACHES[approach]} {RBT_METRICS[metric]}"
            metrics['env'] = env
            all_metrics.append(metrics)

        if len(all_inc_eval) == 0:
            return pd.DataFrame(), pd.DataFrame()
        else: 
            all_inc_eval = pd.concat(all_inc_eval)
            all_metrics = pd.concat(all_metrics)
            
            return all_inc_eval, all_metrics

    def load_pbt_results(self, approach: str, env: str, replay_ratio: float | None = None):
        results = []
        for seed in SEEDS:
            if replay_ratio:
                perf_path =  self.results_dir / env / f"{approach}_{replay_ratio}" / str(seed) / "runhistory.csv"
            else:
                perf_path =  self.results_dir / env / f"{approach}" / str(seed) / "runhistory.csv"
            
            if not perf_path.exists():
                print(f"Skipping {perf_path}")
                continue

            perf = pd.read_csv(perf_path, index_col=False)
            perf['iteration'] = (perf.index // PBT_POPULATION_SIZE) + 1
            perf['returns'] = perf.groupby(['iteration'])['performance'].transform('max') 
            perf['seed'] = seed

            perf['env'] = env   
            perf['steps'] = perf['iteration'] * perf['budget'].values[0]

            results.append(perf)

        if len(results) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(results)
    
    def load_data(
            self,
            env: str,
            rbt_optimizer: str,
            approaches: list[str] | None = None,
            replay_ratios: list[float] | None = None,
            rbt_metrics: list[str] | None = None,
            smooth_rbt: bool = False
        ):
        all_data = []
        all_rbt_metrics = []

        if approaches is None:
            approaches = list(APPROACHES.keys())

        if replay_ratios is None:
            replay_ratios = RBT_REPLAY_RATIO

        if rbt_metrics is None:
            rbt_metrics = list(RBT_METRICS.keys())

        for approach in approaches:
            if approach in ["dqn", "reset_dqn"]:
                data = self.load_baseline_results(approach, env)
                data["approach"] = APPROACHES[approach]
                all_data.append(data)
            elif approach == "redo_dqn":
                for replay_ratio in replay_ratios:
                    data = self.load_baseline_results(approach, env, replay_ratio=replay_ratio)
                    data["approach"] = APPROACHES[approach]
                    all_data.append(data) 
            elif "rbt" in approach:
                for replay_ratio in replay_ratios:
                    # for offline_update_fraction in rbt_fractions:
                    for metric in rbt_metrics:
                        train_info, metrics = self.load_rbt_results(approach, rbt_optimizer, env, metric, replay_ratio, smooth_rbt)
                        train_info["approach"] = APPROACHES[approach]
                        metrics["approach"] = APPROACHES[approach]
                        all_data.append(train_info)
                        all_rbt_metrics.append(metrics)
            elif approach == "pbt":
                data = self.load_pbt_results(approach, env)
                data["approach"] = APPROACHES[approach]
                all_data.append(data)
            elif "pbt_redo" in approach:
                data = self.load_pbt_results(approach, env, replay_ratio=replay_ratio)
                data["approach"] = APPROACHES[approach]
                all_data.append(data)
            else:
                raise ValueError(f"Unknown approach {approach}")
        
        all_data = pd.concat(all_data)
        all_rbt_metrics = pd.concat(all_rbt_metrics)
                
        return all_data, all_rbt_metrics


    def plot_combined(self, env: str, show_rbt_inc: bool = True):
        fig, axs = plt.subplots(len(RBT_REPLAY_RATIO), len(OPTIMIZERS), figsize=(3 * len(OPTIMIZERS), 3 * len(RBT_REPLAY_RATIO)), sharex=True, sharey=True)

        for i, rbt_optimizer in enumerate( OPTIMIZERS.keys()):
            for j, replay_ratio in enumerate(RBT_REPLAY_RATIO):
                if len(axs.shape) == 1:
                    ax = axs[i]
                else:
                    ax = axs[j, i]

                data, _ = self.load_data(env=env, rbt_optimizer=rbt_optimizer, replay_ratios=[replay_ratio], smooth_rbt=show_rbt_inc)
                if len(data) == 0:
                    continue
                
                sns.lineplot(
                    data=data,
                    x="steps",
                    y="returns",
                    hue="approach",
                    ax=ax,
                    hue_order=APPROACHES.values(),
                    errorbar=("ci", 95)
                )
                opt_name = OPTIMIZERS[rbt_optimizer]
                ax.set_title(f"{opt_name}\nRR = {replay_ratio}")
                ax.set_xlabel("Steps")
                ax.set_ylabel("Evaluation Return")

                ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

                # disable legend of axis
                ax.get_legend().remove()

        # Add a single legend to the figure
        # handles, labels = axs[0, 0].get_legend_handles_labels()
        handles, labels = [], []
        for ax in axs.flatten():
            _handles, _labels = ax.get_legend_handles_labels()
            for h, l in zip(_handles, _labels):
                if l not in labels:
                    labels.append(l)
                    handles.append(h)

        fig.legend(handles, labels, title="Approach", loc='center left', bbox_to_anchor=(0, 0.5))
        fig.suptitle(env)

        plt.tight_layout(rect=[0.18, 0, 1, 1])  # Adjust layout to make space for the legend
        
        if show_rbt_inc:
            name = f"{env}_rbt_inc"
        else:
            name = env
        plt.savefig(f"plots/{name}.png", dpi=400)

    def plot_rbt(self, env: str):
        fig, axs = plt.subplots(3, len(RBT_REPLAY_RATIO), figsize=(4 * len(RBT_REPLAY_RATIO), 6), sharex=True, sharey=True)

        for i, rbt_optimizer in enumerate(["random", "smac", "smac_mf"]):
            for ax, budget in zip(axs[i].flatten(), RBT_REPLAY_RATIO):
                data, _ = self.load_data(
                    env=env,
                    rbt_optimizer=rbt_optimizer,
                    approaches=["rbt"],
                    replay_ratios=[budget],
                    rbt_metrics=["eval_return"]
                )
                lineplot = sns.lineplot(data=data, x="steps", y="returns", hue="approach", ax=ax)
                opt_name = "SMAC" if rbt_optimizer == "smac" else "Random Search"
                ax.set_title(f"{opt_name}\nmax budget = {budget}")
                ax.set_xlabel("Steps")
                ax.set_ylabel("Evaluation Return")

                # disable legend of axis
                ax.get_legend().remove()

        # Add a single legend to the figure
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Approach", loc='center left', bbox_to_anchor=(0, 0.5))
        plt.tight_layout(rect=[0.18, 0, 1, 1])  # Adjust layout to make space for the legend
        plt.savefig(f"plots/{env}_rbt.png", dpi=400)

if __name__ == '__main__':
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    plotter = Plotter()
    # plotter.plot_combined("CartPole-v1", show_rbt_inc=False)
    # plotter.plot_combined("CartPole-v1", show_rbt_inc=True)
    # plotter.plot_combined("SpaceInvaders-MinAtar", show_rbt_inc=False)
    plotter.plot_combined("SpaceInvaders-MinAtar", show_rbt_inc=True)
    # plotter.plot_combined("LunarLander-v2", show_rbt_inc=False)
    # plotter.plot_combined("LunarLander-v2", show_rbt_inc=True)

