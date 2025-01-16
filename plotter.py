import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


APPROACHES = {
    "dqn": "DQN",
    "redo_dqn": "ReDo DQN",
    "reset_dqn": "Reset DQN",
    "pbt": "PBT",
    "rbt": "RBT"
}

RBT_BUDGETS = [250, 500, 1000, 2500]
RBT_METRICS = {
    "eval_return": "Evaluation Return",
    "td_error": "TD Error",
    "msbe": "MSBE",
}

SEEDS = range(10)
PBT_POPULATION_SIZE = 8

class Plotter:
    def __init__(self):
        self.results_dir = Path("./results")

    def load_baseline_results(self, approach: str, env: str):
        evals = []
        for seed in SEEDS:
            eval_path = self.results_dir / f"{approach}_{env}" / str(seed) / "evaluation.csv"
            if not eval_path.exists():
                print(f"Skipping {eval_path}")
                continue
            eval = pd.read_csv(eval_path)
            eval['seed'] = seed
            eval['approach'] = APPROACHES[approach]
            eval['env'] = env
            evals.append(eval)

        if len(evals) == 0:
            return pd.DataFrame()
        else:
            return pd.concat(evals)
    
    def load_rbt_results(self, optimizer: str, approach: str, env: str, metric: str, budget: int):
        all_inc_eval = []
        all_metrics = []

        base_dir = f"{approach}_{optimizer}_{budget}_{metric}_{env}"
        
        for seed in SEEDS:
            train_info_path = self.results_dir / base_dir / str(seed) / "train_info.csv"
            if not train_info_path.exists():
                print(f"Skipping {train_info_path}")
                continue
        
            train_info = pd.read_csv(train_info_path)

            inc_eval_path = self.results_dir / base_dir / str(seed) / "incumbent_eval_performances.csv"
            inc_eval = pd.read_csv(inc_eval_path)
            inc_eval['seed'] = seed
            inc_eval['approach'] = f"{APPROACHES[approach]} {RBT_METRICS[metric]}"
            inc_eval['env'] = env   
            inc_eval.loc[:, 'steps'] = (inc_eval['iteration'] + 1) * max(train_info['steps'])
            inc_eval.loc[:, 'returns'] = inc_eval['incumbent_performance'] * -1
            all_inc_eval.append(inc_eval)

            evals_path = self.results_dir / base_dir / str(seed) / "full_evals.csv"
            eval = pd.read_csv(evals_path)

            td_path = self.results_dir / base_dir / str(seed) / "td_errors.csv"
            td = pd.read_csv(td_path)
            metrics = pd.merge(eval, td, on=['iteration', 'config_id'])

            msbe_path = self.results_dir / base_dir / str(seed) / "msbes.csv"
            if msbe_path.exists():
                msbe = pd.read_csv(msbe_path)
                metrics = pd.merge(metrics, msbe, on=['iteration', 'config_id'])

            metrics['seed'] = seed
            metrics['approach'] =  f"{APPROACHES[approach]} {budget} {RBT_METRICS[metric]}"
            metrics['env'] = env
            all_metrics.append(metrics)

        all_inc_eval = pd.concat(all_inc_eval)
        all_metrics = pd.concat(all_metrics)
        
        return all_inc_eval, all_metrics

    def load_pbt_results(self, approach: str, env: str):
        results = []
        for seed in SEEDS:
            perf_path =  self.results_dir / f"{approach}_{env}" / str(seed) / "runhistory.csv"
            if not perf_path.exists():
                print(f"Skipping {perf_path}")
                continue

            perf = pd.read_csv(perf_path, index_col=False)
            perf['seed'] = seed
            perf['approach'] = APPROACHES[approach]
            perf['env'] = env   
            perf['iteration'] = (perf.index // PBT_POPULATION_SIZE) + 1
            perf['steps'] = perf['iteration'] * perf['budget'].values[0]
            perf['returns'] = perf['performance']

            results.append(perf)
        return pd.concat(results)
    
    def load_data(self, env: str, rbt_optimizer: str, rbt_budget: int | None = None, rbt_metric: str | None = None):
        all_data = []
        all_rbt_metrics = []

        for approach in APPROACHES.keys():
            if approach in ["dqn", "redo_dqn", "reset_dqn"]:
                data = self.load_baseline_results(approach, env)
                all_data.append(data)
            elif approach == "rbt":
                for budget in RBT_BUDGETS:
                    if rbt_budget is not None and rbt_budget != budget:
                        continue

                    for metric in RBT_METRICS.keys():
                        if rbt_metric is not None and rbt_metric != metric:
                            continue

                        train_info, metrics = self.load_rbt_results(approach, rbt_optimizer, env, metric, budget)
                        all_data.append(train_info)
                        all_rbt_metrics.append(metrics)
            elif approach == "pbt":
                data = self.load_pbt_results(approach, env)
                all_data.append(data)
            else:
                raise ValueError(f"Unknown approach {approach}")
        
        all_data = pd.concat(all_data)
        all_rbt_metrics = pd.concat(all_rbt_metrics)
                
        return all_data, all_rbt_metrics
    
    def plot(self, env: str, rbt_optimizer: str = "smac"):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)

        for ax, budget in zip(axs.flatten(), RBT_BUDGETS):
            data, _ = self.load_data(env=env, rbt_budget=budget, rbt_optimizer=rbt_optimizer)
            lineplot = sns.lineplot(data=data, x='steps', y='returns', hue='approach', ax=ax)
            ax.set_title(f"Budget: {budget}")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Evaluation Return")

            # disable legend of axis
            ax.get_legend().remove()

        # Add a single legend to the figure
        handles, labels = lineplot.get_legend_handles_labels()
        fig.legend(handles, labels, title="Approach", loc='center left', bbox_to_anchor=(0, 0.5))

        plt.tight_layout(rect=[0.1, 0, 1, 1])  # Adjust layout to make space for the legend
        plt.savefig(f"{env}.png")


if __name__ == '__main__':
    sns.set(style="whitegrid")
    plotter = Plotter()
    plotter.plot("SpaceInvaders-MinAtar", rbt_optimizer="smac")

