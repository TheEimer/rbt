<p align="center">
    <a href="./docs/images/logo_lm.png#gh-light-mode-only">
        <img src="./docs/images/logo_lm.png#gh-light-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
    <a href="./docs/images/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/images/logo_dm.png#gh-dark-mode-only" alt="ARLBench Logo" width="80%"/>
    </a>
</p>

<div align="center">
    
[![PyPI Version](https://img.shields.io/pypi/v/arlbench.svg)](https://pypi.python.org/pypi/arlbench)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![License](https://img.shields.io/badge/License-BSD3-orange)
[![Test](https://github.com/automl/arlbench/actions/workflows/pytest.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/pytest.yaml)
[![Doc Status](https://github.com/automl/arlbench/actions/workflows/docs.yaml/badge.svg)](https://github.com/automl/arlbench/actions/workflows/docs.yaml)
    
</div>

<div align="center">
    <h3>
      <a href="#features">Features</a> |
      <a href="#installation">Installation</a> |
      <a href="#quickstart">Quickstart</a> |
      <a href="#cite-us">Cite Us</a>
    </h3>
</div>

---

# 🦾 Reset-based Tuning

This repo contains a draft project for reset-based tuning.

## Installation

Run
```bash
make install
```

Important: If you want to use MSBE as a criterion, you need the flashbax buffer containing training and validation splits.
Therefore, you need to install this [flashbax fork](https://github.com/becktepe/flashbax).

## Experiments

To run the experiments, you need to execute different scripts to run baselines and RBT versions.
For all of them, you need to define the ```experiment```, e.g. ```cc_cartpole_dqn```.
Additionally, you can specify a compute cluster partition, e.g. ```pc2_cpu```.

### Default DQN

```bash
python run_arlbench.py -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu"
```

### Reset DQN

```bash
python run_arlbench.py --config-name=reset_dqn -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu"
```

### Redo DQN

```bash
python run_redo_dqn.py -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu"
```

### Default PBT

For PBT, we submit the PBT processes to a CPU partition which subsequently submit the actual runs to the partition you specify.
The 

```bash
/submit_pbt_pc2.sh <experiment> <cluster>
```

### Redo PBT

Similar to default PBT, you run

```bash
/submit_pbt_redo_pc2.sh <experiment> <cluster>
```

### RBT

For convenience, you can just execute (or comment out specific parts of)
```bash
./run.sh <experiment> <cluster> <replay_ratio>
```

The default version of RBT is ```medium-reset``` RBT which resets half of the network after each iteration.
However, we provide several variations, optimizing budget or continuing SMAC runs.
You find them in  ```examples/configs```.

For the default RBT, run
```bash
python run_rbt --config-name=rbt_medium_reset -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=rs,smac,rs_mf,smac_mf" "replay_ratio=<replay_ratio>"
```

To continue SMAC runs across iterations (only for SMAC and SMAC+HB), run 
```bash
python run_rbt --config-name=rbt_medium_reset_cont -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=smac,smac_mf" "replay_ratio=<replay_ratio>"
```

To optimize the budget, i.e., number of gradient steps (only for RS and SMAC), run 
```bash
python run_rbt --config-name=rbt_medium_reset_optbudget -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=rs,smac" "replay_ratio=<replay_ratio>"
```

To optimize the budget and continue SMAC runs (only for SMAC), run 
```bash
python run_rbt --config-name=rbt_medium_reset_optbudget_cont -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=smac" "replay_ratio=<replay_ratio>"
```

Additionally, you can use the default configuration for rollouts by running:
```bash
python run_rbt --config-name=rbt_default_rollout -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=rs,smac,rs_mf,smac_mf" "replay_ratio=<replay_ratio>"
```

To reset all network weights, run:
```bash
python run_rbt --config-name=rbt_full_reset -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=rs,smac,rs_mf,smac_mf" "replay_ratio=<replay_ratio>"
```

To reset only the last layer, run:
```bash
python run_rbt --config-name=rbt_light_reset -m "experiment=<experiment>" "autorl.seed=range(10)" "cluster=pc2_cpu" "optimizer=rs,smac,rs_mf,smac_mf" "replay_ratio=<replay_ratio>"
```