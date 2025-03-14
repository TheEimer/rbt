#!/bin/bash

# USAGE: ./run.sh <experiment> <cluster> <replay_ratio>

# DQN
python run_arlbench.py -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2" 

# Reset DQN
python run_arlbench.py --config-name=reset_dqn -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2" 

# Redo DQN
python run_redo_dqn.py -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2" "replay_ratio=$3"

# RBT
python run_rbt.py --config-name=rbt_medium_reset -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2"  "replay_ratio=$3" "optimizer=rs,smac,rs_mf,smac_mf"

# RBT OptBudget
python run_rbt.py --config-name=rbt_medium_reset_optbudget -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2"  "replay_ratio=$3" "optimizer=rs,smac"

# RBT Cont
python run_rbt.py --config-name=rbt_medium_reset_cont -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2"  "replay_ratio=$3" "optimizer=smac,smac_mf"

# RBT OptBudget-Cont
python run_rbt.py --config-name=rbt_medium_reset_optbudget_cont -m "autorl.seed=range(10)" "experiment=$1" "cluster=$2"  "replay_ratio=$3" "optimizer=smac"
