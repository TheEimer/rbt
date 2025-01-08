#!/bin/bash

# USAGE: ./submit_luis.sh <config_name> <environment> <cluster>

mkdir -p "log"
mkdir -p "sbatch_scripts"

echo "#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J ${1}_${2}                                   # TODO enter your job name
#SBATCH -t 4-00:00:00                                   # TODO check for your clusters time limit
#SBATCH --mail-type fail
#SBATCH -p ai                                           # TODO check for your clusters partition
#SBATCH --output log/${1}_${2}_%A_%a.out
#SBATCH --error log/${1}_${2}_%A_%a.err
#SBATCH --array=0-9

module load Miniforge3
conda activate rbt

export JAX_PLATFORMS=cpu

python run_arlbench.py -m --config-name=${1} "autorl.seed=\$SLURM_ARRAY_TASK_ID" "environment=$2" "cluster=$3" 
" > sbatch_scripts/${2}.sh
echo "Submitting $2"
chmod +x sbatch_scripts/${2}.sh
sbatch sbatch_scripts/${2}.sh