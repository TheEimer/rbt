#!/bin/bash

# USAGE: ./submit_pbt_pc2.sh <experiment> <cluster>

mkdir -p "log"
mkdir -p "sbatch_scripts"

echo "#!/bin/bash


#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH -J pbt_${2}                                   # TODO enter your job name
#SBATCH -t 4-00:00:00                                   # TODO check for your clusters time limit
#SBATCH -p normal                                           # TODO check for your clusters partition
#SBATCH --output log/pbt_${2}_%A_%a.out
#SBATCH --error log/pbt_${2}_%A_%a.err
#SBATCH --array=0

conda activate arlb

python run_arlbench.py -m --config-name=pbt "autorl.seed=\$SLURM_ARRAY_TASK_ID" "experiment=$1" "cluster=$2" 
" > sbatch_scripts/${2}.sh
echo "Submitting $2"
chmod +x sbatch_scripts/${2}.sh
sbatch sbatch_scripts/${2}.sh