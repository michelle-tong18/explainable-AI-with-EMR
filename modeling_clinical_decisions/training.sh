#!/bin/bash

#SBATCH --job-name=characterize_dataset
#SBATCH --output=../logs/characterize_data_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
#SBATCH --partition=dgx
##SBATCH --partition=gpu
##SBATCH --gres=gpu:teslav100:1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "Running on node $HOSTNAME"

# Activate conda environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# Your Conda environment name: 'myenv'
CONDA_ENV_NAME=/working/mochila2/mwtong/envs/py_env_EMR

# -------------------- BEGIN CODE --------------------

## Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

## Navigate to the base github directory
cd ../.. || exit
echo "The current directory is:"
pwd

python3 modeling_clinical_decisions/1_train_trees.py 

echo "All processes completed successfully."