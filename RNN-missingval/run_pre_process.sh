#!/bin/bash
#
#SBATCH --job-name=pre-process
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:p40:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=5
#SBATCH --time=168:00:00
#SBATCH --mem=20GB
#SBATCH --output=outputs/train_%A.out
#SBATCH --error=outputs/train_%A.err

##### below is for on nyulmc hpc: bigpurple #####
##### above is for on nyu hpc: prince #####

##!/bin/bash
##SBATCH --partition=gpu8_medium
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=1
##SBATCH --job-name=train_PCNN
##SBATCH --gres=gpu:1
##SBATCH --output=outputs/rq_train1_%A_%a.out
##SBATCH --error=outputs/rq_train1_%A_%a.err
##SBATCH --mem=250GB

#module purge
#module load python/gpu/3.6.5

module purge
module load python3/intel/3.5.3


python3 -u /scratch/sb3923/time_series/pre-process.py
