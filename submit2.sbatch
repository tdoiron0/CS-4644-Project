#!/bin/bash
#SBATCH --job-name=mha-experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu 80G

cd $SLURM_SUBMIT_DIR

# --- Modules ---
module load anaconda3

# --- Config file (pass as first arg, default: standard) ---
CONFIG="${1:-configs/multi-query-step.yaml}"

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_NODELIST"
echo "GPUs         : $SLURM_GPUS_ON_NODE"
echo "Config       : $CONFIG"
echo "Start time   : $(date)"
echo "============================================"

nvidia-smi

# --- Run training ---
#cd /home/hice1/jwarren315/scratch/mha-experiments-union-pace-ice
conda run -n dl python train.py "$CONFIG"

echo "Finished at  : $(date)"
