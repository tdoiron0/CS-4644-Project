#!/bin/bash
#SBATCH --job-name=vlm-wiki
#SBATCH --account=ece
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=/storage/ice1/7/4/shall316/cs4644/slurm-%j.out
#SBATCH --error=/storage/ice1/7/4/shall316/cs4644/slurm-%j.err

# ── Usage ────────────────────────────────────────────────────────────────────
# First run:   sbatch submit-text-train.sbatch 10
# This chains 10 sequential 3-hour jobs.
# Training stops early if all epochs finish (exit code 0 without resubmit).
# ─────────────────────────────────────────────────────────────────────────────

COUNT=${1:-1}

# Update these paths for your PACE account
PROJECT_DIR="/storage/ice1/7/4/shall316/cs4644"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Iterations remaining: ${COUNT}"
echo "============================================"

PYTHON="/home/hice1/shall316/.conda/envs/vlm/bin/python"
export PYTHONPATH="${PROJECT_DIR}:/storage/ice1/7/4/shall316/vlm-pip:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
DONE_FILE="${PROJECT_DIR}/WIKI_TRAINING_COMPLETE"

cd "${PROJECT_DIR}" || exit 1

chain_next_job() {
    if (( COUNT > 1 )) && [ ! -f "${DONE_FILE}" ]; then
        NEXT_COUNT=$((COUNT - 1))
        echo "Submitting next job (${NEXT_COUNT} remaining)..."
        cd "${PROJECT_DIR}" && sbatch submit-text-train.sbatch "${NEXT_COUNT}"
    fi
}
trap chain_next_job EXIT

$PYTHON finetune-wiki.py

EXIT_CODE=$?
echo "finetune-wiki.py exited with code: ${EXIT_CODE}"

# Mark complete if training finished successfully
if [ ${EXIT_CODE} -eq 0 ]; then
    touch "${DONE_FILE}"
fi
