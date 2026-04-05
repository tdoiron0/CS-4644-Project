#!/bin/bash
#SBATCH --job-name=vlm-fgvc
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
# First run:   sbatch submit.sbatch 10
# This chains 10 sequential 2-hour jobs, each resuming from latest checkpoint.
# Training stops early if all epochs finish (exit code 0 without resubmit).
# ─────────────────────────────────────────────────────────────────────────────

COUNT=${1:-1}

# Update these paths for your PACE account
PROJECT_DIR="/storage/ice1/7/4/shall316/cs4644"
DATA_ROOT="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Iterations remaining: ${COUNT}"
echo "Data root:  ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"

PYTHON="/home/hice1/shall316/.conda/envs/vlm/bin/python"
export PYTHONPATH="${PROJECT_DIR}:/storage/ice1/7/4/shall316/vlm-pip:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
DONE_FILE="${OUTPUT_DIR}/TRAINING_COMPLETE"

cd "${PROJECT_DIR}" || exit 1

chain_next_job() {
    if (( COUNT > 1 )) && [ ! -f "${DONE_FILE}" ]; then
        NEXT_COUNT=$((COUNT - 1))
        echo "Submitting next job (${NEXT_COUNT} remaining)..."
        cd "${PROJECT_DIR}" && sbatch submit.sbatch "${NEXT_COUNT}"
    fi
}
trap chain_next_job EXIT

# Run training (always pass --resume; it starts fresh if no checkpoint exists)
$PYTHON train_pace.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --resume "${OUTPUT_DIR}/latest"

echo "train_pace.py exited with code: $?"
