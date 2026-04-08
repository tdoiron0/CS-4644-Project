#!/bin/bash
#SBATCH --job-name=vlm-fgvc-test
#SBATCH --account=ece
#SBATCH --gres=gpu:1
#SBATCH --constraint="L40S|HX00|A40|A100-40GB|A100-80GB|gpu-rtxpro-blackwell"
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/storage/ice1/7/4/shall316/cs4644/slurm-test-%j.out
#SBATCH --error=/storage/ice1/7/4/shall316/cs4644/slurm-test-%j.err

# ── Notes ────────────────────────────────────────────────────────────────────
# Single-shot test job (no chaining). Evaluates two models on the held-out
# FGVC-Aircraft test set:
#   1. baseline  – out-of-the-box InternVL3-2B (training step 0)
#   2. finetuned – best LoRA checkpoint from training (highest val accuracy)
#
# GPU selection uses the "first-available CUDA-13-capable" constraint from
# the PACE docs, which lets the scheduler grab whichever modern GPU is free
# first (L40S, H100, H200, A40, A100-40/80GB, or RTXPro Blackwell). The 2B
# model in bf16 fits comfortably on any of these. Partition and QOS are
# auto-assigned so the broader constraint can match nodes outside coc-gpu.
#
# If you ever need to fall back to a specific GPU, replace the two GPU
# directives above with one of:
#   #SBATCH --gres=gpu:A100:1                       # what training used
#   #SBATCH --gres=gpu:L40S:1
#   #SBATCH --gres=gpu:H100:1
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR="/storage/ice1/7/4/shall316/cs4644"
DATA_ROOT="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
BEST_DIR="${OUTPUT_DIR}/best"
# Test results live under testing/ so everything test-related (scripts
# + outputs) is in one self-contained folder that can be PR'd cleanly.
TEST_RESULTS_DIR="${PROJECT_DIR}/testing/test_results"

echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Started:    $(date)"
echo "Data root:  ${DATA_ROOT}"
echo "Best dir:   ${BEST_DIR}"
echo "Results:    ${TEST_RESULTS_DIR}"
echo "============================================"

PYTHON="/home/hice1/shall316/.conda/envs/vlm/bin/python"
export PYTHONPATH="${PROJECT_DIR}:/storage/ice1/7/4/shall316/vlm-pip:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}" || { echo "FATAL: cannot cd to ${PROJECT_DIR}"; exit 1; }

mkdir -p "${TEST_RESULTS_DIR}"

$PYTHON testing/test_pace.py \
    --data_root "${DATA_ROOT}" \
    --best_dir "${BEST_DIR}" \
    --output_dir "${TEST_RESULTS_DIR}" \
    --which both

EXIT_CODE=$?
echo "============================================"
echo "test_pace.py exited with code: ${EXIT_CODE}"
echo "Finished:   $(date)"
echo "============================================"
exit ${EXIT_CODE}
