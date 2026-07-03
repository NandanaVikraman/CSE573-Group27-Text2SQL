#!/bin/bash
#SBATCH -A class_cse573spring2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -c 4
#SBATCH -t 13:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=text2sql_dynet_final
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
#SBATCH --export=ALL

set -euo pipefail

echo "JOB START: $(date)"
echo "HOST: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"

module load cuda-13.0.1-gcc-12.1.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynet38

export PYTHONPATH="$HOME/dynet-cuda:$HOME/dynet-cuda/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$HOME/dynet-cuda/build/dynet:${LD_LIBRARY_PATH:-}"

cd ~/SWM/SWM/modules/traditional_baseline_dynet

echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

mkdir -p checkpoints

TRAIN_DONE="checkpoints/model_final.dy"
LATEST_CKPT="checkpoints/model_latest.dy"
BEST_CKPT="checkpoints/model_best.dy"

RESUME_ARG=""
if [ -f "$TRAIN_DONE" ]; then
    echo "Training already finished. Skipping training."
else
    if [ -f "$LATEST_CKPT" ]; then
        echo "Found existing latest checkpoint: $LATEST_CKPT"
        RESUME_ARG="--resume_checkpoint $LATEST_CKPT"
    else
        echo "No existing latest checkpoint found. Starting from scratch."
    fi

    echo "===== STEP 1: TRAIN ====="
    python -u train.py \
        --epochs 50 \
        --eval_every 10 \
        --train_dev_subset -1 \
        $RESUME_ARG \
        2>&1 | tee train.log
fi

EVAL_CKPT="$BEST_CKPT"
if [ ! -f "$EVAL_CKPT" ]; then
    if [ -f "$TRAIN_DONE" ]; then
        echo "Best checkpoint not found; using final checkpoint instead."
        EVAL_CKPT="$TRAIN_DONE"
    else
        echo "ERROR: no checkpoint found for evaluation."
        exit 1
    fi
fi

echo "Using checkpoint for evaluation: $EVAL_CKPT"

echo "===== STEP 2: LOCAL FULL EVAL ====="
python -u evaluate.py \
    --limit -1 \
    --checkpoint "$EVAL_CKPT" \
    2>&1 | tee eval.log

echo "===== STEP 3: EXPORT PREDICTIONS ====="
python -u evaluate.py \
    --limit -1 \
    --checkpoint "$EVAL_CKPT" \
    --save_predictions pred.txt \
    --save_gold gold.txt \
    --save_results results.jsonl \
    --export_only \
    2>&1 | tee export.log

echo "===== FINAL STATUS =====" | tee final_status.log
echo "Training completed: $( [ -f "$TRAIN_DONE" ] && echo yes || echo no )" | tee -a final_status.log
echo "Local evaluation completed" | tee -a final_status.log
echo "Prediction export completed" | tee -a final_status.log
echo "Checkpoint used for eval: $EVAL_CKPT" | tee -a final_status.log
echo "JOB END: $(date)" | tee -a final_status.log
