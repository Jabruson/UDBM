#!/bin/sh

set -e

cd "$(dirname "$0")"

VARIANT="${VARIANT:-L}"
VARIANT_LC="$(printf "%s" "$VARIANT" | tr '[:upper:]' '[:lower:]')"
DATAROOT="${DATAROOT:-./datasets/all_in_one}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-$(printf "%s" "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}"
CKPT_PATH_S1="${CKPT_PATH_S1:-./ckpt_universal/udbm_${VARIANT_LC}_s1/model-600.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_${VARIANT_LC}_s2}"

export CUDA_VISIBLE_DEVICES

if [ "$NUM_PROCESSES" -gt 1 ]; then
  accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    train_s2.py \
    --variant "$VARIANT" \
    --dataroot "$DATAROOT" \
    --ckpt_path_s1 "$CKPT_PATH_S1" \
    --results_folder "$RESULTS_FOLDER" \
    --gradient_accumulate_every 2 \
    --task_batch_sizes 8,2,4,4,2 \
    "$@"
else
  python train_s2.py \
    --variant "$VARIANT" \
    --dataroot "$DATAROOT" \
    --ckpt_path_s1 "$CKPT_PATH_S1" \
    --results_folder "$RESULTS_FOLDER" \
    --gradient_accumulate_every 2 \
    --task_batch_sizes 8,2,4,4,2 \
    "$@"
fi
