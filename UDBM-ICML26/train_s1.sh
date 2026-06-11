#!/bin/sh

set -e

cd "$(dirname "$0")"

VARIANT="${VARIANT:-L}"
DATAROOT="${DATAROOT:-./datasets/all_in_one}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-$(printf "%s" "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_$(printf "%s" "$VARIANT" | tr '[:upper:]' '[:lower:]')_s1}"

export CUDA_VISIBLE_DEVICES

if [ "$NUM_PROCESSES" -gt 1 ]; then
  accelerate launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    train_s1.py \
    --variant "$VARIANT" \
    --dataroot "$DATAROOT" \
    --results_folder "$RESULTS_FOLDER" \
    --gradient_accumulate_every 1 \
    --task_batch_sizes 16,4,8,8,4 \
    "$@"
else
  python train_s1.py \
    --variant "$VARIANT" \
    --dataroot "$DATAROOT" \
    --results_folder "$RESULTS_FOLDER" \
    --gradient_accumulate_every 1 \
    --task_batch_sizes 16,4,8,8,4 \
    "$@"
fi
