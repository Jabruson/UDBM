#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VARIANT_LC="$(lower_variant "${VARIANT}")"
CKPT_PATH_S1="${CKPT_PATH_S1:-./ckpt_universal/udbm_${VARIANT_LC}_s1/model-600.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_${VARIANT_LC}_s2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
TASK_BATCH_SIZES="${TASK_BATCH_SIZES:-8,2,4,4,2}"

print_run_header
echo "Stage: 2 uncertainty-aware diffusion bridge"
echo "Stage-1 checkpoint: ${CKPT_PATH_S1}"
echo "Results folder: ${RESULTS_FOLDER}"

args=(
  "--variant" "${VARIANT}"
  "--dataroot" "${DATAROOT}"
  "--ckpt_path_s1" "${CKPT_PATH_S1}"
  "--results_folder" "${RESULTS_FOLDER}"
  "--gradient_accumulate_every" "${GRAD_ACCUM}"
  "--task_batch_sizes" "${TASK_BATCH_SIZES}"
)

if [[ -n "${RESUME_MILESTONE:-}" ]]; then
  args+=("--resume_milestone" "${RESUME_MILESTONE}")
fi

run_training train_s2.py "${args[@]}" "$@"
