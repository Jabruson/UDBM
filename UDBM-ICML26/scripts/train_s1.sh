#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_$(lower_variant "${VARIANT}")_s1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
TASK_BATCH_SIZES="${TASK_BATCH_SIZES:-16,4,8,8,4}"

print_run_header
echo "Stage: 1 uncertainty estimator"
echo "Results folder: ${RESULTS_FOLDER}"

args=(
  "--variant" "${VARIANT}"
  "--dataroot" "${DATAROOT}"
  "--results_folder" "${RESULTS_FOLDER}"
  "--gradient_accumulate_every" "${GRAD_ACCUM}"
  "--task_batch_sizes" "${TASK_BATCH_SIZES}"
)

if [[ -n "${RESUME_MILESTONE:-}" ]]; then
  args+=("--resume_milestone" "${RESUME_MILESTONE}")
fi

run_training train_s1.py "${args[@]}" "$@"
