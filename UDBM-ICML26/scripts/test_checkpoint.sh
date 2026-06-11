#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VARIANT_LC="$(lower_variant "${VARIANT}")"
CKPT_PATH_S1="${CKPT_PATH_S1:-./ckpt_universal/udbm_${VARIANT_LC}_s1/model-600.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_${VARIANT_LC}_s2}"
TASKS="${TASKS:-light_only rain blur fog snow}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-1}"

print_run_header
echo "Mode: test trained checkpoint"
echo "Stage-1 checkpoint: ${CKPT_PATH_S1}"
echo "Stage-2 folder: ${RESULTS_FOLDER}"
echo "Tasks: ${TASKS}"

args=(
  "--variant" "${VARIANT}"
  "--dataroot" "${DATAROOT}"
  "--ckpt_path_s1" "${CKPT_PATH_S1}"
  "--results_folder" "${RESULTS_FOLDER}"
  "--milestone" "${MILESTONE}"
  "--result_dir" "${RESULT_DIR}"
  "--sampling_timesteps" "${SAMPLING_TIMESTEPS}"
  "--tasks"
)

read -r -a task_array <<<"${TASKS}"
args+=("${task_array[@]}")

run_python test_s2.py "${args[@]}" "$@"
