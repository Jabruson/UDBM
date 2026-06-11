#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV-fcl}"
VARIANT="${VARIANT:-L}"
GPUS="${GPUS:-0}"
DATAROOT="${DATAROOT:-./datasets/all_in_one}"
RESULT_DIR="${RESULT_DIR:-./result}"
MILESTONE="${MILESTONE:-600}"

num_processes_from_gpus() {
  local gpu_list="$1"
  if [[ -z "${gpu_list}" ]]; then
    echo 1
    return
  fi
  awk -F',' '{print NF}' <<<"${gpu_list}"
}

NUM_PROCESSES="${NUM_PROCESSES:-$(num_processes_from_gpus "${GPUS}")}"

lower_variant() {
  tr '[:upper:]' '[:lower:]' <<<"$1"
}

python_cmd() {
  if [[ -n "${CONDA_ENV}" ]]; then
    echo "conda run -n ${CONDA_ENV} python"
  else
    echo "python"
  fi
}

accelerate_cmd() {
  if [[ -n "${CONDA_ENV}" ]]; then
    echo "conda run -n ${CONDA_ENV} accelerate"
  else
    echo "accelerate"
  fi
}

run_python() {
  local cmd
  read -r -a cmd <<<"$(python_cmd)"
  CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}" "$@"
}

run_training() {
  local entrypoint="$1"
  shift

  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    local cmd
    read -r -a cmd <<<"$(accelerate_cmd)"
    CUDA_VISIBLE_DEVICES="${GPUS}" "${cmd[@]}" launch \
      --multi_gpu \
      --num_processes "${NUM_PROCESSES}" \
      "${entrypoint}" "$@"
  else
    run_python "${entrypoint}" "$@"
  fi
}

print_run_header() {
  echo "Project: ${PROJECT_ROOT}"
  echo "Conda env: ${CONDA_ENV:-<current shell>}"
  echo "Variant: ${VARIANT}"
  echo "GPUs: ${GPUS}"
  echo "Num processes: ${NUM_PROCESSES}"
  echo "Dataroot: ${DATAROOT}"
}
