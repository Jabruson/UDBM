#!/bin/sh

set -e

cd "$(dirname "$0")"

VARIANT="${VARIANT:-L}"
VARIANT_LC="$(printf "%s" "$VARIANT" | tr '[:upper:]' '[:lower:]')"
DATAROOT="${DATAROOT:-./datasets/all_in_one}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CKPT_PATH_S1="${CKPT_PATH_S1:-./ckpt_universal/udbm_${VARIANT_LC}_s1/model-600.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./ckpt_universal/udbm_${VARIANT_LC}_s2}"
MILESTONE="${MILESTONE:-600}"
RESULT_DIR="${RESULT_DIR:-./result}"
TASKS="${TASKS:-light_only rain blur fog snow}"

export CUDA_VISIBLE_DEVICES

python test_s2.py \
  --variant "$VARIANT" \
  --dataroot "$DATAROOT" \
  --ckpt_path_s1 "$CKPT_PATH_S1" \
  --results_folder "$RESULTS_FOLDER" \
  --milestone "$MILESTONE" \
  --result_dir "$RESULT_DIR" \
  --tasks $TASKS \
  "$@"
