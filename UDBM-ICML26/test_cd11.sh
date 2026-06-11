#!/bin/sh

set -e

cd "$(dirname "$0")"

VARIANT="${VARIANT:-L}"
VARIANT_LC="$(printf "%s" "$VARIANT" | tr '[:upper:]' '[:lower:]')"
DATAROOT="${DATAROOT:-./datasets/all_in_one}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CKPT_PATH_S1="${CKPT_PATH_S1:-./pretrained/udbm_${VARIANT_LC}/stage1.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-./pretrained/udbm_${VARIANT_LC}}"
MILESTONE="${MILESTONE:-600}"
RESULT_DIR="${RESULT_DIR:-./result_cd11}"
TASKS="${TASKS:-cd11}"

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
