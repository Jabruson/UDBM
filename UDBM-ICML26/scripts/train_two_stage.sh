#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/train_s1.sh" "$@"

VARIANT="${VARIANT:-L}"
VARIANT_LC="$(tr '[:upper:]' '[:lower:]' <<<"${VARIANT}")"
export CKPT_PATH_S1="${CKPT_PATH_S1:-./ckpt_universal/udbm_${VARIANT_LC}_s1/model-600.pt}"

"${SCRIPT_DIR}/train_s2.sh" "$@"
