#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py \
    --config "${PARAMS_CONFIG:-config/params.yaml}" \
    --device_config "${DEVICE_CONFIG:-config/device.yaml}" \
    "$@"
