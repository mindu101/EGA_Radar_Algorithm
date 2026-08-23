#!/usr/bin/env bash
set -euo pipefail

PARAMS_CONFIG="${1:-config/params.yaml}"
DEVICE_CONFIG="${2:-config/device.yaml}"
TEST_CONFIG="${3:-config/test.yaml}"

cd "$(dirname "$0")/.."
exec python main.py \
    --config "${PARAMS_CONFIG}" \
    --device_config "${DEVICE_CONFIG}" \
    --test_config "${TEST_CONFIG}"
