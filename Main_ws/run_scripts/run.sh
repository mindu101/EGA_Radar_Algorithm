#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python main.py --config config/params.yaml --device_config config/device.yaml
