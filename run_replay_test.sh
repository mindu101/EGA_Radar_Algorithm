#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/Main_ws/config/test.yaml}"
shift || true

exec /usr/bin/python3 "${ROOT_DIR}/tools/synchronized_replay.py" "${CONFIG_PATH}" "$@"
