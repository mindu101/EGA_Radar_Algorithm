#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if (($# != 0)); then
    echo "拒绝启动：该入口固定使用 Web 页面选择正式链路早/晚注入，不接受命令行参数。" >&2
    echo "正确用法：./run_scripts/run_referee_web_injector.sh" >&2
    exit 2
fi

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

cd "$PROJECT_DIR"
exec python tools/referee_web_injector.py \
    --enable-test-injection \
    --confirm TEST_ONLY \
    --mode web
