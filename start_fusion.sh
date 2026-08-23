#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS_DIR="$ROOT_DIR/Lidar_ws"
APP_DIR="$ROOT_DIR/Main_ws"
LIVOX_WS_DIR="${LIVOX_WS_DIR:-$ROOT_DIR/../livox_ws}"
CONDA_ENV="${CONDA_ENV:-cud128}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
ROS_PID=""

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "缺少运行文件: $1" >&2
        exit 1
    fi
}

cleanup() {
    if [[ -n "$ROS_PID" ]] && kill -0 "$ROS_PID" 2>/dev/null; then
        kill -TERM "$ROS_PID" 2>/dev/null || true
        wait "$ROS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

require_file /opt/ros/humble/setup.bash
require_file "$LIVOX_WS_DIR/install/setup.bash"
require_file "$ROS_WS_DIR/install/setup.bash"
require_file "$APP_DIR/config/params.yaml"
require_file "$APP_DIR/config/device.yaml"

(
    set +u
    source /opt/ros/humble/setup.bash
    source "$LIVOX_WS_DIR/install/setup.bash"
    source "$ROS_WS_DIR/install/setup.bash"
    set -u
    export ROS_LOG_DIR="$ROOT_DIR/logs/ros"
    mkdir -p "$ROS_LOG_DIR"
    ros2 launch rm2026_lidar_launch lidar_pipeline.launch.py \
        launch_rviz:="$LAUNCH_RVIZ"
) &
ROS_PID=$!

set +u
source /opt/ros/humble/setup.bash
source "$ROS_WS_DIR/install/setup.bash"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$APP_DIR"
bash run_scripts/run.sh
