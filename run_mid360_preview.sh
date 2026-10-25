#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVOX_WS_DIR="${LIVOX_WS_DIR:-$ROOT_DIR/../livox_ws}"

set +u
source /opt/ros/jazzy/setup.bash
source "$LIVOX_WS_DIR/install/setup.bash"
source "$ROOT_DIR/Lidar_ws/install/setup.bash"
set -u

export ROS_LOG_DIR="$ROOT_DIR/logs/ros"
mkdir -p "$ROS_LOG_DIR"

exec ros2 launch rm2026_lidar_launch mid360_preview.launch.py "$@"
