#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_WS_DIR="$ROOT_DIR/Lidar_ws"
APP_DIR="$ROOT_DIR/Main_ws"
LIVOX_WS_DIR="${LIVOX_WS_DIR:-$ROOT_DIR/../livox_ws}"
CONDA_ENV="${CONDA_ENV:-cud128}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
LIDAR_MODEL="${LIDAR_MODEL:-hap}"
USE_EXISTING_LIDAR="${USE_EXISTING_LIDAR:-auto}"
DEVICE_CONFIG="${DEVICE_CONFIG:-config/device.yaml}"
PARAMS_CONFIG="${PARAMS_CONFIG:-config/params.yaml}"
DISABLE_REFEREE_IO="${DISABLE_REFEREE_IO:-false}"
START_FIELD_PIPELINE="${START_FIELD_PIPELINE:-true}"
ROS_PID=""
DRIVER_PID=""
RVIZ_PID=""

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "缺少运行文件: $1" >&2
        exit 1
    fi
}

sanitize_ros_library_path() {
    local part
    local -a paths=()
    local -a filtered=()
    IFS=: read -r -a paths <<< "${LD_LIBRARY_PATH:-}"
    for part in "${paths[@]}"; do
        [[ -z "$part" || "$part" == /opt/MVS/lib/* ]] && continue
        filtered+=("$part")
    done
    local joined=""
    for part in "${filtered[@]}"; do
        joined="${joined:+$joined:}$part"
    done
    export LD_LIBRARY_PATH="$joined"
}

cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "$ROS_PID" "$DRIVER_PID" "$RVIZ_PID"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$ROS_PID" "$DRIVER_PID" "$RVIZ_PID"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

has_livox_driver_process() {
    pgrep -u "$(id -u)" -f '^.*/livox_ros_driver2_node( |$)' >/dev/null
}

wait_for_pointcloud() {
    local width
    if ! width=$(
        set +u
        source /opt/ros/jazzy/setup.bash
        source "$LIVOX_WS_DIR/install/setup.bash"
        source "$ROS_WS_DIR/install/setup.bash"
        set -u
        timeout 25s ros2 topic echo /livox/lidar sensor_msgs/msg/PointCloud2 \
            --no-daemon --once --field width 2>/dev/null
    ); then
        echo "25 秒内未收到 /livox/lidar 点云。请检查雷达供电、网线、IP 和驱动输出。" >&2
        return 1
    fi
    width=$(printf '%s\n' "$width" | awk '/^[0-9]+$/ {print; exit}')
    if ! [[ "$width" =~ ^[1-9][0-9]*$ ]]; then
        echo "/livox/lidar 已连接，但收到空点云（width=$width）。" >&2
        return 1
    fi
    echo "已收到 MID360 点云，单帧宽度: $width 点。"
}

require_file /opt/ros/jazzy/setup.bash
require_file "$LIVOX_WS_DIR/install/setup.bash"
require_file "$ROS_WS_DIR/install/setup.bash"
require_file "$APP_DIR/config/params.yaml"
require_file "$APP_DIR/config/device.yaml"

if [[ "$LIDAR_MODEL" != "hap" && "$LIDAR_MODEL" != "mid360" ]]; then
    echo "LIDAR_MODEL 只支持 hap 或 mid360: $LIDAR_MODEL" >&2
    exit 1
fi
require_file "$APP_DIR/$DEVICE_CONFIG"
require_file "$APP_DIR/$PARAMS_CONFIG"

if [[ "$LIDAR_MODEL" == "mid360" && "$USE_EXISTING_LIDAR" == "auto" ]]; then
    topic_info=$(
        set +u
        source /opt/ros/jazzy/setup.bash
        source "$LIVOX_WS_DIR/install/setup.bash"
        source "$ROS_WS_DIR/install/setup.bash"
        set -u
        timeout 5s ros2 topic info /livox/lidar 2>/dev/null || true
    )
    publisher_count=$(printf '%s\n' "$topic_info" | awk '/^Publisher count:/ {print $3}')
    if [[ "$topic_info" == *"Type: sensor_msgs/msg/PointCloud2"* && "$publisher_count" =~ ^[1-9][0-9]*$ ]]; then
        USE_EXISTING_LIDAR=true
        echo "检测到已有 /livox/lidar PointCloud2 发布者，复用当前雷达。"
    else
        if has_livox_driver_process; then
            echo "检测到 MID360 驱动进程，但 /livox/lidar 没有发布者；请先停止失效驱动，避免重复占用 UDP 端口。" >&2
            exit 1
        fi
        USE_EXISTING_LIDAR=false
    fi
fi

if [[ "$LIDAR_MODEL" == "mid360" && "$USE_EXISTING_LIDAR" != "true" ]]; then
    DRIVER_RVIZ=false
    if [[ "$START_FIELD_PIPELINE" != "true" ]]; then
        DRIVER_RVIZ="$LAUNCH_RVIZ"
    fi
    (
        set +u
        source /opt/ros/jazzy/setup.bash
        source "$LIVOX_WS_DIR/install/setup.bash"
        source "$ROS_WS_DIR/install/setup.bash"
        set -u
        sanitize_ros_library_path
        export ROS_LOG_DIR="$ROOT_DIR/logs/ros"
        mkdir -p "$ROS_LOG_DIR"
        exec setsid ros2 launch rm2026_lidar_launch mid360_preview.launch.py \
            launch_rviz:="$DRIVER_RVIZ"
    ) &
    DRIVER_PID=$!
fi

if [[ "$LIDAR_MODEL" == "mid360" ]]; then
    wait_for_pointcloud
fi

if [[ "$LIDAR_MODEL" == "mid360" && "$USE_EXISTING_LIDAR" == "true" &&
      "$START_FIELD_PIPELINE" != "true" && "$LAUNCH_RVIZ" == "true" ]]; then
    (
        set +u
        source /opt/ros/jazzy/setup.bash
        source "$ROS_WS_DIR/install/setup.bash"
        set -u
        sanitize_ros_library_path
        exec setsid rviz2 -d "$ROS_WS_DIR/install/rm2026_lidar_launch/share/rm2026_lidar_launch/config/mid360_preview.rviz"
    ) &
    RVIZ_PID=$!
fi

if [[ "$START_FIELD_PIPELINE" == "true" ]]; then
    (
        set +u
        source /opt/ros/jazzy/setup.bash
        source "$LIVOX_WS_DIR/install/setup.bash"
        source "$ROS_WS_DIR/install/setup.bash"
        set -u
        sanitize_ros_library_path
        export ROS_LOG_DIR="$ROOT_DIR/logs/ros"
        mkdir -p "$ROS_LOG_DIR"
        if [[ "$LIDAR_MODEL" == "mid360" ]]; then
            exec setsid ros2 launch rm2026_lidar_launch lidar_pipeline.launch.py \
                start_livox:=false launch_rviz:="$LAUNCH_RVIZ"
        else
            exec setsid ros2 launch rm2026_lidar_launch lidar_pipeline.launch.py \
                launch_rviz:="$LAUNCH_RVIZ"
        fi
    ) &
    ROS_PID=$!
fi

set +u
source /opt/ros/jazzy/setup.bash
source "$ROS_WS_DIR/install/setup.bash"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

cd "$APP_DIR"
export DEVICE_CONFIG PARAMS_CONFIG
if [[ "$DISABLE_REFEREE_IO" == "true" ]]; then
    bash run_scripts/run.sh --disable-referee-io
else
    bash run_scripts/run.sh
fi
