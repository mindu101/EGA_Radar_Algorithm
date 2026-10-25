#!/usr/bin/env bash
set -euo pipefail

# 一键录制 Livox 点云 rosbag 和本机整屏视频。
# 默认保存到 ./recorder/{yyyyMMdd_HHmmss}/，按 Ctrl+C 停止全部录制。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/$(basename "${BASH_SOURCE[0]}")"
LIDAR_WS_DIR="${LIDAR_WS_DIR:-$ROOT_DIR/Lidar_ws}"
LIVOX_WS_DIR="${LIVOX_WS_DIR:-$ROOT_DIR/../livox_ws}"

TIMESTAMP="${RECORDER_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RECORDER_ROOT="${RECORDER_ROOT:-$ROOT_DIR/recorder}"
SESSION_DIR="$RECORDER_ROOT/$TIMESTAMP"

LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
SCREEN_FPS="${SCREEN_FPS:-30}"
SCREEN_OFFSET="${SCREEN_OFFSET:-0,0}"
SCREEN_EXT="${SCREEN_EXT:-mkv}"
GRACEFUL_STOP_TIMEOUT_SEC="${GRACEFUL_STOP_TIMEOUT_SEC:-15}"

ROSBAG_DIR="$SESSION_DIR/rosbag_livox_lidar"
SCREEN_DIR="$SESSION_DIR/screen"
LOG_DIR="$SESSION_DIR/logs"
SCREEN_OUTPUT="${SCREEN_OUTPUT:-$SCREEN_DIR/screen_record.$SCREEN_EXT}"

PIDS=()
NAMES=()
STOPPING=0

log() {
    printf '[recorder] %s\n' "$*"
}

add_pid() {
    NAMES+=("$1")
    PIDS+=("$2")
}

is_pid_active() {
    local pid="$1"
    local stat
    stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    stat="${stat//[[:space:]]/}"
    [[ -n "$stat" && "$stat" != Z* ]]
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "缺少命令: $command_name" >&2
        exit 1
    fi
}

source_ros_env() {
    local nounset_enabled=0
    if [[ -o nounset ]]; then
        nounset_enabled=1
        set +u
    fi

    if [[ -f /opt/ros/jazzy/setup.bash ]]; then
        # shellcheck disable=SC1091
        source /opt/ros/jazzy/setup.bash
    fi
    if [[ -f "$LIVOX_WS_DIR/install/setup.bash" ]]; then
        # shellcheck disable=SC1090
        source "$LIVOX_WS_DIR/install/setup.bash"
    fi
    if [[ -f "$LIDAR_WS_DIR/install/setup.bash" ]]; then
        # shellcheck disable=SC1090
        source "$LIDAR_WS_DIR/install/setup.bash"
    fi

    if [[ "$nounset_enabled" -eq 1 ]]; then
        set -u
    fi
}

detect_x11_screen_size() {
    local size

    if [[ -n "${SCREEN_SIZE:-}" ]]; then
        printf '%s\n' "$SCREEN_SIZE"
        return
    fi

    if command -v xdpyinfo >/dev/null 2>&1; then
        size="$(xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}' || true)"
        if [[ -n "$size" ]]; then
            printf '%s\n' "$size"
            return
        fi
    fi

    if command -v xrandr >/dev/null 2>&1; then
        size="$(xrandr 2>/dev/null | awk '/current/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "current") {
                    width = $(i + 1)
                    height = $(i + 3)
                    gsub(/,/, "", height)
                    print width "x" height
                    exit
                }
            }
        }' || true)"
        if [[ -n "$size" ]]; then
            printf '%s\n' "$size"
            return
        fi
    fi

    return 1
}

run_lidar_bag() {
    source_ros_env
    cd "$ROOT_DIR"
    exec ros2 bag record "$LIDAR_TOPIC" -o "$ROSBAG_DIR"
}

run_screen_record_x11() {
    local display_name="${DISPLAY:-:0}"
    local screen_size
    screen_size="$(detect_x11_screen_size || true)"
    if [[ -z "$screen_size" ]]; then
        echo "无法自动获取屏幕分辨率，请设置 SCREEN_SIZE，例如 SCREEN_SIZE=1920x1080" >&2
        return 1
    fi

    exec ffmpeg \
        -hide_banner \
        -loglevel warning \
        -f x11grab \
        -video_size "$screen_size" \
        -framerate "$SCREEN_FPS" \
        -i "${display_name}+${SCREEN_OFFSET}" \
        -c:v h264_nvenc \
        -preset p5 \
        -tune hq \
        -rc vbr \
        -cq 20 \
        -b:v 0 \
        -spatial-aq 1 \
        -temporal-aq 1 \
        -pix_fmt yuv420p \
        "$SCREEN_OUTPUT"
}

run_screen_record_wayland() {
    exec wf-recorder \
        --framerate "$SCREEN_FPS" \
        --file "$SCREEN_OUTPUT"
}

run_screen_record() {
    if [[ -n "${WAYLAND_DISPLAY:-}" ]] && command -v wf-recorder >/dev/null 2>&1; then
        run_screen_record_wayland
        return
    fi

    require_command ffmpeg
    run_screen_record_x11
}

stop_all() {
    if [[ "$STOPPING" -eq 1 ]]; then
        return
    fi
    STOPPING=1
    trap - INT TERM EXIT

    log "正在停止录制进程..."
    local pid
    for pid in "${PIDS[@]}"; do
        if is_pid_active "$pid"; then
            kill -INT -- "-$pid" >/dev/null 2>&1 || kill -INT "$pid" >/dev/null 2>&1 || true
        fi
    done

    local deadline=$((SECONDS + GRACEFUL_STOP_TIMEOUT_SEC))
    local all_stopped=0
    while [[ "$SECONDS" -lt "$deadline" ]]; do
        all_stopped=1
        for pid in "${PIDS[@]}"; do
            if is_pid_active "$pid"; then
                all_stopped=0
                break
            fi
        done
        if [[ "$all_stopped" -eq 1 ]]; then
            break
        fi
        sleep 1
    done

    for pid in "${PIDS[@]}"; do
        if is_pid_active "$pid"; then
            kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    done

    for pid in "${PIDS[@]}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done

    log "录制已停止，保存目录: $SESSION_DIR"
}

case "${1:-}" in
    __run_lidar_bag)
        run_lidar_bag
        exit 0
        ;;
    __run_screen_record)
        run_screen_record
        exit 0
        ;;
esac

mkdir -p "$SCREEN_DIR" "$LOG_DIR"

source_ros_env
require_command ros2
require_command setsid

cat > "$SESSION_DIR/session_info.txt" <<EOF
timestamp=$TIMESTAMP
lidar_topic=$LIDAR_TOPIC
rosbag_dir=$ROSBAG_DIR
screen_output=$SCREEN_OUTPUT
screen_fps=$SCREEN_FPS
screen_ext=$SCREEN_EXT
screen_size=${SCREEN_SIZE:-auto}
screen_offset=$SCREEN_OFFSET
graceful_stop_timeout_sec=$GRACEFUL_STOP_TIMEOUT_SEC
EOF

trap stop_all INT TERM EXIT

log "保存目录: $SESSION_DIR"
log "启动 rosbag: ros2 bag record $LIDAR_TOPIC -o $ROSBAG_DIR"
RECORDER_TIMESTAMP="$TIMESTAMP" RECORDER_ROOT="$RECORDER_ROOT" LIDAR_TOPIC="$LIDAR_TOPIC" setsid "$SCRIPT_PATH" __run_lidar_bag >"$LOG_DIR/rosbag.log" 2>&1 &
add_pid "rosbag" "$!"

log "启动整屏录制: 输出=$SCREEN_OUTPUT"
RECORDER_TIMESTAMP="$TIMESTAMP" RECORDER_ROOT="$RECORDER_ROOT" SCREEN_FPS="$SCREEN_FPS" SCREEN_OFFSET="$SCREEN_OFFSET" SCREEN_SIZE="${SCREEN_SIZE:-}" SCREEN_EXT="$SCREEN_EXT" SCREEN_OUTPUT="$SCREEN_OUTPUT" setsid "$SCRIPT_PATH" __run_screen_record >"$LOG_DIR/screen.log" 2>&1 &
add_pid "screen" "$!"

log "两个录制进程已启动。按 Ctrl+C 停止。日志目录: $LOG_DIR"

while true; do
    sleep 2
    for index in "${!PIDS[@]}"; do
        pid="${PIDS[$index]}"
        if ! is_pid_active "$pid"; then
            log "${NAMES[$index]} 进程已退出，停止其它录制。请查看 $LOG_DIR/${NAMES[$index]}.log"
            exit 1
        fi
    done
done
