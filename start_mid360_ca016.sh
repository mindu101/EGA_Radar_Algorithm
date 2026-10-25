#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIDAR_MODEL=mid360
export DEVICE_CONFIG=config/device_ca016.yaml
export DISABLE_REFEREE_IO=true
export START_FIELD_PIPELINE="${START_FIELD_PIPELINE:-false}"
if [[ "$START_FIELD_PIPELINE" != "true" ]]; then
    echo "MID360 + CA016 采集预览：场地聚类和坐标融合未启用。"
    echo "当前相机内参、相机位姿和场地地图不属于这套实机，需先标定再启用场地管线。"
    echo "原始点云显示在单独的 RViz 窗口，不会在主界面生成点云图片；若被主界面遮住，请按 Alt+Tab 切换。"
fi
exec "$ROOT_DIR/start_fusion.sh"
