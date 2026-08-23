# 开源内容清单

## 包含

| 部分 | 内容 | 运行必要性 |
| --- | --- | --- |
| ROS2 点云 | Livox 启动、地图配准、动态点云、聚类和消息接口 | 必需 |
| Python 主程序 | 相机驱动、检测、分类、跟踪、融合、标定 UI | 必需 |
| 模型 | 比赛 engine 及对应 PT、ONNX、PTH | 必需或用于重新部署 |
| 场地资源 | UI 图片、PLY 射线地图、PCD 点云地图 | 必需 |
| 配置 | 相机、模型、融合、外参、ROS2 定位与聚类参数 | 必需 |
| 回放工具 | 同步回放脚本、示例配置和测试代码 | 辅助验证 |
| 回放数据 | 约 7 秒精简样例和约 9.9 分钟完整同步录制，含相机、Livox 点云和标定快照 | 独立网盘分发，用于快速检查与完整复现 |
| 录制工具 | Livox rosbag 与屏幕录制脚本 | 辅助采集 |
| 外部接口 | 裁判串口、信息波结果和自瞄通信接口 | 保留接口，非融合核心 |

## 未包含

- `RM2026_Waves_Analyze/`：信息波和干扰波 MATLAB 解析算法。
- 激光云台自动瞄准项目及其模型、配置和硬件驱动。
- 其他历史 rosbag、相机录像、屏幕录像及压缩包；配套精简样例和一组完整同步录制通过网盘单独分发。
- 运行日志、回放输出、性能统计结果和录制缓存。
- ROS2 `build/`、`install/`、`log/`，需要在目标机器重新构建。
- Python 缓存、编辑器配置、代理配置和内部 Git 元数据。
- 旧场地图、备用 PCD、历史报告和与当前运行无关的演示视频。

## 外部依赖但不随包提供

- ROS2 Humble 与 `livox_ros_driver2`。
- CUDA、TensorRT、显卡驱动和 Conda/Python 环境。
- 海康 MVS SDK 运行库。
- 相机、Livox 雷达和可选裁判系统串口设备。

## 当前正式资源对应关系

| 配置项 | 文件 |
| --- | --- |
| `car_detector.weights_path` | `Main_ws/weights/car/best.engine` |
| `armor_detector.weights_path` | `Main_ws/weights/armor/armor_best.engine` |
| `armor_detector.digit_weights_path` | `Main_ws/weights/MobileNetv2.2.pth` |
| `field.mesh_path` | `Main_ws/field/RMUC2026.ply` |
| UI 场地图 | `Main_ws/field/field_image.PNG` |
| ROS2 点云地图 | `Lidar_ws/src/dynamic_cloud/config/RMUC2026.pcd` |
| RViz 显示配置 | `Lidar_ws/src/rm2026_lidar_launch/config/display_point_cloud_ROS2.rviz` |
