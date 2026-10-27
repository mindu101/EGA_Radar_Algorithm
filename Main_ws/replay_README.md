# 相机与激光雷达同步 Replay 使用说明

本文描述当前工程的同步回放链路、使用方法、输入文件要求，以及 replay 与正式比赛模式中各项参数的真实来源。

文中的路径均相对于仓库根目录。

当前统一入口为工作区根目录下的 `run_replay_test.sh`，默认读取
`Main_ws/config/test.yaml`。不要使用
`Lidar_ws` 中旧的独立 rosbag 测试配置代替该入口。

## 1. 快速使用

### 1.1 仅执行预检

```bash
cd RM2026_Radar_develop
./run_replay_test.sh Main_ws/config/test.yaml --preflight-only
```

预检会检查：

- rosbag 目录及 `metadata.yaml` 是否存在；
- 相机目录中视频、时间戳 CSV 和标定快照是否存在；
- 相机时间戳是否单调递增；
- `calibration_snapshot.yaml` 是否包含完整的相机内参和外参；
- rosbag 与相机时间范围是否存在有效交集；
- 本次实际采用的内参、外参、同步起点和输出目录。

即使只执行预检，也会在 `replay.output_root` 下创建一次运行目录，并写出解析后的配置快照。

### 1.2 启动完整回放

```bash
cd RM2026_Radar_develop
./run_replay_test.sh
```

也可以显式指定配置：

```bash
./run_replay_test.sh Main_ws/config/test.yaml
```

回放配置的 `runtime.model_backend` 默认设为 `pytorch`，启动器会在本次运行的
`runtime_params.yaml` 中使用仓库里的 `.pt` 权重，避免复用与当前 TensorRT
版本不兼容的 `.engine`。该设置不修改正式 `params.yaml`。本机引擎重建后，
可将该项改为 `configured`，让回放使用正式参数文件指定的模型。

启动后的操作顺序如下：

1. 启动器先启动 `localization`、`dynamic_cloud`、`cluster`，但不启动真实 Livox 驱动。
2. 启动 Python 主程序和 UI。
3. rosbag 尚未播放时，UI 持续显示有效回放区间的第一帧，供人工标点。
4. 在 UI 中点击“直接导入外参”，默认导入本次录像保存的相机外参；也可以重新手动标定。
5. 根据需要勾选 `xy maps`，然后点击“启动雷达站”。
6. 主循环完成模型加载并写出就绪标记后，启动器才开始播放 rosbag。
7. rosbag 发布 `/clock`；相机回放线程按照同一 `/clock` 和 CSV 时间戳发布对应视频帧。
8. 有效时间交集播放结束后，是否保留 UI 由 `runtime.keep_ui_after_replay` 决定。

当前没有“先播放一段雷达，再开始相机”的预播放阶段。相机与雷达固定从首个共同有效时间戳同时开始。

## 2. Replay 输入目录要求

### 2.1 相机目录

`replay.camera_session_dir` 可以指向 `sensor_record_*` 会话目录，也可以直接指向其中的 `camera` 目录。最终解析出的 `camera` 目录必须包含：

```text
camera/
├── camera_record.mp4
├── camera_timestamps.csv
├── calibration_snapshot.yaml
└── extrinsics.txt                 # 可选；没有时由快照 R/t 临时生成
```

`camera_timestamps.csv` 的表头为：

```text
frame_index,wall_time_ns,wall_time_s,video_time_s,camera_stamp
```

其中：

- `frame_index` 与 `camera_record.mp4` 中的编码帧一一对应；
- `camera_stamp` 是相机帧携带的时间戳；
- `wall_time_s` 是录像程序接收该帧时的系统墙钟时间；
- `video_time_s` 只描述 MP4 内部帧序号对应的播放时间，不作为同步回放时钟；
- `replay.time_basis=sensor_stamp` 时优先使用 `camera_stamp`，单行为空时回退到 `wall_time_s`；
- `replay.time_basis=arrival_stamp` 时使用 `wall_time_s`。

当前录像程序生成的 `calibration_snapshot.yaml` 结构如下：

```yaml
intrinsics:
  image_width: 4024        # 仅供人工核验
  image_height: 3036       # 仅供人工核验
  K: [[...], [...], [...]]
  dist_coeffs: [...]
extrinsics:
  rotation_matrix: [[...], [...], [...]]
  translation_vector: [[...], [...], [...]]
```

重要约定：

- replay 强制要求 `K`、`dist_coeffs`、`rotation_matrix` 和
  `translation_vector` 存在；`image_width/image_height` 不影响预检结果；
- replay 必须使用该快照的 `intrinsics.K` 和 `intrinsics.dist_coeffs`；
- 快照缺失或内参不完整时直接报错，不允许回退 `config/params.yaml`；
- `image_width/image_height` 只用于人工核验，不参与 K 的缩放；
- 实际视频分辨率由 OpenCV 现场读取 `camera_record.mp4`；
- XY Map 的宽高由当前解码帧的 `img.shape` 决定；
- 不再读取或维护 `camera_matadata.yaml`、`camera_metadata.yaml` 等第二份相机元数据文件。

快照在 UI 开始本次传感器录像时写入：内参取当时运行的算法配置，外参优先取 UI
当前已确认的 R/t；若 UI 尚无当前外参，则尝试读取“直接导入外参”所指向的最新外参。

### 2.2 rosbag 目录

`replay.rosbag_path` 可以指向：

- rosbag 会话目录；
- `rosbag_livox_lidar` 目录；
- 该目录中的 `.db3` 文件。

最终目录必须包含 `metadata.yaml`。回放程序从其中读取 rosbag 的起始时间和持续时间，并直接执行：

```text
ros2 bag play <bag_dir>
    --rate <replay.playback_rate>
    --delay <replay.discovery_delay_sec>
    --start-offset <自动计算值>
    --clock 100.0
    --disable-keyboard-controls
```

## 3. 同步规则

设：

```text
camera_start = 相机 CSV 第一条有效时间戳
camera_end   = 相机 CSV 最后一条有效时间戳
bag_start    = rosbag metadata 起始时间
bag_end      = bag_start + rosbag duration
valid_start  = max(camera_start, bag_start)
valid_end    = min(camera_end, bag_end)
```

如果 `valid_start >= valid_end`，预检失败。

实际 rosbag 起播偏移为：

```text
rosbag_start_offset_sec = valid_start - bag_start
```

相机只装载 `[valid_start, valid_end]` 内的 CSV 记录。rosbag 发布 `/clock` 后，相机线程发布时间戳不晚于当前 `/clock` 的最新一帧。

当前唯一支持的过载策略是 `realtime_drop`：

- 如果处理管线跟不上回放时钟，相机不会拖慢 `/clock`；
- 已经过期的中间相机帧会被覆盖，只发布当前应到达的最新帧；
- 因此 replay 保持真实时间关系，但检测处理帧数可能低于录像帧数；
- `playback_rate=1.0` 表示按原始时间速度回放；改变倍率会同时改变 rosbag `/clock` 和相机发布速度。

## 4. 参数来源总表

下表中的“正式比赛模式”指工作区根目录 `start.sh` 启动的完整比赛链路；如果只在
`Main_ws` 中运行 `bash run_scripts/run.sh`，Python 侧参数来源相同，但激光雷达、信息波和干扰波进程需要另行启动。

| 参数或运行数据 | Replay 模式来源 | Replay 中的实际处理 | 正式比赛模式来源 | 正式比赛中的实际处理 |
|---|---|---|---|---|
| 统一启动入口 | 工作区 `run_replay_test.sh` | 调用 `tools/synchronized_replay.py`，统一管理 ROS、Python/UI 和 rosbag 三类进程 | 工作区 `start.sh` | 同时启动真实雷达管线、Python/UI、信息波、干扰波及其他比赛进程 |
| Replay 主配置 | `Main_ws/config/test.yaml`，也可由 `run_replay_test.sh` 第一个参数替换 | 只保存回放输入、同步、测试状态和进程控制参数 | 不使用 `test.yaml` | `run_scripts/run.sh` 不传 `--test_config`，保持正式逻辑 |
| `replay.enabled` | `test.yaml::replay.enabled`，代码默认 `true` | 为 `false` 时统一启动器拒绝启动 | 不适用 | 正式比赛不读取该键 |
| rosbag 路径 | `test.yaml::replay.rosbag_path` | 解析到含 `metadata.yaml` 的 bag 目录；必填 | 不适用 | 正式比赛使用真实 Livox，不播放 bag |
| 相机会话路径 | `test.yaml::replay.camera_session_dir` | 解析到含 MP4、CSV、标定快照的 `camera` 目录；必填 | 不适用 | 正式比赛使用真实相机 |
| 相机图像数据 | `<camera_dir>/camera_record.mp4` | OpenCV 解码，BGR 转为连续 RGB 数组 | 海康相机 MVS SDK | 按真实采集节奏产生图像 |
| 相机帧时间戳 | `<camera_dir>/camera_timestamps.csv` | 根据 `replay.time_basis` 使用 `camera_stamp` 或 `wall_time_s` | 相机驱动实时帧时间戳 | 主循环读取当前最新真实帧及其时间戳 |
| `replay.time_basis` | `test.yaml`，当前为 `sensor_stamp`，代码默认也是 `sensor_stamp` | `sensor_stamp` 使用 `camera_stamp`；`arrival_stamp` 使用 `wall_time_s` | 不适用 | 不做录像 CSV 时间基准选择 |
| `replay.playback_rate` | `test.yaml`，当前及默认值为 `1.0` | 传给 `ros2 bag play --rate`，相机通过 `/clock` 同倍率运行；必须大于 0 | 不适用 | 真实传感器按硬件频率运行 |
| `replay.overload_policy` | `test.yaml`，当前为 `realtime_drop` | 当前只允许 `realtime_drop`；过期相机帧被覆盖 | 实时相机缓冲区最新帧语义 | 消费者落后时读取最新帧，不通过 replay 参数控制 |
| `replay.discovery_delay_sec` | `test.yaml`，当前及默认值为 `0.5 s` | 传给 `ros2 bag play --delay`，给 DDS 发现留出时间 | 不适用 | 真实传感器持续发布；ROS 管线另有启动延时 |
| `replay.output_root` | `test.yaml` | 空值时为 `Main_ws/logs/replay`；每次创建时间戳子目录 | 不适用 | 正式日志目录由各模块和 `start.sh` 决定 |
| 算法基础配置路径 | `test.yaml::runtime.params_config`，默认 `Main_ws/config/params.yaml` | 先复制为本次运行的 `runtime_params.yaml`，再执行 replay 专用覆盖 | `Main_ws/run_scripts/run.sh` 固定传入 `config/params.yaml` | Python 主进程直接读取该文件 |
| 车辆、装甲、数字模型路径和检测阈值 | 本次 `runtime_params.yaml` 的 `car_detector`、`armor_detector`、`digit_detector` | 不作 replay 特殊修改，继承 `runtime.params_config` 指定文件 | `Main_ws/config/params.yaml` 同名节点 | 与 replay 使用相同算法配置 |
| 追踪、匈牙利匹配、二维 KF、猜点参数 | 本次 `runtime_params.yaml::tracker` | 不作 replay 特殊修改 | `Main_ws/config/params.yaml::tracker` | 与 replay 使用相同算法配置 |
| 投影—雷达统计参数 | 本次 `runtime_params.yaml::projection_lidar_statistics` | 不作 replay 特殊修改，进程启动或来源改变时按现有逻辑清零 | `Main_ws/config/params.yaml::projection_lidar_statistics` | 与 replay 使用相同统计逻辑 |
| XY Map 参数 | 本次 `runtime_params.yaml::ray_projection` | 配置来源不变；映射宽高取实际解码帧 `img.shape` | `Main_ws/config/params.yaml::ray_projection` | 配置来源不变；映射宽高取真实相机帧 `img.shape` |
| Python 场地网格与标定关键点 | 本次 `runtime_params.yaml::field.mesh_path/keypoints` | 不作 replay 特殊修改 | `Main_ws/config/params.yaml::field` | 与 replay 相同 |
| 推理设备 | 本次 `runtime_params.yaml::device` | 不作 replay 特殊修改，当前通常为 CUDA | `Main_ws/config/params.yaml::device` | 与 replay 相同 |
| 可视化、耗时统计、TF 调试、前哨站图像参数 | 本次 `runtime_params.yaml` 的对应节点 | 除指标输出路径外不作 replay 特殊修改 | `Main_ws/config/params.yaml` 的对应节点 | 与 replay 相同 |
| `debug.camera_mode` | 统一启动器强制写入本次 `runtime_params.yaml` | 固定为 `replay`，实例化 `TimestampReplayCamera` | `Main_ws/config/params.yaml::debug.camera_mode` | 正常应为 `camera`，实例化真实海康相机 |
| `debug.camera_color_order` | 统一启动器强制写入本次 `runtime_params.yaml` | 固定为 `rgb`，与 MP4 解码后的 BGR→RGB 转换一致 | `Main_ws/config/params.yaml::debug.camera_color_order` | 当前正式配置为 `rgb` |
| `debug.inference_video` | 统一启动器强制写入本次 `runtime_params.yaml` | 固定为 `false`；相机类型只由 `camera_mode=replay` 决定 | `Main_ws/config/params.yaml` | 仅当没有显式 `camera_mode` 时参与旧模式兼容判断 |
| `debug.video_source` | 统一启动器写为本次 `<camera_dir>/camera_record.mp4` | 用于运行配置可追溯；`TimestampReplayCamera` 的最终路径来自 `resolved_test.yaml::_resolved.camera_video` | `Main_ws/config/params.yaml::debug.video_source` | 只有 `camera_mode=image` 时作为普通视频输入 |
| 性能指标开关和采样参数 | `runtime.params_config` 中的 `debug.performance_metrics` | 若启用，仅把 `output_path` 改为本次运行目录的 `runtime_metrics.jsonl` | `Main_ws/config/params.yaml::debug.performance_metrics` | 按配置中的原始 `output_path` 输出 |
| 相机内参 K、畸变 `dist_coeffs` | `<camera_dir>/calibration_snapshot.yaml::intrinsics` | 强制覆盖本次 `runtime_params.yaml::transform.K/dist_coeffs`；缺失或不完整直接失败 | `Main_ws/config/params.yaml::transform.K/dist_coeffs` | 正式投影不读取 `device.yaml::camera.camera_matrix/dist_coeffs` |
| 相机图像宽高 | OpenCV 从 `camera_record.mp4` 读取，逐帧以 `img.shape` 复核 | 快照 `image_width/image_height` 只供人工核验，禁止据此缩放 K | `Main_ws/config/device.yaml::camera.width/height` 设置真实相机，运行中以实际帧 `img.shape` 为准 | 海康驱动把宽高写入相机并检查实际值 |
| 相机外参 R/t 的运行参数初值 | `<camera_dir>/calibration_snapshot.yaml::extrinsics` | 强制写入本次 `runtime_params.yaml::transform.R/t` | `Main_ws/config/params.yaml::transform.R/t` | 只作为启动阶段配置值；正式定位应采用 UI 最终确认的外参 |
| UI“直接导入外参”的默认文件 | 优先 `<camera_dir>/extrinsics.txt`；不存在时由快照 R/t 在本次运行目录生成 `recorded_extrinsics.txt` | 启动器设置环境变量 `RM_EXTRINSICS_IMPORT_PATH` 指向该文件 | `Main_ws/recorder/extrinstics/latest.txt` | 可直接导入，也可现场 PnP 或手动输入；UI 确认值最终覆盖投影转换器 |
| `calibration_snapshot.yaml::image_width/image_height` | 录像时由相机接口 `camera.width/height` 记录 | 仅显示在预检信息中，不参与投影、缩放和视频解码 | 不作为正式运行配置输入 | 正式相机宽高来自 `device.yaml` 与真实帧 |
| 相机采集率、曝光、增益 | 不参与 replay；`runtime.device_config` 虽会传给 Python，但回放相机不访问这些硬件键 | 帧率由 CSV 时间戳和 `/clock` 决定 | `Main_ws/config/device.yaml::camera.acquisition_rate/exposure_time/gain` | 海康驱动写入真实相机 |
| 相机录像编码参数 | 不用于读取既有录像 | replay 只解码 MP4 | `Main_ws/config/device.yaml::camera.recording_*` | 只在 UI 新录制相机数据时使用 |
| `runtime.device_config` | `test.yaml`，默认 `Main_ws/config/device.yaml` | 仍传给 `main.py` 保持统一接口；真实相机硬件参数不控制 replay 图像 | `run_scripts/run.sh` 固定为 `config/device.yaml` | 控制真实相机及录像相关硬件参数 |
| 阵营 | `test.yaml::runtime.faction`；当前为 `red`，代码默认 `manual` | `red/blue` 直接设置 `RefereeCommManager.faction`；`manual` 保持对象初值或人工状态 | 裁判系统实时 `robot_id` | `robot_id < 100` 为红方，`robot_id >= 100` 为蓝方 |
| 裁判串口、UDP、信息波状态 | `test.yaml::runtime.disable_referee_io`；当前为 `true` | 为 `true` 时不启动裁判串口和 UDP 线程，相关状态保持初值；不模拟哨兵数据 | `params.yaml::referee`，且 `start.sh` 可用 `REFEREE_PORT/REFEREE_BAUDRATE` 覆盖 | 正式启动通信线程并接收真实裁判、哨兵、信息波和密钥数据 |
| Python 获取的雷达聚类点话题 | 本次 `runtime_params.yaml::lidar_fusion.topic/queue_size` | 与正式模式相同，默认订阅 `/livox/lidar_cluster` | `Main_ws/config/params.yaml::lidar_fusion` | 与 replay 相同 |
| 雷达原始点云数据 | `test.yaml::replay.rosbag_path` 对应 rosbag | `start_livox=false`，点云来自录制消息 | 真实 Livox HAP | `start_livox=true`，由 `livox_ros_driver2` 发布 |
| ROS 模拟时钟 | 统一启动器硬编码 `use_sim_time=true`，rosbag 使用 `--clock 100.0` | localization、dynamic_cloud、cluster、RViz 和相机回放共用 `/clock` | `lidar_pipeline.launch.py` 默认 `use_sim_time=false` | 使用系统/ROS 实时时钟 |
| 雷达管线 launch | 源码文件 `Lidar_ws/src/rm2026_lidar_launch/launch/lidar_pipeline.launch.py` | 统一启动器通过绝对路径启动 | `ros2 launch rm2026_lidar_launch lidar_pipeline.launch.py` | 使用 `Lidar_ws/install/...` 中已安装的 launch；修改源码后应重新 `colcon build` |
| 雷达管线主配置 | 源码文件 `Lidar_ws/src/rm2026_lidar_launch/config/params.yaml::common/lidar_pipeline` | 通过 `params_file:=...` 显式传入；不读取其中旧的 `rosbag_player` 测试块 | `Lidar_ws/install/rm2026_lidar_launch/share/rm2026_lidar_launch/config/params.yaml` | 安装副本来自源码同名文件；修改源码后应重新构建/安装 |
| localization 参数 | 雷达主配置 `lidar_pipeline.localization.*` | 与正式参数一致，包括累计帧数、GICP、稳定判据、跳变门限、平滑和体素尺寸 | 已安装雷达主配置同名节点 | 与 replay 相同 |
| dynamic_cloud 参数 | 雷达主配置 `lidar_pipeline.dynamic_cloud.*` | 与正式参数一致，包括近远端累计、半场分界、地图距离阈值、线程数和各 ROI | 已安装雷达主配置同名节点 | 与 replay 相同 |
| cluster 参数 | `lidar_pipeline.launch.py` | 当前只显式设置 `use_sim_time=true` | 同一 launch | 当前只显式设置 `use_sim_time=false` |
| 激光雷达初始外参 | 雷达主配置 `common.extrinsics_file` 指向的源码 YAML，当前为 `Lidar_ws/src/rm2026_lidar_launch/config/lidar_extrinsics.yaml` | 作为 localization/GICP 初值；不是相机 `calibration_snapshot.yaml` | 已安装雷达主配置指向的已安装外参 YAML | 与 replay 算法相同，但输入为真实点云 |
| 雷达静态点云地图 | 雷达主配置 `common.map.package/relative_path` | 从对应 ROS 包的 share 目录解析 | 已安装雷达主配置同名节点 | 与 replay 相同 |
| Python 射线投影场地模型 | 算法 `runtime_params.yaml::field.mesh_path` | Open3D 射线投影和 XY Map 使用 `.ply` 网格 | `Main_ws/config/params.yaml::field.mesh_path` | 与 replay 相同；它和 ROS 雷达 `.pcd` 地图不是同一个配置 |
| RViz 是否启动 | `test.yaml::runtime.launch_rviz`，当前为 `true` | 覆盖 launch 参数 | 雷达主配置 `lidar_pipeline.launch_rviz` | 由正式雷达配置决定 |
| RViz 配置 | 雷达主配置 `lidar_pipeline.rviz_config` | 统一启动器显式复用该路径 | 已安装雷达主配置 `lidar_pipeline.rviz_config` | 与 replay 使用同一显示布局 |
| ROS 节点就绪超时 | `test.yaml::runtime.ros_ready_timeout_sec`，当前及默认 `30 s` | 等待 localization、dynamic_cloud、cluster 出现 | 不适用 | `start.sh` 不等待该统一回放就绪条件 |
| 人工标定/启动等待超时 | `test.yaml::runtime.tracking_ready_timeout_sec`，当前及默认 `600 s` | 等待用户点击“启动雷达站”及主循环写出 `tracking.ready` | 不适用 | 正式 UI 由人工直接操作，不受 replay 启动器超时控制 |
| 子进程停止超时 | `test.yaml::runtime.shutdown_timeout_sec`，当前及默认 `10 s` | 统一停止 rosbag、Python/UI 和 ROS 雷达进程 | 不适用 | 由 `start.sh` 和终端进程管理 |
| 播放结束后保留 UI | `test.yaml::runtime.keep_ui_after_replay`，当前及默认 `true` | 为 `true` 时等待关闭 UI 或 Ctrl+C | 不适用 | 正式 UI 持续运行直到人工退出 |
| `device.yaml::lidar.*` | 当前统一 replay 不使用 | Python 雷达点来自 ROS `lidar_fusion` 订阅，ROS 雷达参数来自 `Lidar_ws` | 当前 `start.sh` 雷达管线同样不读取该节点 | 属于旧接口/保留配置，不要与 ROS 雷达外参混淆 |
| `EGA.../params.yaml::rosbag_player.*` 与 `rosbag_lidar_fusion_test.*` | 当前统一 replay 不使用 | bag 路径、倍率和等待时间只读取 `test.yaml::replay` | 正式比赛不使用 | 仅供其他独立测试 launch 使用 |

### 4.1 配置覆盖优先级

Replay 中 Python 算法配置的最终优先级从高到低为：

```text
1. 录像 calibration_snapshot.yaml
   ├── transform.K
   ├── transform.dist_coeffs
   ├── transform.R
   └── transform.t

2. 统一启动器固定覆盖
   ├── debug.camera_mode = replay
   ├── debug.camera_color_order = rgb
   ├── debug.inference_video = false
   ├── debug.video_source = 本次 MP4
   └── debug.performance_metrics.output_path = 本次运行目录（仅在指标启用时）

3. test.yaml::runtime.params_config 指向的算法配置
   └── 检测、追踪、匹配、卡尔曼、猜点、场地、可视化等其余参数
```

`test.yaml` 本身不会整体合并到 `params.yaml`。它只通过启动器和 `main.py` 中明确实现的键改变回放行为。

正式比赛中不存在上述 replay 覆盖：

```text
config/params.yaml
├── 相机投影内参
├── 检测/追踪/场地/可视化参数
└── Python 雷达融合参数

config/device.yaml
└── 真实相机采集与录像硬件参数

recorder/extrinstics/latest.txt 或现场重新标定
└── UI 最终确认并应用的相机外参
```

## 5. `test.yaml` 参数说明

当前支持的完整结构如下：

```yaml
replay:
  enabled: true
  rosbag_path: "recorder/.../rosbag_livox_lidar"
  camera_session_dir: "Main_ws/recorder/sensor_record_..."
  time_basis: "sensor_stamp"
  playback_rate: 1.0
  overload_policy: "realtime_drop"
  discovery_delay_sec: 0.5
  output_root: ""

runtime:
  params_config: "Main_ws/config/params.yaml"
  device_config: "Main_ws/config/device.yaml"
  faction: "red"
  disable_referee_io: true
  launch_rviz: true
  ros_ready_timeout_sec: 30.0
  tracking_ready_timeout_sec: 600.0
  shutdown_timeout_sec: 10.0
  keep_ui_after_replay: true
```

所有相对路径都以仓库根目录为基准，而不是以 `config/test.yaml` 所在目录为基准。

## 6. 每次运行生成的文件

默认输出目录为：

```text
Main_ws/logs/replay/YYYYMMDD_HHMMSS/
```

其中包含：

| 文件 | 内容 |
|---|---|
| `runtime_params.yaml` | 从 `runtime.params_config` 复制并完成 replay 覆盖后的 Python 最终配置 |
| `resolved_test.yaml` | 原 `test.yaml` 加 `_resolved` 节点，记录绝对路径、时间范围、起播偏移和就绪文件 |
| `recorded_extrinsics.txt` | 仅当相机目录没有 `extrinsics.txt` 时，由快照 R/t 生成 |
| `tracking.ready` | 点击“启动雷达站”且主循环准备完成后写出，触发 rosbag 开始播放 |
| `ros_pipeline.log` | localization、dynamic_cloud、cluster 和 RViz 输出 |
| `python_app.log` | Python 主程序与 UI 的标准输出和错误 |
| `rosbag_play.log` | `ros2 bag play` 输出 |
| `runtime_metrics.jsonl` | 仅在 `debug.performance_metrics.enabled=true` 时生成 |

排查参数来源时，应优先检查本次运行目录中的 `runtime_params.yaml` 和 `resolved_test.yaml`，不要只查看原始 `params.yaml` 或 `test.yaml`。

## 7. 常见问题

### 7.1 为什么 replay 修改了 `params.yaml` 内参却没有生效？

这是预期行为。Replay 的 K 和畸变系数必须来自录像目录的
`calibration_snapshot.yaml::intrinsics`，并覆盖基础 `params.yaml`。正式比赛才直接使用
`config/params.yaml::transform.K/dist_coeffs`。

### 7.2 为什么修改 `device.yaml::camera.camera_matrix` 没有效果？

当前射线投影在 replay 和正式比赛中都不读取该键：

- replay 使用录像 `calibration_snapshot.yaml`；
- 正式比赛使用 `config/params.yaml::transform.K/dist_coeffs`。

`device.yaml` 主要负责真实相机采集和录像硬件参数。

### 7.3 为什么快照显示的宽高与视频不一致？

快照宽高只用于人工核验。程序实际从视频和每帧 `img.shape` 读取宽高，不会根据快照宽高缩放 K。若宽高不一致，应人工确认录像和快照是否属于同一次相机会话。

### 7.4 为什么回放检测帧率低于录像帧率？

`realtime_drop` 保持真实时钟，不会为了等待推理而减慢 rosbag。如果单帧检测耗时超过相机帧间隔，中间已过期帧会被覆盖。这代表当前机器在真实比赛相同输入频率下也可能无法逐帧处理。

### 7.5 为什么启动后 rosbag 没有立即播放？

启动器会等待：

1. ROS 雷达节点就绪；
2. Python/UI 启动；
3. 用户完成外参确认并点击“启动雷达站”；
4. 主循环完成模型初始化并写出 `tracking.ready`。

等待期间 UI 会提供第一帧用于标定。超过
`runtime.tracking_ready_timeout_sec` 仍未就绪时，统一启动器报错并关闭子进程。

### 7.6 修改雷达配置后 replay 和正式比赛为何表现不同？

Replay 显式读取源码目录：

```text
Lidar_ws/src/rm2026_lidar_launch/config/params.yaml
```

正式 `start.sh` 使用 ROS 包的安装副本：

```text
Lidar_ws/install/rm2026_lidar_launch/share/rm2026_lidar_launch/config/params.yaml
```

修改源码雷达配置后，应重新构建工作区，保证正式比赛使用的安装副本同步更新：

```bash
cd Lidar_ws
./build.sh
```

### 7.7 Replay 是否模拟裁判系统、哨兵、串口和波形通信？

默认不模拟。当前 `test.yaml` 设置 `runtime.disable_referee_io=true`，只强制指定测试阵营，其他裁判、哨兵、UDP、信息波和干扰波状态保持初值。该模式主要用于验证车辆检测、相机投影、雷达聚类和融合追踪链路。

## 8. 相关实现文件

- 统一启动与同步计算：`../tools/synchronized_replay.py`
- Replay 相机：`driver/replay_camera/timestamp_replay_camera.py`
- Replay 配置：`config/test.yaml`
- Replay Python 启动脚本：`run_scripts/run_test.sh`
- 正式 Python 启动脚本：`run_scripts/run.sh`
- 标定快照录制：`interface/core.py::_save_sensor_calibration_snapshot`
- Replay 主循环就绪标记：`main_event_loop.py::_signal_replay_tracking_ready`
- ROS 雷达管线：`../Lidar_ws/src/rm2026_lidar_launch/launch/lidar_pipeline.launch.py`
- ROS 雷达参数：`../Lidar_ws/src/rm2026_lidar_launch/config/params.yaml`
