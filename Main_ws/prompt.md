# CascadeMatchTracker 定位与身份匹配重构：完整实现 Prompt

## 0. 本文档的用途与约束

本文档是后续代码实现的唯一需求基线。实现者必须按照本文档修改 `Main_ws`，不得自行改变状态定义、处理顺序、匹配优先级、坐标单位、默认参数或输出有效性规则。本文档中给出的默认值是第一版实现的确定默认值；若需要调整，只能通过 `config/params.yaml` 修改，不得在 Python 代码中散落新的魔法常量。

本文档解决的问题是：重构当前 `tracker/CascadeMatchTracker` 的身份分配、相机射线与激光雷达点关联、二维轨迹维护和丢失猜点流程。匈牙利匹配和卡尔曼滤波的数学内部原理不需要重新设计，但其输入、候选边、门限、代价、调用时机、重置条件和配置来源必须严格遵守本文档。

实现必须同时满足以下硬约束：

1. 定位只使用当前帧真实检测到的整车框获得相机投影点。UI 的 `xy maps` 未勾选时逐框实时投影；勾选且映射生成完成后全部通过原图分辨率 XY Map 查表。
2. 相机射线只用于建立“当前框与雷达点”的对应关系，二维位置卡尔曼只使用激光雷达二维点或新鲜的己方哨兵获取的坐标更新。
3. “有相机但没有匹配雷达点”不能更新二维位置卡尔曼，也不能确认 class 身份。
4. 新鲜己方哨兵位置 + 雷达匹配具有全局最高优先级，先锁定并剔除己方雷达点；在剩余雷达点的敌方身份流程中，相机 + 雷达联合匹配优先级最高，之后依次为历史 + 雷达、灰色模糊猜测、纯雷达随机猜测、二维卡尔曼短期外推和固定点猜测。
5. `RADAR_GUESSED`、`GRAY_RADAR_GUESSED`、`KF_PREDICTED` 和 `FIXED_GUESSED` 均是可发送给裁判系统的有效坐标来源。
6. 内部维护红蓝双方共 10 个地面机器人 class；0x02BB 接收时间距当前单调时钟不超过 0.15 秒时视为新鲜且位置绝对准确，必须先于所有相机和敌方历史匹配；随机雷达猜测和固定点猜测只允许敌方 class。
7. 保留 Ultralytics BoT-SORT，它继续提供当前整车框的 `bot_id`。
8. 所有匹配均为部分匹配：允许 class、当前框和雷达点保持未匹配；绝不能为了填满矩阵强制产生不合法匹配。
9. 每一阶段先完整计算匹配结果，再原子提交；禁止在遍历候选时边匹配、边修改剩余候选，导致结果依赖循环顺序。
10. 保留现有逐阶段耗时、端到端延迟和 GPU 指标导出，不得重新引入逐对象 `.item()`、逐对象 `.cpu().numpy()`、全分辨率多次复制或双路高频可视化。

---

## 1. 当前工程入口与需要修改的模块

运行入口和当前数据流如下：

1. `main.py`
   - 读取 `config/params.yaml`。
   - 初始化相机、`RefereeCommManager` 和 `MainEventLoop`。
2. `main_event_loop.py`
   - 获取最新相机帧。
   - 获取最新地面激光雷达聚类点和飞坡点。
   - 调用 `CascadeMatchTracker.track(...)`。
   - 将 tracker 输出写入 `divisions_pos`，再生成 `Radar2ClientMessage`。
3. `tracker/CascadeMatchTracker/detector.py`
   - 调用整车检测 + BoT-SORT。
   - 批量裁剪整车 ROI。
   - 批量检测装甲板并识别 digit。
4. `model/yolov12/predictor_with_tracker.py`
   - 当前整车检测和 BoT-SORT `bot_id` 来源。
5. `model/yolov12/predictor.py`
   - 两阶段装甲板方案中的装甲位置/颜色检测器。
6. `model/armor_detector.py`
   - 批量装甲板检测、颜色解析和 digit 分类。
7. `tracker/CascadeMatchTracker/tracker.py`
   - 需要按本文档重写主要状态机和多阶段匹配逻辑。
8. `tracker/CascadeMatchTracker/type.py`
   - 需要定义新的观测数据、身份状态、空间状态、位置来源和 class 轨迹数据。
9. `tracker/CascadeMatchTracker/kalman.py`
   - 保留二维匀速卡尔曼；增加配置化初始化和“位置低不确定、速度高不确定”的硬重置接口。
10. `tracker/CascadeMatchTracker/guess_pts.py`
    - 固定点选择器；必须修正当前红蓝 class 名称映射反置问题。
11. `config/guess_pts.yaml`
    - 保存各机器人固定猜点候选和固定点评分参数。
12. `driver/lidar/ros2_lidar_client.py`
    - 需要暴露雷达点快照及其本机单调时钟接收时间，供新鲜度判断。
13. `driver/referee/referee_comm.py`
    - 需要提供线程安全的 0x02BB 己方位置快照及本机单调时钟接收时间。
14. `driver/referee/serial_protocol.py`
    - 0x02BB 已定义为 10 个 `float32`，单位为米；无需修改协议布局。
15. `interface/core.py`
    - 在标定区域新增文字严格为 `xy maps` 的勾选项和生成状态提示。
    - 标定完成且勾选时触发 XY Map 生成；运行中切换时通知 `MainEventLoop`。
16. `transform/xy_map.py`
    - 新增原图分辨率二维 XY Map 的批量生成、有效掩码、查表和标定签名管理。

禁止修改裁判系统串口包格式、0x0305 字段顺序、信息波优先级和发送频率。

---

## 2. class、阵营、兵种和装甲类别的唯一映射

`config/params.yaml::armor_detector.class_names` 必须保持以下顺序：

| class_id | 名称 | 阵营 | 兵种 |
|---:|---|---|---|
| 0 | B1 | 蓝方 | 英雄 |
| 1 | B2 | 蓝方 | 工程 |
| 2 | B3 | 蓝方 | 3 号步兵 |
| 3 | B4 | 蓝方 | 4 号步兵 |
| 4 | B7 | 蓝方 | 哨兵 |
| 5 | R1 | 红方 | 英雄 |
| 6 | R2 | 红方 | 工程 |
| 7 | R3 | 红方 | 3 号步兵 |
| 8 | R4 | 红方 | 4 号步兵 |
| 9 | R7 | 红方 | 哨兵 |
| 10 | G1 | 灰色/死亡 | 英雄，阵营未知 |
| 11 | G2 | 灰色/死亡 | 工程，阵营未知 |
| 12 | G3 | 灰色/死亡 | 3 号步兵，阵营未知 |
| 13 | G4 | 灰色/死亡 | 4 号步兵，阵营未知 |
| 14 | G7 | 灰色/死亡 | 哨兵，阵营未知 |

灰色类别候选映射固定为：

```text
G1 -> {B1, R1} -> {0, 5}
G2 -> {B2, R2} -> {1, 6}
G3 -> {B3, R3} -> {2, 7}
G4 -> {B4, R4} -> {3, 8}
G7 -> {B7, R7} -> {4, 9}
```

己方和敌方 class 集合由裁判系统阵营唯一决定：

```text
己方为红方：己方={5,6,7,8,9}，敌方={0,1,2,3,4}
己方为蓝方：己方={0,1,2,3,4}，敌方={5,6,7,8,9}
```

启动初期 `FACTION.UNKONWN` 时，tracker 按 `tracker.cascade.unknown_faction_fallback` 处理；默认值为 `red`，因此默认把蓝方 class 当作敌方。裁判系统给出确定阵营后立即使用真实阵营。

digit 分类器内部索引固定为：

```text
0 -> 1
1 -> 2
2 -> 3
3 -> 4
4 -> S，映射到裁判机器人编号 7
5 -> Q/前哨站，不属于 10 个地面机器人，直接忽略
```

---

## 3. 坐标系、单位与有效性

### 3.1 相机框和装甲框

- 均使用原始相机图像像素坐标。
- 整车框格式为 `[x1, y1, x2, y2]`。
- 装甲框在检测器内部是相对于整车 ROI 的 `[x1, y1, x2, y2]`。
- 只有当前帧由整车检测器实际返回的框可以生成射线。

### 3.2 射线落点

射线像素固定取当前整车框的底边中心：

```text
u = (x1 + x2) / 2
v = y2
```

`xy maps` 未勾选时，继续使用 `PixelToWorld.pixel_to_world([u, v])` 与场地网格求交。`xy maps` 已勾选且映射就绪时，以 `u_idx=floor(u+0.5)`、`v_idx=floor(v+0.5)` 从原图分辨率 XY Map 查表，禁止再逐框调用 `pixel_to_world()`；像素框已经裁剪为非负坐标，因此该公式是唯一的最近整数取整规则，不使用 Python 的银行家舍入 `round()`。

无论使用哪种模式，无交点、结果包含 NaN/Inf 或落点不在场地范围时，该框没有有效相机二维点，不参与任何相机—雷达联合匹配。

求得的三维点通过 `solidwork2uwb(...)` 转成 tracker 当前使用的二维 UWB 坐标。继续沿用现有规则：己方为蓝方时，相机点和雷达点都按 `field_length - x, field_width - y` 镜像，保证两种数据进入同一个 tracker 坐标系。

### 3.3 激光雷达点

- ROS `/livox/lidar_cluster` 的地面聚类点单位为米。
- tracker 内部始终使用米。
- `main_event_loop.divisions_pos` 和 0x0305 使用厘米，只有在最终输出阶段乘以 100。
- 有效点必须满足：两个坐标有限，且 `0 <= x <= field_length`、`0 <= y <= field_width`。
- 雷达快照超过 `lidar_fusion.max_snapshot_age_sec` 时，本帧按“没有雷达点”处理，不能继续使用旧点。

### 3.4 哨兵 0x02BB 己方坐标

0x0301/0x02BB 提供以下 5 个己方位置，单位为米：

```text
sentry, hero, engineer, standard_3, standard_4
```

包内没有源时间戳，只能使用 `RefereeCommManager` 收包时记录的 `time.monotonic()`。唯一新鲜度定义为：

```text
age_sec = current_monotonic - received_monotonic
0 <= age_sec <= 0.15：新鲜，坐标按绝对准确处理
age_sec > 0.15：过期
received_monotonic is None：从未收到，按过期处理
```

固定选择 `0.15 s` 是因为哨兵约以 `19 Hz` 发送，正常周期约 `52.6 ms`；该门限容许约 3 个发送周期的调度抖动/偶发丢包，又不会把明显滞后的己方位置继续用于剔除雷达点。边界 `age_sec == 0.15` 属于新鲜，不再提供第二套门限。

位置过期后不允许参与雷达关联，也不允许仅凭旧己方 KF 预测继续占用雷达点。

0x02BB 坐标已经位于比赛场地坐标系，不进行额外红蓝镜像。坐标包含 NaN/Inf、越界或恰好为 `(0, 0)` 时按无效处理。

### 3.5 原图分辨率 XY Map

XY Map 唯一采用当前原始输入帧分辨率，不采用 YOLO 的 `1280×1280` 推理分辨率。原因是整车检测虽然在 `1280×1280` 图像上运行，但 `PredictorWithTracker` 使用归一化框乘以原始输入图像宽高，最终交给 tracker 的 `car_box` 已经恢复为原图像素坐标。

映射数据固定为：

```text
xy_map.shape = (frame_height, frame_width, 2)
xy_map.dtype = float32
valid_mask.shape = (frame_height, frame_width)
valid_mask.dtype = bool
```

`xy_map[v,u]` 保存未按己方阵营镜像的场地 mesh 二维 `(x,y)`。查表后再沿用 `solidwork2uwb`/蓝方镜像规则，因此阵营变化不需要重建映射。

映射宽高必须直接读取当前真实帧的 `img.shape`，不能从 YAML 中猜测。以当前
`4024×3036` 为例，XY Map 约 93 MiB，布尔有效掩码约 12 MiB。全分辨率下每个
整数像素都有对应表项，因此不采用低分辨率映射，也不执行插值。框底边中心最多
只有 0.5 像素小数，使用四舍五入后的最近整数像素；这比跨越场地几何边缘进行
双线性插值更稳定。

---

## 4. 必须新增或重构的数据结构

### 4.1 `ArmorObservation`

每个装甲板观测至少包含：

```text
armor_box: List[float]        # 相对整车 ROI 的 xyxy
armor_confidence: float       # 装甲位置/颜色 YOLO 置信度
digit_index: int              # 0..5
digit_confidence: float       # digit softmax 最大概率
color: BLUE | RED | GRAY
raw_class_id: int             # 0..14；Q 为 -1
combined_confidence: float
```

`combined_confidence` 唯一公式为：

```text
combined_confidence =
    digit_detector.digit_confidence_weight * digit_confidence
  + digit_detector.armor_confidence_weight * armor_confidence
```

两个权重之和必须为 1.0；配置加载时不满足则直接抛出 `ValueError`，不能静默归一化。

### 4.2 `CarObservation`

每个当前帧整车观测至少包含：

```text
car_box
car_confidence
bot_id
armor_observations
selected_armor                 # 可能为 None
semantic_candidates            # 彩色时一个 class；灰色时同 digit 红蓝两个 class
projection_mode                # DIRECT | XY_MAP；无有效投影时仍记录当前模式
ray_world_3d                   # 可能为 None
ray_uwb_2d                     # 可能为 None
```

没有可靠 digit 的整车框仍保留在可视化和 BoT-SORT 数据中，但 `semantic_candidates` 为空，不能参加 class 匹配。

### 4.3 三类互相独立的状态

不能再用旧 `State.INACTIVE/TENTATIVE/CONFIRMED/LOST` 同时表达身份可信度、空间轨迹和本帧来源。每个 class 必须分别保存以下三个维度。

身份状态 `IdentityState`：

```text
UNSEEN       从未由雷达猜测、可靠彩色相机 + 雷达或新鲜 0x02BB 建立身份
GUESSED      身份由纯雷达或灰色阵营偏置建立，尚未被可靠彩色证据确认
CONFIRMED    至少一次被可靠彩色相机 + 雷达联合确认，或本帧由新鲜且身份映射确定的 0x02BB 确认
```

空间状态 `SpatialState`：

```text
UNINITIALIZED  二维 KF 尚未初始化
TRACKING       本帧获得了真实位置测量
COASTING       本帧无真实测量，正在使用二维 KF 短期外推
LOST           连续丢失达到 radar_guess_after_frames
```

本帧位置来源 `FrameSource`：

```text
INVALID                 本帧没有可输出位置
CAMERA_LIDAR             彩色相机身份 + 雷达位置
GRAY_CAMERA_LIDAR        灰色装甲板维持已有 CONFIRMED 身份 + 雷达位置
ALLY_LIDAR               新鲜己方位置先验 + 门限内雷达点确认/剔除；输出位置仍取己方先验
ALLY_FEEDBACK            新鲜 0x02BB 己方位置，无对应雷达点
LIDAR_HISTORY            历史二维轨迹 + 雷达位置
GRAY_RADAR_GUESSED       灰色 digit + 雷达位置，但阵营靠敌方偏置或模糊代价决定
RADAR_GUESSED            纯剩余雷达点随机猜测
KF_PREDICTED             二维 KF 外推位置
FIXED_GUESSED            `guess_pts.yaml` 固定候选点
```

### 4.4 每个 class 轨迹必须保存的字段

```text
class_id
name
identity_state
spatial_state
frame_source
output_valid
output_position_2d_m
kalman_filter_2d
kalman_initialized
associated_bot_id          # 没有关联时为 None
last_car_box               # 当前/最后可靠相机框
last_measurement_position
last_camera_frame
last_gray_camera_frame
last_lidar_frame
last_update_monotonic
lidar_miss_count
continuous_loss_count
unseen_frame_count
fixed_guess_point
preempted_this_frame
```

`output_valid` 是输出层唯一应检查的有效性字段。旧 `is_active` 和 `is_start_guess` 不再承担输出判断；若为了 UI 兼容暂时保留，必须由新状态派生，不能反向驱动状态机。

### 4.5 输入快照与 tracker 接口

雷达客户端必须提供以下不可变快照：

```text
LidarSnapshot:
    points_xy_m: Tuple[Tuple[float, float], ...]
    ros_stamp_sec: Optional[float]
    received_monotonic: Optional[float]
```

`LidarClusterSubscriber._callback()` 在替换点列表的同一个锁内记录 `time.monotonic()`。`LidarClient.get_snapshot()` 在同一个锁内复制点元组、ROS 时间戳和本机接收时间。新鲜度一律使用 `received_monotonic`，不使用可能与主机不在同一时钟域的 ROS stamp。

裁判系统通信管理器必须提供：

```text
AllyPositionSnapshot:
    positions_m:
        hero: Optional[Tuple[float, float]]
        engineer: Optional[Tuple[float, float]]
        standard_3: Optional[Tuple[float, float]]
        standard_4: Optional[Tuple[float, float]]
        sentry: Optional[Tuple[float, float]]
    received_monotonic: Optional[float]
```

`RefereeCommManager.get_sentry_self_position_snapshot()` 必须在线程锁内返回坐标副本和 `self_02bb_last_update_time`。不能把可变 ctypes 消息对象直接交给 tracker。

`MainEventLoop` 调用接口统一为：

```text
tracks, detect_vis_img, track_vis_img = tracker.track(
    frame,
    lidar_snapshot=lidar_snapshot,
    ally_snapshot=ally_snapshot,
    faction=faction,
)
```

tracker 自身负责快照新鲜度、坐标过滤和当前阵营镜像。`MainEventLoop` 不提前把传入 tracker 的雷达点转换为厘米。但是要注意无论如何最后单位必须统一。

---

## 5. `config/params.yaml` 的最终配置与默认值

实现时必须把当前的 `digit_detector: None` 替换为以下结构，并在 `tracker`、`lidar_fusion` 中加入完整配置。代码必须从下列路径读取；除明确写出的兜底值外，不得使用类常量覆盖 YAML。

```yaml
car_detector:
  weights_path: "weights/car/best.engine"
  img_size: 1280
  conf_thres: 0.3
  iou_thres: 0.5
  max_det: 10
  tracker_config_path: "config/botsort.yaml"
  verbose: false

armor_detector:
  class_names: ["B1", "B2", "B3", "B4", "B7", "R1", "R2", "R3", "R4", "R7", "G1", "G2", "G3", "G4", "G7"]
  use_enhanced_detector: true
  armor_detector_type: "two_step"
  weights_path: "weights/armor/armor_best.engine"
  img_size: 192
  conf_thres: 0.4
  iou_thres: 0.4
  max_det: 4
  verbose: false
  digit_model_type: "mobilenet"
  digit_weights_path: "weights/MobileNetv2.2.pth"

digit_detector:
  digit_confidence_weight: 0.70
  armor_confidence_weight: 0.30
  min_digit_confidence: 0.55
  min_combined_confidence: 0.55
  colored_anchor_min_combined_confidence: 0.65
  gray_anchor_min_combined_confidence: 0.65
  ignored_digit_index: 5

ray_projection:
  xy_maps_default_enabled: false
  map_dtype: "float32"
  generation_chunk_rows: 64
  lookup_mode: "nearest"
  invalid_value: "nan"

tracker:
  type: "CascadeMatchTracker"
  point_guesser_config_path: "config/guess_pts.yaml"
  cascade:
    unknown_faction_fallback: "red"
    random_seed: 2026

    bot_history:
      queue_size: 30
      matching_window: 10
      decay_tau: 0.5
      association_ttl_frames: 30

    colored_triplet_match:
      max_ray_lidar_distance_m: 1.5
      semantic_cost_weight: 2.0
      ray_distance_cost_weight: 3.0

    gray_match:
      max_ray_lidar_distance_m: 1.5
      semantic_cost_weight: 1.5
      track_distance_cost_weight: 2.0
      ray_distance_cost_weight: 3.0
      unseen_track_cost: 0.5
      side_cost_margin: 0.15
      enemy_tie_bias: 0.05

    history_lidar_match:
      max_distance_m: 2.0
      unmatched_cost: 100.0
      invalid_edge_cost: 1000000.0

    ally_fusion:
      feedback_timeout_sec: 0.15
      max_lidar_match_distance_m: 1.5
      feedback_r_std_m: 0.0
      age_r_std_per_sec: 0.0
      update_kf_without_lidar: true

    position_kalman:
      default_dt_sec: 0.033333
      min_dt_sec: 0.001
      max_dt_sec: 0.20
      process_q_std: 2.0
      lidar_r_std_m: 0.20
      initial_position_std_m: 0.25
      initial_velocity_std_mps: 2.0
      hard_reset_distance_m: 2.5

    loss:
      radar_guess_after_frames: 10
      fixed_guess_after_frames: 30

    guessing:
      enemy_only: true
      role_priority: ["hero", "standard_3", "standard_4", "sentry", "engineer"]

lidar_fusion:
  enabled: true
  topic: "/livox/lidar_cluster"
  queue_size: 10
  max_snapshot_age_sec: 0.20
  mirror_for_blue: true
  field_length: 28.0
  field_width: 15.0
```

旧配置的处理规则：

- `lidar_fusion.max_match_distance` 被新的三个独立门限取代，不再读取。
- `lidar_fusion.camera_r_std` 不再读取，因为相机射线不再作为二维 KF 测量。
- `lidar_fusion.lidar_r_std` 被 `tracker.cascade.position_kalman.lidar_r_std_m` 取代。
- 旧 `CascadeMatchTracker.W1..W5`、`COST_THRESHOLD`、`INACTIVE_COST_THRESHOLD`、`HIT_COUNT_THRESHOLD` 和 `MISS_COUNT_THRESHOLD` 删除，所有新参数从 YAML 读取。

### 5.1 `config/guess_pts.yaml`

固定猜点继续从 `tracker.point_guesser_config_path` 指向的文件读取。保留当前候选点和以下默认评分参数：

```yaml
d_factor: 0.01
cos_factor: 0.003
```

固定点评分继续使用：

```text
d_score = exp(-distance * d_factor)
score = cos_factor * cos_similarity + (1 - cos_factor) * d_score
```

必须用 `track.name` 直接读取同名候选点，禁止继续使用当前反置的 `name_id_convert`。蓝方镜像规则继续以场地中心 `(14, 7.5)` 进行。

### 5.2 `config/botsort.yaml`

BoT-SORT 继续由 `car_detector.tracker_config_path` 读取，第一版不得改变当前默认值：

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: true
gmc_method: null
proximity_thresh: 0.5
appearance_thresh: 0.5
with_reid: true
model: auto
```

BoT-SORT 的职责只是在图像域给当前真实检测框提供 `bot_id`；它不直接创建 class 身份和世界二维位置。

### 5.3 射线投影参数

正式比赛射线投影继续从 `config/params.yaml` 的以下路径读取：

```text
transform.K
transform.dist_coeffs
transform.R
transform.t
field.mesh_path
```

当前第一版默认值为：

```yaml
transform:
  K: [[2576.81837, 0.000000, 1515.37690],
      [0.000000, 2571.61920, 1021.32044],
      [0.000000, 0.000000, 1.000000]]
  dist_coeffs: [-0.08747204, 0.16229478, 0.0002525, 0.00149612, -0.05979831]
  R: [[0.05794831, -0.99802970, 0.02405640],
      [-0.27552369, -0.03914937, -0.96049676],
      [0.95954609, 0.04903105, -0.27724947]]
  t: [[5.96406267], [3.65504223], [1.39573549]]
field:
  mesh_path: "field/RMUC2026.ply"
```

录像只使用各自 `camera/calibration_snapshot.yaml` 固化标定信息。其
`intrinsics` 节点记录 `image_width/image_height`、`K` 和 `dist_coeffs`，
`extrinsics` 节点记录 R/t，不再维护第二份相机元数据文件。

UI 完成外参标定后，`MainEventLoop.set_calibrated_param()` 传入的 `R/T` 覆盖运行时
射线转换器。正式比赛的 `SolvePnp`、逐框投影及 XY Map 内参使用当前
`params.yaml::transform.K/dist_coeffs`；replay 必须使用录像目录中
`calibration_snapshot.yaml::intrinsics.K/dist_coeffs` 覆盖回放运行参数。
快照中的 `image_width/image_height` 仅供人工核验，不得参与 K 缩放。replay
相机目录缺少 `calibration_snapshot.yaml`，或其中内参/外参不完整时，必须预检
失败，不允许回退 `params.yaml` 或其他全局文件。

### 5.4 `xy maps` 配置和 UI 行为

唯一实现方案为“当前原始输入帧分辨率二维 XY Map”。不使用 YOLO 的 `1280×1280` 推理分辨率，也不生成压缩映射。

配置读取规则：

```text
ray_projection.xy_maps_default_enabled=false
    UI 启动时 `xy maps` 默认不勾选，保持旧逐框投影逻辑。

ray_projection.map_dtype=float32
    XY Map 两个坐标的固定数据类型；不允许 float16。

ray_projection.generation_chunk_rows=64
    每次批量生成 64 行射线，控制临时 Tensor 内存。

ray_projection.lookup_mode=nearest
    对框底边中心四舍五入后查最近整数像素。

ray_projection.invalid_value=nan
    无 mesh 交点位置写入 NaN，并令 valid_mask=false。

ray_projection.consistency_check.enabled=true
    每次标定生成新 Map 后执行真实 car 一致性检查。

ray_projection.consistency_check.sample_count=10
    使用最先出现且逐框投影有效的 10 个真实 car 观测。

ray_projection.consistency_check.max_error_m=0.10
    查表二维点与同像素逐框投影二维点的最大允许误差。
```

UI 和运行时行为严格如下：

1. 在标定控制区域新增 `QCheckBox("xy maps")` 和只读状态文本；除生成状态外还显示一致性检查进度、通过或失败。
2. 默认不勾选；未勾选时始终使用旧的逐 box `PixelToWorld.pixel_to_world()`。
3. 尚未完成有效外参标定时勾选：状态为 `等待标定`，继续使用旧逐框投影，不使用 `params.yaml` 的备用 R/t 生成表。
4. 外参标定成功时若已经勾选，立即提交后台生成任务。
5. 已完成标定后再勾选，立即提交后台生成任务。
6. 生成任务每帧从当前输入图像 `img.shape` 获取原始 `width/height`，读取操作为 `O(1)`，不产生图像复制。
7. XY Map 和逐框投影必须直接共用同一份 `transform.K/dist_coeffs`，禁止根据任何 YAML 图像尺寸再次缩放 K。
8. 按 64 行分块批量执行畸变校正、相机射线生成和 Open3D `cast_rays`；禁止逐像素调用 `pixel_to_world()`。
9. 生成期间继续使用新的标定 R/t 执行旧逐框投影，不能继续使用旧标定生成的 XY Map。
10. 完整生成 `xy_map + valid_mask` 后，在锁内一次性替换活动映射并开始收集 10 个真实 car 验证样本；验证期间正式定位仍使用逐框投影。
11. 查表像素越界或 `valid_mask=false` 时直接返回无效投影；已就绪模式不得对单个无效像素回退到实时投影。
12. 取消勾选后下一帧立即切回逐框投影，状态为 `未启用`；内存中的同签名映射允许保留。
13. 再次勾选时，如果缓存映射的完整签名与当前标定完全一致，直接切换为 `已就绪`；否则重新生成。
14. 映射签名必须包含：当前帧宽高、K、dist_coeffs、R、t、场地 mesh 文件路径及 mesh 文件修改时间。
15. 任何新标定、手动输入新外参、内参变化、分辨率变化或 mesh 变化都立即使旧映射失效；勾选状态下自动重建。
16. 生成失败时记录完整异常、状态改为 `生成失败`、自动取消勾选，并回到旧逐框投影，主循环不得退出。
17. 10 个样本全部满足二维误差不超过 `0.10 m` 时，在 UI 信息窗口输出成功信息，从下一次 car 投影开始只查表。
18. 任一样本查表无效或最大误差超过阈值时，在 UI 信息窗口输出 warning、自动取消勾选并继续使用逐框投影；同一标定不得直接重新启用失败 Map。
19. 窗口关闭和主程序停止时必须终止/回收生成线程；过期生成任务即使稍后完成也不能覆盖更新标定后的映射。

`MainEventLoop` 必须提供：

```text
set_xy_maps_enabled(enabled: bool)
get_xy_maps_status() -> str
```

`set_calibrated_param(R,T)` 更新外参后必须通知 XY Map 管理器失效，并在勾选状态下立即重建。

### 5.5 性能、可视化和裁判输出兼容参数

下列现有配置不属于匹配代价，但重构时必须继续读取并保持当前默认值：

```yaml
visualization:
  enabled: true
  single_view: true
  separate_detection_view: false
  update_hz: 30.0
  output_max_width: 1280
  output_max_height: 960
  box_thickness: 5
  label_font_scale: 1.5
  label_font_thickness: 2
  label_offset_y: 10

debug:
  timing_debug_enabled: true
  timing_ema_alpha: 0.2
  log_inference_fps: false
  performance_metrics:
    enabled: true
    output_path: "logs/performance/runtime_metrics.jsonl"
    flush_interval_sec: 1.0
    gpu_sample_interval_sec: 0.5
    gpu_device_index: 0
    queue_size: 512

referee:
  info_wave_position_timeout_sec: 3.0
```

0x0305 发送频率继续由 `RefereeCommManager.RADAR2CLIENT_TX_HZ=5.0` 控制。信息波 0x0A01 对方坐标在 3.0 秒内新鲜时仍覆盖 tracker 对方坐标；这与 tracker 内部 0x02BB 己方先验 0.15 秒门限是两套独立超时，不能共用。

### 5.6 配置加载校验

tracker 初始化时一次性校验配置，失败就用包含完整 YAML 路径的 `ValueError` 终止初始化，禁止运行中静默回退：

- 所有 confidence、IoU 和 probability 参数必须位于 `[0, 1]`。
- `digit_confidence_weight + armor_confidence_weight` 必须在 `1.0 ± 1e-6` 内。
- 所有距离门限、标准差、超时和 cost weight 必须大于等于 0；三个匹配距离门限必须严格大于 0。
- `min_dt_sec > 0`、`max_dt_sec >= min_dt_sec`、`default_dt_sec` 必须位于二者之间。
- `fixed_guess_after_frames >= radar_guess_after_frames >= 1`。
- `bot_history.matching_window <= bot_history.queue_size`，二者都至少为 1。
- `enemy_tie_bias < side_cost_margin`。
- `ray_projection.map_dtype` 必须严格等于 `float32`。
- `ray_projection.lookup_mode` 必须严格等于 `nearest`。
- `ray_projection.generation_chunk_rows >= 1`。
- `ray_projection.consistency_check.enabled` 必须为布尔值。
- `ray_projection.consistency_check.sample_count >= 1`。
- `ray_projection.consistency_check.max_error_m > 0`。
- `armor_detector.class_names` 必须与第 2 节 15 个名称完全一致。
- `guessing.enemy_only` 若被配置成 false，直接报错；第一版禁止己方随机或固定猜测。

---

## 6. 检测与语义观测的确定流程

检测链路的图像通道契约固定如下：海康相机、视频模拟相机和 ROS 相机在发布给
主程序前都统一为 RGB，`PredictorWithTracker` 和 `Predictor` 的输入也固定为 RGB。
Ultralytics 的 NumPy 接口会把输入解释为 BGR，并在预处理时执行 BGR→RGB，因此
两个 Predictor 必须只在调用 `model.track()` / `model.predict()` 的边界传入
`image[..., ::-1]` 的 BGR 共享内存视图。禁止直接把项目内部 RGB 数组传给
Ultralytics，也禁止为此新增逐帧全分辨率 `cvtColor` 或 `copy`。

### 6.1 整车检测

1. 继续调用 `PredictorWithTracker.predict()`。
2. 继续使用 `model.track(..., persist=True, tracker=config/botsort.yaml)`。
3. 只接收本帧 `results.boxes` 中真实存在的框；BoT-SORT 没有在本帧输出的历史预测框不能进入 tracker。
4. `boxes.xyxyn/conf/cls/id` 继续拼接后执行一次 `.detach().cpu().numpy()`，禁止恢复为逐框同步。
5. `boxes.id is None` 的框不进入身份匹配，因为没有稳定 `bot_id`；可保留在单纯检测可视化中。
6. 还原到原图像素后，将整车框裁剪到 `[0,width] x [0,height]`；`x2 <= x1` 或 `y2 <= y1` 的框直接丢弃。

### 6.2 多装甲板检测与选择

1. 每个整车 ROI 最多保留 `armor_detector.max_det=4` 个装甲板。
2. 装甲检测器必须把 `armor_confidence` 保留到最终输出，不能像当前两阶段实现一样丢弃 `yolo_conf`、只返回 digit 置信度。
3. digit 分类继续对所有整车 ROI 中的全部装甲板一次性批量推理。
4. digit 类别和概率继续一次性搬到 CPU，禁止逐装甲板 `.item()`。
5. 本地 `weights/armor/armor_best.pt` 的模型元数据为 `{0: dead, 1: red, 2: blue}`，因此两阶段装甲颜色检测器的原始颜色索引固定解释为 `0=GRAY`、`1=RED`、`2=BLUE`；其他索引视为无效颜色。合成后的统一身份仍为 `0~4=BLUE`、`5~9=RED`、`10~14=GRAY`。
6. 装甲框裁剪到对应整车 ROI 边界；空框不送入 digit 分类器。
7. 对每个装甲板计算 `combined_confidence`。
8. 以下任一条件成立时，该装甲板不能生成语义候选：
   - `digit_index == digit_detector.ignored_digit_index`；
   - `digit_confidence < digit_detector.min_digit_confidence`；
   - `combined_confidence < digit_detector.min_combined_confidence`。
9. 每个整车框从剩余装甲板中选 `combined_confidence` 最大者作为 `selected_armor`。
10. 完全相同时依次按更高 `digit_confidence`、更高 `armor_confidence`、更小装甲框左上角 `(x1,y1)` 决定，保证回放可复现。
11. 没有合格装甲板时，该整车框的 class 为 `None`，不参加 class 匹配。

### 6.3 BotID 语义历史

当前仍保留 BoT-SORT `bot_id` 的短期语义记录和 30 帧 TTL，以兼容现有数据结构及关联记录；任何彩色、灰色或模糊匹配代价均不读取该历史，它不参与候选门控、优先级或最终匹配选择。后续删除 BoT-SORT 时再整体移除这些兼容状态。

---

## 7. 每帧处理总流程

每帧必须严格执行以下顺序。所有“已使用”集合均只在当前帧有效：

```text
used_class_ids
used_car_indices
used_lidar_indices
```

### 阶段 0：读取并冻结输入快照

1. 获取当前相机帧和时间戳。
2. 获取本帧开始时最新雷达点快照及其本机接收单调时间。
3. 获取最新 0x02BB 己方位置快照及其本机接收单调时间。
4. 复制必要的小型列表后冻结为本帧输入；本帧匹配期间不读取会被其他线程修改的共享对象。
5. 过滤无效、过期和越界雷达点。
6. 根据当前阵营统一处理相机点和雷达点镜像。

### 阶段 1：二维 KF 统一预测

1. 计算 `real_dt = clamp(now_monotonic - last_frame_monotonic, min_dt_sec, max_dt_sec)`。
2. 首帧使用 `default_dt_sec`。
3. 所有已初始化 class 的二维 KF 恰好执行一次 `predict(real_dt)`。
4. 此阶段只产生预测状态，不直接决定最终来源。
5. 未初始化的 KF 不执行 predict。

### 阶段 2：整车、装甲板、digit、语义候选和当前射线

按照第 6 节生成 `CarObservation`。只对存在可靠 `selected_armor` 的当前整车框生成 class 候选，并根据 `xy maps` UI 状态使用逐框投影或 XY Map 查表得到底边中心二维投影点。

当前框没有雷达匹配时，射线结果在本帧结束后丢弃，不能作为位置测量保留。

### 阶段 3：新鲜己方哨兵位置优先匹配并剔除雷达点

本阶段在所有相机—雷达和敌方历史—雷达匹配之前执行。0x02BB 的 `age_sec <= 0.15` 时，所有有效己方坐标按身份和位置绝对准确处理；超过 0.15 秒时完全跳过本阶段。

候选行是 0x02BB 中有效的己方英雄、工程、3 号、4 号和哨兵，候选列是本帧全部有效地面雷达点。使用增广匈牙利匹配：

- 己方坐标到雷达点的距离不超过 `ally_fusion.max_lidar_match_distance_m=1.5` 才建立合法边。
- 合法边代价为 `distance / 1.5`。
- 超门限边使用 `history_lidar_match.invalid_edge_cost=1000000.0`。
- 每个己方行配置独立 dummy 列，代价为 `history_lidar_match.unmatched_cost=100.0`。
- 优化目标先最大化合法匹配数量，再最小化总距离。

必须进行一对一匹配，禁止简单删除每个己方坐标半径内的全部雷达点。一个己方 class 最多锁定一个雷达点，一个雷达点最多被一个己方 class 锁定；己方没有匹配雷达时不删除任何雷达点。

己方成功匹配雷达后：

- 将对应 class 和雷达点加入 `used_class_ids`、`used_lidar_indices`，后续相机和敌方匹配不可再使用。
- `IdentityState=CONFIRMED`。
- `SpatialState=TRACKING`。
- `FrameSource=ALLY_LIDAR`。
- 二维位置和二维 KF 测量使用绝对准确的 0x02BB 坐标；对应雷达点只用于确认并剔除，不替换己方哨兵坐标。

己方没有匹配雷达时：

- 不消耗任何雷达点。
- 该己方 class 仍加入 `used_class_ids`，因为新鲜 0x02BB 已经确定其身份和位置，无需再由相机覆盖。
- `IdentityState=CONFIRMED`、`SpatialState=TRACKING`、`FrameSource=ALLY_FEEDBACK`。
- 使用 0x02BB 坐标更新或初始化己方二维 KF。

新鲜 0x02BB 本帧处理完毕后，具有有效 0x02BB 坐标的己方 class 不再进入后续 class 匹配；对应坐标无效或缺失的己方 class 仍允许由当前可靠彩色相机 + 雷达识别。后续阶段主要在剩余雷达点上建立敌方身份。

### 阶段 4：彩色 `class-car-lidar` 联合匹配

彩色装甲板只有在：

```text
combined_confidence >= colored_anchor_min_combined_confidence
```

时才能进入本阶段。

为每个合法 `class-car-lidar` 三元组建立候选，必须同时满足：

1. car 已按第 6.2 节从多个装甲板中选出唯一的 `selected_armor`，且该彩色 `selected_armor` 唯一指向该 class；同一 car 内其他 digit/color 不一致的装甲板不会导致整车废弃，也不再各自产生额外 class 候选。
2. car 有有效当前射线落点。
3. 雷达快照有效且该雷达点未使用。
4. 射线二维点与雷达点的欧氏距离不超过 1.5 米。
5. class、car、雷达点均未被本阶段其他结果占用。

三元组代价固定为：

```text
semantic_cost = 1 - combined_confidence
ray_cost = ray_lidar_distance / max_ray_lidar_distance_m

total_cost =
    semantic_cost_weight * semantic_cost
  + ray_distance_cost_weight * ray_cost
```

本阶段不能先固定 class—car，再独立做 car—lidar，因为这会在重复 digit 时丢失有效三元组。实现方式固定为：枚举全部合法三元组，使用小规模精确搜索/分支限界求解互斥集合，优化目标按以下字典序执行：

1. 最大化匹配三元组数量。
2. 在数量相同时最小化总代价。
3. 仍相同时按 `(class_id, bot_id, lidar_index)` 字典序选择，保证确定性。

最多只有 10 个 class、10 个整车框和少量雷达点，因此精确求解不会成为性能瓶颈。

匹配成功后：

- 位置测量严格使用对应雷达点。
- `IdentityState` 设为 `CONFIRMED`。
- `SpatialState` 设为 `TRACKING`。
- `FrameSource` 设为 `CAMERA_LIDAR`。
- 计数器由最终提交阶段依据 `CAMERA_LIDAR` 统一更新。
- 更新 `associated_bot_id`、`last_car_box`、相机/雷达帧号。
- 锁定 class、car、雷达点。
- 若原身份为 `UNSEEN` 或 `GUESSED`，二维 KF 在雷达点硬重置。
- 若原身份为 `CONFIRMED`，预测位置到雷达点距离小于 2.5 米时正常更新；大于或等于 2.5 米时在雷达点硬重置。
- 相机射线本身永远不进入二维 KF。

### 阶段 5：灰色装甲板维持已确认身份

只有满足以下条件的灰色观测进入本阶段：

1. `combined_confidence >= gray_anchor_min_combined_confidence`。
2. 候选 class 的 `IdentityState == CONFIRMED`。
3. class 与灰色 digit 对应。
4. car 有有效当前射线。
5. 射线到雷达点距离不超过 1.5 米。

灰色维持三元组代价为：

```text
semantic_cost = 1 - combined_confidence
track_cost = distance(predicted_track_position, lidar_point) / history_lidar_match.max_distance_m
ray_cost = distance(ray_position, lidar_point) / gray_match.max_ray_lidar_distance_m

total_cost =
    semantic_cost_weight * semantic_cost
  + track_distance_cost_weight * track_cost
  + ray_distance_cost_weight * ray_cost
```

若预测轨迹到雷达点距离超过 `history_lidar_match.max_distance_m`，候选边直接删除。

本阶段只接受已有 `CONFIRMED` 空间轨迹，因此 `track_cost` 必须存在，否则该候选无效。

本阶段同样按“最大匹配数、最小总代价、确定性字典序”求全局互斥三元组。两个同 digit 灰色框同时出现时，必须一起求解，不能逐框贪心。

成功后：

- 身份保持 `CONFIRMED`，灰色证据不能新建 `CONFIRMED`。
- `SpatialState=TRACKING`。
- `FrameSource=GRAY_CAMERA_LIDAR`。
- 二维 KF 使用雷达点正常更新。
- 计数器由最终提交阶段依据 `GRAY_CAMERA_LIDAR` 统一更新。
- 更新 `last_gray_camera_frame`，但不更新代表“最后一次彩色确认”的 `last_camera_frame`。

### 阶段 6：敌方历史轨迹与剩余雷达点匹配

本阶段使用增广匈牙利匹配。候选行只有未在前面阶段匹配、二维 KF 已初始化的敌方 class；新鲜己方 class 已经在阶段 3 处理并锁定，过期己方 class 禁止进入本阶段。

敌方行使用阶段 1 的 KF 预测位置。

候选边门限：

- 敌方历史轨迹到雷达点不超过 `history_lidar_match.max_distance_m=2.0`。
- 超门限边在求解前设置为 `invalid_edge_cost=1000000.0`。

有效边代价为对应距离除以该行门限。为每一行增加独立 dummy 未匹配列，dummy 代价固定为 `unmatched_cost=100.0`。必须有足够 dummy 列让所有行都能选择未匹配。求解后只接收真实雷达列且边合法的结果。这个大 dummy 代价保证先最大化合法匹配数量，再最小化距离；禁止先对全矩阵求普通 Hungarian、再事后删除远距离匹配。

敌方成功匹配：

- `FrameSource=LIDAR_HISTORY`。
- `SpatialState=TRACKING`。
- 二维 KF 使用雷达点和 `lidar_r_std_m` 更新。
- `GUESSED` 保持 `GUESSED`；`CONFIRMED` 保持 `CONFIRMED`；历史雷达匹配不能把身份升级为 `CONFIRMED`。
- 计数器由最终提交阶段依据 `LIDAR_HISTORY` 统一更新。

### 阶段 7：灰色模糊身份 + 剩余雷达点匹配

本阶段只处理前面未使用的灰色 car、未使用雷达点以及以下 class：

- `IdentityState` 为 `UNSEEN` 或 `GUESSED`；
- class 尚未被本帧高优先级阶段锁定；
- 灰色 digit 与 class 兵种一致。

`CONFIRMED` class 只能在阶段 5 由灰色维持，不能在本阶段被模糊灰色结果重新猜测。

本阶段中，`UNSEEN` class 没有历史空间位置，因此使用无信息常量 `track_cost=gray_match.unseen_track_cost=0.5`；`GUESSED` class 使用已初始化二维 KF 计算 `track_cost`，距离超过 2.0 米则删除候选。

候选仍必须有有效射线，且射线—雷达距离不超过 1.5 米。先按第 5 阶段公式计算不含阵营偏置的基础代价。对于同一灰色 car 的敌我两个候选，如果两者都未被占用且：

```text
abs(enemy_base_cost - ally_base_cost) <= gray_match.side_cost_margin
```

则执行：

```text
enemy_total_cost = enemy_base_cost - gray_match.enemy_tie_bias
ally_total_cost = ally_base_cost
```

默认 `side_cost_margin=0.15`，`enemy_tie_bias=0.05`。敌方偏置只能打破接近平局，不能覆盖明显的轨迹历史差异。

本阶段也必须全局求解所有灰色 `class-car-lidar` 三元组，并允许全部对象未匹配：

- 单个灰色框、同 digit 红蓝双方均空缺、代价接近时，优先分给敌方。
- 两个灰色框出现时，由全局代价和 class 唯一性决定最优分配。
- 即使有两个框，也不能强制“一敌一友”；没有合法第二条边时第二个框保持 `None`。

匹配成功后：

- `IdentityState=GUESSED`，不能设为 `CONFIRMED`。
- `FrameSource=GRAY_RADAR_GUESSED`。
- `SpatialState=TRACKING`。
- 在雷达点硬重置二维 KF，位置速度重新初始化。
- 记录关联 bot 和当前相机框。

### 阶段 8：纯剩余雷达点随机猜测

本阶段只允许敌方 class，且 `guessing.enemy_only` 必须保持为 true。候选 class 分两类：

1. `IdentityState=UNSEEN`：从第一帧起立即有资格猜剩余雷达点。
2. 已经建立过轨迹的 class：先计算 `pending_loss_count = continuous_loss_count + 1`；只有 `pending_loss_count >= radar_guess_after_frames` 才有资格。这里的加 1 表示阶段 3 至阶段 7 已经确认本帧仍无真实测量，保证第 10 个连续丢失帧就能进入猜测，不产生一帧偏移。

class 选择顺序固定为：

1. 所有 `UNSEEN` class 优先于所有已见过 class。
2. 已见过 class 按 `continuous_loss_count` 从大到小。
3. 再按兵种顺序：英雄 > 3 号步兵 > 4 号步兵 > 哨兵 > 工程。
这样可以唯一的决定class的选择顺序。

使用 `numpy.random.Generator(numpy.random.PCG64(2026))`。每帧只对剩余雷达点索引做一次随机排列，然后依照上述 class 顺序一一分配。禁止调用全局 `random` 或未设种子的 NumPy 随机接口。

成功后：

- `IdentityState=GUESSED`，包括原先已经 `CONFIRMED`、但持续丢失后重新随机猜测的 class。
- `FrameSource=RADAR_GUESSED`。
- `SpatialState=TRACKING`。
- 在雷达点硬重置二维 KF。
- 计数器由最终提交阶段依据 `RADAR_GUESSED` 统一更新。
- 不产生或更新相机框或 bot 关联。

若一个 `GUESSED` class 的有效历史边指向的雷达点在本帧被更高优先级 `CAMERA_LIDAR` 分给其他 class，则该猜测 class 标记 `preempted_this_frame=true`，本帧只能进入 KF 外推，禁止立即在阶段 8 再抢另一个雷达点。这避免同一帧发生身份连锁漂移。

### 阶段 9：无测量 class 的 KF 外推与固定点猜测

对本帧仍未得到真实位置测量的敌方 class：

1. 对已见过 class 使用阶段 8 计算的 `pending_loss_count`；对 `UNSEEN` class 使用 `pending_unseen_count = unseen_frame_count + 1`。所有计数器只在最终来源确定后一次性提交。
2. 如果二维 KF 已初始化且 `pending_loss_count < radar_guess_after_frames`：
   - `SpatialState=COASTING`；
   - `FrameSource=KF_PREDICTED`；
   - 输出阶段 1 的预测位置；
   - `output_valid=true`。
3. 如果二维 KF 已初始化且 `pending_loss_count >= radar_guess_after_frames`、但 `pending_loss_count < fixed_guess_after_frames`：
   - `SpatialState=LOST`；
   - 仍使用 `FrameSource=KF_PREDICTED`；
   - 继续输出预测位置；
   - 等待后续帧出现剩余雷达点进行 `RADAR_GUESSED`。
4. 如果 `pending_loss_count >= fixed_guess_after_frames` 且本帧仍无可用雷达点：
   - 通过 `PointGuesser` 选择固定候选点；
   - `SpatialState=LOST`；
   - `FrameSource=FIXED_GUESSED`；
   - `output_valid=true`；
   - 绝不更新或重置二维 KF；
   - 身份状态保持不变；
   - 后续每帧仍保持 loss，直到真实雷达或相机 + 雷达重新建立空间轨迹。
5. `UNSEEN` class 在没有雷达、且 `pending_unseen_count < fixed_guess_after_frames` 时：
   - `SpatialState=UNINITIALIZED`；
   - `FrameSource=INVALID`；
   - `output_valid=false`。
6. `UNSEEN` class 的 `pending_unseen_count >= fixed_guess_after_frames` 且仍没有雷达时，输出固定点，但身份仍保持 `UNSEEN`，二维 KF 仍未初始化。

己方 class 不执行随机雷达猜测和固定点猜测。己方 0x02BB 过期且本帧没有更高优先级真实证据时，其 `FrameSource=INVALID`。

### 阶段 10：统一提交输出

每个 class 每帧最多有一个最终来源。来源优先级固定为：

```text
ALLY_LIDAR
> ALLY_FEEDBACK
> CAMERA_LIDAR
> GRAY_CAMERA_LIDAR
> LIDAR_HISTORY
> GRAY_RADAR_GUESSED
> RADAR_GUESSED
> KF_PREDICTED
> FIXED_GUESSED
> INVALID
```

前面阶段一旦锁定结果，后面阶段不能覆盖。

---

## 8. 二维位置卡尔曼的确定行为

二维 KF 状态保持 `[x, y, vx, vy]`，使用匀速模型。实现从 `tracker.cascade.position_kalman` 读取全部参数。

### 8.1 正常更新来源

只有以下来源可以作为位置测量：

```text
CAMERA_LIDAR       使用雷达点
GRAY_CAMERA_LIDAR  使用雷达点
ALLY_LIDAR         使用新鲜 0x02BB 坐标；雷达点只用于匹配和剔除
LIDAR_HISTORY      使用雷达点
GRAY_RADAR_GUESSED 使用雷达点并硬重置
RADAR_GUESSED      使用雷达点并硬重置
ALLY_FEEDBACK      使用新鲜 0x02BB 坐标
```

以下内容绝不能更新二维 KF：

```text
相机射线落点
当前框底边中心换算出的世界点
KF_PREDICTED 输出
FIXED_GUESSED 输出
过期的 0x02BB 坐标
```

### 8.2 硬重置

硬重置接口必须在“新测量点”处设置；该点通常是雷达点，`ALLY_LIDAR/ALLY_FEEDBACK` 时则是新鲜 0x02BB 坐标：

```text
x, y = 新测量点
vx, vy = 0
P_position = initial_position_std_m^2
P_velocity = initial_velocity_std_mps^2
交叉协方差 = 0
```

必须硬重置的情况：

- 己方二维 KF 未初始化时首次获得 `ALLY_LIDAR` 或 `ALLY_FEEDBACK`，在新鲜 0x02BB 坐标硬初始化
- `UNSEEN -> CAMERA_LIDAR`
- `GUESSED -> CAMERA_LIDAR`
- 任意 class 获得 `GRAY_RADAR_GUESSED`
- 任意 class 获得 `RADAR_GUESSED`
- `CONFIRMED -> CAMERA_LIDAR` 且预测位置与雷达位置距离大于等于 2.5 米

普通 `LIDAR_HISTORY` 只允许门限内更新，不触发大跳重置。

---

## 9. 灰色/死亡装甲板规则汇总

灰色装甲板永远只确定 digit，不直接确定阵营。所有行为固定如下：

1. 灰色观测产生同 digit 红蓝两个 class 候选。
2. 有已确认身份和雷达轨迹时，灰色观测可以维持 `CONFIRMED`，来源为 `GRAY_CAMERA_LIDAR`。
3. 灰色观测本身不能把 `UNSEEN` 或 `GUESSED` 升级为 `CONFIRMED`。
4. 单个灰色框、红蓝双方对应 class 都空缺、两边代价差不超过 0.15 时，给敌方代价减 0.05，优先猜为敌方。
5. 该敌方优先结果的身份为 `GUESSED`，来源为 `GRAY_RADAR_GUESSED`。
6. 两个灰色框同时出现时必须全局匹配，充分使用历史；禁止逐框先到先得。
7. 匹配必须允许 dummy 未匹配，不能强制第二个灰色框归属己方。
8. 后续可靠彩色 `CAMERA_LIDAR` 与灰色猜测冲突时，彩色结果无条件优先并纠正身份。

---

## 10. 己方 10-class 混合维护规则

内部始终创建 10 个 class 轨迹。0x02BB 的唯一新鲜度门限为 0.15 秒：

```text
0 <= age <= 0.15 s：坐标新鲜且按绝对准确处理
age > 0.15 s 或没有接收时间：坐标过期
```

数据新鲜时，阶段 3 必须先执行己方 5 类到全部雷达点的一对一增广匈牙利匹配，锁定匹配到的己方雷达点后，才允许相机和敌方历史使用剩余点。这一优先级高于 `CAMERA_LIDAR`。

不能简单删除己方坐标半径内的全部雷达点；每个有效己方坐标最多剔除一个通过 1.5 米距离门限的雷达点。没有门限内雷达点时，只输出 `ALLY_FEEDBACK`，不剔除任何点。

数据过期时，所有己方历史/KF 行完全退出雷达匹配，不能依靠旧预测占用点；但当前可靠彩色己方相机 + 雷达仍可正常建立 `CAMERA_LIDAR`。

己方坐标到 class 的映射按当前己方阵营和兵种表执行。0x02BB 不包含无人机；无人机继续走现有 `lidar_fly_points_cm` 和 0x0305 独立字段，不进入这 10 个地面 class。

---

## 11. 输出到 `MainEventLoop` 和裁判系统

### 12.1 tracker 输出

tracker 继续返回 10 个 class 轨迹，顺序固定为 class_id 0..9。每个轨迹至少提供：

```text
class_id
name
identity_state
spatial_state
frame_source
output_valid
output_position_2d_m
car_box
associated_bot_id
```

### 12.2 `divisions_pos`

`main_event_loop.py` 不再根据 `is_active` 或 `is_start_guess` 分支，统一执行：

```text
if track.output_valid:
    divisions_pos[class_id] = round(output_position_2d_m * 100), valid=True
else:
    divisions_pos[class_id] = (-1, -1), valid=False
```

下列来源都必须设置 `output_valid=true`：

```text
CAMERA_LIDAR
GRAY_CAMERA_LIDAR
LIDAR_HISTORY
GRAY_RADAR_GUESSED
RADAR_GUESSED
KF_PREDICTED
FIXED_GUESSED
ALLY_LIDAR
ALLY_FEEDBACK
```

只有 `INVALID` 为 false。

### 12.3 敌方 0x0305 选择

- 己方为红方时，从 class 0..4 取敌方英雄、工程、3 号、4 号、哨兵。
- 己方为蓝方时，从 class 5..9 取同样兵种。
- `RADAR_GUESSED`、`GRAY_RADAR_GUESSED`、`KF_PREDICTED`、`FIXED_GUESSED` 均按有效坐标发送。
- 坐标最终转换为厘米并钳制到 `uint16`。
- `RefereeCommManager` 当前已有的信息波 0x0A01 对方位置优先逻辑保持不变：信息波新鲜时覆盖 tracker 对方坐标；信息波超时后回退到 tracker 输出。
- 0x02BB 己方位置继续覆盖 0x0305 己方字段；tracker 雷达剔除使用严格的 0.15 秒超时，不改变串口协议结构。

---

## 12. 计数器和状态转换表

### 12.1 计数器更新

- 最终来源确定后，每个 class 的计数器恰好更新一次。
- `lidar_miss_count` 在任何真实雷达来源时清零，包括 `CAMERA_LIDAR`、`GRAY_CAMERA_LIDAR`、`ALLY_LIDAR`、`LIDAR_HISTORY` 和两种雷达猜测；其他来源均加 1。
- `continuous_loss_count` 在 `CAMERA_LIDAR`、`GRAY_CAMERA_LIDAR`、`ALLY_LIDAR`、`LIDAR_HISTORY`、`GRAY_RADAR_GUESSED`、`RADAR_GUESSED` 时清零。
- `ALLY_FEEDBACK` 对己方 class 清零空间丢失，但不消耗雷达点。
- 已见过的敌方 class 本帧没有任何真实位置测量时，将 `continuous_loss_count` 提交为阶段 8 计算的 `pending_loss_count`。
- `UNSEEN` 敌方 class 本帧没有获得雷达位置时，`unseen_frame_count += 1`；身份离开 `UNSEEN` 时将该计数清零。
- `FIXED_GUESSED` 不清零任何丢失计数。

### 12.2 主要转换

| 原身份 | 证据 | 新身份 | 空间处理 | 来源 |
|---|---|---|---|---|
| UNSEEN | 彩色相机 + 雷达 | CONFIRMED | KF 硬初始化 | CAMERA_LIDAR |
| UNSEEN | 灰色模糊 + 雷达 | GUESSED | KF 硬初始化 | GRAY_RADAR_GUESSED |
| UNSEEN | 剩余雷达随机猜测 | GUESSED | KF 硬初始化 | RADAR_GUESSED |
| UNSEEN | 无雷达达到固定猜点阈值 | UNSEEN | 不初始化 KF | FIXED_GUESSED |
| GUESSED | 彩色相机 + 雷达 | CONFIRMED | KF 硬重置 | CAMERA_LIDAR |
| GUESSED | 历史 + 雷达 | GUESSED | KF 正常更新 | LIDAR_HISTORY |
| GUESSED | 灰色模糊 + 雷达 | GUESSED | KF 硬重置 | GRAY_RADAR_GUESSED |
| CONFIRMED | 彩色相机 + 雷达，小跳变 | CONFIRMED | KF 正常更新 | CAMERA_LIDAR |
| CONFIRMED | 彩色相机 + 雷达，大跳变 | CONFIRMED | KF 硬重置 | CAMERA_LIDAR |
| CONFIRMED | 灰色 + 历史 + 雷达 | CONFIRMED | KF 正常更新 | GRAY_CAMERA_LIDAR |
| CONFIRMED | 历史 + 雷达 | CONFIRMED | KF 正常更新 | LIDAR_HISTORY |
| CONFIRMED | 持续丢失后纯雷达猜测 | GUESSED | KF 硬重置 | RADAR_GUESSED |
| 任意已初始化身份 | 短期无测量 | 身份不变 | 只 predict | KF_PREDICTED |
| 任意身份 | 达到固定猜点阈值且无雷达 | 身份不变 | 不更新 KF | FIXED_GUESSED |

---

## 13. 性能与可观测性要求

重构不得破坏已经完成的性能优化：

1. 整车 `boxes.xyxyn/conf/cls/id` 合并后仅一次 GPU→CPU。
2. 装甲检测的所有 ROI 结果合并后仅一次 GPU→CPU。
3. digit 分类的类别和完整概率矩阵合并后仅一次 GPU→CPU。
4. 不允许在 per-car、per-armor、per-class、per-match 循环内调用 CUDA Tensor 的 `.item()`、`.tolist()` 或 `.cpu()`。
5. 匹配和状态机进入 CPU 前，GPU 结果已经以一个或少数几个批量数组同步完成。
6. 可视化继续在缩小后的图像上绘制；`visualization.separate_detection_view=false` 时不得生成第二张全分辨率检测图。
7. 原始相机帧只保留引用；只有真正导出数据集时才复制。
8. 新增以下逐帧耗时阶段名：

```text
输入快照
车辆与装甲检测
观测预处理与投影
二维KF预测
己方哨兵雷达剔除
彩色三元组匹配
灰色身份维持匹配
历史雷达匈牙利匹配
灰色模糊猜测匹配
纯雷达猜测
丢失与固定点处理
输出提交
```

这些阶段继续写入现有 `RuntimeMetricsExporter` 的 `stages_ms.tracker`。不得在主处理线程直接频繁写磁盘。

XY Map 生成不计入单帧 tracker 耗时，单独记录一次性指标：`xy_map_generate_ms`、`xy_map_width`、`xy_map_height`、`xy_map_valid_ratio` 和 `xy_map_generation_error`。生成任务必须在后台线程运行。

---

## 14. 异常和边界情况

以下行为必须确定：

1. 本帧没有整车框：跳过相机阶段，继续历史—雷达、雷达猜测、KF/固定点流程。
2. 本帧没有可靠 digit：该框 class 为 `None`，不能进入身份匹配。
3. 本帧有相机 digit、但逐框射线无交点或 XY Map 查表无效：不能进入相机—雷达匹配。
4. 本帧有相机 digit 和有效投影、但没有门限内雷达：不更新二维 KF，不确认身份。
5. 雷达快照过期：按空雷达列表处理。
6. 0x02BB 年龄大于 0.15 秒：己方历史/KF 不参与雷达匹配，但当前彩色己方相机证据仍可参与。
7. 匹配代价出现 NaN/Inf：对应边直接无效，不传入求解器。
8. 一个雷达点最多属于一个 class；一个当前 car 最多属于一个 class；一个 class 每帧最多属于一个 car 和一个雷达点。
9. 重复 digit：通过全局三元组匹配选择总代价最优者，其余框保持未分配。
10. 两个 class 的合法边都很远：门限外边预先断开，允许双方未匹配。
11. 灰色第二候选不合法：保留为 `None`，不能为了满射强制分配。
12. 固定猜点列表为空：该 class `FrameSource=INVALID`、`output_valid=false`，记录一次限频错误日志，不能抛异常终止主循环。
13. tracker 保存上一帧的“有效阵营”，其中未知阵营先映射为 `unknown_faction_fallback`。如果新一帧的有效阵营与上一帧不同，则在处理该帧前清空全部 10 个 class 的二维 KF、空间状态、位置输出、固定猜点状态和丢失计数；保留 class_id 和按颜色记录的 bot 语义历史；不交换 class_id。这样红蓝镜像切换不会继续使用旧坐标系轨迹。

---

## 15. 必须采用的代码组织方式

`tracker.py` 不得继续把所有逻辑塞进单个 `track()`。至少拆出以下私有函数，并保持下列职责边界：

```text
_prepare_frame_inputs(...)
_predict_position_filters(...)
_build_car_observations(...)
_match_fresh_ally_to_lidar(...)
_commit_fresh_ally_results(...)
_build_colored_triplets(...)
_solve_disjoint_triplets(...)
_commit_colored_matches(...)
_build_gray_maintenance_triplets(...)
_commit_gray_maintenance_matches(...)
_match_enemy_history_to_lidar(...)
_commit_enemy_history_matches(...)
_match_ambiguous_gray(...)
_assign_random_radar_guesses(...)
_handle_unmatched_tracks(...)
_finalize_outputs(...)
```

所有求解函数只接收不可变候选数据并返回匹配索引，不得直接修改 track。所有 `_commit_*` 函数负责集中更新状态。

`transform/xy_map.py` 至少提供以下职责明确的接口；名称允许按现有风格微调，但职责不能合并回 UI 或 tracker：

```text
XYMapManager.set_enabled(enabled, calibration_ready, frame_size)
XYMapManager.invalidate(calibration_signature)
XYMapManager.request_build(K, dist_coeffs, R, t, mesh, frame_size)
XYMapManager.lookup(u, v) -> Optional[(x, y)]
XYMapManager.status -> 未启用 | 等待标定 | 生成中 | 已就绪 | 生成失败
```

UI 只改变启用状态和展示状态；映射生成、签名比较、后台任务代际检查、原子替换和查表全部由 `XYMapManager` 管理。

`type.py` 必须使用 `dataclass` 和 `Enum` 表达观测、匹配结果和状态，不得继续使用大量相互矛盾的布尔值。

---

## 16. 必须完成的测试场景

即使仓库当前没有正式测试框架，也必须新增可独立运行的 CPU 合成测试，至少覆盖：

1. 单个可靠彩色框 + 单个门限内雷达点，产生 `CAMERA_LIDAR/CONFIRMED`。
2. 两个 car 同时识别成同一个彩色 digit，只允许一个占用该 class。
3. 第一候选 car 无合法雷达、第二候选有合法雷达时，三元组求解选择第二候选。
4. 相机有 digit 但无雷达，二维 KF 不更新。
5. 已确认 class 无相机但历史门限内有雷达，产生 `LIDAR_HISTORY`。
6. 历史与雷达距离超过门限，候选边被断开。
7. `UNSEEN` 敌方 class 有剩余雷达时立即产生 `RADAR_GUESSED`。
8. 已见 class 短期无雷达，先输出 `KF_PREDICTED`，未提前随机猜测。
9. 连续丢失达到 10 帧后有剩余雷达，产生 `RADAR_GUESSED` 并降级为 `GUESSED`。
10. 连续丢失达到 30 帧且无雷达，产生 `FIXED_GUESSED`，KF 状态完全不变。
11. 单个灰色框、同 digit 红蓝空缺且代价接近，优先分配敌方并标记 `GUESSED`。
12. 两个灰色框和两个历史身份，全局匹配得到最小总代价。
13. 两个灰色框但只有一条合法边，第二个保持未匹配。
14. 灰色框维持已有 `CONFIRMED` 身份，但不刷新最后彩色确认帧。
15. 新鲜 0x02BB 己方位置在任何相机匹配之前参与增广匈牙利匹配，门限内一对一消耗己方雷达点，且最终位置严格等于 0x02BB 坐标。
16. 新鲜 0x02BB 己方位置没有门限内雷达点时产生 `ALLY_FEEDBACK`，不消耗任何雷达点，剩余点仍可进入敌方匹配。
17. 用可控单调时钟验证 0x02BB 年龄恰好 `0.15` 秒时仍新鲜，`0.15 + epsilon` 秒时过期；过期后己方历史/KF 完全不占雷达点，但当前可靠彩色己方相机 + 雷达仍可匹配。
18. 雷达快照超过 0.20 秒后按空雷达处理。
19. 相机锚点抢占旧 `GUESSED` 轨迹附近雷达点时，被抢占 class 当帧不能连锁重猜。
20. `RADAR_GUESSED`、`GRAY_RADAR_GUESSED`、`KF_PREDICTED`、`FIXED_GUESSED` 均在输出层标记有效并转换为厘米。
21. 同一合成输入重复运行两次，随机雷达分配和全部匹配结果完全一致。
22. `xy maps` 未勾选时不创建映射，每个有语义候选的 box 使用旧 `pixel_to_world()` 逐框投影。
23. 未完成外参标定时勾选 `xy maps`，状态为 `等待标定`，仍走旧逐框投影；标定成功后立刻进入 `生成中`。
24. 映射生成完成后，前 10 个有效真实 car 同时执行逐框投影与查表比较，并继续用逐框结果参与正式定位。
25. 10 个样本全部通过后，后续多个 box 均只查表，测试替身断言没有任何逐框 `pixel_to_world()` 调用。
26. 已就绪映射中的越界像素或 `valid_mask=false` 像素直接返回无效，不回退逐框投影。
27. 取消勾选后下一帧立即恢复逐框投影；再次勾选且签名一致时复用缓存并直接就绪。
28. 外参、内参、当前帧分辨率或 mesh 修改时间任一变化都会使旧映射失效并触发重建；旧后台任务晚完成时不能覆盖新任务结果。
29. 当前帧尺寸变化时重建对应宽高的 Map，但 K 不缩放，确保与同一帧上的逐框投影使用完全一致的射线模型。
30. 一致性检查失败时输出 UI warning、自动取消勾选，并且正式定位全程保持逐框投影。
31. 后台映射生成异常时，状态变为 `生成失败`、勾选项自动取消、下一帧回到逐框投影且主循环继续运行。

---

## 17. 完成标准

实现只有同时满足以下条件才算完成：

1. `config/params.yaml` 已包含第 5 节全部配置和默认值。
2. 当前整车框、装甲板置信度、digit 置信度、bot_id、射线点、雷达点和 0x02BB 坐标的数据接口清晰且单位一致。
3. 相机位置不再直接更新二维 KF。
4. 彩色三元组、灰色身份维持、历史雷达、灰色猜测、纯雷达猜测、KF 外推和固定点严格按顺序运行。
5. 距离过大的边在求解前断开，所有阶段允许未匹配。
6. 灰色单框敌方优先和双框全局最优逻辑均已实现。
7. 年龄不超过 0.15 秒的己方 0x02BB 以全局最高优先级先执行一对一雷达剔除；超时后己方旧历史/KF 不占点，敌方流程和当前彩色己方证据仍按本文运行。
8. 随机猜测只对敌方，优先级和随机种子符合本文档。
9. 固定猜点不更新 KF且持续标记 loss。
10. 猜测坐标和 KF 外推坐标均能有效发送给裁判系统。
11. 现有性能指标导出继续工作，且没有恢复逐对象 GPU 同步和高频双路全分辨率可视化。
12. UI 已提供文字严格为 `xy maps` 的勾选项；未启用时保留逐框投影，启用并就绪后只使用原始输入帧分辨率 XY Map，映射生成、失效、并发替换和失败回退均符合第 5.4 节。
13. 相机内参和畸变参数仍只读取 `config/params.yaml::transform.K/dist_coeffs`，未迁移到 `device.yaml`。
14. 第 16 节合成测试全部通过，并完成一次 `main.py` 录像源 smoke run。

实现过程中若现有代码行为与本文档冲突，以本文档为准；若某项配置不存在，必须按第 5 节补充，而不是在代码中猜测默认值。
