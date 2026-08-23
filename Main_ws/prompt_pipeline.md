# 车辆检测—装甲识别两级流水线：完整实现 Prompt

## 0. 文档用途、目标与约束

本文档是后续实现 `Main_ws` 两级检测流水线的唯一需求基线。实现者必须按照本文档完成代码修改、配置、指标、测试和回退路径，不得自行改变阶段边界、线程数量、丢帧策略、时间戳语义、状态提交顺序或默认参数。

本文档的目标是把当前单线程串行链路：

```text
取最新相机帧
→ 车辆 YOLO
→ BoT-SORT
→ 裁剪车辆 ROI
→ 装甲 YOLO
→ 裁剪实际装甲小图
→ digit 分类
→ 观测组装
→ 相机/雷达融合状态机
→ 输出提交
```

改为以下两级跨帧流水线：

```text
Stage A：取最新相机帧 → 车辆 YOLO → BoT-SORT → 裁剪车辆 ROI
Stage B：装甲 YOLO → 裁剪实际装甲小图 → digit 分类
         → 观测组装 → 相机/雷达融合状态机 → 输出提交
```

稳定状态下允许：

```text
Stage A 正在处理 Frame N+1
Stage B 同时处理 Frame N
```

必须同时满足以下硬约束：

1. Stage A 只有一个执行线程，BoT-SORT 必须严格按 Stage A 取得帧的先后顺序调用。
2. Stage B 只有一个执行线程，融合状态机必须严格按 Stage B 实际接收数据包的先后顺序调用。
3. 禁止建立多个“取图→车辆→BoT-SORT→装甲→融合”的完整并行副本。
4. A→B 不是普通多元素 FIFO，而是容量固定为 1 的 latest-wins 邮箱；尚未进入 Stage B 的旧包允许被新包覆盖。
5. Stage B 已经取走并开始处理的数据包不尝试取消 GPU 推理；是否提交由 epoch、顺序和陈旧门限共同决定。
6. 不允许队列无限积压，不允许为了逐帧完整性持续增加端到端延迟。
7. 所有图像、雷达、己方反馈、阵营、时间戳和可视化必须来自同一个帧数据包，禁止在 Stage B 执行时临时读取“当前最新值”拼接到旧图像结果。
8. 第一版只改项目内的阶段拆分、线程调度和状态提交，不修改 Ultralytics、TensorRT、PyTorch 或 CUDA Stream 后端。
9. 必须保留完全串行的回退路径；流水线关闭时，输出行为必须与修改前一致。
10. 代码描述、注释、日志和新增文档使用中文。

---

## 1. 当前事实与性能基线

### 1.1 当前代码位置

当前关键调用关系如下：

1. `main_event_loop.py::MainEventLoop.main_loop_thread()`
   - 使用 `camera.get_image_latest("tracker", timeout=0.1)` 获取最新帧。
   - 获取阵营、地面雷达、飞坡点和己方哨兵位置快照。
   - 调用 `CascadeMatchTracker.track(...)`。
   - 整理 UI、TF 和裁判消息。
2. `tracker/CascadeMatchTracker/tracker.py::CascadeMatchTracker.track()`
   - 调用 `self.detect(img)`。
   - 调用 `_build_car_observations(...)`。
   - 调用 `process_observations(...)`。
3. `tracker/CascadeMatchTracker/detector.py::BaseDetector.detect()`
   - 调用 `car_detector.predict(img)`。
   - 裁剪车辆 ROI。
   - 调用 `armor_detector.predict_batch(crop_imgs)`。
   - 组装 `SingleDetectionResult`。
4. `model/yolov12/predictor_with_tracker.py::PredictorWithTracker.predict()`
   - 车辆 YOLO 与 BoT-SORT 的唯一正式入口。
5. `model/armor_detector.py::TwoStepArmorDetectorClassifier.predict_batch()`
   - 输入是车辆 ROI，不是实际装甲小图。
   - 先执行装甲位置/颜色 YOLO，再裁剪实际装甲小图，最后批量执行 digit 分类。

### 1.2 BoT-SORT 当前对外结果

当前 Ultralytics 内部 BoT-SORT 每条结果为：

```text
[x1, y1, x2, y2, track_id, score, cls, detection_index]
```

经过 Ultralytics `Results.boxes` 和项目包装后，Stage A 当前需要保留的唯一正式输出为：

```text
car_box: [x1, y1, x2, y2]  # 原图像素坐标
car_conf: float             # 当前帧车辆 YOLO 检测置信度
track_id: int               # BoT-SORT 跨帧轨迹 ID
car_class_id: int           # 当前车辆模型类别；现有下游不依赖其语义
```

`car_conf` 不是轨迹身份置信度。BoT-SORT 内部 Kalman 均值、协方差、速度、轨迹年龄等不在本次改造中新增为下游接口。

### 1.3 实测基线

基线来源固定为：

```text
logs/replay/20260730_225535/runtime_metrics.jsonl
统计范围：record_type == "frame" 且 frame_id > 100
有效帧数：1262
```

实测均值：

| 指标 | 均值 |
|---|---:|
| 车辆 resize | 5.751 ms |
| 车辆 Ultralytics Track | 37.795 ms |
| 车辆结果解析 | 0.207 ms |
| 车辆检测总计 | 43.764 ms |
| 车辆框裁剪 | 0.017 ms |
| 装甲批量识别 | 15.871 ms |
| 检测结果组装 | 0.130 ms |
| 检测总计 | 60.231 ms |
| 主循环总计 | 62.376 ms |

因此第一版只需要一个 Stage A 和一个 Stage B；Stage A 是稳定吞吐瓶颈，禁止预先增加 Stage B 线程池。

理想完全重叠时，周期上限由约 43.8 ms 的 Stage A 决定；考虑融合和提交开销，预期约为 21～22 FPS。该数字是性能参考，不是功能正确性的替代标准。

---

## 2. 第一版唯一阶段边界

### 2.1 Stage A 必须包含

Stage A 由现有 `MainEventLoop.main_loop_thread` 所在线程承担，不新增多个 Stage A 工作者。每个循环严格执行：

1. 读取一次当前阵营快照。
2. 调用 `camera.get_image_latest("tracker", timeout=...)` 取得最新且尚未被该消费组读取的相机帧。
3. 若返回空帧，本轮结束，不生成 `source_sequence`，继续等待。
4. 为有效帧递增 `source_sequence`。
5. 记录相机输入延迟。
6. 使用该帧判断前哨站状态，但暂不写入跨线程共享的最终输出。
7. 获取地面雷达快照、飞坡点快照和己方哨兵位置快照。
8. 在快照全部取得后、车辆推理开始前记录 `observation_monotonic = time.monotonic()`。
9. 调用唯一的车辆 `PredictorWithTracker.predict(frame)`。
10. 按当前 `_clip_box` 规则裁剪车辆框。
11. 生成车辆 ROI 视图和对应的 `car_box/car_conf/track_id`。
12. 生成不可变 `StageAPacket`。
13. 将数据包发布到 latest-wins 邮箱；若邮箱中已有尚未被 Stage B 取走的旧包，原子覆盖旧包并记录丢弃。
14. 立即开始下一次取最新图，不等待 Stage B 完成。

“裁剪车辆 ROI”是 Stage A 终点。Stage A 不执行装甲 YOLO，也不裁剪真正的装甲板小图。

### 2.2 Stage B 必须包含

Stage B 由唯一后台线程承担。每次从 latest-wins 邮箱取走一个数据包后严格执行：

1. 校验 pipeline 是否仍在运行。
2. 校验 `pipeline_epoch`。
3. 校验 `source_sequence > last_committed_source_sequence`。
4. 使用单调时钟校验数据包是否已超过 `max_commit_age_sec`；取包时已经过期则直接丢弃，不执行装甲推理。
5. 若车辆 ROI 列表非空，调用唯一的 `armor_detector.predict_batch(vehicle_rois)`。
6. 在 `TwoStepArmorDetectorClassifier` 内保持原有串行子步骤：
   - 装甲位置/颜色 YOLO；
   - 从车辆 ROI 裁剪实际装甲小图；
   - 对所有实际装甲小图进行一次批量 digit 分类；
   - 将结果分配回对应车辆 ROI。
7. 按原有逻辑解析全部 `ArmorObservation`、选择 `selected_armor`、计算相机投影并组装 `SingleDetectionResult`。
8. 构建 `CarObservation`。
9. armor 和观测组装完成后、修改 tracker 之前，再次校验 stop、epoch、source_sequence 和 `max_commit_age_sec`。
10. 获取唯一的 `pipeline_state_lock`，并在锁内再次校验 stop、epoch、source_sequence 和处理年龄。
11. 校验通过后，在同一次状态事务中调用唯一的 `process_observations(...)`、生成 `TrackOutputSnapshot/DivisionPositionSnapshot/CompletedFrame`，并更新兼容输出字段。
12. 在同一锁内更新 `last_committed_source_sequence`、`commit_sequence` 和 `_latest_pipeline_output`，随后释放 `pipeline_state_lock`。
13. 锁外使用不可变 `CompletedFrame` 提交异步可视化、TF、裁判消息和指标；这些后提交动作失败时记录 `commit_error`，但不回滚已提交 tracker 状态。
14. 完成后提交动作后，立即从邮箱取下一份最新数据包。

`process_observations` 会修改 tracker，且第一版不实现事务回滚，因此一旦在锁内通过最终校验并开始调用它，本包必须完成状态与 `_latest_pipeline_output` 提交；不得在 `process_observations` 之后因为刚好跨过 0.20 秒门限而丢弃。`max_commit_age_sec` 的精确定义是“进入状态修改事务前允许的最大处理年龄”，最终提交年龄允许比门限多出本次状态机和原子提交临界区的实际耗时，该超出必须计入指标。

Stage B 在处理某个在途包时，即使发现邮箱已有更新包，也不能仅以“有更新包”为由放弃当前包，否则 Stage B 永久慢于 Stage A 时可能持续丢弃而不产生任何输出。只有以下情况可以丢弃在途结果：

```text
pipeline 已停止
pipeline_epoch 已变化
source_sequence 不大于最后已提交序号
进入状态修改事务前，单调时钟处理年龄超过 max_commit_age_sec
本包发生无法形成有效状态提交的异常
```

---

## 3. 数据结构的唯一要求

新增 `tracker/CascadeMatchTracker/pipeline.py`，并把以下 pipeline 专用数据结构和单槽邮箱全部放在该文件中：

```text
CarStageItem
StageAPacket
TrackOutputSnapshot
DivisionPositionSnapshot
CompletedFrame
LatestPacketMailbox
PipelineDiagnostics
```

现有 `ArmorObservation`、`SingleDetectionResult`、`CarObservation` 和 `TrackingState` 继续保留在 `tracker/CascadeMatchTracker/type.py`。禁止把线程、Condition 或邮箱状态塞进 `type.py` 的业务观测类型。

### 3.1 `CarStageItem`

必须是不可变数据结构，至少包含：

```text
car_class_id: int
car_box: Tuple[float, float, float, float]
car_conf: float
track_id: int
vehicle_roi: np.ndarray
```

约束：

- `car_box` 为原图像素坐标。
- `vehicle_roi` 是从不可变原图裁剪得到的只读视图；不得在 Stage A 主动复制整块 ROI。
- `track_id` 不允许为空；沿用当前行为，BoT-SORT 尚未分配 ID 的车辆不进入 Stage B。
- 数据包内顺序必须与车辆检测输出的稳定顺序一致，Stage B 通过相同顺序把 armor batch 结果配回车辆。

### 3.2 `StageAPacket`

必须是不可变数据结构，至少包含：

```text
pipeline_epoch: int
source_sequence: int
frame: np.ndarray
frame_stamp: Optional[float]
frame_width: int
frame_height: int
input_frame_age_ms: Optional[float]
ingress_monotonic: float
observation_monotonic: float
published_monotonic: float
faction
lidar_snapshot
ally_snapshot
lidar_fly_points
outpost_alive: int
cars: Tuple[CarStageItem, ...]
stage_a_timings_ms: Mapping[str, float]
```

约束：

- `source_sequence` 在每次 Stage A 接受有效相机帧时递增，从 1 开始；允许下游看到跳号，禁止倒退和重复。
- `frame_stamp` 保持相机原始时钟域，不修改、不替换。
- `ingress_monotonic` 在有效图像被取出后立即记录，专门用于流水线内部处理年龄。
- `observation_monotonic` 在雷达/己方快照取得后、车辆推理开始前记录，专门作为本包状态机的“观测时刻”。
- `published_monotonic` 在车辆 ROI 裁剪完成、数据包发布前立即记录，专门用于精确计算 Stage A→B 邮箱等待时间；不得用 `observation_monotonic` 代替，否则会把模块 3–4 的执行时间误算为排队。
- `faction/lidar_snapshot/ally_snapshot/lidar_fly_points/outpost_alive` 必须在 Stage A 采集后随包传递。
- `frame` 必须保持引用存活到 Stage B 和异步可视化取得所需引用为止，不得依赖相机环形缓冲区的槽位对象继续保存该引用。
- `stage_a_timings_ms` 不允许引用仍会被 Stage A 修改的可变字典；发布前必须冻结或复制。

### 3.3 `TrackOutputSnapshot`

必须是不可变数据结构，由 Stage B 在 `process_observations` 完成后从每个可变 `TrackingState` 复制形成，至少包含：

```text
class_id: int
name: str
identity_state
spatial_state
frame_source
output_valid: bool
output_position_2d_m: Optional[Tuple[float, float]]
is_guessed_output: bool
pos_3d: Tuple[float, float, float]
last_car_box: Optional[Tuple[float, float, float, float]]
```

禁止把可变 `TrackingState` 本体保存到 `CompletedFrame`，否则 Stage B 处理下一包时会在 UI/异步线程读取旧结果期间修改同一对象。

### 3.4 `DivisionPositionSnapshot`

必须是不可变数据结构，避免 `pipeline.py` 反向导入 `main_event_loop.py` 中定义的 `RefereeDivision2dPosition` 形成循环依赖：

```text
x: int
y: int
is_valid: bool
source: str
is_guessed: bool
```

Stage B 使用它保存 10 个 class 的完成输出；提交兼容字段时再构造或更新 `RefereeDivision2dPosition`。

### 3.5 `CompletedFrame`

Stage B 完成计算后形成局部结果，不得边计算边写入 `MainEventLoop` 公共字段。至少包含：

```text
pipeline_epoch: int
source_sequence: int
commit_sequence: int
source_gap_from_previous_commit: int
frame: np.ndarray
frame_stamp: Optional[float]
input_frame_age_ms: Optional[float]
faction
outpost_alive: int
detections: Tuple[SingleDetectionResult, ...]
tracks: Tuple[TrackOutputSnapshot, ...]
tracker_lidar_points: Tuple[Tuple[float, float], ...]
lidar_point_owners: Mapping[int, int]
projection_points_cm: Tuple[Tuple[float, float], ...]
lidar_fly_points_cm: Tuple[Tuple[float, ...], ...]
divisions_pos: Tuple[DivisionPositionSnapshot, ...]
stage_a_timings_ms: Mapping[str, float]
stage_b_timings_ms: Mapping[str, float]
pipeline_timings_ms: Mapping[str, float]
processing_age_ms: float
commit_interval_ms: Optional[float]
commit_fps: Optional[float]
error: Optional[str]
```

`tracks` 必须是 10 个不可变 `TrackOutputSnapshot`。TF 整理、UI 和 `AsyncTrackingVisualizer.submit(...)` 全部使用这些快照；`MainEventLoop.self.tracks` 可以继续作为兼容字段指向 tracker 内部可变状态，但任何异步消费者不得保存或跨帧使用其中对象。

`commit_sequence` 在持有 `pipeline_state_lock`、通过最终校验并完成 `process_observations` 后确定为“上一次成功提交值 + 1”。所有 Mapping 在放入对象前必须复制并冻结，所有列表型输出必须转成 tuple 或独立副本；提交后不得再修改 `CompletedFrame`。

提交前必须完成所有字段计算。提交阶段不得再读取 Stage A 当前帧或最新相机帧。

### 3.6 `PipelineDiagnostics`

必须提供线程安全的累计诊断，所有计数初值为 0：

```text
source_sequence
commit_sequence
published_total
taken_total
committed_total
pending_replaced_total
stale_before_stage_b_total
stale_before_state_total
epoch_drop_total
out_of_order_drop_total
shutdown_drop_total
stage_a_error_total
stage_b_error_total
armor_error_total
tracker_error_total
commit_error_total
last_commit_monotonic: Optional[float]
```

计数修改必须发生在对应状态改变的同一临界区，禁止先改 mailbox 再在锁外补计数。对外指标读取通过一次不可变 snapshot 完成，禁止逐字段无锁读取导致统计跨时刻拼接。

- `published_total`：邮箱开启时 `publish_latest` 成功的次数，包含覆盖旧 pending 的新发布。
- `taken_total`：`take_latest` 实际返回数据包的次数。
- `committed_total`：成功完成状态事务和输出提交的次数，必须始终等于 `commit_sequence`。
- `pending_replaced_total`：成功发布时覆盖旧 pending 的次数。
- 拒绝向已关闭邮箱发布只增加 `shutdown_drop_total`，不增加 `published_total`。

### 3.7 序号定义

必须同时维护：

```text
source_sequence：Stage A 已接受有效相机帧并准备执行车辆检测的输入序号
commit_sequence：Stage B 成功提交的结果序号
tracker.frame_count：Stage B 实际调用 process_observations 的次数
```

三者不要求数值相等：

- Stage A 异常和 latest-wins 覆盖都会使 `source_sequence` 在 Stage B 观察到跳号。
- 陈旧结果或 epoch 变化会使部分 `source_sequence` 永不提交。
- `tracker.frame_count` 只随实际进入状态机的数据包递增。

禁止为了让序号相等而伪造空调用或回滚 BoT-SORT。

---

## 4. latest-wins 邮箱与丢帧语义

### 4.1 邮箱必须是单槽结构

第一版固定只允许：

```text
0 个等待包
或
1 个等待包
```

Stage B 已经取走的在途包不占“等待槽”。因此同一时刻最多可能存在：

```text
Stage B 正在处理的 1 个包
邮箱等待的 1 个最新包
Stage A 正在生成的 1 个包
```

不得使用会阻塞 Stage A 的 `queue.Queue(maxsize=1).put()` 语义。必须使用 `Lock + Condition` 或等价原语实现原子 `publish_latest(packet)`：

```text
若 pending is None：pending = packet
若 pending 已存在：记录旧 pending 被覆盖；pending = packet
唤醒 Stage B
```

`LatestPacketMailbox` 的公开接口固定为：

```python
publish_latest(packet: StageAPacket) -> bool
take_latest() -> Optional[StageAPacket]
clear(reason: str) -> Optional[StageAPacket]
close() -> Optional[StageAPacket]
```

唯一语义：

- `publish_latest` 永不等待 Stage B；邮箱开启时发布成功返回 `True`，邮箱已关闭时拒绝并返回 `False`。
- 覆盖旧 pending 时在同一锁内更新 `pending_replaced` 诊断和 drop 事件所需字段。
- `take_latest` 在无 pending 且邮箱未关闭时使用 Condition 等待，不使用忙轮询；允许伪唤醒，必须用 while 条件重新检查。
- `take_latest` 返回包前把 pending 置空并释放邮箱锁；Stage B 不得持锁执行模型。
- `clear` 清空并返回当时 pending，供调用方按传入 reason 记录一次 drop；没有 pending 返回 `None`。
- `close` 原子标记关闭、清空 pending、唤醒全部等待者并返回被清除包；关闭后永久拒绝发布，本实例不重新开启，pipeline 重启必须创建新邮箱实例。
- Stage A 在 stop 发生时可能刚从车辆 GPU 调用返回；其后 `publish_latest` 会因邮箱关闭返回 `False`，该包记录为 shutdown drop，不能提交。

### 4.2 允许丢弃的位置

允许：

1. 相机自身只保留最新帧，因此 Stage A 天然跳过未读取的旧相机帧。
2. 新 Stage A 包覆盖尚未被 Stage B 取走的旧 pending 包。
3. Stage B 取包前发现陈旧、epoch 失效或顺序失效时丢弃。
4. Stage B 完成 armor/观测组装但尚未调用 `process_observations` 时，发现陈旧、epoch 失效、顺序失效或 stop 则丢弃。

不允许：

1. 回滚已经执行的 BoT-SORT。
2. 尝试中止正在运行的 TensorRT/PyTorch GPU 调用。
3. 把被丢包重新插回队列。
4. 为被覆盖包调用 `process_observations` 伪造一次空帧。
5. 让旧包覆盖新包。

### 4.3 丢包对状态机的确定语义

BoT-SORT 可以看到比后级融合状态机更多的帧，这是允许且预期的：

- BoT-SORT 继续利用这些帧维护 `track_id` 连续性。
- 后级把下一份提交包视为间隔更长的新观测。
- `BotIdTrack` 语义历史只记录 Stage B 实际处理的 armor 观测，不补偿被覆盖的包。
- `continuous_loss_count`、`unseen_frame_count`、`association_ttl_frames`、`radar_guess_after_frames` 和 `fixed_guess_after_frames` 第一版继续按 Stage B 实际处理帧计数，不转换为时间门限。
- 必须记录丢包率；若实际丢包率较高，后续另行评估把上述帧数门限改成时间门限，本次禁止顺带修改。

---

## 5. 时间、快照与融合语义

### 5.1 四种时间不得混用

```text
frame_stamp
```

- 来源：相机/ROS/回放。
- 用途：UI、数据集、已有端到端 wall-time 延迟计算。
- 可能不属于本机 wall clock；沿用 `_get_frame_age_ms` 的时钟域合法性判断。

```text
ingress_monotonic
```

- 来源：Stage A 取得有效帧后调用 `time.monotonic()`。
- 用途：`max_commit_age_sec` 和流水线处理年龄；不用于邮箱排队时间。

```text
observation_monotonic
```

- 来源：本包雷达/己方快照取得完成后、车辆推理开始前调用 `time.monotonic()`。
- 用途：调用 `process_observations(..., now_monotonic=observation_monotonic)`。
- 用途：所有本帧 `SingleDetectionResult.time_stamp` 使用同一个值。

```text
published_monotonic
```

- 来源：Stage A 完成车辆 ROI 裁剪、发布 `StageAPacket` 前调用 `time.monotonic()`。
- 用途：`mailbox_wait_ms = Stage B 取包时刻 - published_monotonic`。
- 不参与 KF `dt`、最大年龄或相机时间戳计算。

禁止用 Stage B 完成时刻替代 `observation_monotonic`，否则流水线调度抖动会被错误地解释为传感器观测时间。

### 5.2 雷达和己方快照

- 地面雷达快照必须在 Stage A 随图像打包。
- 己方哨兵位置快照必须在 Stage A 随图像打包。
- 飞坡点必须在 Stage A 随图像打包。
- Stage B 禁止再次调用 `get_snapshot()` 获取更新值。
- 雷达新鲜度仍由 `lidar_fusion.max_snapshot_age_sec` 判断，并使用本包 `observation_monotonic`。
- 当前 `lidar_fusion.max_snapshot_age_sec = 0.20 s` 保持不变。

这会保持与当前串行链路相同的因果关系：图像和在车辆推理前取得的快照共同形成一次观测，推理耗时不会改变它们的观测时刻。

### 5.3 KF 的 dt

`process_observations` 继续使用相邻两次实际处理包的 `observation_monotonic` 差值：

```text
dt = clamp(current_observation_monotonic - last_observation_monotonic,
           min_dt_sec,
           max_dt_sec)
```

当前默认：

```text
min_dt_sec = 0.001
default_dt_sec = 0.033333
max_dt_sec = 0.20
```

latest-wins 丢包会自然增大 dt。超过 0.20 秒时继续沿用现有上限，不在本次改造中改变 KF 数学模型。

### 5.4 阵营

- 每个包使用 Stage A 取图时读取的 `faction`。
- Stage B 禁止使用执行时的 `self.faction` 最新值替换包内 faction。
- 普通 faction 变化不自动增加 pipeline epoch；旧 faction 包和新 faction 包按 source_sequence 顺序进入 Stage B，现有 tracker 阵营变化重置逻辑继续生效。
- 显式 tracker reset、回放源重启、标定几何整体替换等事件使用 pipeline epoch 使旧在途包失效，详见生命周期章节。

### 5.5 分辨率、标定和 XY Map

- Stage A 从每个有效 `frame.shape` 取得宽高并写入包。
- 稳定分辨率下禁止每帧跨线程修改 XY Map 状态；只有检测到宽高变化时才进入重置流程。
- 分辨率变化时，Stage A 获取 `pipeline_state_lock`，递增 epoch、清空 pending、调用 `xy_map_manager.set_frame_size(...)`，然后以新 epoch 处理该帧。
- UI/标定线程应用新的相机外参、场地几何或 XY Map 标定签名时，也必须通过持有 `pipeline_state_lock` 的统一重置入口递增 epoch。
- Stage B 的投影使用包内 `frame_width/frame_height` 和该包 epoch 对应的当前标定；禁止拿旧尺寸车辆框查询新尺寸 XY Map。
- 仅从“直接投影”异步切换为已经生成完成、但标定签名和分辨率完全相同的 XY Map 时不递增 epoch，因为两者表示同一几何；继续沿用现有一致性检查。

---

## 6. 线程所有权与共享状态

### 6.1 Stage A 唯一所有

以下对象和操作只能在 Stage A 线程执行：

```text
camera.get_image_latest("tracker", ...)
car_detector.predict(...)
Ultralytics vehicle model.track(..., persist=True)
BoT-SORT 可变状态
车辆框裁剪
source_sequence 递增
latest-wins mailbox 发布
```

禁止从 Stage B、UI 线程或线程池再次调用同一个 `car_detector.predict`。

### 6.2 Stage B 唯一所有

以下对象和操作只能在 Stage B 线程执行：

```text
armor_detector.predict_batch(...)
digit_classifier.predict_batch(...)
ArmorObservation 解析与选择
SingleDetectionResult 组装
_build_car_observations(...)
process_observations(...)
tracker.frame_count 和所有 TrackingState 修改
commit_sequence 递增
last_committed_source_sequence 更新
```

禁止建立第二个 Stage B 工作者。若以后实测 Stage B 成为瓶颈，必须另写需求处理多工作者乱序重排，不能在本次实现中预留未验证的线程池。

### 6.3 指标字典

Stage A 和 Stage B 禁止同时修改：

```text
detector_timing_stats
detector_frame_timing_ms
track_timing_stats
track_frame_timing_ms
loop_frame_timing_ms
```

唯一规则：

1. Stage A 只写本地 timing 字典。
2. timing 字典冻结后放入 `StageAPacket`。
3. Stage B 只写本地 Stage B timing 字典。
4. 成功提交时由 Stage B 将两阶段 timing 合并为“该 source_sequence 的完整指标”。
5. EMA/最大值等累计统计只能由 Stage B 提交线程统一更新；禁止选择“Stage A/B 各自加锁写共享统计”的替代实现。

### 6.4 输出原子性

Stage B 必须先在局部变量中形成完整 `CompletedFrame`，之后在同一个提交临界区更新兼容字段：

```text
self.tracks
self.latest_detection_results
self.latest_detection_source_frame
self.latest_detection_source_stamp
self.lidar_points_cm
self.lidar_point_owners
self.projection_points_cm
self.lidar_fly_points_cm
self.outpost_alive
self.divisions_pos
tracker.latest_detections
tracker.latest_lidar_points
tracker.latest_lidar_matches
```

`MainEventLoop` 必须新增：

```text
self._pipeline_output_lock
self._pipeline_state_lock
self._latest_pipeline_output: Optional[CompletedFrame]
get_latest_pipeline_output() -> Optional[CompletedFrame]
```

`_pipeline_state_lock` 负责把最终有效性校验、`process_observations` 和状态提交串成不可被 reset/stop/epoch 变化穿插的事务。`_pipeline_output_lock` 只保护完成输出对象和兼容字段的短时间替换，不包围 armor GPU 推理，也不包围可视化、TF、裁判消息或指标导出。

Stage B 在输出临界区内先更新所有兼容字段，最后替换 `_latest_pipeline_output`。UI 中需要同时读取图像、detections、车辆位置、雷达点或投影点的调用点必须改为调用一次 `get_latest_pipeline_output()`，并从同一个返回对象读取，禁止分别 `getattr` 多个可能属于不同 source_sequence 的字段。串行模式也必须形成并发布同样的 `CompletedFrame`，保证 UI 不需要按模式分支。

锁顺序固定为：

```text
pipeline_state_lock → mailbox 内部锁 → pipeline_output_lock
```

- Stage B 从 mailbox 取包后必须立即释放 mailbox 锁，armor 推理期间不持有任何上述锁。
- reset/stop 在持有 `pipeline_state_lock` 时可以短暂获取 mailbox 锁清空 pending。
- 状态提交在持有 `pipeline_state_lock` 时可以短暂获取 `pipeline_output_lock`。
- `get_latest_pipeline_output()` 只获取 `pipeline_output_lock`，不得反向获取 state/mailbox 锁。
- 任何代码不得在持有 mailbox 锁或 output 锁时申请 `pipeline_state_lock`。

异步可视化提交必须直接使用本次 `CompletedFrame` 的局部 `frame/detections/tracks/frame_stamp`，不能在锁外重新读取可能已经更新的公共字段。

TF 发布和 `referee.update_tracking_radar2client_msg(...)` 必须发生在 tracker 状态成功提交之后、`pipeline_state_lock` 之外，使用同一份已完成结果。

### 6.5 运行时参数修改

现有 UI 修改车辆置信度阈值时：

- 新值允许从下一次尚未开始的 Stage A 推理生效。
- 不要求中断当前 Stage A。
- 简单浮点阈值写入可以沿用现有接口；若实现中需要重建 predictor 或 engine，则必须在 Stage A 安全点执行，本次不应引入模型重建。

---

## 7. 串行回退与代码拆分要求

### 7.1 必须保留同步组合接口

在 `BaseDetector` 中新增以下确定方法名：

```python
detect_cars_and_crop(...)
detect_armors_and_assemble(...)
```

`BaseDetector.detect()` 必须保留，并严格依次调用以上两个方法，作为同步组合接口。

该接口用于：

- `detection_pipeline.enabled == false` 的正式串行回退。
- 当前 tracker 预热。
- 拆分等价性测试。
- 不具备线程环境的单元测试。

禁止删除现有 `CascadeMatchTracker.track(...)` 公共接口。流水线关闭时它必须继续依次完成 Stage A、Stage B 和状态融合。流水线开启时，由 `MainEventLoop.main_loop_thread` 直接调用 `detect_cars_and_crop(...)` 并向 `LatestPacketMailbox` 发布；唯一 Stage B 工作者调用 `detect_armors_and_assemble(...)`、`_build_car_observations(...)` 和 `process_observations(...)`。禁止再引入第三种协调架构。

`detection_pipeline.enabled` 只在 `MainEventLoop` 初始化时读取一次，第一版不支持运行中热切换；修改后必须重启进程。串行模式不创建 Stage B 线程和 mailbox，但仍为每个有效输入分配 `source_sequence`、形成 `CompletedFrame` 并通过同一个输出提交函数发布。没有异常时串行模式的 `source_sequence == commit_sequence == tracker.frame_count`。

### 7.2 第一版禁止修改

以下内容不属于本次范围：

- 车辆 YOLO、armor YOLO、digit 模型结构或权重。
- 车辆/armor 推理尺寸、置信度、IoU、max_det。
- BoT-SORT 配置、ReID、Kalman 或 `persist=True`。
- Ultralytics 安装目录代码。
- TensorRT `execute_v2` 到 `execute_async_v3` 的替换。
- CUDA Stream 或 CUDA Event 管理。
- 多 GPU 分配。
- 多个完整流水线副本。
- 下游 Cascade 匹配算法、代价、优先级或阈值。
- 把按帧门限改为按时间门限。
- 裁判系统消息格式和发送频率。

---

## 8. 生命周期、epoch 与停止语义

### 8.1 `pipeline_epoch`

- 类型固定为非负整数。
- pipeline 每次启动时从 1 开始。
- 每次发生需要使旧在途数据失效的重置时递增 1。
- 允许自然溢出前长期运行；Python 整数不需要手动回绕。

必须递增 epoch 的事件：

1. 显式重启 detection pipeline。
2. 显式重置整个 tracker/BoT-SORT 组合状态。
3. 回放 seek、回放源重新从头开始或输入源实例替换。
4. 相机输入分辨率变化，导致旧 ROI/投影尺寸不再适用。
5. 标定几何或 XY Map 对应的标定版本整体替换，且旧包不能再使用新几何投影。

不单独递增 epoch 的事件：

- 普通相机掉一帧或超时。
- 普通 latest-wins 覆盖。
- 普通 faction 变化。
- armor 某一帧没有结果。
- 某一帧没有地面雷达点。

epoch 递增必须在持有 `pipeline_state_lock` 时完成，并按既定锁顺序原子清空邮箱 pending 包。Stage B 在途包返回后检测到 epoch 不同，丢弃而不提交。

### 8.2 启动

1. 按现有流程加载模型。
2. 保持当前 `warmup_num = 10`，通过同步组合接口预热，不在预热阶段启用流水线并发。
3. 初始化 source/commit/drop/error 计数器。
4. 初始化 epoch。
5. 启动唯一 Stage B daemon 线程。
6. 进入 Stage A 主循环。

本次不改变预热图像、预热次数和预热对模型内部状态的现有影响。

### 8.3 停止

停止顺序固定为：

1. 获取 `pipeline_state_lock`。若 Stage B 已经进入状态事务，等待该包完成提交。
2. 在锁内设置统一 stop event，禁止 Stage A 接收新帧和 Stage B 开始新状态事务。
3. 按固定锁顺序关闭 latest-wins 邮箱并清除 pending 包，记录 `shutdown_pending_drop`，随后释放 `pipeline_state_lock`。
4. 唤醒正在等待邮箱的 Stage B。
5. 若 Stage B 仍在 GPU 推理，不能强杀；GPU 调用自然返回后因最终校验失败而不得进入状态事务。
6. 最多等待 `worker_join_timeout_sec`。
7. 超时后记录 critical 日志；Stage B 使用 daemon 线程，不能让进程关闭无限阻塞。
8. 再关闭可视化、雷达客户端、TF、XY Map 和指标导出器。

停止时不排空 pending 队列，默认实时关闭优先。

---

## 9. 异常处理

### 9.1 总原则

- 任何异常都不能让 Stage B 线程静默死亡。
- 必须记录完整 traceback、epoch、source_sequence、阶段和错误计数。
- 不允许在异常后提交一半更新的公共输出。
- 第一版不实现 tracker 状态事务回滚，保持与当前串行主循环相同的“记录异常并继续下一帧”策略。

### 9.2 Stage A 异常

车辆推理或车辆结果解析异常时：

1. 记录 `stage_a_error`。
2. 本帧不发布到 Stage B。
3. 不自动重建模型，不自动重置或回滚 BoT-SORT。
4. 下一轮继续取得最新相机帧。
5. 若异常持续，指标和日志必须能看出连续错误；本次不增加自动切换串行模式。

相机超时返回空帧不是异常，不生成错误日志风暴。

### 9.3 Stage B armor 异常

若 armor YOLO 或 digit 分类对整批抛出异常：

1. 记录完整错误。
2. 对本包所有有效车辆生成“有车辆、无可靠 armor”的检测结果：
   - `class_id = -1`
   - `class_conf = 0`
   - `selected_armor = None`
   - `armor_observations = ()`
   - 保留 `car_box/car_conf/bot_id`
3. 继续构建 `CarObservation` 并调用融合状态机。
4. 禁止把 armor 异常误处理为“整帧没有车辆”。

### 9.4 无车辆

Stage A 正常完成但没有带 track_id 的有效车辆时：

- 仍发布数据包。
- Stage B 不调用 armor 模型。
- 使用空 detections/cars 调用 `process_observations`。
- 正常推进下游丢失、KF 和猜测状态。

### 9.5 融合或提交异常

- `process_observations` 异常时不更新公共输出和 `last_committed_source_sequence`。
- 记录错误后继续下一包。
- 如果状态机已经发生部分内部修改，第一版不回滚；该行为与当前串行实现异常策略一致。
- 输出提交临界区应尽量只包含不可失败的引用/基础类型赋值；耗时操作和可能抛异常的格式转换应在形成 `CompletedFrame` 时完成。

---

## 10. 配置、默认值及来源

必须在 `config/params.yaml` 顶层新增，位置固定在 `digit_detector` 配置块之后、`projection_lidar_statistics` 之前：

```yaml
detection_pipeline:
  enabled: true
  camera_timeout_sec: 0.10
  max_commit_age_sec: 0.20
  worker_join_timeout_sec: 5.0
```

### 10.1 新增运行参数

| 参数 | 默认值 | 合法范围 | 唯一语义 | 默认值来源 |
|---|---:|---|---|---|
| `detection_pipeline.enabled` | `true` | bool | `true` 使用本文档流水线；`false` 使用原串行组合接口 | 本需求的正式目标是启用流水线，同时保留显式回退 |
| `camera_timeout_sec` | `0.10 s` | `> 0` | Stage A 每次等待新相机帧的最长时间 | 当前 `main_loop_thread` 已使用 `timeout=0.1`，保持行为不变 |
| `max_commit_age_sec` | `0.20 s` | `> 0` | `time.monotonic() - ingress_monotonic` 在“Stage B 取包”和“进入 tracker 状态事务前”两个安全点的最大允许值；边界 `== 0.20` 允许进入，`> 0.20` 丢弃。状态事务开始后不再因年龄丢弃 | 当前 `lidar_fusion.max_snapshot_age_sec=0.20` 且 `position_kalman.max_dt_sec=0.20`；选择同一上限，避免明显超过现有融合时间尺度的数据进入状态机 |
| `worker_join_timeout_sec` | `5.0 s` | `>= 1.0` | stop 后等待 Stage B 自然退出的最长时间 | 正常单帧约 0.06 s；5 秒远大于正常抖动，同时避免 GPU/驱动异常时无限阻塞关闭 |

代码读取兼容规则：

- 正式 `config/params.yaml` 必须包含以上完整配置。
- 对旧外部配置缺少整个 `detection_pipeline` 节点时，代码安全回退为 `enabled=false`，其余键使用表中默认值，并输出一次 warning。
- 节点存在但字段类型或范围非法时启动失败，禁止静默修正。

### 10.2 固定实现常量，不提供配置

| 项目 | 固定值 | 来源与理由 |
|---|---:|---|
| Stage A 工作者数量 | 1 | BoT-SORT 是有序可变状态；当前 Stage A 也是性能瓶颈 |
| Stage B 工作者数量 | 1 | Stage B 约 16 ms，小于 Stage A 约 44 ms；避免乱序重排和共享 tracker 状态竞争 |
| pending 容量 | 1 | 用户确认实时性优先，使用 latest-wins，禁止多帧积压 |
| pending 满时策略 | 覆盖最旧 pending | 保留最新 Stage A 输出，不阻塞 Stage A |
| Stage B 在途取消 | 禁止 | 当前 GPU 同步调用无法安全取消；避免永久饥饿 |
| shutdown drain | `false` | 停止实时系统时不继续输出旧结果 |
| Stage B 线程 daemon | `true` | GPU/驱动异常时进程关闭不能永久被后台线程阻塞 |
| warmup 次数 | 10 | 当前 `MainEventLoop` 使用 `tracker.warmup(warmup_num=10)`，本次保持 |
| drop 后空帧补偿 | 禁止 | 下游只按实际处理包推进，禁止伪造传感器观测 |
| pipeline 初始 epoch | 1 | 0 保留为“尚未启动/无有效 epoch”哨兵，便于测试和诊断 |
| source/commit 初值 | 0 | 第一个有效输入/提交递增为 1，与现有 `processed_frame_count` 从 0 递增的习惯一致 |

这些固定值若未来需要修改，必须先更新本文档或新需求，不得在第一版中擅自参数化。

### 10.3 受流水线影响但本次保持不变的现有参数

| 参数 | 当前默认值 | 来源 | 本次语义 |
|---|---:|---|---|
| 相机采集率 | `30.0 Hz` | `config/device.yaml::camera.acquisition_rate` | 相机可快于 Stage A；`get_latest` 自然跳过未读取旧帧 |
| 车辆输入尺寸 | `1280` | `config/params.yaml::car_detector.img_size` | 不修改 |
| 车辆置信度 | `0.3` | `car_detector.conf_thres` | 不修改 |
| 车辆 IoU | `0.5` | `car_detector.iou_thres` | 不修改 |
| 车辆最大框数 | `10` | `car_detector.max_det` | 不修改 |
| armor 输入尺寸 | `192` | `armor_detector.img_size` | 不修改 |
| armor 置信度 | `0.4` | `armor_detector.conf_thres` | 不修改 |
| armor IoU | `0.4` | `armor_detector.iou_thres` | 不修改 |
| 每个车辆最大 armor 数 | `4` | `armor_detector.max_det` | 不修改 |
| 雷达快照最大年龄 | `0.20 s` | `lidar_fusion.max_snapshot_age_sec` | 相对本包 `observation_monotonic` 判断 |
| KF 最小 dt | `0.001 s` | `tracker.cascade.position_kalman.min_dt_sec` | 不修改 |
| KF 默认 dt | `0.033333 s` | 同上 `default_dt_sec` | 不修改 |
| KF 最大 dt | `0.20 s` | 同上 `max_dt_sec` | 不修改 |
| bot 语义历史容量 | `30` | `tracker.cascade.bot_history.queue_size` | 只记录 Stage B 实际处理包 |
| bot 匹配窗口 | `10` | `bot_history.matching_window` | 不修改 |
| bot 关联 TTL | `30` 帧 | `bot_history.association_ttl_frames` | 按 Stage B 处理帧计数 |
| 雷达猜测门限 | `10` 帧 | `tracker.cascade.loss.radar_guess_after_frames` | 按 Stage B 处理帧计数 |
| 固定点门限 | `30` 帧 | `tracker.cascade.loss.fixed_guess_after_frames` | 按 Stage B 处理帧计数 |
| timing EMA alpha | `0.2` | `debug.timing_ema_alpha` | 由 Stage B 统一合并更新 |
| GPU 采样间隔 | `0.5 s` | `debug.performance_metrics.gpu_sample_interval_sec` | 保持；该粗粒度数据不能单独证明 CUDA kernel 重叠 |

---

## 11. 指标、日志与可观测性

### 11.1 每个成功提交帧必须新增的指标

现有 `runtime_metrics.jsonl` frame 记录中新增：

```json
{
  "pipeline": {
    "enabled": true,
    "epoch": 1,
    "source_sequence": 123,
    "commit_sequence": 120,
    "source_gap_from_previous_commit": 1,
    "stage_a_ms": 43.8,
    "stage_b_ms": 16.0,
    "mailbox_wait_ms": 0.2,
    "processing_age_ms": 60.5,
    "commit_interval_ms": 44.5,
    "commit_interval_ema_ms": 44.8,
    "commit_fps": 22.47,
    "commit_fps_ema": 22.32,
    "published_total": 123,
    "taken_total": 121,
    "committed_total": 120,
    "pending_replaced_total": 0,
    "stale_before_stage_b_total": 0,
    "stale_before_state_total": 0,
    "epoch_drop_total": 0,
    "out_of_order_drop_total": 0,
    "shutdown_drop_total": 0,
    "stage_a_error_total": 0,
    "stage_b_error_total": 0,
    "armor_error_total": 0,
    "tracker_error_total": 0,
    "commit_error_total": 0
  }
}
```

定义：

- `stage_a_ms`：从有效帧取出到 `StageAPacket` 发布完成。
- `stage_b_ms`：从 Stage B 取包到完成或决定丢弃。
- `mailbox_wait_ms`：Stage B 取包时刻减去 Stage A 发布时刻。
- UI 的“稳定提交间隔”使用原始 `commit_interval_ms` 的指数移动平均，`alpha` 复用 `debug.timing_ema_alpha`，默认 `0.2`；默认值来源为现有耗时调试 EMA 配置。新增 `commit_interval_ema_ms` 与 `commit_fps_ema=1000/commit_interval_ema_ms` 作为界面稳定值。原有 `commit_interval_ms` 与 `commit_fps` 仍保存当前两次提交的原始值，指标文件必须同时保留原始值和 EMA，避免隐藏瞬时抖动。
- `processing_age_ms`：提交时刻减去 `ingress_monotonic`。
- `commit_interval_ms`：当前成功提交的单调时钟时刻减去上一次成功提交时刻；第一次提交为 `null`。
- `commit_fps`：`commit_interval_ms > 0` 时等于 `1000 / commit_interval_ms`；第一次提交为 `null`。流水线吞吐统一使用该字段，禁止用单包延迟的倒数代替吞吐。
- `source_gap_from_previous_commit`：当前 source_sequence 减去上次成功提交 source_sequence；正常连续值为 1。
- `stage_b_error_total`：固定等于 `armor_error_total + tracker_error_total + commit_error_total`；三类错误互斥归类，每次异常只增加其中一个子计数和总计数。
- 所有 `*_total` 是本次进程累计值。

流水线启用时，现有顶层 frame 指标语义固定为：

```text
frame_id = source_sequence
frame_timestamp = StageAPacket.frame_stamp
input_frame_age_ms = StageAPacket.input_frame_age_ms
end_to_end_latency_ms = 提交时使用现有 _get_frame_age_ms 计算
processing_ms = processing_age_ms
processing_fps = 1000 / processing_age_ms
```

其中顶层 `processing_fps` 只为兼容旧日志结构，表示单包处理年龄的倒数，不代表稳定提交吞吐；稳定吞吐必须读取 `pipeline.commit_fps`。Stage A 不再在原主循环 `finally` 中为每个 source 包导出 frame 记录；只有 Stage B 成功提交时导出一条 frame 记录。被覆盖或丢弃的包只导出 `pipeline_drop`。

### 11.2 被丢包记录

每次 pending 覆盖、陈旧丢弃、epoch 丢弃和 shutdown 丢弃必须向指标导出器提交低开销事件：

```json
{
  "record_type": "pipeline_drop",
  "reason": "pending_replaced | stale_before_stage_b | stale_before_state | epoch_changed | out_of_order | shutdown",
  "epoch": 1,
  "source_sequence": 122,
  "replacement_source_sequence": 123,
  "processing_age_ms": 45.0
}
```

`replacement_source_sequence` 只对 `pending_replaced` 必填，其余为 `null`。

指标队列自身满时继续沿用 `RuntimeMetricsExporter` 的旧记录覆盖策略，不能反向阻塞检测流水线。

Stage A/B 捕获到异常时必须额外提交：

```json
{
  "record_type": "pipeline_error",
  "stage": "stage_a | armor | tracker | commit",
  "epoch": 1,
  "source_sequence": 123,
  "error_type": "RuntimeError",
  "error_text": "简要错误文本"
}
```

完整 traceback 写入正常日志；JSONL 只保存异常类型和简要文本，避免单条指标过大。Stage A 在创建 source_sequence 后失败时填写该序号；若取图前失败或没有有效帧则 `source_sequence=null`。

### 11.3 控制台诊断

现有耗时诊断文本应增加：

```text
流水线：启用/关闭
epoch
Stage A 最近/平均/最大耗时
Stage B 最近/平均/最大耗时
处理年龄最近/平均/最大值
pending 覆盖累计数
陈旧丢弃累计数
Stage A/B 异常累计数
最近提交 source_sequence / commit_sequence
```

禁止逐帧默认打印；沿用现有 UI/按需诊断和 JSONL 指标，避免日志 IO 影响性能。

---

## 12. 正确性测试

新增测试文件名固定为：

```text
tests/test_detection_pipeline_mailbox.py
tests/test_detection_pipeline_state.py
tests/test_detection_pipeline_equivalence.py
```

- 前两个测试必须纯 CPU、使用 `unittest`，可通过 `python -m unittest tests.test_detection_pipeline_mailbox tests.test_detection_pipeline_state` 执行。
- 等价性测试允许依赖现有模型/GPU环境；无 GPU 时必须明确 skip，不能伪造通过。
- 不得用真实相机、真实串口或裁判系统作为 mailbox/state 单元测试前提。

### 12.1 阶段拆分等价性

在线程化之前必须先完成同步拆分测试。在相同预加载图像序列、相同初始模型和 tracker 状态下，对比：

```text
旧 detect()
新 Stage A → 新 Stage B 同步组合
```

至少检查：

- 每帧车辆数量。
- 每辆车 `car_box/car_conf/bot_id`。
- 每辆车 armor observations 数量与字段。
- selected armor。
- `SingleDetectionResult` 除统一时间戳语义外的字段。
- 最终 10 个 `TrackingState` 的身份、来源、有效性和坐标。

整数、枚举、bot_id、列表长度和离散状态必须完全相等。模型置信度、投影坐标、KF 状态和其他浮点字段使用 `numpy.testing.assert_allclose(rtol=1e-5, atol=1e-5, equal_nan=True)`；不能为了通过测试降低业务阈值或扩大该容差。`1e-5` 来源是本次只重排调用边界、不改变数值算法，同一硬件上的差异应仅来自 float32/float64 表达和容器转换，而不应接近业务阈值尺度。

### 12.2 邮箱确定性测试

必须覆盖：

1. 邮箱空时发布/取走。
2. pending N 被 N+1 原子覆盖。
3. Stage B 在途 N 时 pending N+1 被 N+2 覆盖。
4. 被覆盖包不会进入 Stage B。
5. source_sequence 跳号但严格递增。
6. 多次 wakeup 不产生重复包。
7. stop 能唤醒空等待。
8. epoch 变化清空 pending，在途旧 epoch 不能提交。

该测试不得加载 GPU 模型。

### 12.3 无车辆与异常测试

必须覆盖：

- 无车辆包仍调用一次 `process_observations`。
- 车辆有框但没有 armor。
- armor 整批异常时保留车辆框和 bot_id，并按无 armor 处理。
- 陈旧包在 armor 前丢弃。
- armor 后、状态事务前变陈旧时不调用 `process_observations`，也不提交。
- stop 后 in-flight 返回不提交。
- source_sequence 小于等于最后提交序号时拒绝提交。

### 12.4 回放集成测试

使用现有 replay 流程验证：

- 无死锁。
- Stage A/B 线程持续存活。
- frame/source/可视化时间戳一致。
- UI 不把旧检测框画到新图。
- BotSort ID 不因线程化出现同帧重复调用。
- `process_observations` 始终单线程。
- 裁判输出使用最新一次成功提交结果。
- 停止可在 `worker_join_timeout_sec` 内完成。

---

## 13. 性能验收与决策规则

### 13.1 测量方法

- 使用与基线相同的机器、模型、配置和回放输入。
- 丢弃至少前 100 个成功提交帧作为预热。
- 至少统计后续 1000 个成功提交帧。
- 同时报告均值、P50、P90、P95，不只报告单帧最好值。
- 主要指标：
  - 成功提交 FPS；
  - `processing_age_ms`；
  - Stage A/B 耗时；
  - pending 覆盖率；
  - 陈旧丢弃率；
  - GPU 利用率和显存；
  - 业务输出数量与异常率。

### 13.2 判定

与当前约 16 FPS 基线相比：

| 稳定吞吐提升 | 结论 |
|---|---|
| `>= 15%` | 第一版调度流水线具有明确价值，继续保持默认启用并进行长时间稳定性测试 |
| `5% ～ 15%` | 功能实现可接受，保持本文档规定的 `enabled=true`，但报告中必须注明收益有限并建议后续分析 GPU 时间线 |
| `< 5%` | 认为现有同步 TensorRT/默认 Stream 基本没有形成有效 GPU 重叠；仍保持本文档规定的 `enabled=true`，不得自动改配置或擅自改后端，必须在交付报告中明确建议用户人工选择 `enabled=false` 回退 |

判定参数来源：

- `15%` 是“明确收益”门限：理论理想吞吐提升约 35%，取不足其一半的 15% 作为第一版调度仍值得保留的保守下限。
- `5%` 是“基本无有效重叠”门限：低于 5% 的变化容易被 GPU 温度、回放调度和输入目标数量波动覆盖，不能证明两级 GPU 推理有效并发。
- 丢弃前 100 帧来源于现有基线统计同样使用 `frame_id > 100` 排除预热。
- 后续至少 1000 帧来源于当前基线有效样本 1262 帧；选择 1000 保证覆盖足够多的目标数量与调度变化，同时保持一次测试可执行。

附加要求：

- 正常基线场景 pending 覆盖率应 `<= 1%`。来源：Stage B 实测约 16 ms，显著短于 Stage A 约 44 ms；若仍频繁覆盖，说明 GPU 争用、线程调度或实现存在问题。
- 基线 `processing_ms` P95 为 `75.561 ms`，来源与 1.3 节相同。流水线成功提交包的 `processing_age_ms` P95 必须 `<= 85.561 ms`，即只允许最多 `10 ms` 的线程调度/提交额外预算。`max_commit_age_sec=0.20 s` 按 10.1 节定义在两个安全点检查；已进入状态事务的包允许因事务本身耗时在最终指标中略微超过 200 ms。
- 任何 bot_id 乱序、融合状态并发、UI 帧错配、死锁或静默线程退出都属于功能失败，无论 FPS 多高。

---

## 14. CUDA/TensorRT 边界说明

当前本机 Ultralytics 8.4.7 TensorRT 后端调用：

```python
self.context.execute_v2(list(self.binding_addrs.values()))
```

该接口是同步调用，并且当前项目没有为车辆和 armor 显式创建不同 CUDA Stream。因此：

- 两个 Python 线程可以重叠 CPU resize、裁剪、结果解析和等待。
- 不能仅凭线程数量断言车辆与 armor GPU kernel 已并发。
- `.cpu().numpy()` 只保证调用方需要的数据完成，不应在需求中错误描述为必然 `cudaDeviceSynchronize()`。
- 是否形成 GPU 重叠必须通过实测时间线判断。

第一版严格禁止：

- 修改当前 Python 环境中的 `site-packages/ultralytics`。
- monkey patch `AutoBackend`。
- 自行替换 `execute_v2`。
- 为同一个 engine/context 从多个线程并发调用。

若第一版性能不足，后续任务应单独设计：

```text
每个模型独立 TensorRT execution context
独立 CUDA Stream
execute_async_v3/兼容接口
CUDA Event 依赖和输出生命周期
PyTorch digit 分类 Stream
预处理/后处理 Stream 归属
正确的性能与数值一致性验证
```

不得把这部分隐式夹带进本次两级调度流水线。

---

## 15. 实现完成清单

实现者提交代码前必须逐项确认：

- [ ] 新增完整 `detection_pipeline` 配置和严格校验。
- [ ] 拆分 Stage A/Stage B，同时保留同步 `detect()` 组合接口。
- [ ] Stage A 只有现有主循环线程一个调用者。
- [ ] Stage B 只有一个 daemon 工作者。
- [ ] 使用单槽 latest-wins mailbox，不使用阻塞 FIFO。
- [ ] pending 覆盖原子且有计数/事件记录。
- [ ] frame、faction、雷达、己方、飞坡点、前哨站状态全部随包传递。
- [ ] 严格区分 frame_stamp、ingress_monotonic、observation_monotonic。
- [ ] 状态机使用 observation_monotonic。
- [ ] source_sequence、commit_sequence、tracker.frame_count 语义正确。
- [ ] 无车辆仍推进 Stage B 状态机。
- [ ] armor 异常保留车辆和 bot_id。
- [ ] Stage A/B 不并发修改 timing 字典。
- [ ] 输出先形成 CompletedFrame，再原子提交。
- [ ] UI/可视化使用提交包自己的原图和时间戳。
- [ ] epoch 重置和 stop 能使旧在途包失效。
- [ ] stop 不排空 pending，不无限等待 GPU 工作者。
- [ ] 串行配置回退可运行。
- [ ] 未修改 Ultralytics/TensorRT/CUDA 后端。
- [ ] 完成同步拆分等价性测试。
- [ ] 完成 mailbox CPU 测试。
- [ ] 完成 replay 集成和性能测试。
- [ ] 报告吞吐、P50/P90/P95 延迟、丢包率、异常数和 GPU 指标。

只有全部满足后，才可以认为“车辆检测—装甲识别小范围两级流水线”实现完成。

---

## 16. “跟踪定位耗时调试”详情窗口的最终展示要求

本节属于本任务的强制验收内容，不是可选 UI 优化。

### 16.1 窗口布局

点击“详情”后必须至少有两个视觉上独立的区域：

1. “详细管线流程（模块编号与计时覆盖）”：固定按 01–12 的执行语义展示模块、A/B 阶段边界、总时间覆盖关系和最新一次成功提交的分模块耗时。
2. “实时耗时统计”：展示每项 `last / EMA平均 / 历史最大`，允许滚动查看历史统计。

“拷贝到剪贴板”必须同时复制两个区域，不能只复制实时统计。

### 16.2 唯一模块编号与易懂名称

详情窗口使用以下唯一编号；内部旧名字可以为兼容保留，但对用户显示时必须转换：

```text
01 获取最新相机帧
02 前哨站判断 + 雷达/己方/飞坡点同源快照
03 车辆神经网络推理 + BoT-SORT（含 GPU→CPU 结果回传）
04 车辆 ROI 裁剪
05 latest-wins 邮箱排队等待
06 装甲板位置神经网络推理
07 装甲板 ROI 裁剪 + 数字神经网络输入预处理
08 数字神经网络推理
09 装甲结果组装、车辆底边投影、跟踪观测构建
10 相机/雷达跟踪融合状态机
11 不可变输出快照与原子提交
12 异步可视化、TF、裁判消息和性能指标发布
```

命名规则：

- 任何调用神经网络模型的计时名称必须显式含“神经网络推理”。
- 车辆模型的 `model.track()` 必须显示为“车辆神经网络推理 + BoT-SORT”，不能只写 `UltralyticsTrack`。
- armor YOLO 必须显示为“装甲板位置神经网络推理”。
- digit classifier 必须显示为“数字神经网络推理”。
- 每个合计项必须在名称中写明“模块 N 总时间”或“模块 N–M 总时间”；细分项写明“模块 N 细分”。
- `last/EMA平均/历史最大` 的单位统一为毫秒，标题必须明确三者顺序。

编号必须使用点分层，例如 `03 → 03.2 → 03.2.4 → 03.2.4.1`。每个大模块都必须在“详细管线流程”中继续展开，直到对应一段不再拆分计时的原子操作。层级编号不得用 `03/xxx` 或无编号缩进文本替代。

计时覆盖标记固定为：

- `[计时/原子]`：该原子模块有独立数值。
- `[计时]`：该父模块有总数值，但子步骤没有全部独立插桩。
- `[父级计时/原子]`：该原子步骤的耗时只计入最近父模块，UI 不得伪造独立数值。

上述方括号必须在冒号后同时标明主要执行位置，例如 `[计时/原子: CPU]`、`[计时/原子: GPU]`、`[计时: CPU+GPU]`、`[计时/原子: CPU→GPU]` 或 `[计时/原子: GPU→CPU]`。设备标记规则为：

- `CPU`：主要为 Python、NumPy、OpenCV CPU、PIL、Kalman、线性分配、数据结构或消息处理。
- `GPU`：主要为 TensorRT/CUDA/PyTorch CUDA 张量操作。
- `CPU+GPU`：父模块内同时含 CPU 调度和 GPU 工作，禁止强行标成单一设备。
- `CPU→GPU` / `GPU→CPU`：跨设备传输；墙钟时间可包含对之前 CUDA 任务的同步等待，不代表纯复制带宽耗时。
- 大模块标题也必须使用 `[模块: XXX]` 标明设备，不只标原子行。

BoT-SORT 的设备标记必须与当前配置一致：`with_reid=false` 时 `BOTSORT.update` 为 CPU；若以后重新启用基于 CUDA 网络的 ReID，必须改为 `CPU+GPU→CPU`，并明确该耗时包含 ReID 神经网络推理和特征回传。

模块 03 的最小强制计时项为：

```text
03.1.1 项目侧原图缩放
03.1.2 RGB→BGR共享视图
03.2.1 Ultralytics内部预处理总时间
03.2.2 车辆神经网络TensorRT推理
03.2.3 YOLO后处理总时间
03.2.4 BoT-SORT回调总时间
03.2.4.1 BOTSORT.update纯跟踪更新
03.2.4.2 跟踪输入CPU转换与结果写回
03.2.5 Ultralytics框架墙钟余量
03.3.1 跟踪结果张量合并与CPU转换
03.3.2 坐标还原与Python结果组装
```

`03.2.1` 必须在参考区内展开 LetterBox、batch/通道与布局转换、连续内存与 Tensor 构造、CPU→GPU、精度转换与归一化。`03.2.3` 必须展开置信度/类别过滤与 NMS、坐标缩放、`Results/Boxes` 构造。`03.2.4.1` 必须继续展开高低置信度分组、Kalman/GMC、第一/第二轮关联、未确认轨迹、新轨迹/丢失/删除状态、去重与输出。模块 06 的 Ultralytics 预处理和 YOLO 后处理也必须在其自身 `06.x` 编号下完整展开，不能只写“同 03”。

### 16.2.1 参考区刷新与滚动位置

管线参考区允许随最新提交帧更新动态数值，但任何定时刷新都必须保留用户当前的垂直和水平滚动位置：

- 用户位于中间时，刷新后恢复刷新前的绝对滚动值，只在新最大值更小时做上限裁剪。
- 用户已在底部时，刷新后继续跟随新底部。
- 首次填充文本时从顶部开始。
- 文本未变化时不得重设 `QTextEdit` 内容。

“实时耗时统计”区域执行相同的滚动位置保留规则。

实时耗时词条必须按数字模块编号稳定排序，不得按 `last/EMA/最大值` 排序，否则词条会随耗时波动而跳动。排序规则为：

- 主模块按 `01 → 02 → … → 12` 排列。
- 同一主模块按数字层级排列，例如 `10.2` 必须位于 `10.10` 之前。
- 父模块位于子模块之前，范围合计项按结束模块排在该模块细分之后。
- 不含模块编号的兼容指标放在编号模块之后，使用稳定文本顺序。

主循环、检测器、跟踪器和异步可视化中的实时统计必须先按用户可见名称合并再显示，同一词条只允许出现一次。父模块与子模块、不同覆盖范围的合计不属于重复；完全同名项、旧兼容合计与新编号合计覆盖相同调用时属于重复，必须去除。“最慢模块”摘要仍允许按 EMA 耗时判定，不影响下方词条顺序。

### 16.3 总时间与 FPS 的唯一口径

必须并列展示以下数据，不得只显示一个“总耗时/FPS”：

```text
A 阶段线程周期 = 模块 01–04
B 阶段处理时间 = 模块 06–11
邮箱等待 = Stage B 取包 monotonic - published_monotonic
单帧端到端处理延迟 = 模块 01 取得有效帧后的 ingress_monotonic 到模块 11 提交
原始提交间隔 = 相邻两次模块 11 成功提交的 monotonic 差
稳定提交间隔 = 原始提交间隔的 EMA
原始 commit_fps = 1000 / 原始提交间隔毫秒
界面稳定吞吐 FPS = 1000 / 稳定提交间隔毫秒
跨帧重叠节省估算 = max(0, A有效帧处理时间 + B处理时间 - 原始提交间隔)
```

边界说明：

- 模块 01 的相机阻塞等待会计入“A 阶段线程周期”，但发生在有效帧 `ingress_monotonic` 之前，因此不计入该帧“端到端处理延迟”。详情窗口必须把这两个口径分开描述。
- 第一次成功提交没有原始提交间隔，指标写 `null`，界面显示“首帧暂无”，不得显示无穷大或用单帧延迟倒推 FPS。
- EMA 系数复用 `debug.timing_ema_alpha`，默认 `0.2`；来源是现有耗时调试配置，不新增重复参数。
- 主界面的“定位帧率”使用 `commit_fps_ema`；`processing_fps=1000/processing_age_ms` 只为兼容旧指标分析器保留，禁止把它称为流水线吞吐。
- 丢弃包不更新提交间隔、EMA 或定位 FPS。

### 16.4 最新状态与累计诊断

独立管线区域还必须展示：

- 当前运行模式：两级流水线或兼容串行。
- 当前 `pipeline_epoch`。
- 最新 `source_sequence → commit_sequence`。
- 最新单帧延迟、原始/EMA 提交间隔、原始/EMA FPS。
- `published/taken/committed/pending_replaced` 累计值。
- 两个 stale 丢弃计数以及 stage A/B/armor/tracker/commit 异常计数，可在实时统计区或指标文件完整展示。

UI 的场地投影、检测结果导出和可视化必须从一次 `get_latest_pipeline_output()` 得到的同一个 `CompletedFrame` 读取。禁止分别读取公共字段拼出“车辆框来自帧 N、定位点来自帧 N-1”的混合画面。
