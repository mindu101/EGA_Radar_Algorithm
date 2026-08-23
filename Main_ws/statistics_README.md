# 相机投影点—雷达点可靠匹配残差统计说明

## 1. 功能目标

本功能用于从实时定位链路中筛选出一部分可信度较高、局部不易混淆的“相机投影点—雷达点”直接匹配，并在线统计两者在场地二维坐标系中的残差：

```text
delta = lidar_xy - camera_projection_xy
      = (delta_x, delta_y)
```

所有坐标和残差的单位均为米。这里的正负号定义固定为“雷达减相机”：

- `delta_x > 0`：雷达点的 x 比相机投影点的 x 大。
- `delta_y > 0`：雷达点的 y 比相机投影点的 y 大。
- 若要用统计均值补偿相机投影点，应当使用 `camera_projection_xy + mean_delta`，而不是减去均值。

统计结果包括：

- 全部机器人合并后的全局样本数、均值、样本协方差和标准差。
- 蓝方 5 个机器人合并后的阵营统计。
- 红方 5 个机器人合并后的阵营统计。
- B1、B2、B3、B4、B7、R1、R2、R3、R4、R7 共 10 个 class 各自的样本数、均值、样本协方差和标准差。
- 最近一次被接受样本的残差、匹配距离和双向歧义比例。
- 所有被考察匹配的接受数、拒绝数和首个拒绝原因计数。

统计仅保存在当前进程内，不保存历史样本，也不写入文件。

## 2. 相关代码与配置

主要实现位置如下：

- `tracker/CascadeMatchTracker/projection_lidar_statistics.py`
  - `ProjectionLidarStatistics`：可靠性筛选、线程保护、清零、快照和文本输出。
  - `_OnlineResidualStatistics`：二维 Welford 在线均值和协方差统计。
- `tracker/CascadeMatchTracker/tracker.py`
  - 在彩色三元组匹配和灰色身份维持匹配提交后，把匹配结果交给统计器。
  - 提供统计清零、快照和格式化文本接口。
- `main_event_loop.py`
  - 负责在外参、阵营、XY Maps 等坐标条件变化时清零。
  - 把统计文本追加到“跟踪定位耗时调试”文本中。
- `config/params.yaml`
  - `projection_lidar_statistics` 节点保存本功能的全部业务阈值和开关。
- `tests/test_projection_lidar_statistics.py`
  - 验证在线统计、采样间隔、互为最近邻、比例定义、灰色来源开关和清零。
- `tests/test_cascade_match_tracker.py`
  - 验证实际追踪器接入和阵营变化清零。

## 3. 数据进入统计器的时机

每一帧追踪处理的大致顺序为：

1. 获取本帧雷达点和相机车辆观测。
2. 用新鲜的己方哨兵反馈匹配并剔除确定属于己方的雷达点。
3. 保存统计可靠性检查使用的候选池：
   - 雷达候选池：当前帧全部雷达点中，排除已被新鲜己方哨兵信息占用的雷达点。
   - 相机候选池：当前帧所有已经成功产生二维投影点的 car。
4. 构建并求解彩色 `class_id + car + lidar` 三元组匹配。
5. 提交彩色匹配，更新追踪状态。
6. 将已经提交的彩色直接匹配作为 `CAMERA_LIDAR` 候选交给统计器。
7. 构建、求解并提交灰色装甲板的身份维持匹配。
8. 将灰色直接匹配作为 `GRAY_CAMERA_LIDAR` 候选交给统计器；是否允许它进入统计由配置决定。
9. 继续执行历史雷达匹配、灰色模糊猜测、纯雷达猜点和丢失处理。这些结果不会进入本统计。

统计器不会重新决定哪一对点应该匹配。它只对状态机已经提交的直接匹配进行第二层、更严格的可靠性筛选。

## 4. 会参与和不会参与统计的数据来源

### 4.1 默认参与

- `CAMERA_LIDAR`：有彩色装甲板身份依据，并完成相机投影点和雷达点直接匹配。

### 4.2 可选参与

- `GRAY_CAMERA_LIDAR`：灰色/死亡装甲板依靠已经确认的身份历史完成的相机—雷达直接匹配。
- 默认关闭，可设置 `include_gray_camera_lidar: true` 开启。

### 4.3 不参与

以下来源即使最终输出了有效坐标，也不会被送入统计器：

- `ALLY_LIDAR`
- `ALLY_FEEDBACK`
- `LIDAR_HISTORY`
- `GRAY_RADAR_GUESSED`
- `RADAR_GUESSED`
- `KF_PREDICTED`
- `FIXED_GUESSED`
- `INVALID`

原因是这些结果缺少“本帧相机投影点和本帧雷达点的直接对应关系”，或身份本身仍属于猜测，不能作为投影误差的可靠样本。

## 5. class 映射

统计器只维护前 10 个确定阵营的 class：

| class_id | 名称 | 含义 |
|---:|---|---|
| 0 | B1 | 蓝方英雄 |
| 1 | B2 | 蓝方工程 |
| 2 | B3 | 蓝方 3 号步兵 |
| 3 | B4 | 蓝方 4 号步兵 |
| 4 | B7 | 蓝方哨兵 |
| 5 | R1 | 红方英雄 |
| 6 | R2 | 红方工程 |
| 7 | R3 | 红方 3 号步兵 |
| 8 | R4 | 红方 4 号步兵 |
| 9 | R7 | 红方哨兵 |

G1～G7 是装甲板检测阶段的灰色语义类别，不是独立的二维轨迹。灰色观测只有在维持到某个已经确认的红/蓝 class 后，才可能计入该红/蓝 class 的统计。

## 6. 当前配置与默认阈值

配置位于 `config/params.yaml`：

```yaml
projection_lidar_statistics:
  enabled: true
  include_gray_camera_lidar: false
  min_armor_confidence: 0.70
  min_digit_confidence: 0.70
  min_combined_confidence: 0.75
  max_match_distance_m: 0.75
  min_lidar_margin_m: 0.50
  min_camera_margin_m: 0.50
  max_lidar_ambiguity_ratio: 0.50
  max_camera_ambiguity_ratio: 0.50
  min_sample_interval_sec: 0.20
```

各参数含义如下：

| 参数 | 当前值 | 接受条件 | 含义 |
|---|---:|---|---|
| `enabled` | `true` | 必须为 `true` | 统计总开关。关闭后不增加评估、接受或拒绝计数。 |
| `include_gray_camera_lidar` | `false` | 灰色来源需要为 `true` | 是否允许 `GRAY_CAMERA_LIDAR` 进入统计。 |
| `min_armor_confidence` | `0.70` | `armor_confidence >= 0.70` | 装甲板检测置信度下限。 |
| `min_digit_confidence` | `0.70` | `digit_confidence >= 0.70` | 数字分类置信度下限。 |
| `min_combined_confidence` | `0.75` | `combined_confidence >= 0.75` | 数字和装甲板加权综合置信度下限。 |
| `max_match_distance_m` | `0.75` | `d1 <= 0.75 m` | 当前匹配的投影点—雷达点最大欧氏距离。 |
| `min_lidar_margin_m` | `0.50` | `d_lidar_2 - d1 >= 0.50 m` | 匹配雷达点相对其他雷达候选必须具有足够绝对优势。 |
| `min_camera_margin_m` | `0.50` | `d_camera_2 - d1 >= 0.50 m` | 匹配相机点相对其他相机候选必须具有足够绝对优势。 |
| `max_lidar_ambiguity_ratio` | `0.50` | `d1 / d_lidar_2 <= 0.50` | 雷达候选方向的相对歧义上限。越小越可靠。 |
| `max_camera_ambiguity_ratio` | `0.50` | `d1 / d_camera_2 <= 0.50` | 相机候选方向的相对歧义上限。越小越可靠。 |
| `min_sample_interval_sec` | `0.20` | 同一 class 距上次接受样本至少 `0.20 s` | 降低连续视频帧强相关造成的重复采样。 |

配置加载时会检查：

- 三个置信度和两个 ratio 必须位于 `[0, 1]`。
- `max_match_distance_m` 必须大于 0。
- 两个 margin 和采样间隔必须大于等于 0。
- 两个开关必须是 YAML 布尔值。

## 7. 综合置信度的来源

统计器不会重新计算综合置信度，而是读取本帧选中装甲板 `selected_armor` 中已经计算好的三个值：

```text
combined_confidence
    = digit_confidence_weight * digit_confidence
    + armor_confidence_weight * armor_confidence
```

当前 `config/params.yaml` 中：

```yaml
digit_detector:
  digit_confidence_weight: 0.70
  armor_confidence_weight: 0.30
```

因此当前公式为：

```text
combined_confidence = 0.70 * digit_confidence
                    + 0.30 * armor_confidence
```

需要区分两类阈值：

- `digit_detector` 和普通检测器配置中的阈值，决定观测能否进入匹配链路以及选择哪个装甲板。
- `projection_lidar_statistics` 中的阈值，只决定已经完成的匹配能否成为残差统计样本。

统计阈值默认更严格。降低普通检测阈值不会自动降低统计阈值。

## 8. 完整筛选算法

统计器按照以下固定顺序检查一对已经提交的匹配。一个样本在遇到第一个不满足的条件时立即被拒绝，并且只给该首个拒绝原因加一。

### 8.1 总开关

若 `enabled == false`，直接返回，不增加 `evaluated`，也不记录拒绝原因。

### 8.2 来源检查

若来源为 `GRAY_CAMERA_LIDAR`，但 `include_gray_camera_lidar == false`，拒绝并记为“灰色来源未启用”。

彩色 `CAMERA_LIDAR` 不受该开关影响。

### 8.3 身份状态检查

对应二维轨迹的 `IdentityState` 必须为 `CONFIRMED`。猜测身份会被拒绝并记为“猜测身份”。

彩色三元组匹配在提交时会把身份置为 `CONFIRMED`。灰色身份维持匹配只允许维持已有确认身份，不应凭灰色观测建立一个新身份。

### 8.4 装甲板观测检查

car 必须具有 `selected_armor`。没有选中装甲板时拒绝并记为“缺少装甲观测”。

随后依次检查：

```text
armor_confidence    >= min_armor_confidence
digit_confidence    >= min_digit_confidence
combined_confidence >= min_combined_confidence
```

边界值相等时允许通过。

### 8.5 class 和坐标合法性检查

- `class_id` 必须位于 `[0, 9]`。
- 投影点和雷达点必须都能转换为二维 `float64` 数组。
- 两个二维点的 x、y 必须全部为有限数，不能含 `NaN` 或正负无穷。

任何一项失败均记为“坐标非法”。

### 8.6 当前匹配距离检查

设：

```text
C = 当前相机投影点
L = 当前匹配雷达点
d1 = ||L - C||₂
```

必须满足：

```text
d1 <= max_match_distance_m
```

当前默认要求 `d1 <= 0.75 m`。

### 8.7 构造双向第二候选距离

对当前匹配 `(C, L)`，分别在两个方向寻找竞争点。

雷达侧第二候选距离：

```text
d_lidar_2 = min ||L_other - C||₂
```

其中 `L_other` 遍历统计雷达候选池中除当前 `L` 以外的其他雷达点。

相机侧第二候选距离：

```text
d_camera_2 = min ||L - C_other||₂
```

其中 `C_other` 遍历本帧所有成功产生二维投影的其他 car 投影点。

注意：这里使用的是完整的相关候选池，而不是只在最终被匈牙利算法选中的配对中寻找第二候选。这样可以发现“另一个点虽然最终没有被选中，但局部上其实很容易混淆”的情况。

如果某个方向没有任何其他候选，该方向的第二距离记为正无穷。

### 8.8 互为最近邻检查

当前匹配必须在两个方向上都不存在更近的竞争点：

```text
d_lidar_2 >= d1 - 1e-9
d_camera_2 >= d1 - 1e-9
```

`1e-9` 仅用于吸收浮点计算误差，不是业务可调阈值。

若另一个雷达点离当前相机投影点更近，或者另一个相机投影点离当前雷达点更近，均拒绝并记为“非互为最近邻”。

距离完全相等不会在这一阶段直接拒绝，但通常会被后续 margin 或 ratio 判据拒绝。

### 8.9 绝对 margin 检查

存在第二候选时，要求：

```text
lidar_margin  = d_lidar_2  - d1 >= min_lidar_margin_m
camera_margin = d_camera_2 - d1 >= min_camera_margin_m
```

当前两个默认下限均为 `0.50 m`。

margin 衡量的是绝对距离优势。例如：

```text
d1 = 0.20 m
d_lidar_2 = 0.80 m
lidar_margin = 0.60 m
```

该方向可以通过默认 margin 阈值。

若某个方向没有竞争点，则该方向不执行 margin 拒绝。

### 8.10 相对歧义比例检查

两个方向分别计算：

```text
lidar_ratio  = d1 / d_lidar_2
camera_ratio = d1 / d_camera_2
```

接受条件为：

```text
lidar_ratio  <= max_lidar_ambiguity_ratio
camera_ratio <= max_camera_ambiguity_ratio
```

当前默认最大值均为 `0.50`。

特殊情况：

- 没有竞争点，即第二距离为正无穷时，ratio 定义为 `0.0`。
- `d1 <= 0` 时，ratio 定义为 `0.0`。
- 第二距离异常地小于等于 0 时，ratio 定义为 `1.0`。
- 正常结果会限制在 `[0, 1]` 内，以抑制浮点误差越界。

ratio 衡量相对优势。数值越接近 0，当前匹配相对第二候选越明显；越接近 1，两个候选越难区分。

绝对 margin 和相对 ratio 同时存在是有意设计：

- margin 避免在距离都很小时，比例看起来较好但绝对间隔仍太小。
- ratio 避免在距离都很大时，绝对间隔看起来足够但相对区分度仍不足。

### 8.11 同 class 采样间隔检查

统计器记录每个 class 上一次被接受样本的单调时钟时间。必须满足：

```text
current_time - last_accepted_time[class_id]
    >= min_sample_interval_sec
```

当前默认间隔为 `0.20 s`，因此单个 class 理论上最多约以 5 Hz 进入统计。

采样间隔按 class 独立维护。B1 刚接受一个样本不会阻止 R2 同时接受样本。

只有最终被接受的样本才会更新时间戳；被其他条件拒绝的样本不会延后下次采样。

### 8.12 接受样本并更新统计

通过上述所有条件后，计算：

```text
delta = lidar_xy - camera_projection_xy
```

同一个 `delta` 同时更新：

- 全局统计器。
- 对应阵营的统计器：`class_id` 为 0～4 时更新蓝方总计，为 5～9 时更新红方总计。
- 对应 `class_id` 的独立统计器。

随后更新该 class 的最后接受时间，并将 `accepted` 加一。

## 9. 在线均值和协方差算法

### 9.1 为什么使用 Welford 算法

程序不会保存全部历史残差。若直接维护 `sum(x)` 和 `sum(x²)`，在样本较多、均值较大而方差较小时可能出现浮点消减误差。Welford 算法可以在常量内存下更稳定地更新均值和协方差。

内部数据全部使用 NumPy `float64`：

```text
n       样本数
mean    形状为 (2,) 的均值向量
M2      形状为 (2, 2) 的离差乘积累积矩阵
```

### 9.2 每个新样本的更新公式

设新样本为二维向量：

```text
x = (delta_x, delta_y)
```

旧状态为 `n_old`、`mean_old`、`M2_old`，更新过程为：

```text
n_new = n_old + 1
difference = x - mean_old
mean_new = mean_old + difference / n_new
difference_after = x - mean_new
M2_new = M2_old + outer(difference, difference_after)
```

其中 `outer(a, b)` 是两个二维向量的外积，结果为 2×2 矩阵。

### 9.3 均值

输出均值为：

```text
mean = (mean_delta_x, mean_delta_y)
```

它表示当前可靠样本集合中，雷达点相对相机投影点的平均偏移。

### 9.4 样本协方差

当 `n >= 2` 时，输出无偏样本协方差：

```text
covariance = M2 / (n - 1)
```

矩阵布局为：

```text
[[var(delta_x),          cov(delta_x, delta_y)],
 [cov(delta_y, delta_x), var(delta_y)         ]]
```

理论上两个非对角元素相等，仅可能因浮点误差出现极小差别。

当 `n < 2` 时，样本协方差没有定义，UI 显示 `N/A`，不会用全零矩阵冒充已有协方差。

### 9.5 标准差

当 `n >= 2` 时：

```text
std_x = sqrt(max(covariance[0, 0], 0))
std_y = sqrt(max(covariance[1, 1], 0))
```

对角元素先与 0 取最大值，是为了避免极小的浮点负数导致开方异常。

### 9.6 内存与计算复杂度

每个统计器只保存样本数、二维均值、2×2 的 `M2` 和最近样本信息：

- 不保存历史残差序列。
- 单次接受更新的时间复杂度为 `O(1)`。
- 全局统计、蓝/红两个阵营统计和 10 个 class 统计占用固定内存。

可靠性筛选中寻找双向第二候选需要遍历本帧候选点，但当前机器人和聚类点数量很小，不会产生与历史样本数相关的开销。

## 10. 统计快照字段

`snapshot()` 返回内存状态的副本，主要字段为：

```text
enabled                 是否启用
evaluated               被统计器实际考察的匹配数量
accepted                最终接受数量
rejected                evaluated - accepted
rejections              按首个拒绝原因分类的计数
reset_count             本对象创建后的清零次数
last_reset_reason       最近一次清零原因
global                  全局统计
by_faction              blue/red 两个阵营的独立统计
by_class                10 个 class 的独立统计
```

`global`、每个 `by_faction` 项和每个 `by_class` 项都包含：

```text
count                    样本数
mean                     二维均值
covariance               2×2 样本协方差；样本不足时为 None
standard_deviation       二维标准差；样本不足时为 None
last_delta               最近接受残差
last_match_distance      最近接受匹配的 d1
last_lidar_ratio         最近接受匹配的 d1/d_lidar_2
last_camera_ratio        最近接受匹配的 d1/d_camera_2
```

快照和更新受同一个线程锁保护，UI 读取时不会获得更新到一半的均值或协方差。

## 11. UI 输出说明

统计文本被追加到 UI 的“跟踪定位耗时调试”区域，同时也会出现在“详情”按钮打开的窗口中。

示意格式：

```text
投影-雷达可靠匹配残差统计（单位 m，delta=lidar-camera）
评估/接受/拒绝: 120/35/85，最近清零: 启动雷达站

所有机器人总计: n=35
  mean=(+0.1234, -0.0567)
  cov=[[0.012345, 0.001234],
       [0.001234, 0.023456]]
  std=(+0.1111, +0.1532)
  last=(+0.0800, -0.0300) d=0.0854 ratio(lidar/camera)=0.120/0.090

蓝方总计: n=18
  ...

蓝方机器人:
  B1: n=8
    mean=(...)
    cov=[[..., ...],
         [..., ...]]

  B2: n=3
  ...

================================

红方总计: n=17
  ...

红方机器人:
  R1: n=4
    ...

拒绝原因:
  采样间隔未到: 40
  相机侧margin不足: 20
  ...
```

其中：

- `评估`：进入统计器并开始检查的直接匹配数量。
- `接受`：通过全部条件并更新均值/协方差的数量。
- `拒绝`：评估减接受。
- `n`：对应统计器的有效样本数，不是检测帧数。
- `所有机器人总计`：蓝方和红方全部接受样本的合并统计。
- `蓝方总计`：B1、B2、B3、B4、B7 接受样本的合并统计。
- `红方总计`：R1、R2、R3、R4、R7 接受样本的合并统计。
- `mean`：平均 `(delta_x, delta_y)`。
- `cov`：二维样本协方差矩阵。为了便于阅读，矩阵的两行分别占一行文本。
- `std`：x、y 两个方向的标准差。
- `last`：该统计器最近一次接受的残差。
- `d`：最近一次接受匹配的欧氏距离 `d1`。
- `ratio(lidar/camera)`：最近一次接受匹配的两个歧义比例。

如果关闭 `projection_lidar_statistics.enabled`，该区域显示“投影-雷达可靠匹配残差统计已关闭”。

## 12. 拒绝原因

当前拒绝原因及含义如下：

| UI 原因 | 内部键 | 含义 |
|---|---|---|
| 灰色来源未启用 | `gray_disabled` | 灰色直接匹配未被配置允许。 |
| 猜测身份 | `guessed_identity` | 对应 class 不是确认身份。 |
| 缺少装甲观测 | `missing_armor` | car 没有 `selected_armor`。 |
| armor置信度不足 | `armor_confidence` | 装甲板检测置信度低于阈值。 |
| digit置信度不足 | `digit_confidence` | 数字置信度低于阈值。 |
| combined置信度不足 | `combined_confidence` | 综合置信度低于阈值。 |
| 坐标非法 | `invalid_point` | class 越界、点格式错误、NaN 或无穷。 |
| 匹配距离过大 | `match_distance` | `d1` 大于最大距离。 |
| 非互为最近邻 | `not_mutual_nearest` | 任一方向存在比当前配对更近的候选。 |
| 雷达侧margin不足 | `lidar_margin` | `d_lidar_2-d1` 太小。 |
| 相机侧margin不足 | `camera_margin` | `d_camera_2-d1` 太小。 |
| 雷达侧ratio过大 | `lidar_ratio` | `d1/d_lidar_2` 太大。 |
| 相机侧ratio过大 | `camera_ratio` | `d1/d_camera_2` 太大。 |
| 采样间隔未到 | `sample_interval` | 同 class 距上次接受样本时间太短。 |

由于只记录首个失败条件，拒绝计数适合判断当前主要瓶颈，但不能直接解释一条样本同时违反了多少项条件。

## 13. 自动清零条件

下列事件会清空所有统计样本和计数：

1. 点击并启动雷达站主循环，原因显示为“启动雷达站”。
2. 运行中有效阵营发生红蓝切换，原因显示为“阵营变化”。
3. 导入或更新了与当前值不同的相机外参，原因显示为“外参更新”。
4. XY Maps 从关闭切换为开启，原因显示为“XY Maps 开启”。
5. XY Maps 从开启切换为关闭，原因显示为“XY Maps 关闭”。
6. XY Maps 重新生成成功，原因显示为“XY Maps 重新生成完成”。
7. XY Maps 生成失败并回退直接投影，原因显示为“XY Maps 生成失败并回退直接投影”。

清零时会重置：

- 全局和全部 class 的样本数、均值、`M2`、最近样本信息。
- 蓝方总计和红方总计的样本数、均值、`M2`、最近样本信息。
- 每个 class 的最后接受时间。
- `evaluated`、`accepted` 和拒绝原因计数。

清零不会重置机器人追踪状态；统计清零和轨迹状态清零是不同操作。阵营切换自身还会按追踪器原逻辑处理空间状态。

外参传入值与当前外参完全一致时，程序会直接返回，不重复清零。仅刷新 UI、打开统计详情窗口或切换耗时调试显示不会清零。

## 14. 阈值关系示例

假设当前配对距离和竞争距离为：

```text
d1 = 0.20 m
d_lidar_2 = 0.80 m
d_camera_2 = 1.00 m
```

则：

```text
lidar_margin = 0.80 - 0.20 = 0.60 m
camera_margin = 1.00 - 0.20 = 0.80 m
lidar_ratio = 0.20 / 0.80 = 0.25
camera_ratio = 0.20 / 1.00 = 0.20
```

在当前默认阈值下，距离、margin 和 ratio 均通过。

另一个例子：

```text
d1 = 0.30 m
d_lidar_2 = 0.55 m
```

此时当前点仍然是最近点，但：

```text
lidar_margin = 0.25 m < 0.50 m
lidar_ratio = 0.545... > 0.50
```

该样本会先在 margin 检查阶段被拒绝，并且只增加“雷达侧margin不足”计数。

## 15. 参数调整建议

建议先积累一段实际比赛或回放数据，再结合“接受率”和“拒绝原因”调节：

- 接受样本极少：逐项观察主要拒绝原因，不要一次同时放宽全部参数。
- 大量被“匹配距离过大”拒绝：检查外参、XY Maps、坐标转换和雷达聚类质量后，再考虑增大距离阈值。
- 大量被 margin 拒绝：场上机器人或雷达点经常聚集，可适当降低对应 margin，但会提高误配样本进入统计的风险。
- 大量被 ratio 拒绝：当前配对相对第二候选的优势不足。放宽 ratio 会直接容纳更模糊的局部关系。
- 大量被采样间隔拒绝：这是正常的时间降采样现象；若只关注长期稳定统计，不必追求高接受率。
- 每个 class 样本太少：可适当降低 `min_sample_interval_sec`，但连续帧的强相关样本会增加，样本数不再等价于独立观测数。
- 若决定统计死亡装甲板：开启 `include_gray_camera_lidar` 后应单独观察灰色身份维持匹配的可靠性，不能假设其误配率与彩色匹配相同。

任何阈值修改均需重启程序后生效；当前没有为这些统计阈值提供运行时 UI 热更新。

## 16. 统计结果的解释边界

该功能输出的是“通过筛选的匹配残差统计”，不是严格的传感器真值误差：

- 雷达点本身存在聚类误差和时间累积误差。
- 相机投影点受到检测框底边位置、内参、外参、地面假设和 XY Map 离散化影响。
- 相机帧和雷达帧即使按时间戳对齐，运动目标仍可能存在采样时差。
- 可靠性筛选会产生选择偏差：距离小、周围孤立、置信度高的机器人更容易进入统计。
- 全局统计会混合不同位置、距离、机器人外形和观测角度。
- 各 class 统计仍可能混合该机器人在不同区域和不同距离下的误差。

因此，当前均值适合用于观察系统性偏移，协方差适合用于评估被接受样本的离散程度，但不应未经进一步验证就直接作为二维卡尔曼滤波器的测量噪声矩阵。若要用于自动外参修正或 KF 参数估计，建议同时保存位置、距离、时间同步误差等分层信息，并使用独立数据验证修正后的效果。

## 17. 测试方式

在 `Main_ws` 目录运行：

```bash
python -m unittest \
  tests.test_projection_lidar_statistics \
  tests.test_cascade_match_tracker
```

当前测试覆盖：

- 两个二维样本的全局和分 class Welford 均值、样本协方差。
- 蓝方与红方阵营总计相互独立，并与全局统计同步更新。
- 详情文本中的阵营分区、机器人间空行和两行协方差矩阵。
- 同 class 最小采样间隔。
- 存在更近竞争点时的互为最近邻拒绝。
- ratio 严格按照 `d1 / d_second` 计算。
- 灰色来源默认关闭。
- 手动清零。
- 追踪器彩色匹配接入。
- 阵营变化自动清零。
