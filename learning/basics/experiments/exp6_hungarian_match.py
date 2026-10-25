# 实验 6：匈牙利算法 + 距离门限——多目标跟踪「谁是谁」的最小模型
#
# 融合层的核心问题：视觉说「红3在 (17.9, 6.1) 附近」，雷达聚类说
# 「(17.4, 6.4) 和 (23.0, 2.0) 有两个目标」——视觉预测该认领哪个点？
# 项目用「代价矩阵 + 匈牙利算法求全局最优 + 距离门限拒配」解决（tracker.py
# 的级联匹配里反复出现）。这个实验用 4 个跟踪器 x 5 个雷达点演示全流程，
# 并对比贪心法为什么会配错。
#
# 对应项目模块：tracker/CascadeMatchTracker/tracker.py::process_observations、learning/06。
# 运行：python3 exp6_hungarian_match.py
# 输出：output/exp6_match.png

import os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FIELD_X, FIELD_Y = 28.0, 15.0
PIXELS_PER_METER = 40
GATE = 2.0  # 项目 colored_triplet_match 同款门限量级（1.5~2 m）


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4 个视觉跟踪器的当前预测位置（卡尔曼 predict 的输出）
    tracks = {
        "B3": (6.2, 4.1),
        "R1": (17.9, 6.3),
        "R3": (21.5, 11.8),
        "R7": (12.0, 2.5),   # 这个跟踪器其实已经飘了（真目标早走了）
    }
    # 5 个雷达聚类中心（含一个新出现的未知目标）
    clusters = [(6.4, 4.3), (17.4, 6.6), (21.9, 11.5), (27.0, 1.2), (5.0, 14.0)]

    names = list(tracks)
    track_pos = np.array([tracks[n] for n in names])
    cluster_pos = np.array(clusters)

    # ---- 代价矩阵 = 预测位置与聚类中心的场地平面距离 ----
    cost = np.linalg.norm(
        track_pos[:, None, :] - cluster_pos[None, :, :], axis=2)
    print("代价矩阵（行=视觉跟踪器, 列=雷达聚类, 单位米）:")
    header = "        " + "".join(f"c{j:<8}" for j in range(len(clusters)))
    print(header)
    for i, n in enumerate(names):
        row = f"  {n:>4}  " + "".join(f"{cost[i, j]:<10.3f}" for j in range(len(clusters)))
        print(row)

    # ---- 匈牙利算法：全局总代价最小的一对一分配 ----
    row_ind, col_ind = linear_sum_assignment(cost)
    print("\n匈牙利最优分配:")
    matched = []
    for i, j in zip(row_ind, col_ind):
        d = cost[i, j]
        if d <= GATE:
            matched.append((names[i], j))
            print(f"  {names[i]} <-> 聚类{j}: 距离 {d:.2f} m  (<= {GATE} m, 接受)")
        else:
            print(f"  {names[i]} <-> 聚类{j}: 距离 {d:.2f} m  (> {GATE} m, 拒绝! 走降级链)")
    unassigned = [j for j in range(len(clusters)) if j not in [c for _, c in matched]]
    print(f"  未被认领的聚类: {unassigned} -> 项目里进入「新目标出生」逻辑")

    # ---- 对照：贪心（每行挑当前最近的）会犯什么错 ----
    print("\n贪心法（每个跟踪器各自挑最近的聚类）:")
    greedy_cols = []
    for i, n in enumerate(names):
        j = int(np.argmin(cost[i]))
        greedy_cols.append(j)
        print(f"  {n} -> 聚类{j} (距离 {cost[i, j]:.2f} m)")
    if len(greedy_cols) != len(set(greedy_cols)):
        dup = [j for j in set(greedy_cols) if greedy_cols.count(j) > 1]
        print(f"  两个跟踪器抢了同一个聚类 {dup}！局部最优 != 全局最优，")
        print("  这就是必须用匈牙利（或拍卖算法）做数据关联的原因。")

    # ---- 画图 ----
    W, H = int(FIELD_X * PIXELS_PER_METER), int(FIELD_Y * PIXELS_PER_METER)
    canvas = np.full((H, W, 3), 25, np.uint8)

    def to_px(p):
        return int(p[0] * PIXELS_PER_METER), H - 1 - int(p[1] * PIXELS_PER_METER)

    for i, (name, p) in enumerate(tracks.items()):
        X, Y = to_px(p)
        cv2.circle(canvas, (X, Y), 8, (0, 140, 255), 2)
        cv2.putText(canvas, name, (X + 10, Y - 10), 0, 0.5, (0, 140, 255), 1)
    for j, p in enumerate(clusters):
        X, Y = to_px(p)
        cv2.circle(canvas, (X, Y), 8, (0, 255, 0), 2)
        cv2.putText(canvas, f"c{j}", (X + 10, Y + 5), 0, 0.5, (0, 255, 0), 1)
    for name, j in matched:
        X1, Y1 = to_px(tracks[name])
        X2, Y2 = to_px(clusters[j])
        cv2.line(canvas, (X1, Y1), (X2, Y2), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "orange=vision tracks  green=lidar clusters  white=accepted match (gate 2.0m)",
                (10, 25), 0, 0.55, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, "exp6_match.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n匹配示意图已保存: {out_path}")


if __name__ == "__main__":
    main()
