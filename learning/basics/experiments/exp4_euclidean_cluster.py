# 实验 4：欧式聚类——把「一团点」变成「一个目标」，Lidar_ws/cluster 的最小复刻
#
# 雷达每帧给出成千上万个动态点，融合层要的不是点，是「这里有 N 个目标、
# 每个目标的中心」。Lidar_ws/src/cluster 用 PCL 的欧式聚类做这件事。
# 这个实验不用 PCL，用「体素网格 + 广度优先搜索」手写一个等价算法，
# 在合成的比赛场景上跑通，并体会距离阈值 cluster_tolerance 的作用。
#
# 对应项目模块：Lidar_ws/src/cluster（cluster.cpp）、learning/03-雷达管线三节点。
# 运行：python3 exp4_euclidean_cluster.py
# 输出：output/exp4_clusters.png

import os
from collections import deque
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FIELD_X, FIELD_Y = 28.0, 15.0
PIXELS_PER_METER = 40


def euclidean_cluster(points, tol):
    """朴素欧式聚类：体素网格加速的连通分量搜索。

    points: (N, 3)；tol: 距离阈值（米），小于它的两点算「连通」。
    返回 labels（-1 = 噪声/未分到足夠大的簇时不处理，这里全部分配）。
    PCL 的 EuclideanClusterExtraction 本质就是这 + KD 树。
    """
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    # 1) 建体素哈希：voxel 大小 = tol，邻居只可能在周围 27 格里
    vox = np.floor(points / tol).astype(int)
    table = {}
    for i, key in enumerate(map(tuple, vox)):
        table.setdefault(key, []).append(i)

    cluster_id = 0
    min_size = 10  # 小于 10 个点的簇当噪声丢掉（项目里也有类似下限）
    for seed in range(n):
        if labels[seed] != -1:
            continue
        queue = deque([seed])
        labels[seed] = cluster_id
        members = []
        while queue:
            idx = queue.popleft()
            members.append(idx)
            key = vox[idx]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in table.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            if labels[j] != -1:
                                continue
                            if np.linalg.norm(points[j] - points[idx]) <= tol:
                                labels[j] = cluster_id
                                queue.append(j)
        if len(members) >= min_size:
            cluster_id += 1
        else:  # 太小，回退成噪声
            for m in members:
                labels[m] = -2
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)

    # ---- 造一个「比赛某瞬间」：5 台车 + 1 架无人机 + 地面散点噪声 ----
    def car_blob(cx, cy, n=300):
        # 车 ≈ 0.6m x 0.6m x 0.4m 的盒子表面散点
        pts = rng.uniform(-0.35, 0.35, (n, 3))
        pts[:, 2] = np.abs(pts[:, 2]) * 0.4
        return pts + np.array([cx, cy, 0])

    def drone_blob(cx, cy, cz, n=120):
        pts = rng.normal(0, 0.25, (n, 3))
        return pts + np.array([cx, cy, cz])

    clouds = [
        car_blob(6, 4), car_blob(14, 11), car_blob(15.2, 11.9),  # 后两台只隔 1.5 m
        car_blob(21, 3), car_blob(23, 12), drone_blob(14, 5, 3.0),
    ]
    noise = np.column_stack([
        rng.uniform(0, FIELD_X, 150),
        rng.uniform(0, FIELD_Y, 150),
        rng.uniform(0, 0.1, 150),
    ])
    points = np.vstack(clouds + [noise])
    print(f"合成点云: {len(points)} 个点（6 台车/无人机 + 150 个散点）")

    # ---- 聚类 ----
    import time
    t0 = time.time()
    labels = euclidean_cluster(points, tol=0.5)  # 项目同量级的容忍半径
    dt = (time.time() - t0) * 1000
    n_clusters = labels.max() + 1
    print(f"聚类完成: {n_clusters} 个目标, 耗时 {dt:.0f} ms（纯 Python，C++/PCL 会快百倍）")

    centers = []
    for c in range(n_clusters):
        member = points[labels == c]
        centers.append(member.mean(axis=0)[:2])
        print(f"  目标 {c}: {len(member):4d} 点, 中心 ({member[:,0].mean():5.2f},"
              f" {member[:,1].mean():5.2f}) m, 平均高 {member[:,2].mean():4.2f} m")

    # ---- 画俯视图 ----
    W, H = int(FIELD_X * PIXELS_PER_METER), int(FIELD_Y * PIXELS_PER_METER)
    canvas = np.full((H, W, 3), 25, np.uint8)
    rng_colors = rng.integers(60, 255, (n_clusters, 3))
    palette = np.array([[40, 40, 40]] + list(rng_colors), dtype=np.uint8)  # 0=噪声
    lut = np.where(labels >= 0, labels, 0)
    px = (points[:, 0] * PIXELS_PER_METER).astype(int)
    py = (points[:, 1] * PIXELS_PER_METER).astype(int)
    canvas[H - 1 - py, px] = palette[lut]

    for i, (cx, cy) in enumerate(centers):
        X, Y = int(cx * PIXELS_PER_METER), H - 1 - int(cy * PIXELS_PER_METER)
        cv2.circle(canvas, (X, Y), 6, (255, 255, 255), 1)
        cv2.putText(canvas, str(i), (X + 8, Y - 8), 0, 0.5, (255, 255, 255), 1)

    cv2.putText(canvas, "euclidean clustering (tol=0.5m): 5 cars + 1 drone (hovering 3m)",
                (10, 25), 0, 0.55, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, "exp4_clusters.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n聚类结果图已保存: {out_path}")

    # ---- 阈值敏感性：tol 大了/小了分别会怎样 ----
    print("\n对照实验（同一份点云，不同容忍半径）：")
    for tol in (0.3, 0.5, 1.6):
        lab = euclidean_cluster(points, tol)
        n = lab.max() + 1
        sizes = sorted([int((lab == c).sum()) for c in range(n)], reverse=True)
        print(f"  tol={tol:4.1f} m -> {n:2d} 个聚类, 各簇点数 {sizes[:8]}")
    print("\n读数：tol=0.3 时挨近的两台车分得很开、但目标可能碎成几团；")
    print("tol=1.6 时两台 1.5 m 间距的车粘成一个几百点的大簇（还混进噪声）。")
    print("这就是聚类阈值调参的经典两难：太小一台车碎成几团，太大两台车粘成一团。")


if __name__ == "__main__":
    main()
