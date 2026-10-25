# 实验 7：ICP 点云配准——「雷达每次开机怎么知道自己在场地哪」的直觉
#
# 雷达站每天架的位置都不同。localization 节点要解的问题是：
# 已知一张建好的场地地图（RMUC2026.pcd），现在来了一帧新扫描（livox_frame 坐标系），
# 求一个刚体变换把它们对齐——对齐了，就得到了「雷达坐标系 -> 场地坐标系」的 TF。
# 项目用 GICP 先粗配、再拿结果当初值喂 NDT 精配。这个实验用最经典的 ICP
# 在 2D 上复现同样的思想：最近邻关联 + 最小二乘求刚体变换 + 迭代。
#
# 对应项目模块：Lidar_ws/src/localization（GICP->NDT 两阶段）、learning/03。
# 运行：python3 exp7_icp_2d.py
# 输出：output/exp7_icp.png

import os
import numpy as np
import cv2
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FIELD_X, FIELD_Y = 28.0, 15.0
PIXELS_PER_METER = 40


def make_map(rng, n=900):
    """合成「场地地图」：四周围墙 + 中央两个环形高地的轮廓点。"""
    t = np.linspace(0, 1, 250)
    xs = np.concatenate([t * FIELD_X, t * FIELD_X, np.zeros_like(t),
                         np.full_like(t, FIELD_X)])
    ys = np.concatenate([np.zeros_like(t), np.full_like(t, FIELD_Y),
                         t * FIELD_Y, t * FIELD_Y])
    wall = np.column_stack([xs, ys])
    a1 = np.linspace(0, 2 * np.pi, 200)
    ring1 = np.column_stack([14 + 3.2 * np.cos(a1), 7.5 + 2.4 * np.sin(a1)])
    a2 = np.linspace(0, 2 * np.pi, 150)
    ring2 = np.column_stack([7 + 2.0 * np.cos(a2), 10.5 + 2.0 * np.sin(a2)])
    m = np.vstack([wall, ring1, ring2])
    m = m[rng.choice(len(m), size=min(n, len(m)), replace=False)]
    return m


def transform(pts, theta, dx, dy):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([dx, dy])


def icp(scan, map_tree, map_pts, max_iter=40, inlier_dist=0.8):
    """最近邻 ICP：反复「配对 -> 求最优刚体变换 -> 应用」。

    返回累计变换和每轮的平均配对残差。对应项目里 GICP 的角色。
    """
    cur = scan.copy()
    errors = []
    for it in range(max_iter):
        dist, idx = map_tree.query(cur)                     # 每个扫描点找地图最近点
        mask = dist < inlier_dist                          # 距离太远的对当外点剔除
        src, dst = cur[mask], map_pts[idx[mask]]
        if len(src) < 10:
            break
        # SVD 求两点集间最优旋转平移（Umeyama 的刚体特例）
        mu_s, mu_d = src.mean(0), dst.mean(0)
        H = (src - mu_s).T @ (dst - mu_d)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:                           # 防反射
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_d - R @ mu_s
        cur = cur @ R.T + t
        errors.append(dist[mask].mean())
        if errors[-1] < 1e-4:
            break
    return cur, errors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    map_pts = make_map(rng)
    tree = cKDTree(map_pts)

    # 真值变换：雷达架歪了 4 度、偏了 (0.5, -0.4) 米。ICP 要把它找回来。
    theta_true, d_true = np.radians(4.0), np.array([0.5, -0.4])
    scan = transform(map_pts[rng.choice(len(map_pts), 400, replace=False)],
                     theta_true, *d_true)
    scan += rng.normal(0, 0.02, scan.shape)                # 扫描本身有厘米级噪声

    aligned, errors = icp(scan, tree, map_pts)
    nn_dist, _ = tree.query(aligned)  # 每个配准后点到地图的最近邻距离

    # 反解出 ICP 估计的「雷达->场地」旋转角，和真值比
    mu_a, mu_m = aligned.mean(0), scan.mean(0)
    H = (scan - mu_m).T @ (aligned - mu_a)
    U, _, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    theta_est = np.degrees(np.arctan2(R_est[1, 0], R_est[0, 0]))
    t_est = mu_a - R_est @ mu_m
    print(f"ICP 迭代 {len(errors)} 轮, 配对平均残差 {errors[0]*100:.1f} cm -> {errors[-1]*100:.1f} cm")
    print(f"对齐后扫描点到地图最近邻平均距离: {nn_dist.mean()*100:.1f} cm")
    print(f"估计的「扫描->地图」旋转: {theta_est:.2f} 度（安装歪了 +4.00 度，")
    print(f"  转回去自然是 -4 度；平移 {np.round(t_est, 3)} m 与真值 (0.5,-0.4) 互为逆变换）")

    # ICP 只吃小初值：再演示一下初值歪太多时的失败模式
    scan_bad = transform(scan, np.radians(15.0), 3.0, 2.0)
    _, errors_bad = icp(scan_bad, tree, map_pts)
    print(f"\n对照实验：初值偏差放大到 15 度 + 3.6 m -> 收敛残差 {errors_bad[-1]*100:.1f} cm")
    print("  收敛到局部极小值，配准失败。这就是项目『GICP 先粗配 -> 给 NDT 当初值』")
    print("  两阶段设计的原因（learning/03 引用了作者的注释）。")

    # ---- 画图（点云没有顺序，画散点而不是折线）----
    W, H = int(FIELD_X * PIXELS_PER_METER), int(FIELD_Y * PIXELS_PER_METER)
    canvas = np.full((H, W, 3), 25, np.uint8)

    def draw(pts, color, radius=2):
        p = np.column_stack([(pts[:, 0] * PIXELS_PER_METER).astype(int),
                             H - 1 - (pts[:, 1] * PIXELS_PER_METER).astype(int)])
        for X, Y in p:
            cv2.circle(canvas, (int(X), int(Y)), radius, color, -1, cv2.LINE_AA)

    draw(map_pts, (0, 220, 0), 2)      # 地图：绿
    draw(scan, (0, 0, 255), 2)         # 原始扫描：红（架歪的雷达视角）
    draw(aligned, (0, 200, 255), 1)    # 配准后：黄（应基本盖住绿点）
    cv2.putText(canvas, "green=map  red=raw scan (mis-installed lidar)  yellow=after ICP",
                (10, 25), 0, 0.55, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, "exp7_icp.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n配准对比图已保存: {out_path}")


if __name__ == "__main__":
    main()
