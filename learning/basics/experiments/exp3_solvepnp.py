# 实验 3：solvePnP 恢复外参——复刻项目「启动时点关键点标定」的原理
#
# 项目每次架设后，操作员在界面里点 4 个以上的场地关键点（transform/keypoints.txt），
# 用 cv2.solvePnPRansac(EPnP) + solvePnP(ITERATIVE) 精修求出 R/t（transform/solvepnp.py）。
# 这个实验用合成数据走一遍同样的流程：
#   已知真值外参 -> 把关键点投影成像素 -> 加 1~2 px 噪声 -> solvePnP 反解 -> 和真值比。
#
# 对应项目模块：transform/solvepnp.py、interface/core.py 的 KeypointCalibrationDialog。
# 运行：python3 exp3_solvepnp.py

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

FIELD_X, FIELD_Y = 28.0, 15.0
K = np.array([
    [3299.5283562470486, 0.0, 2024.0633490912842],
    [0.0, 3300.8856957717744, 1609.1450726022888],
    [0.0, 0.0, 1.0],
])
DIST = np.array([
    -0.058072516483303048, 0.09529462517050756, 0.00095029382183982921,
    -0.0011825558881310113, -0.052444936481243022,
])


def look_at(cam_pos, target):
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    return R, -R @ cam_pos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)

    # 真值外参（假装这是今天架好的雷达站）
    cam_pos_true = np.array([FIELD_X + 5.0, FIELD_Y / 2 + 2.0, 5.5])
    R_true, t_true = look_at(cam_pos_true, np.array([FIELD_X / 2, FIELD_Y / 2, 0.0]))

    # 场地关键点：项目里 keypoints.txt 存的就是这类坐标（单位米）
    world = np.array([
        [0, 0, 0], [FIELD_X, 0, 0], [FIELD_X, FIELD_Y, 0], [0, FIELD_Y, 0],
        [FIELD_X / 2, 0, 0], [FIELD_X / 2, FIELD_Y, 0],      # 中线两端
        [9.5, FIELD_Y / 2, 0], [18.5, FIELD_Y / 2, 0],       # 资源岛两侧
    ], dtype=float)

    # 正向投影 + 像素噪声（模拟人手点不准 / 检测抖动）
    rvec_true, _ = cv2.Rodrigues(R_true)
    px, _ = cv2.projectPoints(world, rvec_true, t_true, K, DIST)
    px = px.reshape(-1, 2)
    px_noisy = px + rng.normal(0, 1.5, px.shape)  # 1.5 px 标准差的手动点击噪声

    # ---- 项目同款两步求解 ----
    ok, rvec_r, t_r, inliers = cv2.solvePnPRansac(
        world, px_noisy, K, DIST, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=6.0, confidence=0.99,
    )
    print(f"solvePnPRansac(EPnP) 成功: {ok}, 内点 {0 if inliers is None else len(inliers)}/{len(world)}")
    # 用 RANSAC 内点做 ITERATIVE 精修（项目同款套路）
    if inliers is not None and len(inliers) >= 4:
        world_ref, px_ref = world[inliers.ravel()], px_noisy[inliers.ravel()]
    else:
        world_ref, px_ref = world, px_noisy
    ok2, rvec, tvec = cv2.solvePnP(
        world_ref, px_ref, K, DIST, rvec=rvec_r, tvec=t_r,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    print(f"solvePnP(ITERATIVE) 精修成功: {ok2}")

    R_est, _ = cv2.Rodrigues(rvec)

    # ---- 和真值比：相机位置误差 + 旋转角误差 + 重投影误差 ----
    cam_pos_est = (-R_est.T @ tvec).ravel()
    pos_err = np.linalg.norm(cam_pos_est - cam_pos_true)
    rot_err_deg = np.degrees(np.arccos(
        np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)))

    px_reproj, _ = cv2.projectPoints(world, rvec, tvec, K, DIST)
    reproj_err = np.linalg.norm(px_reproj.reshape(-1, 2) - px_noisy, axis=1)

    print(f"\n相机位置真值: {np.round(cam_pos_true, 3)}")
    print(f"相机位置估计: {np.round(cam_pos_est, 3)}")
    print(f"相机位置误差: {pos_err*100:.2f} cm")
    print(f"旋转误差:     {rot_err_deg:.3f} 度")
    print(f"重投影误差:   mean {reproj_err.mean():.2f} px, max {reproj_err.max():.2f} px")

    # ---- 更要紧的指标：场地平面上的定位误差 ----
    # 外参准不准，最终看「把像素反投回场地」偏了多少。
    # 对每个关键点：像素 -> 归一化射线 -> 和 z=0 平面求交（ray_renderer 的 2 行版）
    print("\n各关键点的地面反投影误差（像素 -> 场地平面）：")
    for i, (w, p) in enumerate(zip(world, px_noisy)):
        # 逆投影三步（ray_renderer.py 的 2 行版）：
        # 1) 去畸变（P=K：输出仍是像素坐标）2) 乘 K^-1 得相机系方向 3) 转世界系
        und = cv2.undistortPoints(p.reshape(1, 1, 2), K, DIST, P=K).reshape(2)
        d_cam = np.linalg.inv(K) @ np.array([und[0], und[1], 1.0])
        d_world = R_est.T @ d_cam
        o_world = cam_pos_est
        if abs(d_world[2]) < 1e-6:
            continue
        s = -o_world[2] / d_world[2]
        hit = o_world + s * d_world
        err = np.linalg.norm(hit[:2] - w[:2])
        print(f"  点{i} ({w[0]:5.1f},{w[1]:5.1f}): 反投影偏移 {err*100:6.1f} cm")

    print("\n结论：1.5 px 的点选噪声 -> 厘米级场地误差。这就是项目坚持")
    print("「现场人工点关键点标定」而不是抄固定外参的原因（learning/05 有完整分析）。")


if __name__ == "__main__":
    main()
