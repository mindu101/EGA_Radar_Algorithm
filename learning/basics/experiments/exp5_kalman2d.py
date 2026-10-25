# 实验 5：二维卡尔曼滤波——项目 tracker/kalman.py::KalmanFilter2d 的可运行复刻
#
# 项目给每个身份（B1~R7 共 10 个）维护一个 [x, y, vx, vy] 状态的卡尔曼：
# 雷达聚类中心来了就 update（位置可信），目标丢了就只 predict 续航。
# 这个实验合成一段「绕圈跑 + 急停」的轨迹，加 0.3 m 测量噪声，
# 对比三种估计的 RMSE：原始测量 / 卡尔曼 / 目标消失后纯预测续航。
#
# 对应项目模块：tracker/CascadeMatchTracker/kalman.py、tracker.py 的丢失降级链。
# 运行：python3 exp5_kalman2d.py
# 输出：output/exp5_kalman.png

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
FIELD_X, FIELD_Y = 28.0, 15.0
PIXELS_PER_METER = 40
DT = 0.1          # 10 Hz，和项目融合频率同量级
LOST_AT = 130     # 从第 130 步起目标消失（雷达再也测不到）


class Kalman2d:
    """和项目 KalmanFilter2d 相同结构的匀速模型 KF（精简注释版）。"""

    def __init__(self, pos, q_std=1.0, r_std=0.3, dt=DT):
        self.dt = dt
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])      # [x, y, vx, vy]
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 100.0                          # 初始很不确定
        self.Q = (np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                            [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
                  * q_std**2)
        self.R = np.eye(2) * r_std**2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, z):
        y = np.asarray(z) - self.H @ self.x                 # 残差：测量比预测多出多少
        S = self.H @ self.P @ self.H.T + self.R             # 残差的不确定度
        K = self.P @ self.H.T @ np.linalg.inv(S)            # 卡尔曼增益：信预测还是信测量
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x[:2]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)
    n_steps = 200

    # ---- 真值轨迹：前 100 步绕圈（机器人常见的走位），100 步后直线急刹 ----
    gt = []
    for k in range(n_steps):
        t = k * DT
        if k < 100:
            ang = 0.8 * t
            gt.append([14 + 6 * np.cos(ang), 7.5 + 4 * np.sin(ang)])
        else:
            v = np.array(gt[-1]) - np.array(gt[-2])         # 上一帧速度
            gt.append(np.array(gt[-1]) + v * np.exp(-(k - 100) * 0.5))
    gt = np.array(gt)

    meas = gt + rng.normal(0, 0.3, gt.shape)                # 雷达聚类中心 = 真值 + 噪声

    # ---- 跑 KF：LOST_AT 之前正常 update，之后只 predict（项目的「续航」）----
    kf = Kalman2d(meas[0], q_std=4.0)
    est, est_phase = [], []
    for k in range(1, n_steps):
        kf.predict()
        if k < LOST_AT:
            pos = kf.update(meas[k])
            est_phase.append("update")
        else:
            pos = kf.x[:2]                                  # 没有测量，只能靠模型滑行
            est_phase.append("coast")
        est.append(pos)
    est = np.array(est)

    # ---- 统计误差（est[0] 对应 gt[1]，先对齐再比较）----
    def rmse(a, b):
        return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

    gt_c = gt[1:]
    m = slice(0, LOST_AT - 1)
    m_lost = slice(LOST_AT - 1, None)
    print(f"目标出现期间（前 {LOST_AT-1} 步）:")
    print(f"  直接用雷达测量   RMSE = {rmse(meas[1:][m], gt_c[m]):.3f} m")
    print(f"  卡尔曼融合       RMSE = {rmse(est[m], gt_c[m]):.3f} m")
    print(f"目标消失后（纯 predict 续航 {n_steps-LOST_AT} 步 = {(n_steps-LOST_AT)*DT:.1f} s）:")
    print(f"  续航误差 RMSE = {rmse(est[m_lost], gt_c[m_lost]):.3f} m")
    print("  项目里也是这么撑的：丢失 <10 帧靠卡尔曼，>=10 帧允许雷达猜点，")
    print("  >=30 帧退化为固定出生点（params.yaml: loss 三档）。")

    # ---- 画图：真值/测量/估计三条轨迹 ----
    W, H = int(FIELD_X * PIXELS_PER_METER), int(FIELD_Y * PIXELS_PER_METER)
    canvas = np.full((H, W, 3), 25, np.uint8)

    def polyline(arr, color, thickness=1, dotted_from=None):
        pts = np.column_stack([
            (arr[:, 0] * PIXELS_PER_METER).astype(int),
            H - 1 - (arr[:, 1] * PIXELS_PER_METER).astype(int),
        ]).reshape(-1, 1, 2)
        if dotted_from is None:
            cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)
        else:
            solid = pts[:dotted_from]
            coast = pts[dotted_from - 1:]
            cv2.polylines(canvas, [solid], False, color, thickness, cv2.LINE_AA)
            cv2.polylines(canvas, [coast], False, color, 1, cv2.LINE_AA)  # 细一点表续航

    polyline(meas, (90, 90, 90))                            # 测量：暗灰
    polyline(gt, (0, 220, 0), 2)                            # 真值：绿
    polyline(est, (0, 140, 255), 2, dotted_from=LOST_AT - 1)  # 估计：橙
    x0 = int(meas[0, 0] * PIXELS_PER_METER)
    y0 = H - 1 - int(meas[0, 1] * PIXELS_PER_METER)
    cv2.circle(canvas, (x0, y0), 6, (255, 255, 255), 2)

    cv2.putText(canvas, "green=truth  gray=noisy lidar  orange=kalman (thin=coasting after loss)",
                (10, 25), 0, 0.55, (255, 255, 255), 1)
    out_path = os.path.join(OUT_DIR, "exp5_kalman.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n轨迹对比图已保存: {out_path}")


if __name__ == "__main__":
    main()
