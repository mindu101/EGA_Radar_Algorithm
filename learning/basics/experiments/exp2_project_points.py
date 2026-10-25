# 实验 2：针孔相机投影——把场地三维点投到照片上（项目投影的「逆过程」）
#
# 项目的主链路是「像素 -> 场地坐标」（transform/ray_renderer.py:pixel_to_world，
# 从像素发一条射线、和场地网格求交）。这个实验做反方向、也是更好理解的方向：
# 「场地坐标 -> 像素」（cv2.projectPoints）。
# 两个方向用的是同一组参数：内参 K、畸变系数、外参 R/t。
# 内参直接抄 Main_ws/config/device.yaml 里的真实标定值。
#
# 对应项目模块：config/device.yaml、transform/ray_renderer.py。
# 运行：python3 exp2_project_points.py
# 输出：output/exp2_projection.png

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

FIELD_X, FIELD_Y = 28.0, 15.0

# ---- 海康 MV-CS120 的真实内参（Main_ws/config/device.yaml: camera_matrix）----
K = np.array([
    [3299.5283562470486, 0.0, 2024.0633490912842],
    [0.0, 3300.8856957717744, 1609.1450726022888],
    [0.0, 0.0, 1.0],
])
DIST = np.array([
    -0.058072516483303048, 0.09529462517050756, 0.00095029382183982921,
    -0.0011825558881310113, -0.052444936481243022,
])
IMG_W, IMG_H = 4024, 3036


def look_at(cam_pos, target):
    """由「相机位置 + 看向哪」构造世界->相机的外参 R, t。

    约定：相机 z 轴朝前（视线方向）、y 轴向下、x 轴向右，和 OpenCV 一致。
    X_cam = R @ X_world + t，其中 t = -R @ cam_pos。
    """
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])  # 场地 z 轴朝上
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    t = -R @ cam_pos
    return R, t


def project(R, t, world_points, use_distortion):
    """世界点 -> 像素。use_distortion=False 时传零畸变，看差别。"""
    rvec, _ = cv2.Rodrigues(R)
    dist = DIST if use_distortion else np.zeros(5)
    pts, _ = cv2.projectPoints(world_points.astype(np.float64), rvec, t, K, dist)
    return pts.reshape(-1, 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 模拟一次典型的雷达站架设：场边 6 米高台，斜俯视全场
    cam_pos = np.array([FIELD_X + 6.0, FIELD_Y / 2, 6.0])
    target = np.array([FIELD_X / 2, FIELD_Y / 2, 0.0])
    R, t = look_at(cam_pos, target)
    print(f"相机位置（世界坐标）: {cam_pos}")
    print(f"R =\n{np.round(R, 3)}")
    print(f"t = {np.round(t, 3)}")

    # 要投影的三维点：四个场地角 + 中线点阵 + 一面墙的顶边
    corners = np.array([
        [0, 0, 0], [FIELD_X, 0, 0],
        [FIELD_X, FIELD_Y, 0], [0, FIELD_Y, 0],
    ], dtype=float)
    labels = ["(0,0)", f"({FIELD_X:.0f},0)", f"({FIELD_X:.0f},{FIELD_Y:.0f})", f"(0,{FIELD_Y:.0f})"]

    grid = np.array([[x, y, 0] for x in np.linspace(4, 24, 6)
                     for y in np.linspace(3, 12, 4)], dtype=float)
    wall = np.array([[x, 0, 1.2] for x in np.linspace(2, 26, 13)], dtype=float)  # 红方墙顶边

    px_no_dist = project(R, t, corners, use_distortion=False)
    px_dist = project(R, t, corners, use_distortion=True)
    grid_px = project(R, t, grid, use_distortion=True)
    wall_px = project(R, t, wall, use_distortion=True)

    print("\n四个角点投影（无畸变 vs 有畸变，像素）：")
    for i, lab in enumerate(labels):
        print(f"  {lab:>9}: ({px_no_dist[i,0]:8.1f},{px_no_dist[i,1]:8.1f})"
              f"  ->  ({px_dist[i,0]:8.1f},{px_dist[i,1]:8.1f})")
    print("  注意两件事：")
    print("  1) 画面中心的角点畸变只差 1~3 px，画面边缘差几十上百 px——畸变是径向的；")
    print("  2) 近处两个角点投影在画面外（负坐标）。6mm 镜头水平视场约 62 度，")
    print("     贴着场边架根本看不全 28m 场地——这是真实几何约束，不是 bug。")

    # ---- 画到缩小 4 倍的「照片」上 ----
    scale = 0.25
    canvas = np.zeros((int(IMG_H * scale), int(IMG_W * scale), 3), np.uint8)
    canvas[:] = (40, 40, 40)

    def draw(pts, color, radius=4, texts=None):
        for i, p in enumerate(pts):
            x, y = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(canvas, (x, y), radius, color, -1)
            if texts:
                cv2.putText(canvas, texts[i], (x + 8, y), 0, 0.5, color, 1)

    draw(grid_px, (180, 180, 180), 2)               # 地面点阵：白
    draw(wall_px, (0, 200, 255), 3)                 # 墙顶：黄（悬空 1.2 m）
    draw(px_no_dist, (255, 0, 0), 5)                # 角点-无畸变：红
    draw(px_dist, (0, 0, 255), 3, labels)           # 角点-有畸变：蓝
    cv2.putText(canvas, "white=ground grid  yellow=wall top(z=1.2m)", (10, 25),
                0, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, "red=corners w/o distortion  blue=corners with distortion",
                (10, 50), 0, 0.6, (255, 255, 255), 1)

    out_path = os.path.join(OUT_DIR, "exp2_projection.png")
    cv2.imwrite(out_path, canvas)
    print(f"\n投影图已保存: {out_path}")
    print("注意看：地面点阵在照片上不是等距的——越远越密。这就是透视，")
    print("也是「像素距离」和「米」不能直接比、必须投影回场地平面的原因。")


if __name__ == "__main__":
    main()
