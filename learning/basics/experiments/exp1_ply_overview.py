# 实验 1：不用 open3d，亲手读项目的场地 PLY 网格
#
# 项目里 Main_ws/field/RMUC2026.ply 是一张约 100 万顶点、200 万三角面的
# RMUC2026 场地三维网格。主程序用 open3d 加载它做射线求交（transform/ray_renderer.py）。
# 这个实验只用 numpy 手写一个二进制 PLY 读取器，把顶点读出来，
# 再用 OpenCV 画一张「俯视图」，按高度 z 上色——你会直观看到整张场地。
#
# 对应项目模块：field/RMUC2026.ply、interface 的 UI 场地图 field_image.PNG。
# 运行：python3 exp1_ply_overview.py
# 输出：output/exp1_field_topview.png

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PLY_PATH = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "Main_ws", "field", "RMUC2026.ply")
)
OUT_DIR = os.path.join(HERE, "output")

# 场地实际尺寸（米），来自 Main_ws/config/params.yaml: lidar_fusion
FIELD_X, FIELD_Y = 28.0, 15.0
PIXELS_PER_METER = 40  # 俯视图分辨率：40 px/m -> 1120x600


def read_ply_vertices(path):
    """读取二进制 little-endian PLY 的顶点（只解析 vertex 元素，跳过 face）。

    PLY 结构 = 文本头 + 二进制数据。头里声明了每个元素有几个、
    每个顶点有哪些属性，据此拼一个 numpy dtype 一次读入。
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 头部没有 end_header")
            header_lines.append(line.decode("ascii").strip())
            if line.startswith(b"end_header"):
                break
        data_offset = f.tell()

        in_vertex = False
        count = None
        fields = []  # (属性名, numpy 类型)
        type_map = {"float": "f4", "float32": "f4", "double": "f8",
                    "uchar": "u1", "uint8": "u1", "char": "i1",
                    "ushort": "u2", "uint": "u4", "int": "i4"}
        for line in header_lines:
            parts = line.split()
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex and len(parts) == 3:
                fields.append((parts[2], type_map[parts[1]]))

        dtype = np.dtype(fields)
        vertices = np.fromfile(path, dtype=dtype, count=count, offset=data_offset)
    return vertices


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"读取 {PLY_PATH} ...（约 50 MB，几秒钟）")
    verts = read_ply_vertices(PLY_PATH)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    print(f"顶点数: {len(xyz):,}")
    print(f"x 范围: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}] m")
    print(f"y 范围: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}] m")
    print(f"z 范围: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}] m")

    # ---- 俯视图：把每个顶点撒到一张画布上，颜色编码高度 ----
    W, H = int(FIELD_X * PIXELS_PER_METER), int(FIELD_Y * PIXELS_PER_METER)
    canvas = np.zeros((H, W, 3), np.uint8)

    in_field = (
        (xyz[:, 0] >= 0) & (xyz[:, 0] < FIELD_X)
        & (xyz[:, 1] >= 0) & (xyz[:, 1] < FIELD_Y)
    )
    pts = xyz[in_field]
    px = (pts[:, 0] * PIXELS_PER_METER).astype(int)
    py = (pts[:, 1] * PIXELS_PER_METER).astype(int)

    # 高度归一化到 0~1：蓝色 = 低（地面），红色 = 高（墙、高台）
    z = pts[:, 2]
    z_norm = np.clip((z - z.min()) / (max(z.max() - z.min(), 1e-6)), 0, 1)
    colors = np.stack([
        (0.3 + 0.7 * z_norm) * 255,          # B 通道
        90 + 40 * np.sin(z_norm * np.pi),    # G 通道
        (255 - 200 * z_norm),                # R 通道…… 下面再交换
    ], axis=1).astype(np.uint8)
    # OpenCV 是 BGR：地面偏蓝、高处偏红
    canvas[H - 1 - py, px] = colors[:, [2, 1, 0]]

    # 画 1 米网格线和边框，方便对照真实场地
    for m in range(0, int(FIELD_X) + 1):
        cv2.line(canvas, (m * PIXELS_PER_METER, 0),
                 (m * PIXELS_PER_METER, H - 1), (60, 60, 60), 1)
    for m in range(0, int(FIELD_Y) + 1):
        cv2.line(canvas, (0, H - 1 - m * PIXELS_PER_METER),
                 (W - 1, H - 1 - m * PIXELS_PER_METER), (60, 60, 60), 1)
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (200, 200, 200), 2)
    cv2.putText(canvas, "RMUC2026 field top view (color = height z), 1 px = 2.5 cm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    out_path = os.path.join(OUT_DIR, "exp1_field_topview.png")
    cv2.imwrite(out_path, canvas)
    print(f"俯视图已保存: {out_path}")
    print("提示：蓝色大片是地面，红黄色是环形高墙和资源岛——"
          "这就是雷达看到的世界，也是 UI 左侧场地图的来源。")


if __name__ == "__main__":
    main()
