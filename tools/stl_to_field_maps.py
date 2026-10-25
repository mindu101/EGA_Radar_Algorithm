#!/usr/bin/env python3
"""场地资产转换：官方 STL/STEP 导出的三角网格 -> 本项目三件套。

生成物（与 RMUC2026 资产同格式，可直接被现有代码加载）：
  1. RMUC2027.ply   二进制 PLY 网格（transform/ray_renderer.py 射线求交用）
  2. RMUC2027.pcd   ASCII PCD 点云（localization / dynamic_cloud 地图，体素降采样）
  3. RMUC2027_preview.png  按高度着色的俯视图（人工核对朝向/比例用）

用法：
  python3 tools/stl_to_field_maps.py /path/to/rmuc_2027.stl \
      [--voxel 0.05] [--name RMUC2027] [--field-x 28] [--field-y 15]

约定：输出对齐项目坐标系——x∈[0,28] y∈[0,15] z 向上、单位米；
STL 若为毫米（CAD 常见）自动除以 1000，并把包围盒最小角平移到原点。
"""

import argparse
import os
import struct
import sys

import numpy as np

# 二进制 STL：80 字节头 + 4 字节三角形数 + 每三角形 50 字节
_STL_TRI_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
)


def read_binary_stl(path):
    """读取二进制 STL，返回 (去重顶点 N×3 float64, 面 N×3 int32)。"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.read(80)
        (n_tri,) = struct.unpack("<I", f.read(4))
        if 84 + 50 * n_tri != size:
            raise ValueError(
                "文件长度与二进制 STL 结构不符（可能是 ASCII STL），"
                "请先用 CAD/meshlab 另存为二进制 STL"
            )
        tris = np.fromfile(f, dtype=_STL_TRI_DTYPE, count=n_tri)
    verts = tris["vertices"].reshape(-1, 3).astype(np.float64)
    unique, inverse = np.unique(verts, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    header_text = header[:60].rstrip(b"\x00 ").decode("ascii", "ignore")
    print(f"读取 STL: {n_tri:,} 三角形 -> 去重后 {len(unique):,} 顶点")
    print(f"（头部: {header_text!r}）")
    return unique, faces


def detect_floor_and_anchor(vertices, faces, field_x, field_y):
    """自动找地板平面并确定坐标系原点。

    官方 CAD 模型的坐标原点/朝向和本项目约定（场地一角为原点、地面
    z=0）不一致，且模型常带场外结构（操作间、围挡外底座），不能直接用
    整体包围盒归零。这里按三步走：

    1) 地板 = 面积最大的近水平面所在 z 层，把该层平移到 z=0；
    2) 地板的 x/y 范围可能比赛场大（如 28×16），用「滑动的 field_x×field_y
       窗口取地板面积最大者」确定场地真实边界（多出的条带是场外结构）；
    3) 平移使场地一角为 (0,0)。
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    norm = np.linalg.norm(cross, axis=1)
    area = 0.5 * norm
    cz = (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0
    horiz = (np.abs(cross[:, 2]) / np.maximum(norm, 1e-12)) > 0.9

    bin_size = 0.05
    z_lo = np.floor(cz[horiz].min())
    hist, edges = np.histogram(
        cz[horiz], bins=np.arange(z_lo, cz[horiz].max() + bin_size, bin_size),
        weights=area[horiz],
    )
    floor_z = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    on_floor = horiz & (np.abs(cz - floor_z) < bin_size)
    print(f"地板平面: z={floor_z:+.3f} m, 面积 {area[on_floor].sum():.0f} m²")

    floor_v = vertices[np.unique(faces[on_floor])]

    def anchor_axis(vals, tri_axis_normals, tri_area, tri_zmax, tri_zmin,
                    tri_centroid_axis, field_len, axis_name):
        """用边界墙线锚定场地边界；找不到墙再退回地板滑窗。

        墙线 = 面法向垂直于该轴、且穿过「地板上方 0.5~1.2m」条带的竖直
        面积聚集处。有两侧墙且间距≈field_len 直接用；只有一侧墙时，
        由「地板必须覆盖整个赛场」约束定方向。
        """
        band = (tri_zmax > floor_z + 0.5) & (tri_zmin < floor_z + 1.2)
        walls = band & tri_axis_normals
        if walls.sum() > 0:
            hist_w, edges_w = np.histogram(
                tri_centroid_axis[walls], bins=np.arange(
                    vals.min(), vals.max() + 0.2, 0.2),
                weights=tri_area[walls],
            )
            centers = 0.5 * (edges_w[:-1] + edges_w[1:])
            peaks = centers[hist_w > 5.0]  # 面积聚集 >5m² 视为墙线
            if len(peaks):
                groups = np.split(
                    peaks, np.where(np.diff(peaks) > 1.0)[0] + 1
                )
                lines = np.array([g.mean() for g in groups])
                near = lines[np.abs(lines - lines.mean()) < field_len]
                lo_side = np.sort(near[near < near.mean()])[::-1]  # 近到远
                hi_side = np.sort(near[near > near.mean()])
                # 优先找「一对间距≈field_len」的线（真边墙），避免误选
                # 场内高地的侧壁
                for w_lo in lo_side:
                    for w_hi in hi_side:
                        if abs((w_hi - w_lo) - field_len) < 0.5:
                            print(f"{axis_name} 轴: 两侧墙线 {w_lo:+.2f}/{w_hi:+.2f}")
                            return w_lo
                for w in near:  # 单侧墙：由地板覆盖范围定方向
                    if vals.min() <= w - field_len + 0.3:
                        print(f"{axis_name} 轴: 单侧墙线 {w:+.2f}（作为远界）")
                        return w - field_len
                    if vals.max() >= w + field_len - 0.3:
                        print(f"{axis_name} 轴: 单侧墙线 {w:+.2f}（作为近界）")
                        return w
        # 兜底：滑窗取地板面积最大者
        on_floor_pts = tri_centroid_axis[on_floor]
        on_floor_area = tri_area[on_floor]
        lo_cut = vals.min() + field_len
        hi_cut = vals.max() - field_len
        area_lo = on_floor_area[on_floor_pts <= lo_cut].sum()
        area_hi = on_floor_area[on_floor_pts >= hi_cut].sum()
        origin = vals.min() if area_lo >= area_hi else vals.max() - field_len
        print(
            f"警告: {axis_name} 轴未识别到墙线，用地板滑窗锚定 "
            f"(lo={area_lo:.0f}m² hi={area_hi:.0f}m²)"
        )
        return origin

    cross_dir_x = np.abs(cross[:, 0]) / np.maximum(norm, 1e-12) > 0.9
    cross_dir_y = np.abs(cross[:, 1]) / np.maximum(norm, 1e-12) > 0.9
    tri_zmax = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
    tri_zmin = np.minimum(np.minimum(a[:, 2], b[:, 2]), c[:, 2])
    cx = (a[:, 0] + b[:, 0] + c[:, 0]) / 3.0
    cy = (a[:, 1] + b[:, 1] + c[:, 1]) / 3.0

    x0 = anchor_axis(floor_v[:, 0], cross_dir_x, area, tri_zmax, tri_zmin,
                     cx, field_x, "x")
    y0 = anchor_axis(floor_v[:, 1], cross_dir_y, area, tri_zmax, tri_zmin,
                     cy, field_y, "y")
    print(f"场地锚定: x 从 {x0:+.3f}, y 从 {y0:+.3f}, 地板 z={floor_z:+.3f} -> 归零")
    vertices = vertices - np.array([x0, y0, floor_z])
    return vertices


def normalize_to_field_coords(vertices, field_x, field_y):
    """毫米->米 + 地板平面检测与锚定（见 detect_floor_and_anchor）。"""
    span = vertices.max(axis=0) - vertices.min(axis=0)
    if span[0] > 100 or span[1] > 100:  # CAD 毫米模型
        vertices = vertices / 1000.0
        print("检测到毫米单位，已除以 1000 转米。")
    return vertices


def write_ply(path, vertices, faces):
    """二进制 little-endian PLY（无颜色，open3d 可直接读）。"""
    n_v, n_f = len(vertices), len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment generated by tools/stl_to_field_maps.py\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {n_f}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    v_arr = vertices.astype("<f4")
    f_rec = np.empty(
        n_f,
        dtype=np.dtype(
            [("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")]
        ),
    )
    f_rec["n"] = 3
    f_rec["a"], f_rec["b"], f_rec["c"] = (
        faces[:, 0],
        faces[:, 1],
        faces[:, 2],
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(v_arr.tobytes())
        f.write(f_rec.tobytes())
    print(f"PLY 已写: {path}（{n_v:,} 顶点 / {n_f:,} 面, "
          f"{os.path.getsize(path)/1e6:.1f} MB）")


def clip_outside_field(vertices, faces, field_x, field_y, margin=0.05):
    """删除越出场地边界的三角形（官方模型带操作间/道具等场外杂项）。

    任一顶点超出 [−margin, field+margin] 的三角形整面删除，随后重索引。
    """
    tri = vertices[faces]  # (n,3,3)
    keep = (
        (tri[:, :, 0] >= -margin) & (tri[:, :, 0] <= field_x + margin)
        & (tri[:, :, 1] >= -margin) & (tri[:, :, 1] <= field_y + margin)
    ).all(axis=1)
    if keep.all():
        return vertices, faces
    faces = faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    print(
        f"场外裁剪: 删除 {(~keep).sum():,} 个越界三角形、"
        f"{len(vertices) - len(used):,} 个孤立顶点"
    )
    return vertices[used], inverse.reshape(-1, 3).astype(np.int32)


def add_perimeter_walls(vertices, faces, field_x, field_y, height=2.4,
                        thickness=0.05):
    """按规则手册补一圈 2.4m 围挡（官方 STL 通常不含围挡，只给场地模块）。

    围挡保证相机射线总能命中（2026 资产的环形高墙即来自此），
    也给雷达配准提供强边缘特征。
    """
    t = thickness
    boxes = [
        ((-t, field_x + t), (-t, 0.0)),              # y=0 长边
        ((-t, field_x + t), (field_y, field_y + t)),  # y=field_y 长边
        ((-t, 0.0), (-t, field_y + t)),              # x=0 短边
        ((field_x, field_x + t), (-t, field_y + t)),  # x=field_x 短边
    ]
    new_verts, new_faces = [], []
    base = len(vertices)
    for (x0, x1), (y0, y1) in boxes:
        # 盒子的 6 个面（12 个三角形），射线求交不需要剔除背面，绕向不敏感
        quads = [
            [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)],        # 底
            [(x0, y0, height), (x1, y0, height),
             (x1, y1, height), (x0, y1, height)],                        # 顶
            [(x0, y0, 0), (x0, y1, 0), (x0, y1, height), (x0, y0, height)],  # x=x0 面
            [(x1, y0, 0), (x1, y1, 0), (x1, y1, height), (x1, y0, height)],  # x=x1 面
            [(x0, y0, 0), (x1, y0, 0), (x1, y0, height), (x0, y0, height)],  # y=y0 面
            [(x0, y1, 0), (x1, y1, 0), (x1, y1, height), (x0, y1, height)],  # y=y1 面
        ]
        for quad in quads:
            for p in quad:
                new_verts.append(p)
            new_faces.append([base, base + 1, base + 2])
            new_faces.append([base, base + 2, base + 3])
            base += 4
    vertices = np.vstack([vertices, np.asarray(new_verts, dtype=np.float64)])
    faces = np.vstack([faces, np.asarray(new_faces, dtype=np.int32)])
    print(f"已补围挡: {height}m 高、{t}m 厚, 新增 {len(new_faces)} 个三角形")
    return vertices, faces


def sample_surface(vertices, faces, spacing, seed=2027):
    """按三角形面积加权做重心采样，再体素去重。

    只取网格顶点会漏掉大三角形（CAD 模型的地面往往就是一整块板），
    雷达地图需要面覆盖，因此必须按面积撒点：每 spacing² 至少 1 个采样。
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n_per_tri = np.maximum(np.ceil(area / spacing**2).astype(int), 1)
    rng = np.random.default_rng(seed)
    r1, r2 = rng.random(int(n_per_tri.sum())), rng.random(int(n_per_tri.sum()))
    tri_idx = np.repeat(np.arange(len(faces)), n_per_tri)
    su = np.sqrt(r1)
    w1, w2, w3 = (1 - su), su * (1 - r2), su * r2
    pts = w1[:, None] * a[tri_idx] + w2[:, None] * b[tri_idx] + w3[:, None] * c[tri_idx]
    keys = np.floor(pts / spacing).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def write_pcd(path, surface_points, voxel):
    """体素去重后的表面采样点写 ASCII PCD（x,y,z）。"""
    keys = np.floor(surface_points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    pts = surface_points[np.sort(first)]
    with open(path, "w", encoding="ascii") as f:
        f.write(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(pts)}\nDATA ascii\n"
        )
        np.savetxt(f, pts, fmt="%.4f")
    print(f"PCD 已写: {path}（体素 {voxel}m, {len(pts):,} 点, "
          f"{os.path.getsize(path)/1e6:.1f} MB）")


def write_preview_png(path, vertices, field_x, field_y, px_per_m=40):
    """按高度着色的俯视图（无 cv2 依赖时跳过）。"""
    try:
        import cv2
    except ImportError:
        print("未安装 cv2，跳过预览图。")
        return
    w, h = int(field_x * px_per_m), int(field_y * px_per_m)
    canvas = np.zeros((h, w, 3), np.uint8)
    in_field = (
        (vertices[:, 0] >= 0) & (vertices[:, 0] < field_x)
        & (vertices[:, 1] >= 0) & (vertices[:, 1] < field_y)
    )
    pts = vertices[in_field]
    if len(pts) == 0:
        print("场地范围内没有顶点，跳过预览图。")
        return
    px = (pts[:, 0] * px_per_m).astype(int)
    py = (pts[:, 1] * px_per_m).astype(int)
    z = pts[:, 2]
    z_norm = np.clip(z / max(z.max(), 1e-6), 0, 1)
    # BGR：蓝=地面，红=高处（围挡/高地/堡垒）
    color = np.stack(
        [255 - 200 * z_norm, 90 + 60 * np.sin(z_norm * np.pi),
         60 + 195 * z_norm], axis=1
    ).astype(np.uint8)
    canvas[h - 1 - py, px] = color
    for m in range(0, int(field_x) + 1, 2):
        cv2.line(canvas, (m * px_per_m, 0), (m * px_per_m, h - 1), (60, 60, 60), 1)
    for m in range(0, int(field_y) + 1, 2):
        cv2.line(canvas, (0, h - 1 - m * px_per_m), (w - 1, h - 1 - m * px_per_m),
                 (60, 60, 60), 1)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (200, 200, 200), 2)
    cv2.putText(canvas, "top view, color=height, 2m grid", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(path, canvas)
    print(f"预览图已写: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stl_path", help="二进制 STL 路径")
    parser.add_argument("--out-dir", default="Main_ws/field",
                        help="输出目录（默认 Main_ws/field）")
    parser.add_argument("--pcd-out", default=None,
                        help="PCD 额外输出目录（如 Lidar_ws 配置目录）")
    parser.add_argument("--name", default="RMUC2027", help="资产名（默认 RMUC2027）")
    parser.add_argument("--voxel", type=float, default=0.05,
                        help="PCD 体素降采样尺寸，米（默认 0.05）")
    parser.add_argument("--no-wall", action="store_true",
                        help="不补 2.4m 外围围挡（默认按规则手册补）")
    parser.add_argument("--field-x", type=float, default=28.0)
    parser.add_argument("--field-y", type=float, default=15.0)
    args = parser.parse_args()

    vertices, faces = read_binary_stl(args.stl_path)
    vertices = normalize_to_field_coords(vertices, args.field_x, args.field_y)
    vertices = detect_floor_and_anchor(
        vertices, faces, args.field_x, args.field_y
    )
    vertices, faces = clip_outside_field(
        vertices, faces, args.field_x, args.field_y, margin=0.3
    )
    if not args.no_wall:
        vertices, faces = add_perimeter_walls(
            vertices, faces, args.field_x, args.field_y
        )

    os.makedirs(args.out_dir, exist_ok=True)
    write_ply(os.path.join(args.out_dir, f"{args.name}.ply"), vertices, faces)
    surface_points = sample_surface(vertices, faces, spacing=args.voxel)
    write_pcd(
        os.path.join(args.out_dir, f"{args.name}.pcd"), surface_points, args.voxel
    )
    write_preview_png(
        os.path.join(args.out_dir, f"{args.name}_preview.png"),
        vertices, args.field_x, args.field_y,
    )
    if args.pcd_out:
        os.makedirs(args.pcd_out, exist_ok=True)
        write_pcd(
            os.path.join(args.pcd_out, f"{args.name}.pcd"),
            surface_points,
            args.voxel,
        )
    print(
        "\n下一步（人工确认预览图朝向后）：\n"
        "  1. 对照规则手册核对 x/y 朝向与红蓝半场是否和 2026 一致（不一致需旋转/镜像）\n"
        "  2. 切换 map 路径：Lidar_ws 两个 params.yaml 的 map 与 Main_ws params.yaml 的场地资源\n"
        "  3. 重新制作 guess_pts.yaml 与雷达外参初值"
    )


if __name__ == "__main__":
    sys.exit(main())
