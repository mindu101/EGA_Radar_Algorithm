#!/usr/bin/env python3
"""官方 STEP 场地模型 -> 二进制 STL（供 stl_to_field_maps.py 消费）。

官方开源的是 STEP（CAD B-rep），需要 OCC 内核做网格化。gmsh 的 Python
wheel 自带 OCC，用它读 STEP、做 2D 表面网格、导出二进制 STL。

用法（gmsh 装在独立 venv，避免污染系统/项目环境）：
  python3 -m venv /tmp/step2stl-venv
  /tmp/step2stl-venv/bin/pip install gmsh
  /tmp/step2stl-venv/bin/python tools/step_to_stl.py 输入.stp 输出.stl \
      [--cl-max 40] [--cl-min 5]

--cl-max/--cl-min 是网格特征长度（模型单位，官方 STEP 通常为毫米）：
40mm ≈ 每平方分米一个三角形，对场地结构足够细。
"""

import argparse
import sys

import gmsh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_path")
    parser.add_argument("stl_path")
    parser.add_argument("--cl-max", type=float, default=40.0)
    parser.add_argument("--cl-min", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=1,
                        help="gmsh 线程数（默认 1：本机是多用户共用，"
                             "网格化吃满所有核会卡死整机）")
    args = parser.parse_args()

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.NumThreads", args.threads)
        print(f"读取 STEP（大文件需几分钟）: {args.step_path}")
        gmsh.model.occ.importShapes(args.step_path)
        gmsh.model.occ.synchronize()

        # gmsh 4.15 的 occ.getBoundingBox 不支持 (-1,-1) 通配，用 model 版
        n_solids = len(gmsh.model.getEntities(3))
        n_surfaces = len(gmsh.model.getEntities(2))
        print(f"实体: {n_solids} 个体 / {n_surfaces} 个面")
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        print(
            f"包围盒(模型单位): x[{xmin:.1f},{xmax:.1f}] "
            f"y[{ymin:.1f},{ymax:.1f}] z[{zmin:.1f},{zmax:.1f}]"
        )

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.cl_max)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.cl_min)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        print("网格化中……")
        gmsh.model.mesh.generate(2)  # 只要表面网格

        n_tri = 0
        for dim, tag in gmsh.model.getEntities(2):
            _, types, _, nodes = gmsh.model.mesh.getElements(dim, tag)
            for t in types:
                if t == 2:  # 三角形
                    n_tri += len(nodes) // 3
        print(f"三角形总数: {n_tri:,}")

        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.write(args.stl_path)
        print(f"二进制 STL 已写: {args.stl_path}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    sys.exit(main())
