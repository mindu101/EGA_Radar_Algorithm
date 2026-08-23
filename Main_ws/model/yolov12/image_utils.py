"""YOLO 图像输入格式转换工具。"""

from __future__ import annotations

import numpy as np


def rgb_to_ultralytics_bgr_view(image: np.ndarray) -> np.ndarray:
    """把项目内部 RGB 图像转换为 Ultralytics NumPy 接口所需的 BGR 视图。

    项目相机链路统一使用 RGB，而 Ultralytics 的 NumPy 输入接口会把输入当作
    BGR，并在预处理阶段转换为 RGB。这里只反转通道维并共享底层内存，避免
    为每帧额外复制整张图像。
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image 必须为 numpy.ndarray，实际为 {type(image)!r}")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image 必须为 HxWx3，实际形状为 {image.shape}")
    return image[..., ::-1]
