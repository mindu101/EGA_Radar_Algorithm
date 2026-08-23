from pathlib import Path

from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_orig_export = torch.onnx.export

def _export_with_fallback(*args, **kwargs):
    kwargs.setdefault("dynamo", True)
    kwargs.setdefault("fallback", True)  # dynamo失败时回退TorchScript exporter
    return _orig_export(*args, **kwargs)

torch.onnx.export = _export_with_fallback
#car
model = YOLO(PROJECT_ROOT / "weights/car/best.pt")
model.export(
    format="engine",  # 导出为TensorRT engine格
    dynamic=False,     # 启用动态批大小
    imgsz=1280, # 固定输入尺寸或范围
    half=True,
    device=0,
    int8=False,
    batch=1,  # 最大批大小(批量处理的最大图像数量)   
) 

#armor
# model = YOLO(PROJECT_ROOT / "weights/armor/armor_best.pt")
# model.export(
#     format="engine",  # 导出为TensorRT engine格
#     dynamic=True,     # 启用动态批大小
#     imgsz=192, # 固定输入尺寸或范围
#     half=True,
#     device=0,
#     int8=False,
#     batch=10,  # 最大批大小(批量处理的最大图像数量)
# )
