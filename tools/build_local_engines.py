#!/usr/bin/env python3
"""Build FP16 engines for this GPU from the supplied ONNX models (TensorRT 10)."""

import argparse
import json
from pathlib import Path
import struct

import torch
import tensorrt as trt
from ultralytics.utils.export.engine import onnx2engine


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "Main_ws/weights"
MODELS = {
    "car": (WEIGHTS / "car/best", (1, 3, 1280, 1280), False),
    "armor": (WEIGHTS / "armor/armor_best", (10, 3, 192, 192), True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--workspace", type=float, default=1.0, help="Workspace limit in GiB")
    args = parser.parse_args()
    if not trt.__version__.startswith("10."):
        parser.error("This FP16 build uses TensorRT 10; see learning/12-本机部署记录.md")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is unavailable")
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, TensorRT: {trt.__version__}", flush=True)
    output_dir = WEIGHTS / "local"
    output_dir.mkdir(exist_ok=True)
    for name, (source, shape, dynamic) in MODELS.items():
        if args.model not in ("all", name):
            continue
        # Keep class labels and preprocessing metadata from the original export.
        with source.with_suffix(".engine").open("rb") as stream:
            length = struct.unpack("<I", stream.read(4))[0]
            if not 0 < length < 100_000:
                raise ValueError(f"Invalid Ultralytics metadata: {source}")
            metadata = json.loads(stream.read(length))
        destination = output_dir / source.with_suffix(".engine").name
        temporary = destination.with_suffix(".building.engine")
        try:
            onnx2engine(
                str(source.with_suffix(".onnx")),
                output_file=str(temporary),
                workspace=args.workspace,
                quantize=16,
                dynamic=dynamic,
                shape=shape,
                metadata=metadata,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Built {destination}", flush=True)


if __name__ == "__main__":
    main()
