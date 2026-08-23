# -*- coding: utf-8 -*-
"""运行时性能指标的低开销后台导出工具。"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime


logger = logging.getLogger(__name__)


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlGpuSampler:
    """直接调用 NVML，避免依赖 pynvml 或高频创建 nvidia-smi 进程。"""

    def __init__(self, device_index: int = 0):
        self.available = False
        self.error = None
        self._lib = None
        self._handle = ctypes.c_void_p()
        try:
            self._lib = ctypes.CDLL("libnvidia-ml.so.1")
            self._configure_functions()
            self._check(self._lib.nvmlInit_v2(), "nvmlInit_v2")
            self._check(
                self._get_handle(int(device_index), ctypes.byref(self._handle)),
                "nvmlDeviceGetHandleByIndex",
            )
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def _configure_functions(self):
        self._lib.nvmlInit_v2.restype = ctypes.c_int
        self._lib.nvmlShutdown.restype = ctypes.c_int

        self._get_handle = getattr(
            self._lib, "nvmlDeviceGetHandleByIndex_v2", None
        )
        if self._get_handle is None:
            self._get_handle = self._lib.nvmlDeviceGetHandleByIndex
        self._get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self._get_handle.restype = ctypes.c_int

        self._lib.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlUtilization),
        ]
        self._lib.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self._lib.nvmlDeviceGetMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlMemory),
        ]
        self._lib.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, function_name: str):
        if code != 0:
            raise RuntimeError(f"{function_name} 返回 NVML 错误码 {code}")

    def sample(self) -> dict:
        if not self.available:
            return {"available": False, "error": self.error}

        utilization = _NvmlUtilization()
        memory = _NvmlMemory()
        try:
            self._check(
                self._lib.nvmlDeviceGetUtilizationRates(
                    self._handle, ctypes.byref(utilization)
                ),
                "nvmlDeviceGetUtilizationRates",
            )
            self._check(
                self._lib.nvmlDeviceGetMemoryInfo(
                    self._handle, ctypes.byref(memory)
                ),
                "nvmlDeviceGetMemoryInfo",
            )
        except Exception as exc:
            return {"available": False, "error": str(exc)}

        mib = 1024.0 * 1024.0
        return {
            "available": True,
            "gpu_util_percent": int(utilization.gpu),
            "memory_util_percent": int(utilization.memory),
            "memory_used_mib": round(memory.used / mib, 1),
            "memory_total_mib": round(memory.total / mib, 1),
        }

    def close(self):
        if self._lib is not None and self.available:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass
        self.available = False


class RuntimeMetricsExporter:
    """将逐帧指标放入有界队列，由后台线程写入 JSONL。"""

    def __init__(self, config: dict | None):
        config = config if isinstance(config, dict) else {}
        self.enabled = bool(config.get("enabled", True))
        self.output_path = os.path.abspath(
            str(config.get("output_path", "logs/performance/runtime_metrics.jsonl"))
        )
        self.flush_interval_sec = max(
            0.1, float(config.get("flush_interval_sec", 1.0))
        )
        self.gpu_sample_interval_sec = max(
            0.1, float(config.get("gpu_sample_interval_sec", 0.5))
        )
        self.device_index = max(0, int(config.get("gpu_device_index", 0)))
        self._queue = queue.Queue(maxsize=max(16, int(config.get("queue_size", 512))))
        self._dropped_records = 0
        self._stop_token = object()
        self._thread = None

        if self.enabled:
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="runtime-metrics-writer",
                daemon=True,
            )
            self._thread.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    def submit(self, record: dict):
        """非阻塞提交；磁盘过慢时丢弃最旧记录，不能反向阻塞主流程。"""
        if not self.enabled:
            return
        record = dict(record)
        record.setdefault("record_type", "frame")
        record.setdefault("wall_time", time.time())
        try:
            self._queue.put_nowait(record)
            return
        except queue.Full:
            self._dropped_records += 1

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_records += 1

    def _writer_loop(self):
        sampler = NvmlGpuSampler(self.device_index)
        latest_gpu = sampler.sample()
        last_gpu_sample = time.monotonic()
        last_flush = time.monotonic()

        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8", buffering=1) as file:
                session_record = {
                    "record_type": "session_start",
                    "wall_time": time.time(),
                    "local_time": datetime.now().isoformat(timespec="milliseconds"),
                    "pid": os.getpid(),
                    "gpu": latest_gpu,
                }
                file.write(json.dumps(session_record, ensure_ascii=False) + "\n")

                while True:
                    try:
                        item = self._queue.get(timeout=self.flush_interval_sec)
                    except queue.Empty:
                        item = None

                    now = time.monotonic()
                    if now - last_gpu_sample >= self.gpu_sample_interval_sec:
                        latest_gpu = sampler.sample()
                        last_gpu_sample = now

                    if item is self._stop_token:
                        break
                    if item is not None:
                        item["gpu"] = latest_gpu
                        item["metrics_queue_dropped"] = self._dropped_records
                        file.write(json.dumps(item, ensure_ascii=False) + "\n")

                    if now - last_flush >= self.flush_interval_sec:
                        file.flush()
                        last_flush = now

                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is self._stop_token:
                        continue
                    item["gpu"] = latest_gpu
                    item["metrics_queue_dropped"] = self._dropped_records
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                file.flush()
        except Exception:
            logger.exception("运行时性能指标导出线程异常")
        finally:
            sampler.close()

    def close(self):
        if not self.enabled or self._thread is None:
            return
        try:
            self._queue.put_nowait(self._stop_token)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(self._stop_token)
            except queue.Full:
                pass
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logger.warning("性能指标导出线程未在超时时间内退出")
        self._thread = None
