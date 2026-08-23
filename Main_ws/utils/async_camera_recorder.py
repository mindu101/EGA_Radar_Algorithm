# -*- coding: utf-8 -*-
"""相机异步录像器。

采集线程只负责把图像引用和采集时间戳放入有界队列；视频编码、CSV 写入和
统计落盘都在后台线程中完成，避免高分辨率编码阻塞相机或 Qt 主线程。
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import cv2


@dataclass(frozen=True)
class _FramePacket:
    frame: np.ndarray
    camera_stamp: Optional[float]
    wall_time_ns: int
    enqueue_monotonic_ns: int


class PyNvVideoCodecWriter:
    """进程内 NVENC 编码器，避免通过管道传输每秒约 1GiB 的 RGB 数据。"""

    backend = "pynvvideocodec_h264_nvenc"

    @staticmethod
    def check_available() -> None:
        try:
            import PyNvVideoCodec  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "缺少 PyNvVideoCodec；请在 cud128 环境执行 "
                "python -m pip install PyNvVideoCodec==2.1.0"
            ) from error

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        input_pixel_format: str,
        log_path: str,
        ffmpeg_path: str = "ffmpeg",
        preset: str = "p3",
        bitrate_mbps: float = 60.0,
        max_bitrate_mbps: float = 100.0,
    ) -> None:
        self.check_available()
        import PyNvVideoCodec as nvc

        executable = shutil.which(ffmpeg_path)
        if executable is None:
            raise RuntimeError(f"找不到 FFmpeg 可执行文件: {ffmpeg_path}")
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError(f"NVENC 要求正偶数分辨率，当前为 {width}x{height}")

        pixel_format = str(input_pixel_format).lower()
        if pixel_format in ("rgb24", "rgba"):
            surface_format = "ABGR"
        elif pixel_format in ("bgr24", "bgra"):
            surface_format = "ARGB"
        elif pixel_format == "gray":
            surface_format = "ABGR"
        else:
            raise ValueError(f"PyNvVideoCodec 不支持输入格式: {input_pixel_format}")

        self.output_path = str(output_path)
        self.log_path = str(log_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.input_pixel_format = pixel_format
        self._ffmpeg_path = executable
        self._closed = False
        self._aborted = False
        self._packed_frame = (
            None
            if pixel_format in ("rgba", "bgra")
            else np.empty((self.height, self.width, 4), dtype=np.uint8)
        )
        self._raw_path = str(Path(self.output_path).with_suffix(".h264"))
        self._raw_file = open(self._raw_path, "wb")
        self._log_file = open(self.log_path, "w", encoding="utf-8")
        config = {
            "gpu_id": 0,
            "codec": "h264",
            # 2.1.0 的预设名称区分大小写，P3 兼顾速度与画质。
            "preset": str(preset).upper(),
            "tuning_info": "high_quality",
            "rc": "vbr",
            "fps": max(1, int(round(self.fps))),
            "bitrate": max(1, int(float(bitrate_mbps) * 1_000_000)),
            "maxbitrate": max(1, int(float(max_bitrate_mbps) * 1_000_000)),
            "vbvbufsize": max(1, int(float(max_bitrate_mbps) * 1_000_000)),
        }
        try:
            self._encoder = nvc.CreateEncoder(
                self.width,
                self.height,
                surface_format,
                True,
                **config,
            )
        except Exception:
            self._raw_file.close()
            self._log_file.close()
            raise

    def _pack_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.input_pixel_format in ("rgba", "bgra"):
            packed = frame
        elif self.input_pixel_format == "rgb24":
            packed = cv2.cvtColor(
                frame, cv2.COLOR_RGB2RGBA, dst=self._packed_frame
            )
        elif self.input_pixel_format == "bgr24":
            packed = cv2.cvtColor(
                frame, cv2.COLOR_BGR2BGRA, dst=self._packed_frame
            )
        elif self.input_pixel_format == "gray":
            packed = cv2.cvtColor(
                frame, cv2.COLOR_GRAY2RGBA, dst=self._packed_frame
            )
        else:
            raise ValueError(f"不支持的输入格式: {self.input_pixel_format}")
        return packed if packed.flags.c_contiguous else np.ascontiguousarray(packed)

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("PyNvVideoCodec 编码器已经关闭")
        packed = self._pack_frame(frame)
        try:
            bitstream = self._encoder.Encode(packed.reshape(-1))
            if bitstream:
                self._raw_file.write(bytearray(bitstream))
        except Exception as error:
            raise RuntimeError(f"PyNvVideoCodec 编码失败: {error}") from error

    def abort(self) -> None:
        self._aborted = True

    def _remux_mp4(self) -> None:
        self._log_file.flush()
        command = [
            self._ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-framerate",
            f"{self.fps:.9f}",
            "-i",
            self._raw_path,
            "-c:v",
            "copy",
            # 裸 H.264 没有容器时间戳；按目标帧率补齐 PTS/DTS，保证 MP4
            # 时长与 camera_timestamps.csv 的 video_time_s 一致。
            "-bsf:v",
            f"setts=ts=N/({self.fps:.9f}*TB)",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
            timeout=60.0,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"H.264 封装 MP4 失败，返回码={result.returncode}；"
                f"原始码流保留在 {self._raw_path}"
            )
        Path(self._raw_path).unlink(missing_ok=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not self._aborted:
                bitstream = self._encoder.EndEncode()
                if bitstream:
                    self._raw_file.write(bytearray(bitstream))
            self._raw_file.flush()
            self._raw_file.close()
            if not self._aborted and Path(self._raw_path).stat().st_size > 0:
                self._remux_mp4()
        finally:
            if not self._raw_file.closed:
                self._raw_file.close()
            self._log_file.close()


class AsyncCameraRecorder:
    """有界队列异步录像器，CSV 与视频帧严格一一对应。"""

    CSV_HEADER = "frame_index,wall_time_ns,wall_time_s,video_time_s,camera_stamp\n"

    def __init__(
        self,
        camera,
        output_path: str,
        timestamp_path: str,
        metrics_path: str,
        target_fps: float,
        color_order: str = "rgb",
        group_id: str = "sensor_recorder",
        queue_size: int = 8,
        encoder_preset: str = "p3",
        encoder_bitrate_mbps: float = 60.0,
        encoder_max_bitrate_mbps: float = 100.0,
        ffmpeg_path: str = "ffmpeg",
        encoder_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("录像帧率必须大于 0")
        color_order = str(color_order).lower()
        if color_order not in ("rgb", "bgr"):
            raise ValueError("color_order 只能是 rgb 或 bgr")

        self.camera = camera
        self.output_path = str(output_path)
        self.timestamp_path = str(timestamp_path)
        self.metrics_path = str(metrics_path)
        self.target_fps = float(target_fps)
        self.color_order = color_order
        self.group_id = str(group_id)
        self.queue_size = max(1, int(queue_size))
        self.encoder_preset = str(encoder_preset)
        self.encoder_bitrate_mbps = float(encoder_bitrate_mbps)
        self.encoder_max_bitrate_mbps = float(encoder_max_bitrate_mbps)
        self.ffmpeg_path = str(ffmpeg_path)
        self._using_default_encoder = encoder_factory is None
        self._encoder_factory = encoder_factory or PyNvVideoCodecWriter

        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._sentinel = object()
        self._stats_lock = threading.Lock()
        self._metrics_write_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._accepting = threading.Event()
        self._poll_stop = threading.Event()
        self._encoder_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._encoder_ready = threading.Event()
        self._listener_token = None
        self._active_writer = None
        self._timestamp_file = None
        self._started = False
        self._stopped = False
        self._fatal_error: Optional[str] = None
        self._next_due_stamp: Optional[float] = None
        self._last_observed_stamp: Optional[float] = None
        self._source_period_ema: Optional[float] = None
        self._last_metrics_write_monotonic = 0.0
        self._started_wall_time = 0.0
        self._stopped_wall_time = None
        self._first_observed_stamp = None
        self._last_observed_for_rate = None
        self._first_encoded_stamp = None
        self._last_encoded_stamp = None
        self._capture_seen = 0
        self._sampled_out = 0
        self._enqueued = 0
        self._queue_overflow_dropped = 0
        self._shutdown_dropped = 0
        self._encoded = 0
        self._queue_max_depth = 0
        self._encoder_write_total_ms = 0.0
        self._encoder_write_max_ms = 0.0
        self._encoder_init_ms = 0.0
        self._queue_wait_total_ms = 0.0
        self._queue_wait_max_ms = 0.0
        self._record_latency_total_ms = 0.0
        self._record_latency_max_ms = 0.0
        self._encoder_backend = "pynvvideocodec_h264_nvenc"
        self._width = None
        self._height = None
        self._input_pixel_format = None

    @staticmethod
    def _finite_stamp(value) -> Optional[float]:
        if value is None:
            return None
        try:
            stamp = float(value)
        except (TypeError, ValueError):
            return None
        return stamp if math.isfinite(stamp) else None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started and not self._stopped:
                return
            if self._stopped:
                raise RuntimeError("异步录像器停止后不能再次启动")
            if self._using_default_encoder:
                PyNvVideoCodecWriter.check_available()
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.timestamp_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.metrics_path).parent.mkdir(parents=True, exist_ok=True)
            self._timestamp_file = open(
                self.timestamp_path, "w", encoding="utf-8", newline=""
            )
            self._timestamp_file.write(self.CSV_HEADER)
            self._timestamp_file.flush()
            self._started_wall_time = time.time()
            # 必须在线程和监听器启动前置位；若后续任一步骤失败，stop() 才能
            # 完整关闭时间戳文件和已经启动的编码线程。
            self._started = True
            self._set_camera_layout_hint()
            self._encoder_thread = threading.Thread(
                target=self._encoder_loop,
                name="camera-record-encoder",
                daemon=True,
            )
            self._encoder_thread.start()
            if not self._encoder_ready.wait(timeout=10.0):
                raise RuntimeError("NVENC 编码器初始化超时")
            if self._fatal_error:
                raise RuntimeError(self._fatal_error)
            # 编码器预热完成后才接收相机帧，避免启动阶段填满有界队列。
            self._accepting.set()

            add_listener = getattr(self.camera, "add_frame_listener", None)
            if callable(add_listener):
                self._listener_token = add_listener(self.submit_frame)
            else:
                self._poll_thread = threading.Thread(
                    target=self._poll_camera_loop,
                    name="camera-record-capture",
                    daemon=True,
                )
                self._poll_thread.start()
        self._write_metrics(force=True)

    def _poll_camera_loop(self) -> None:
        while self._accepting.is_set() and not self._poll_stop.is_set():
            try:
                frame, camera_stamp = self.camera.get_image_latest(
                    self.group_id, timeout=0.1
                )
            except Exception as error:
                self._set_fatal_error(f"读取相机录像帧失败: {error}")
                return
            if frame is not None:
                self.submit_frame(frame, camera_stamp)

    def _should_sample(self, sample_stamp: float) -> bool:
        period = 1.0 / self.target_fps
        previous = self._last_observed_stamp
        if previous is not None:
            delta = sample_stamp - previous
            if delta > 1e-6:
                if self._source_period_ema is None:
                    self._source_period_ema = delta
                else:
                    self._source_period_ema = (
                        0.9 * self._source_period_ema + 0.1 * delta
                    )
            elif delta < -period:
                self._next_due_stamp = None
                self._source_period_ema = None
        self._last_observed_stamp = sample_stamp

        if self._next_due_stamp is None:
            self._next_due_stamp = sample_stamp + period
            return True

        source_period = self._source_period_ema or period
        tolerance = min(0.45 * period, 0.4 * source_period)
        if sample_stamp + tolerance < self._next_due_stamp:
            return False
        while self._next_due_stamp <= sample_stamp + tolerance:
            self._next_due_stamp += period
        return True

    def submit_frame(self, frame: np.ndarray, camera_stamp=None) -> bool:
        """由相机采集线程调用；不得执行图像复制、编码或文件 IO。"""
        if not self._accepting.is_set() or frame is None:
            return False
        wall_time_ns = time.time_ns()
        normalized_stamp = self._finite_stamp(camera_stamp)
        sample_stamp = (
            normalized_stamp
            if normalized_stamp is not None
            else wall_time_ns / 1_000_000_000.0
        )

        with self._stats_lock:
            self._capture_seen += 1
            if self._first_observed_stamp is None:
                self._first_observed_stamp = sample_stamp
            self._last_observed_for_rate = sample_stamp
            if not self._should_sample(sample_stamp):
                self._sampled_out += 1
                return False

        packet = _FramePacket(
            frame=frame,
            camera_stamp=normalized_stamp,
            wall_time_ns=wall_time_ns,
            enqueue_monotonic_ns=time.perf_counter_ns(),
        )
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            with self._stats_lock:
                self._queue_overflow_dropped += 1
            return False

        depth = self._queue.qsize()
        with self._stats_lock:
            self._enqueued += 1
            self._queue_max_depth = max(self._queue_max_depth, depth)
        return True

    def _frame_layout(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        if frame.dtype != np.uint8:
            raise ValueError(f"录像仅支持 uint8 图像，当前为 {frame.dtype}")
        if frame.ndim == 2:
            pixel_format = "gray"
        elif frame.ndim == 3 and frame.shape[2] == 3:
            pixel_format = "rgb24" if self.color_order == "rgb" else "bgr24"
        elif frame.ndim == 3 and frame.shape[2] == 4:
            pixel_format = "rgba" if self.color_order == "rgb" else "bgra"
        else:
            raise ValueError(f"不支持的录像图像形状: {frame.shape}")
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        return frame, pixel_format

    def _set_camera_layout_hint(self) -> None:
        """从相机配置预取三通道图像尺寸，用于在收帧前预热编码器。"""
        camera_args = getattr(self.camera, "args", None)
        width = getattr(self.camera, "width", None)
        height = getattr(self.camera, "height", None)
        if width is None and camera_args is not None:
            width = getattr(camera_args, "width", None)
        if height is None and camera_args is not None:
            height = getattr(camera_args, "height", None)
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError):
            return
        if width <= 0 or height <= 0:
            return
        self._width = width
        self._height = height
        self._input_pixel_format = (
            "rgb24" if self.color_order == "rgb" else "bgr24"
        )

    def _create_writer(self, width: int, height: int, pixel_format: str):
        log_path = str(Path(self.output_path).with_name("camera_encoder.log"))
        writer = self._encoder_factory(
            output_path=self.output_path,
            width=width,
            height=height,
            fps=self.target_fps,
            input_pixel_format=pixel_format,
            log_path=log_path,
            ffmpeg_path=self.ffmpeg_path,
            preset=self.encoder_preset,
            bitrate_mbps=self.encoder_bitrate_mbps,
            max_bitrate_mbps=self.encoder_max_bitrate_mbps,
        )
        self._width = int(width)
        self._height = int(height)
        self._input_pixel_format = pixel_format
        self._encoder_backend = getattr(writer, "backend", self._encoder_backend)
        self._active_writer = writer
        return writer

    def _write_timestamp_row(self, packet: _FramePacket, frame_index: int) -> None:
        file = self._timestamp_file
        if file is None:
            raise RuntimeError("相机时间戳文件未打开")
        wall_time_s = packet.wall_time_ns / 1_000_000_000.0
        video_time_s = frame_index / self.target_fps
        camera_stamp_text = (
            "" if packet.camera_stamp is None else f"{packet.camera_stamp:.9f}"
        )
        file.write(
            f"{frame_index},{packet.wall_time_ns},{wall_time_s:.9f},"
            f"{video_time_s:.9f},{camera_stamp_text}\n"
        )
        file.flush()

    def _encoder_loop(self) -> None:
        writer = None
        try:
            if (
                self._width is not None
                and self._height is not None
                and self._input_pixel_format is not None
            ):
                begin = time.perf_counter()
                writer = self._create_writer(
                    self._width, self._height, self._input_pixel_format
                )
                self._encoder_init_ms = (time.perf_counter() - begin) * 1000.0
            self._encoder_ready.set()
            while True:
                packet = self._queue.get()
                try:
                    if packet is self._sentinel:
                        return
                    frame, pixel_format = self._frame_layout(packet.frame)
                    if writer is None:
                        begin = time.perf_counter()
                        writer = self._create_writer(
                            frame.shape[1], frame.shape[0], pixel_format
                        )
                        self._encoder_init_ms = (
                            time.perf_counter() - begin
                        ) * 1000.0
                    elif (
                        frame.shape[1] != self._width
                        or frame.shape[0] != self._height
                        or pixel_format != self._input_pixel_format
                    ):
                        raise ValueError(
                            "录像期间图像格式发生变化: "
                            f"{frame.shape[1]}x{frame.shape[0]}/{pixel_format}"
                        )

                    queue_wait_ms = (
                        time.perf_counter_ns() - packet.enqueue_monotonic_ns
                    ) / 1_000_000.0
                    begin = time.perf_counter()
                    writer.write(frame)
                    elapsed_ms = (time.perf_counter() - begin) * 1000.0
                    with self._stats_lock:
                        frame_index = self._encoded
                    self._write_timestamp_row(packet, frame_index)
                    encoded_stamp = (
                        packet.camera_stamp
                        if packet.camera_stamp is not None
                        else packet.wall_time_ns / 1_000_000_000.0
                    )
                    record_latency_ms = (
                        time.perf_counter_ns() - packet.enqueue_monotonic_ns
                    ) / 1_000_000.0
                    with self._stats_lock:
                        self._encoded += 1
                        self._encoder_write_total_ms += elapsed_ms
                        self._encoder_write_max_ms = max(
                            self._encoder_write_max_ms, elapsed_ms
                        )
                        self._queue_wait_total_ms += queue_wait_ms
                        self._queue_wait_max_ms = max(
                            self._queue_wait_max_ms, queue_wait_ms
                        )
                        self._record_latency_total_ms += record_latency_ms
                        self._record_latency_max_ms = max(
                            self._record_latency_max_ms, record_latency_ms
                        )
                        if self._first_encoded_stamp is None:
                            self._first_encoded_stamp = encoded_stamp
                        self._last_encoded_stamp = encoded_stamp
                    self._write_metrics()
                finally:
                    self._queue.task_done()
        except Exception as error:
            self._set_fatal_error(str(error))
        finally:
            self._encoder_ready.set()
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:
                    self._set_fatal_error(str(error))
            self._active_writer = None

    def _set_fatal_error(self, message: str) -> None:
        with self._stats_lock:
            if self._fatal_error is None:
                self._fatal_error = str(message)
        self._accepting.clear()
        self._poll_stop.set()

    @staticmethod
    def _rate(count: int, first_stamp, last_stamp) -> float:
        if count <= 1 or first_stamp is None or last_stamp is None:
            return 0.0
        duration = float(last_stamp) - float(first_stamp)
        return (count - 1) / duration if duration > 1e-9 else 0.0

    def get_diagnostics(self) -> dict:
        with self._stats_lock:
            encoded = self._encoded
            elapsed = (
                max(
                    0.0,
                    (self._stopped_wall_time or time.time())
                    - self._started_wall_time,
                )
                if self._started_wall_time > 0.0
                else 0.0
            )
            return {
                "version": 1,
                "status": (
                    "error"
                    if self._fatal_error
                    else ("stopped" if self._stopped else "recording")
                ),
                "backend": self._encoder_backend,
                "target_fps": self.target_fps,
                "width": self._width,
                "height": self._height,
                "input_pixel_format": self._input_pixel_format,
                "queue_capacity": self.queue_size,
                "queue_depth": self._queue.qsize(),
                "queue_max_depth": self._queue_max_depth,
                "capture_seen": self._capture_seen,
                "capture_observed_fps": self._rate(
                    self._capture_seen,
                    self._first_observed_stamp,
                    self._last_observed_for_rate,
                ),
                "sampled_out": self._sampled_out,
                "enqueued": self._enqueued,
                "queue_overflow_dropped": self._queue_overflow_dropped,
                "shutdown_dropped": self._shutdown_dropped,
                "encoded": encoded,
                "encoded_source_fps": self._rate(
                    encoded, self._first_encoded_stamp, self._last_encoded_stamp
                ),
                "encoder_write_avg_ms": (
                    self._encoder_write_total_ms / encoded if encoded else 0.0
                ),
                "encoder_write_max_ms": self._encoder_write_max_ms,
                "encoder_init_ms": self._encoder_init_ms,
                "queue_wait_avg_ms": (
                    self._queue_wait_total_ms / encoded if encoded else 0.0
                ),
                "queue_wait_max_ms": self._queue_wait_max_ms,
                "record_latency_avg_ms": (
                    self._record_latency_total_ms / encoded if encoded else 0.0
                ),
                "record_latency_max_ms": self._record_latency_max_ms,
                "started_wall_time": self._started_wall_time,
                "stopped_wall_time": self._stopped_wall_time,
                "elapsed_wall_time_s": elapsed,
                "fatal_error": self._fatal_error,
                "video_path": self.output_path,
                "timestamp_path": self.timestamp_path,
                "metrics_path": self.metrics_path,
            }

    def _write_metrics(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_metrics_write_monotonic < 1.0:
            return
        if not self._metrics_write_lock.acquire(blocking=False):
            return
        try:
            self._last_metrics_write_monotonic = now
            metrics = self.get_diagnostics()
            temporary_path = f"{self.metrics_path}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(metrics, file, ensure_ascii=False, indent=2)
                file.write("\n")
            os.replace(temporary_path, self.metrics_path)
        finally:
            self._metrics_write_lock.release()

    def stop(self) -> dict:
        with self._lifecycle_lock:
            if self._stopped:
                return self.get_diagnostics()
            if not self._started:
                self._stopped = True
                return self.get_diagnostics()
            self._accepting.clear()
            self._poll_stop.set()

            if self._listener_token is not None:
                remove_listener = getattr(self.camera, "remove_frame_listener", None)
                if callable(remove_listener):
                    try:
                        remove_listener(self._listener_token)
                    except Exception as error:
                        self._set_fatal_error(f"移除相机录像监听器失败: {error}")
                self._listener_token = None

        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            if self._poll_thread.is_alive():
                self._set_fatal_error("相机录像取帧线程停止超时")

        encoder = self._encoder_thread
        if encoder is not None and encoder.is_alive():
            deadline = time.monotonic() + 10.0
            while encoder.is_alive():
                try:
                    self._queue.put(self._sentinel, timeout=0.1)
                    break
                except queue.Full:
                    if time.monotonic() >= deadline:
                        self._set_fatal_error("录像队列排空超时")
                        shutdown_dropped = 0
                        while True:
                            try:
                                packet = self._queue.get_nowait()
                            except queue.Empty:
                                break
                            else:
                                if packet is not self._sentinel:
                                    shutdown_dropped += 1
                                self._queue.task_done()
                        with self._stats_lock:
                            self._shutdown_dropped += shutdown_dropped
                        self._queue.put(self._sentinel)
                        break
            encoder.join(timeout=40.0)
            if encoder.is_alive():
                writer = self._active_writer
                if writer is not None and hasattr(writer, "abort"):
                    writer.abort()
                encoder.join(timeout=5.0)
                if encoder.is_alive():
                    self._set_fatal_error("相机编码线程停止超时")

        if self._timestamp_file is not None:
            self._timestamp_file.flush()
            self._timestamp_file.close()
            self._timestamp_file = None
        with self._stats_lock:
            self._stopped_wall_time = time.time()
            self._stopped = True
        self._write_metrics(force=True)
        return self.get_diagnostics()

    def close(self) -> dict:
        return self.stop()
