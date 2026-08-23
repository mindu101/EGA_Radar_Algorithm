from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections import deque
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

from transform.ray_renderer import _get_open3d


logger = logging.getLogger(__name__)


class XYMapManager:
    """管理原图分辨率像素到场地二维坐标的异步映射。"""

    STATUS_DISABLED = "未启用"
    STATUS_WAITING = "等待标定"
    STATUS_BUILDING = "生成中"
    STATUS_READY = "已就绪"
    STATUS_FAILED = "生成失败"
    STATUS_CONSISTENCY_FAILED = "一致性检查失败，已禁用"

    def __init__(self, config: dict, metrics_callback: Optional[Callable[[dict], None]] = None):
        projection_cfg = config.get("ray_projection", {})
        self.chunk_rows = int(projection_cfg.get("generation_chunk_rows", 64))
        self.field_length = float(config.get("lidar_fusion", {}).get("field_length", 28.0))
        self.field_width = float(config.get("lidar_fusion", {}).get("field_width", 15.0))
        self.mesh_path = str(config.get("field", {}).get("mesh_path", ""))
        consistency_cfg = projection_cfg.get("consistency_check", {})
        self.consistency_enabled = bool(consistency_cfg.get("enabled", True))
        self.consistency_sample_count = max(
            1, int(consistency_cfg.get("sample_count", 10))
        )
        self.consistency_max_error_m = max(
            0.0, float(consistency_cfg.get("max_error_m", 0.10))
        )
        self._lock = threading.RLock()
        self._enabled = bool(projection_cfg.get("xy_maps_default_enabled", False))
        self._calibration_ready = False
        self._status = self.STATUS_WAITING if self._enabled else self.STATUS_DISABLED
        self._frame_size: Optional[Tuple[int, int]] = None
        self._converter = None
        self._xy_map: Optional[np.ndarray] = None
        self._valid_mask: Optional[np.ndarray] = None
        self._active_signature: Optional[str] = None
        self._building_signature: Optional[str] = None
        self._generation = 0
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._metrics_callback = metrics_callback
        self._consistency_state = (
            "waiting" if self.consistency_enabled else "disabled"
        )
        self._consistency_errors: list[float] = []
        self._consistency_invalid_count = 0
        self._consistency_signature: Optional[str] = None
        self._consistency_events = deque()
        self.last_metrics = {
            "xy_map_generate_ms": None,
            "xy_map_width": None,
            "xy_map_height": None,
            "xy_map_valid_ratio": None,
            "xy_map_generation_error": None,
        }

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def status_text(self) -> str:
        """返回包含初期一致性检查进度的 UI 状态文本。"""
        with self._lock:
            if self._status != self.STATUS_READY:
                return self._status
            if not self.consistency_enabled:
                return self._status
            if self._consistency_state == "checking":
                return (
                    f"{self._status}；一致性检查 "
                    f"{len(self._consistency_errors)}/"
                    f"{self.consistency_sample_count}"
                )
            if self._consistency_state == "passed":
                return f"{self._status}；一致性检查通过"
            return self._status

    def _reset_consistency_locked(self) -> None:
        self._consistency_state = (
            "waiting" if self.consistency_enabled else "disabled"
        )
        self._consistency_errors = []
        self._consistency_invalid_count = 0
        self._consistency_signature = None

    def _start_consistency_check_locked(self, signature: str) -> None:
        if not self.consistency_enabled:
            self._consistency_state = "disabled"
            return
        self._consistency_state = "checking"
        self._consistency_errors = []
        self._consistency_invalid_count = 0
        self._consistency_signature = signature

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            if bool(enabled) and self._consistency_state == "failed":
                self._enabled = False
                self._status = self.STATUS_CONSISTENCY_FAILED
                return
            self._enabled = bool(enabled)
            if not self._enabled:
                self._status = self.STATUS_DISABLED
                if self._consistency_state == "checking":
                    self._reset_consistency_locked()
                return
            if not self._calibration_ready:
                self._status = self.STATUS_WAITING
                return
            if self._active_signature is not None and self._xy_map is not None:
                expected = self._make_signature_locked()
                if expected == self._active_signature:
                    self._status = self.STATUS_READY
                    if self._consistency_state == "waiting":
                        self._start_consistency_check_locked(expected)
                    return
        self._request_build_if_possible()

    def set_frame_size(self, width: int, height: int) -> None:
        size = (int(width), int(height))
        if size[0] <= 0 or size[1] <= 0:
            return
        with self._lock:
            self._frame_size = size
        # 每帧轻量核对完整签名，因此运行中替换 mesh 也会自动失效重建。
        self._request_build_if_possible()

    def set_calibration(self, converter) -> None:
        """接收已完成 UI 标定的新转换器，并使旧映射失效。"""
        with self._lock:
            self._converter = converter
            self._calibration_ready = converter is not None
            self._generation += 1
            self._building_signature = None
            self._reset_consistency_locked()
            if self._enabled:
                self._status = self.STATUS_BUILDING if self._frame_size else self.STATUS_WAITING
            else:
                self._status = self.STATUS_DISABLED
        self._request_build_if_possible()

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._active_signature = None
            self._building_signature = None
            self._reset_consistency_locked()
            if self._enabled:
                self._status = self.STATUS_WAITING if not self._calibration_ready else self.STATUS_BUILDING
            else:
                self._status = self.STATUS_DISABLED
        self._request_build_if_possible()

    def _make_signature_locked(self) -> Optional[str]:
        if self._converter is None or self._frame_size is None:
            return None
        mesh_mtime = os.path.getmtime(self.mesh_path) if os.path.exists(self.mesh_path) else -1.0
        digest = hashlib.sha256()
        for value in (
            self._frame_size,
            np.asarray(self._converter.camera_matrix, dtype=np.float64),
            np.asarray(self._converter.dist_coeffs, dtype=np.float64),
            np.asarray(self._converter.R, dtype=np.float64),
            np.asarray(self._converter.T, dtype=np.float64),
            self.mesh_path,
            mesh_mtime,
        ):
            if isinstance(value, np.ndarray):
                digest.update(value.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
        return digest.hexdigest()

    def _request_build_if_possible(self) -> None:
        with self._lock:
            if not self._enabled or not self._calibration_ready or self._frame_size is None:
                if self._enabled and not self._calibration_ready:
                    self._status = self.STATUS_WAITING
                return
            signature = self._make_signature_locked()
            if signature is None:
                return
            if signature == self._active_signature and self._xy_map is not None:
                self._status = self.STATUS_READY
                return
            if signature == self._building_signature and self._worker is not None and self._worker.is_alive():
                self._status = self.STATUS_BUILDING
                return

            self._generation += 1
            generation = self._generation
            self._building_signature = signature
            self._status = self.STATUS_BUILDING
            converter = self._converter
            frame_size = self._frame_size
            worker = threading.Thread(
                target=self._build_worker,
                args=(generation, signature, converter, frame_size),
                name=f"xy-map-{generation}",
                daemon=True,
            )
            self._worker = worker
            worker.start()

    def _build_worker(self, generation: int, signature: str, converter, frame_size: Tuple[int, int]) -> None:
        started = time.perf_counter()
        width, height = frame_size
        error_text = None
        try:
            o3d = _get_open3d()
            # Map 的宽高直接来自当前真实帧 img.shape；内参和逐框直接投影
            # 使用同一份 K，不再依据 YAML 中的历史图像尺寸进行二次缩放。
            camera_matrix = np.asarray(
                converter.camera_matrix, dtype=np.float64
            ).copy()
            inv_k = np.linalg.inv(camera_matrix)
            dist_coeffs = np.asarray(converter.dist_coeffs, dtype=np.float64)
            rotation_t = np.asarray(converter.R, dtype=np.float64).T
            translation = np.asarray(converter.T, dtype=np.float64).reshape(3)
            origin = -rotation_t @ translation
            # 映射线程使用独立 RaycastingScene，避免和主线程逐框投影并发访问同一场景。
            scene = converter.scene
            if hasattr(o3d, "t") and hasattr(converter, "mesh"):
                isolated_scene = o3d.t.geometry.RaycastingScene()
                isolated_scene.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(converter.mesh)
                )
                scene = isolated_scene
            xy_map = np.full((height, width, 2), np.nan, dtype=np.float32)
            valid_mask = np.zeros((height, width), dtype=bool)

            for row_start in range(0, height, self.chunk_rows):
                if self._stop_event.is_set():
                    return
                with self._lock:
                    if generation != self._generation:
                        return
                row_end = min(height, row_start + self.chunk_rows)
                vv, uu = np.indices((row_end - row_start, width), dtype=np.float32)
                vv += float(row_start)
                pixels = np.stack((uu.reshape(-1), vv.reshape(-1)), axis=1)
                if dist_coeffs.size and not np.all(dist_coeffs == 0):
                    pixels = cv2.undistortPoints(
                        pixels.reshape(-1, 1, 2),
                        camera_matrix,
                        dist_coeffs,
                        P=camera_matrix,
                    ).reshape(-1, 2)
                pixel_hom = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float32)))
                camera_dirs = (inv_k @ pixel_hom.T).T
                world_dirs = (rotation_t @ camera_dirs.T).T
                origins = np.broadcast_to(origin, world_dirs.shape)
                rays = np.column_stack((origins, world_dirs)).astype(np.float32, copy=False)
                result = scene.cast_rays(
                    o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
                )
                t_hit = result["t_hit"].numpy().reshape(-1)
                valid = np.isfinite(t_hit) & (t_hit >= 0.0)
                hits = np.full((len(t_hit), 2), np.nan, dtype=np.float32)
                if np.any(valid):
                    hit_xyz = origin + t_hit[valid, None] * world_dirs[valid]
                    hit_xy = hit_xyz[:, :2]
                    inside = (
                        np.isfinite(hit_xy).all(axis=1)
                        & (hit_xy[:, 0] >= 0.0)
                        & (hit_xy[:, 0] <= self.field_length)
                        & (hit_xy[:, 1] >= 0.0)
                        & (hit_xy[:, 1] <= self.field_width)
                    )
                    valid_indices = np.flatnonzero(valid)
                    accepted = valid_indices[inside]
                    hits[accepted] = hit_xy[inside].astype(np.float32, copy=False)
                    valid[:] = False
                    valid[accepted] = True
                xy_map[row_start:row_end] = hits.reshape(row_end - row_start, width, 2)
                valid_mask[row_start:row_end] = valid.reshape(row_end - row_start, width)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            metrics = {
                "xy_map_generate_ms": elapsed_ms,
                "xy_map_width": width,
                "xy_map_height": height,
                "xy_map_valid_ratio": float(valid_mask.mean()),
                "xy_map_generation_error": None,
            }
            with self._lock:
                if generation != self._generation or signature != self._building_signature:
                    return
                self._xy_map = xy_map
                self._valid_mask = valid_mask
                self._active_signature = signature
                self._building_signature = None
                self._status = self.STATUS_READY if self._enabled else self.STATUS_DISABLED
                if self._enabled:
                    self._start_consistency_check_locked(signature)
                self.last_metrics = metrics
            if self._metrics_callback is not None:
                self._metrics_callback(dict(metrics))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            logger.exception("XY Map 生成失败")
            metrics = {
                "xy_map_generate_ms": (time.perf_counter() - started) * 1000.0,
                "xy_map_width": width,
                "xy_map_height": height,
                "xy_map_valid_ratio": None,
                "xy_map_generation_error": error_text,
            }
            with self._lock:
                if generation != self._generation:
                    return
                self._enabled = False
                self._building_signature = None
                self._status = self.STATUS_FAILED
                self._reset_consistency_locked()
                self.last_metrics = metrics
            if self._metrics_callback is not None:
                self._metrics_callback(dict(metrics))

    def project_pixel(
        self,
        u: float,
        v: float,
        frame_size: Tuple[int, int],
        direct_project: Callable[[Sequence[float]], Optional[Sequence[float]]],
    ) -> tuple[Optional[Tuple[float, float, float]], str]:
        """就绪时只查表；等待/生成期间按规范使用当前外参直接投影。"""
        map_point = None
        map_is_valid = False
        consistency_signature = None
        with self._lock:
            authoritative_map = (
                self._enabled
                and self._status == self.STATUS_READY
                and self._frame_size == tuple(frame_size)
                and self._xy_map is not None
                and self._valid_mask is not None
            )
            if authoritative_map:
                u_idx = int(math.floor(float(u) + 0.5))
                v_idx = int(math.floor(float(v) + 0.5))
                width, height = frame_size
                if not (0 <= u_idx < width and 0 <= v_idx < height):
                    map_is_valid = False
                elif bool(self._valid_mask[v_idx, u_idx]):
                    point = self._xy_map[v_idx, u_idx]
                    map_point = (float(point[0]), float(point[1]), 0.0)
                    map_is_valid = True
                if self._consistency_state == "checking":
                    consistency_signature = self._active_signature

        if authoritative_map:
            if consistency_signature is not None:
                direct_point = direct_project((float(u), float(v)))
                self._record_consistency_sample(
                    consistency_signature,
                    map_point if map_is_valid else None,
                    direct_point,
                )
                # 一致性检查通过前，10 个验证样本仍以逐框投影作为正式输出；
                # 避免错误 Map 在被自动关闭前污染定位匹配。
                if direct_point is None:
                    return None, "direct"
                direct_values = tuple(float(value) for value in direct_point)
                if len(direct_values) < 3 or not all(
                    math.isfinite(value) for value in direct_values[:3]
                ):
                    return None, "direct"
                return (
                    direct_values[0],
                    direct_values[1],
                    direct_values[2],
                ), "direct"
            return map_point if map_is_valid else None, "xy_map"

        point = direct_project((float(u), float(v)))
        if point is None:
            return None, "direct"
        values = tuple(float(value) for value in point)
        if len(values) < 3 or not all(math.isfinite(value) for value in values[:3]):
            return None, "direct"
        return (values[0], values[1], values[2]), "direct"

    def _record_consistency_sample(
        self,
        signature: str,
        map_point: Optional[Sequence[float]],
        direct_point: Optional[Sequence[float]],
    ) -> None:
        """用真实 car 像素的逐框投影验证对应 XY Map 查表结果。"""
        if direct_point is None:
            return
        try:
            direct_xy = np.asarray(direct_point[:2], dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return
        if direct_xy.shape != (2,) or not np.isfinite(direct_xy).all():
            return

        invalid_map = False
        if map_point is None:
            error_m = float("inf")
            invalid_map = True
        else:
            try:
                map_xy = np.asarray(map_point[:2], dtype=np.float64)
            except (TypeError, ValueError, IndexError):
                map_xy = np.full(2, np.nan, dtype=np.float64)
            invalid_map = map_xy.shape != (2,) or not np.isfinite(map_xy).all()
            error_m = (
                float("inf")
                if invalid_map
                else float(np.linalg.norm(map_xy - direct_xy))
            )

        with self._lock:
            if (
                self._consistency_state != "checking"
                or signature != self._consistency_signature
                or signature != self._active_signature
            ):
                return
            self._consistency_errors.append(error_m)
            if invalid_map:
                self._consistency_invalid_count += 1
            if len(self._consistency_errors) < self.consistency_sample_count:
                return

            finite_errors = [
                value for value in self._consistency_errors if math.isfinite(value)
            ]
            max_error = max(finite_errors, default=float("inf"))
            mean_error = (
                float(np.mean(finite_errors)) if finite_errors else float("inf")
            )
            failed = (
                self._consistency_invalid_count > 0
                or not math.isfinite(max_error)
                or max_error > self.consistency_max_error_m
            )
            if failed:
                self._consistency_state = "failed"
                self._enabled = False
                self._status = self.STATUS_CONSISTENCY_FAILED
                message = (
                    "警告：XY Maps 初期一致性检查失败，已自动关闭。"
                    f"样本={len(self._consistency_errors)}，"
                    f"无效查表={self._consistency_invalid_count}，"
                    f"最大误差={max_error:.4f} m，"
                    f"允许误差={self.consistency_max_error_m:.4f} m。"
                )
                level = "warning"
            else:
                self._consistency_state = "passed"
                message = (
                    "XY Maps 初期一致性检查通过。"
                    f"样本={len(self._consistency_errors)}，"
                    f"平均误差={mean_error:.4f} m，"
                    f"最大误差={max_error:.4f} m。"
                )
                level = "info"
            self._consistency_events.append(
                {
                    "level": level,
                    "message": message,
                    "sample_count": len(self._consistency_errors),
                    "invalid_count": self._consistency_invalid_count,
                    "mean_error_m": mean_error,
                    "max_error_m": max_error,
                }
            )

    def consume_consistency_events(self) -> list[dict]:
        """由 UI 定时取走尚未展示的一致性检查结果。"""
        with self._lock:
            events = list(self._consistency_events)
            self._consistency_events.clear()
            return events

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._generation += 1
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
