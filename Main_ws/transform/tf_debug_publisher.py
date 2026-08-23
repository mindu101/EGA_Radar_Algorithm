from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class CameraTfDebugPublisher:
    """发布标定后的相机位姿与目标检测位姿调试信息（tf + Pose）。"""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.root_frame = str(cfg.get("root_frame", "rm_frame"))
        self.parent_frame = str(cfg.get("parent_frame", "field"))
        self.child_frame = str(cfg.get("child_frame", "camera"))
        self.publish_rm_to_field = bool(cfg.get("publish_rm_to_field", True))
        self.pose_topic = str(cfg.get("pose_topic", "/debug/camera_pose"))
        self.rm_to_field_tf_cfg = cfg.get("rm_to_field_tf", {})
        self.rm_to_field_translation = self._parse_vector3(
            name="rm_to_field_tf.translation_xyz",
            value=self.rm_to_field_tf_cfg.get("translation_xyz", [0.0, 0.0, 0.0]),
            default=[0.0, 0.0, 0.0],
        )
        self.rm_to_field_rpy_deg = self._parse_vector3(
            name="rm_to_field_tf.rotation_rpy_deg",
            value=self.rm_to_field_tf_cfg.get("rotation_rpy_deg", [0.0, 0.0, 0.0]),
            default=[0.0, 0.0, 0.0],
        )
        self.rm_to_field_rotation_mode = str(
            self.rm_to_field_tf_cfg.get("rotation_mode", "rpy")
        ).lower()
        self.rm_to_field_rotation_matrix = self._parse_rotation_matrix(
            name="rm_to_field_tf.rotation_matrix_row_major",
            value=self.rm_to_field_tf_cfg.get("rotation_matrix_row_major", None),
            default=None,
        )
        if self.rm_to_field_rotation_mode not in ("rpy", "matrix"):
            logger.warning(
                "rotation_mode=%s 非法，回退为 rpy",
                self.rm_to_field_rotation_mode,
            )
            self.rm_to_field_rotation_mode = "rpy"

        # 目标调试发布配置
        self.publish_target_tf = bool(cfg.get("publish_target_tf", True))
        self.publish_target_pose = bool(cfg.get("publish_target_pose", True))
        self.target_pose_topic = str(cfg.get("target_pose_topic", "/debug/target_poses"))
        self.target_child_prefix = str(cfg.get("target_child_prefix", "target_"))

        self.publish_rate_hz = float(cfg.get("publish_rate_hz", 2.0))
        if self.publish_rate_hz <= 0.0:
            self.publish_rate_hz = 2.0

        self._ready = False
        self._lock = threading.Lock()
        self._latest_r = None
        self._latest_t = None
        self._latest_targets: list[tuple[int, np.ndarray]] = []

        self._node = None
        self._executor = None
        self._thread = None
        self._tf_broadcaster = None
        self._pose_publisher = None
        self._target_pose_publisher = None

        if not self.enabled:
            return

        try:
            import rclpy
            from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
            from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from tf2_ros import TransformBroadcaster
        except ImportError as exc:
            logger.warning("tf debug 依赖导入失败，已关闭发布功能: %s", exc)
            self.enabled = False
            return

        self._rclpy = rclpy
        self._Pose = Pose
        self._PoseArray = PoseArray
        self._PoseStamped = PoseStamped
        self._FloatingPointRange = FloatingPointRange
        self._ParameterDescriptor = ParameterDescriptor
        self._SetParametersResult = SetParametersResult
        self._TransformStamped = TransformStamped
        self._Node = Node
        self._SingleThreadedExecutor = SingleThreadedExecutor
        self._TransformBroadcaster = TransformBroadcaster

        if not self._rclpy.ok():
            self._rclpy.init()

        self._node = self._Node("rm2026_camera_tf_debug_pub")
        self._tf_broadcaster = self._TransformBroadcaster(self._node)
        self._pose_publisher = self._node.create_publisher(self._PoseStamped, self.pose_topic, 10)

        if self.publish_target_pose:
            self._target_pose_publisher = self._node.create_publisher(
                self._PoseArray, self.target_pose_topic, 10
            )

        self._rm_to_field_rotation = self._resolve_rm_to_field_rotation()
        self._register_dynamic_tf_parameters()
        self._node.create_timer(1.0 / self.publish_rate_hz, self._timer_callback)

        self._executor = self._SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

        self._ready = True
        logger.info(
            "tf debug 发布器已启动: %s -> %s, pose_topic=%s, rate=%.2fHz",
            self.parent_frame,
            self.child_frame,
            self.pose_topic,
            self.publish_rate_hz,
        )
        if self.publish_rm_to_field:
            logger.info(
                "固定 tf 已启用: %s -> %s, xyz=(%.3f, %.3f, %.3f), mode=%s, rpy_deg=(%.3f, %.3f, %.3f)",
                self.root_frame,
                self.parent_frame,
                float(self.rm_to_field_translation[0]),
                float(self.rm_to_field_translation[1]),
                float(self.rm_to_field_translation[2]),
                self.rm_to_field_rotation_mode,
                float(self.rm_to_field_rpy_deg[0]),
                float(self.rm_to_field_rpy_deg[1]),
                float(self.rm_to_field_rpy_deg[2]),
            )
        if self.publish_target_tf or self.publish_target_pose:
            logger.info(
                "目标调试发布已启用: tf=%s, pose=%s, prefix=%s, topic=%s",
                self.publish_target_tf,
                self.publish_target_pose,
                self.target_child_prefix,
                self.target_pose_topic,
            )

    @staticmethod
    def _build_rm_to_field_rotation(rpy_deg: np.ndarray) -> np.ndarray:
        """根据 roll/pitch/yaw（度）构造 field->rm 旋转矩阵。

        约定：
        1. 角度定义在 rm_frame 固定坐标轴（外旋）。
        2. 矩阵形式为 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
        3. 作用到列向量时，实际执行顺序是先 Rx，再 Ry，最后 Rz。
        """
        roll, pitch, yaw = np.deg2rad(rpy_deg.astype(np.float64))
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        r_x = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float64,
        )
        r_y = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float64,
        )
        r_z = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return r_z @ r_y @ r_x

    def _parse_vector3(self, name: str, value, default: list[float]) -> np.ndarray:
        """解析长度为3的向量配置，格式非法时回退默认值。"""
        try:
            vec = np.array(value, dtype=np.float64).reshape(-1)
            if vec.size != 3:
                raise ValueError(f"{name} 必须是长度为3的数组，当前长度={vec.size}")
            return vec
        except Exception as exc:
            logger.warning(
                "配置 %s 无效，已回退默认值 %s，错误: %s",
                name,
                default,
                exc,
            )
            return np.array(default, dtype=np.float64)

    def _parse_rotation_matrix(
        self,
        name: str,
        value,
        default: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """解析 3x3 旋转矩阵（行优先 9 元素），并做正交化。"""
        if value is None:
            return default
        try:
            raw = np.array(value, dtype=np.float64).reshape(-1)
            if raw.size != 9:
                raise ValueError(f"{name} 必须是长度为9的数组，当前长度={raw.size}")
            mat = raw.reshape(3, 3)
            # 将近似旋转矩阵投影到 SO(3)，避免数值误差导致非正交
            u, _, vt = np.linalg.svd(mat)
            rot = u @ vt
            if np.linalg.det(rot) < 0.0:
                u[:, -1] *= -1.0
                rot = u @ vt
            return rot
        except Exception as exc:
            logger.warning("配置 %s 无效，忽略该矩阵，错误: %s", name, exc)
            return default

    def _resolve_rm_to_field_rotation(self) -> np.ndarray:
        """按 rotation_mode 解析最终姿态，统一表示为 field->rm_frame 旋转矩阵。"""
        if self.rm_to_field_rotation_mode == "matrix":
            if self.rm_to_field_rotation_matrix is None:
                logger.warning(
                    "rotation_mode=matrix 但未提供 rotation_matrix_row_major，回退为 rpy 模式"
                )
                self.rm_to_field_rotation_mode = "rpy"
            else:
                return self.rm_to_field_rotation_matrix.copy()
        return self._build_rm_to_field_rotation(self.rm_to_field_rpy_deg)

    def _register_dynamic_tf_parameters(self):
        """注册可动态修改的 tf 参数（用于运行时在线调参）。"""
        if self._node is None:
            return
        self._declare_float_parameter(
            "rm_to_field.tx",
            float(self.rm_to_field_translation[0]),
            "field->rm 平移 x（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "rm_to_field.ty",
            float(self.rm_to_field_translation[1]),
            "field->rm 平移 y（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "rm_to_field.tz",
            float(self.rm_to_field_translation[2]),
            "field->rm 平移 z（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "rm_to_field.roll_deg",
            float(self.rm_to_field_rpy_deg[0]),
            "field->rm 旋转 roll（度，Rz(yaw)*Ry(pitch)*Rx(roll)）",
            -360.0,
            360.0,
        )
        self._declare_float_parameter(
            "rm_to_field.pitch_deg",
            float(self.rm_to_field_rpy_deg[1]),
            "field->rm 旋转 pitch（度，Rz(yaw)*Ry(pitch)*Rx(roll)）",
            -360.0,
            360.0,
        )
        self._declare_float_parameter(
            "rm_to_field.yaw_deg",
            float(self.rm_to_field_rpy_deg[2]),
            "field->rm 旋转 yaw（度，Rz(yaw)*Ry(pitch)*Rx(roll)）",
            -360.0,
            360.0,
        )
        self._declare_string_parameter(
            "rm_to_field.rotation_mode",
            self.rm_to_field_rotation_mode,
            "姿态表示方式：rpy 或 matrix（统一表示 field->rm 旋转）",
        )

        # 数组写法（脚本批量修改）
        self._node.declare_parameter(
            "rm_to_field.translation_xyz",
            [float(self.rm_to_field_translation[0]), float(self.rm_to_field_translation[1]), float(self.rm_to_field_translation[2])],
        )
        self._node.declare_parameter(
            "rm_to_field.rotation_rpy_deg",
            [float(self.rm_to_field_rpy_deg[0]), float(self.rm_to_field_rpy_deg[1]), float(self.rm_to_field_rpy_deg[2])],
        )
        self._node.declare_parameter(
            "rm_to_field.rotation_matrix_row_major",
            (
                self.rm_to_field_rotation_matrix.reshape(-1).tolist()
                if self.rm_to_field_rotation_matrix is not None
                else self._rm_to_field_rotation.reshape(-1).tolist()
            ),
        )

        # 为兼容旧命名，保留 field_to_rm 参数别名（语义仍为 field->rm）。
        self._declare_float_parameter(
            "field_to_rm.tx",
            float(self.rm_to_field_translation[0]),
            "兼容别名: field->rm 平移 x（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "field_to_rm.ty",
            float(self.rm_to_field_translation[1]),
            "兼容别名: field->rm 平移 y（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "field_to_rm.tz",
            float(self.rm_to_field_translation[2]),
            "兼容别名: field->rm 平移 z（单位米，坐标在 rm_frame 下）",
            -100.0,
            100.0,
        )
        self._declare_float_parameter(
            "field_to_rm.roll_deg",
            float(self.rm_to_field_rpy_deg[0]),
            "兼容别名: field->rm 旋转 roll（度）",
            -360.0,
            360.0,
        )
        self._declare_float_parameter(
            "field_to_rm.pitch_deg",
            float(self.rm_to_field_rpy_deg[1]),
            "兼容别名: field->rm 旋转 pitch（度）",
            -360.0,
            360.0,
        )
        self._declare_float_parameter(
            "field_to_rm.yaw_deg",
            float(self.rm_to_field_rpy_deg[2]),
            "兼容别名: field->rm 旋转 yaw（度）",
            -360.0,
            360.0,
        )
        self._declare_string_parameter(
            "field_to_rm.rotation_mode",
            self.rm_to_field_rotation_mode,
            "兼容别名: 姿态表示方式 rpy 或 matrix",
        )
        self._node.declare_parameter(
            "field_to_rm.translation_xyz",
            [float(self.rm_to_field_translation[0]), float(self.rm_to_field_translation[1]), float(self.rm_to_field_translation[2])],
        )
        self._node.declare_parameter(
            "field_to_rm.rotation_rpy_deg",
            [float(self.rm_to_field_rpy_deg[0]), float(self.rm_to_field_rpy_deg[1]), float(self.rm_to_field_rpy_deg[2])],
        )
        self._node.declare_parameter(
            "field_to_rm.rotation_matrix_row_major",
            (
                self.rm_to_field_rotation_matrix.reshape(-1).tolist()
                if self.rm_to_field_rotation_matrix is not None
                else self._rm_to_field_rotation.reshape(-1).tolist()
            ),
        )
        self._node.add_on_set_parameters_callback(self._on_set_parameters)

    def _declare_float_parameter(
        self,
        name: str,
        default_value: float,
        description: str,
        min_value: float,
        max_value: float,
    ):
        descriptor = self._ParameterDescriptor()
        descriptor.description = description
        fp_range = self._FloatingPointRange()
        fp_range.from_value = float(min_value)
        fp_range.to_value = float(max_value)
        fp_range.step = 0.0
        descriptor.floating_point_range = [fp_range]
        self._node.declare_parameter(name, float(default_value), descriptor)

    def _declare_string_parameter(self, name: str, default_value: str, description: str):
        descriptor = self._ParameterDescriptor()
        descriptor.description = description
        self._node.declare_parameter(name, str(default_value), descriptor)

    def _parse_param_vector3(self, param_name: str, value) -> np.ndarray:
        vec = np.array(value, dtype=np.float64).reshape(-1)
        if vec.size != 3:
            raise ValueError(f"{param_name} 必须是长度为3的数组，当前长度={vec.size}")
        return vec

    def _parse_param_rotation_mode(self, param_name: str, value: Any) -> str:
        mode = str(value).strip().lower()
        if mode not in ("rpy", "matrix"):
            raise ValueError(f"{param_name} 仅支持 'rpy' 或 'matrix'，当前={mode}")
        return mode

    def _parse_param_matrix3(self, param_name: str, value) -> np.ndarray:
        parsed = self._parse_rotation_matrix(param_name, value, default=None)
        if parsed is None:
            raise ValueError(f"{param_name} 解析失败")
        return parsed

    def _publish_fixed_tf_only(self):
        """仅发布 fixed tf，便于在未完成外参标定前也能查看坐标系变化。"""
        if not self._ready or not self.publish_rm_to_field:
            return
        stamp_msg = self._node.get_clock().now().to_msg()
        fixed_tf_msg = self._compose_fixed_rm_to_field_tf(stamp_msg)
        self._tf_broadcaster.sendTransform(fixed_tf_msg)

    def _on_set_parameters(self, params):
        """动态参数回调：实时更新 field<->rm_frame 的 tf。"""
        updated_translation = None
        updated_rpy_deg = None
        updated_rotation_mode = None
        updated_rotation_matrix = None
        support_param_names = {
            "rm_to_field.tx",
            "rm_to_field.ty",
            "rm_to_field.tz",
            "rm_to_field.roll_deg",
            "rm_to_field.pitch_deg",
            "rm_to_field.yaw_deg",
            "rm_to_field.translation_xyz",
            "rm_to_field.rotation_rpy_deg",
            "rm_to_field.rotation_matrix_row_major",
            "rm_to_field.rotation_mode",
            "field_to_rm.tx",
            "field_to_rm.ty",
            "field_to_rm.tz",
            "field_to_rm.roll_deg",
            "field_to_rm.pitch_deg",
            "field_to_rm.yaw_deg",
            "field_to_rm.translation_xyz",
            "field_to_rm.rotation_rpy_deg",
            "field_to_rm.rotation_matrix_row_major",
            "field_to_rm.rotation_mode",
        }
        try:
            translation_next = self.rm_to_field_translation.copy()
            rpy_next = self.rm_to_field_rpy_deg.copy()
            rotation_mode_next = self.rm_to_field_rotation_mode
            rotation_matrix_next = (
                None
                if self.rm_to_field_rotation_matrix is None
                else self.rm_to_field_rotation_matrix.copy()
            )
            for param in params:
                if param.name not in support_param_names:
                    continue
                if param.name in ("rm_to_field.tx", "field_to_rm.tx"):
                    translation_next[0] = float(param.value)
                    updated_translation = translation_next
                elif param.name in ("rm_to_field.ty", "field_to_rm.ty"):
                    translation_next[1] = float(param.value)
                    updated_translation = translation_next
                elif param.name in ("rm_to_field.tz", "field_to_rm.tz"):
                    translation_next[2] = float(param.value)
                    updated_translation = translation_next
                elif param.name in ("rm_to_field.roll_deg", "field_to_rm.roll_deg"):
                    rpy_next[0] = float(param.value)
                    updated_rpy_deg = rpy_next
                elif param.name in ("rm_to_field.pitch_deg", "field_to_rm.pitch_deg"):
                    rpy_next[1] = float(param.value)
                    updated_rpy_deg = rpy_next
                elif param.name in ("rm_to_field.yaw_deg", "field_to_rm.yaw_deg"):
                    rpy_next[2] = float(param.value)
                    updated_rpy_deg = rpy_next
                elif param.name.endswith("rotation_mode"):
                    rotation_mode_next = self._parse_param_rotation_mode(
                        param.name, param.value
                    )
                    updated_rotation_mode = rotation_mode_next
                elif param.name.endswith("translation_xyz"):
                    updated_translation = self._parse_param_vector3(param.name, param.value)
                elif param.name.endswith("rotation_rpy_deg"):
                    updated_rpy_deg = self._parse_param_vector3(param.name, param.value)
                elif param.name.endswith("rotation_matrix_row_major"):
                    rotation_matrix_next = self._parse_param_matrix3(
                        param.name, param.value
                    )
                    updated_rotation_matrix = rotation_matrix_next
        except Exception as exc:
            return self._SetParametersResult(successful=False, reason=str(exc))

        if (
            updated_translation is None
            and updated_rpy_deg is None
            and updated_rotation_mode is None
            and updated_rotation_matrix is None
        ):
            return self._SetParametersResult(successful=True, reason="")

        with self._lock:
            if updated_translation is not None:
                self.rm_to_field_translation = updated_translation
            if updated_rpy_deg is not None:
                self.rm_to_field_rpy_deg = updated_rpy_deg
            if updated_rotation_mode is not None:
                self.rm_to_field_rotation_mode = updated_rotation_mode
            if updated_rotation_matrix is not None:
                self.rm_to_field_rotation_matrix = updated_rotation_matrix
            self._rm_to_field_rotation = self._resolve_rm_to_field_rotation()

            r_cw = None if self._latest_r is None else self._latest_r.copy()
            t_cw = None if self._latest_t is None else self._latest_t.copy()
            targets = list(self._latest_targets)

        self._publish_fixed_tf_only()
        if r_cw is not None and t_cw is not None:
            self._publish_once(r_cw, t_cw, targets=targets, log_pose=False)

        self._node.get_logger().info(
            "动态更新 field->rm tf 成功: "
            f"xyz=({float(self.rm_to_field_translation[0]):.3f}, "
            f"{float(self.rm_to_field_translation[1]):.3f}, "
            f"{float(self.rm_to_field_translation[2]):.3f}), "
            f"mode={self.rm_to_field_rotation_mode}, "
            f"rpy_deg=({float(self.rm_to_field_rpy_deg[0]):.3f}, "
            f"{float(self.rm_to_field_rpy_deg[1]):.3f}, "
            f"{float(self.rm_to_field_rpy_deg[2]):.3f})"
        )
        return self._SetParametersResult(successful=True, reason="")

    @staticmethod
    def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
        """将旋转矩阵转换为四元数 (x, y, z, w)。"""
        trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rotation[2, 1] - rotation[1, 2]) / s
            qy = (rotation[0, 2] - rotation[2, 0]) / s
            qz = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)

    def _compose_fixed_rm_to_field_tf(self, stamp_msg):
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(self._rm_to_field_rotation)
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp_msg
        tf_msg.header.frame_id = self.root_frame
        tf_msg.child_frame_id = self.parent_frame
        tf_msg.transform.translation.x = float(self.rm_to_field_translation[0])
        tf_msg.transform.translation.y = float(self.rm_to_field_translation[1])
        tf_msg.transform.translation.z = float(self.rm_to_field_translation[2])
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        return tf_msg

    def _compose_camera_msgs(self, r_cw: np.ndarray, t_cw: np.ndarray, stamp_msg):
        # solvePnP 输出是 世界->相机，tf 需要发布 场地->相机位姿（相机在场地中的位置）
        r_wc = r_cw.T
        t_wc = (-r_wc @ t_cw).reshape(3)
        qx, qy, qz, qw = self._rotation_matrix_to_quaternion(r_wc)

        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp_msg
        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id = self.child_frame
        tf_msg.transform.translation.x = float(t_wc[0])
        tf_msg.transform.translation.y = float(t_wc[1])
        tf_msg.transform.translation.z = float(t_wc[2])
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        pose_msg = self._PoseStamped()
        pose_msg.header = tf_msg.header
        pose_msg.pose.position.x = tf_msg.transform.translation.x
        pose_msg.pose.position.y = tf_msg.transform.translation.y
        pose_msg.pose.position.z = tf_msg.transform.translation.z
        pose_msg.pose.orientation = tf_msg.transform.rotation
        return tf_msg, pose_msg, t_wc

    def _compose_target_tf_msg(self, class_id: int, pos_3d: np.ndarray, stamp_msg):
        tf_msg = self._TransformStamped()
        tf_msg.header.stamp = stamp_msg
        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id = f"{self.target_child_prefix}{class_id}"
        tf_msg.transform.translation.x = float(pos_3d[0])
        tf_msg.transform.translation.y = float(pos_3d[1])
        tf_msg.transform.translation.z = float(pos_3d[2])
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        return tf_msg

    def _compose_target_pose_array(self, targets: list[tuple[int, np.ndarray]], stamp_msg):
        pose_array = self._PoseArray()
        pose_array.header.stamp = stamp_msg
        pose_array.header.frame_id = self.parent_frame
        for _, pos_3d in targets:
            pose_msg = self._Pose()
            pose_msg.position.x = float(pos_3d[0])
            pose_msg.position.y = float(pos_3d[1])
            pose_msg.position.z = float(pos_3d[2])
            pose_msg.orientation.x = 0.0
            pose_msg.orientation.y = 0.0
            pose_msg.orientation.z = 0.0
            pose_msg.orientation.w = 1.0
            pose_array.poses.append(pose_msg)
        return pose_array

    def _normalize_targets(self, targets: list[dict[str, Any]] | None) -> list[tuple[int, np.ndarray]]:
        normalized_targets: list[tuple[int, np.ndarray]] = []
        if not targets:
            return normalized_targets

        for target in targets:
            if not isinstance(target, dict):
                continue
            class_id = int(target.get("class_id", target.get("id", -1)))
            pos_3d_raw = target.get("pos_3d", None)
            if pos_3d_raw is None:
                continue
            try:
                pos_3d = np.array(pos_3d_raw, dtype=np.float64).reshape(3)
            except Exception:
                continue
            normalized_targets.append((class_id, pos_3d))

        return normalized_targets

    def update_extrinsics(self, r_cw, t_cw) -> bool:
        """更新标定外参并立即发布一次。"""
        if not self._ready:
            return False
        try:
            r_np = np.array(r_cw, dtype=np.float64).reshape(3, 3)
            t_np = np.array(t_cw, dtype=np.float64).reshape(3, 1)
            with self._lock:
                self._latest_r = r_np
                self._latest_t = t_np
                targets = list(self._latest_targets)
            self._publish_once(r_np, t_np, targets=targets, log_pose=True)
            return True
        except Exception as exc:
            logger.error("更新 tf debug 外参失败: %s", exc)
            return False

    def update_targets(self, targets: list[dict[str, Any]] | None) -> bool:
        """更新目标检测结果（场地坐标系）用于 rviz 可视化。"""
        if not self._ready:
            return False
        try:
            normalized_targets = self._normalize_targets(targets)
            with self._lock:
                self._latest_targets = normalized_targets
                r_cw = None if self._latest_r is None else self._latest_r.copy()
                t_cw = None if self._latest_t is None else self._latest_t.copy()

            if r_cw is not None and t_cw is not None:
                self._publish_once(r_cw, t_cw, targets=normalized_targets, log_pose=False)
            return True
        except Exception as exc:
            logger.error("更新目标调试发布数据失败: %s", exc)
            return False

    def _publish_once(
        self,
        r_cw: np.ndarray,
        t_cw: np.ndarray,
        targets: list[tuple[int, np.ndarray]] | None = None,
        log_pose: bool = False,
    ):
        stamp_msg = self._node.get_clock().now().to_msg()
        if self.publish_rm_to_field:
            fixed_tf_msg = self._compose_fixed_rm_to_field_tf(stamp_msg)
            self._tf_broadcaster.sendTransform(fixed_tf_msg)

        tf_msg, pose_msg, t_wc = self._compose_camera_msgs(r_cw, t_cw, stamp_msg)
        self._tf_broadcaster.sendTransform(tf_msg)
        self._pose_publisher.publish(pose_msg)

        if targets is None:
            with self._lock:
                targets = list(self._latest_targets)

        if self.publish_target_tf and targets:
            for class_id, pos_3d in targets:
                target_tf_msg = self._compose_target_tf_msg(class_id, pos_3d, stamp_msg)
                self._tf_broadcaster.sendTransform(target_tf_msg)

        if self.publish_target_pose and self._target_pose_publisher is not None:
            target_pose_array = self._compose_target_pose_array(targets or [], stamp_msg)
            self._target_pose_publisher.publish(target_pose_array)

        if log_pose:
            self._node.get_logger().info(
                f"发布相机位姿: frame={self.parent_frame}, child={self.child_frame}, "
                f"pos=({float(t_wc[0]):.3f}, {float(t_wc[1]):.3f}, "
                f"{float(t_wc[2]):.3f})"
            )

    def _timer_callback(self):
        if not self._ready:
            return
        with self._lock:
            if self._latest_r is None or self._latest_t is None:
                return
            r_cw = self._latest_r.copy()
            t_cw = self._latest_t.copy()
            targets = list(self._latest_targets)
        self._publish_once(r_cw, t_cw, targets=targets, log_pose=False)

    def stop(self):
        if not self._ready:
            return
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception as exc:
            logger.warning("关闭 tf debug 发布器时出现异常: %s", exc)
        finally:
            self._ready = False
