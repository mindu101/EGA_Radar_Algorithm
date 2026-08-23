#!/usr/bin/env python3
"""测试专用裁判链路 Web 注入器。

该工具不会由 start.sh 自动启动。只有显式提供双重测试确认后才允许运行，
页面还需再次确认。serial 模式直接发送最终串口帧；info-wave-attach
只附着显式授权的 start.sh 测试态。
"""

from __future__ import annotations

import argparse
import atexit
import copy
import ctypes
import json
import math
import os
import secrets
import signal
import socket
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from driver.referee.serial_protocol import (  # noqa: E402
    InfoWaveMessage,
    InfoWaveRobotMulticastMessage,
    OBJECT_ID,
    Radar2ClientMessage,
    Radar2DartMessage,
    Radar2SentryMessage,
    RadarDecisionMessage,
    RefereeGenericMessage,
    RawMessage,
)
from driver.referee.tx_policy import RefereeTxPolicy  # noqa: E402
from driver.referee.udp_protocol import InfoMsgBag  # noqa: E402
from driver.referee.web_test_override import (  # noqa: E402
    DEFAULT_PORT as WEB_LATE_DEFAULT_PORT,
    encode_snapshot as encode_web_late_snapshot,
)


PID_FILE = Path("/tmp/rm2026_referee_web_injector.pid")
FULL_CHAIN_TEST_MARKER = Path("/tmp/rm2026_info_wave_full_chain_test.env")
HTML_FILE = Path(__file__).with_name("referee_web_injector.html")
TEST_CONFIRMATION = "TEST_ONLY"
PAGE_CONFIRMATION = "仅用于测试"

ROBOT_KEYS = (
    ("enemy_hero", "对方英雄"),
    ("enemy_engineer", "对方工程"),
    ("enemy_standard_3", "对方3号步兵"),
    ("enemy_standard_4", "对方4号步兵"),
    ("enemy_drone", "对方空中"),
    ("enemy_sentry", "对方哨兵"),
    ("self_hero", "己方英雄"),
    ("self_engineer", "己方工程"),
    ("self_standard_3", "己方3号步兵"),
    ("self_standard_4", "己方4号步兵"),
    ("self_drone", "己方空中"),
    ("self_sentry", "己方哨兵"),
)

MESSAGE_KEYS = (
    "radar_map",
    "decision",
    "dart",
    "engineer",
    "sentry",
    "robot_multicast",
)
INFO_WAVE_MESSAGE_KEYS = ("radar_map", "health", "engineer", "sentry")
MULTICAST_ROBOT_KEYS = (
    ("hero", "英雄"),
    ("engineer", "工程"),
    ("standard_3", "3号步兵"),
    ("standard_4", "4号步兵"),
    ("sentry", "哨兵"),
)
_OPTIONAL_SLOT_WEIGHTS = RefereeTxPolicy.OPTIONAL_MULTICAST_SLOT_WEIGHTS
DEFAULT_FREQUENCIES_HZ = {
    "radar_map": RefereeTxPolicy.RADAR2CLIENT_TX_HZ,
    "decision": float(_OPTIONAL_SLOT_WEIGHTS[0]),
    "dart": float(_OPTIONAL_SLOT_WEIGHTS[1]),
    "engineer": float(_OPTIONAL_SLOT_WEIGHTS[2]),
    "sentry": float(_OPTIONAL_SLOT_WEIGHTS[3]),
    "robot_multicast": float(_OPTIONAL_SLOT_WEIGHTS[4]),
    "health": 5.0,
}
MIN_TEST_FREQUENCY_HZ = 0.1
MAX_TEST_FREQUENCY_HZ = 20.0


def default_config() -> dict[str, Any]:
    coordinates = {}
    for robot_key, _ in ROBOT_KEYS:
        coordinates[f"{robot_key}_x"] = 0
        coordinates[f"{robot_key}_y"] = 0
    return {
        "side": "red",
        "messages": {
            "radar_map": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["radar_map"],
                "coordinates": coordinates,
            },
            "decision": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["decision"],
                "radar_cmd": 0,
                "key_command": 0,
                "key_text": "AAAAAA",
            },
            "dart": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["dart"],
                "trigger_state": 0,
                "remaining_count": 0,
                "autoaim_stage": 0,
            },
            "engineer": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["engineer"],
                "aerial_ammo": 0,
                "remaining_coins": 0,
            },
            "sentry": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["sentry"],
                "hero_targetable": True,
                "engineer_targetable": False,
                "standard_3_targetable": True,
                "standard_4_targetable": True,
                "sentry_targetable": True,
                "prefer_long_hole": False,
            },
            "robot_multicast": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["robot_multicast"],
                "hero": {
                    "hp_percent": 100,
                    "ammo": 0,
                    "defense": 0,
                    "negative_defense": 0,
                    "targetable": True,
                },
                "engineer": {
                    "hp_percent": 100,
                    "ammo": 0,
                    "defense": 0,
                    "negative_defense": 0,
                    "targetable": True,
                },
                "standard_3": {
                    "hp_percent": 100,
                    "ammo": 0,
                    "defense": 0,
                    "negative_defense": 0,
                    "targetable": True,
                },
                "standard_4": {
                    "hp_percent": 100,
                    "ammo": 0,
                    "defense": 0,
                    "negative_defense": 0,
                    "targetable": True,
                },
                "sentry": {
                    "hp_percent": 100,
                    "ammo": 0,
                    "defense": 0,
                    "negative_defense": 0,
                    "targetable": True,
                },
            },
            "health": {
                "enabled": False,
                "frequency_hz": DEFAULT_FREQUENCIES_HZ["health"],
                "hero": 0,
                "engineer": 0,
                "standard_3": 0,
                "standard_4": 0,
                "reserved": 0,
                "sentry": 0,
            },
        },
    }


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是布尔值")
    return value


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} 必须在 {minimum}～{maximum} 之间")
    return result


def _require_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} 必须在 {minimum:g}～{maximum:g} Hz 之间")
    return result


def _frequency_hz(message: dict[str, Any], label: str) -> float:
    return _require_float(
        message.get("frequency_hz"),
        f"{label}发送频率",
        MIN_TEST_FREQUENCY_HZ,
        MAX_TEST_FREQUENCY_HZ,
    )


def normalize_config(raw: Any) -> dict[str, Any]:
    """校验页面提交的语义字段，不接受原始 bit 或任意 ID。"""
    root = _require_dict(raw, "配置")
    side = str(root.get("side", "")).lower()
    if side not in ("red", "blue"):
        raise ValueError("己方阵营只能是 red 或 blue")

    messages = _require_dict(root.get("messages"), "messages")
    for key in MESSAGE_KEYS:
        if key not in messages:
            raise ValueError(f"缺少消息配置：{key}")

    radar_map = _require_dict(messages["radar_map"], "radar_map")
    coordinates_raw = _require_dict(radar_map.get("coordinates"), "coordinates")
    coordinates = {}
    for robot_key, robot_label in ROBOT_KEYS:
        for axis in ("x", "y"):
            field_name = f"{robot_key}_{axis}"
            coordinates[field_name] = _require_int(
                coordinates_raw.get(field_name),
                f"{robot_label} {axis.upper()}",
                0,
                0xFFFF,
            )

    decision = _require_dict(messages["decision"], "decision")
    key_command = _require_int(
        decision.get("key_command"), "密钥指令", 0, 2
    )
    if key_command not in (0, 2):
        raise ValueError("密钥指令只能选择 0=不验证 或 2=验证")
    key_text = str(decision.get("key_text", ""))
    if len(key_text) != 6 or not key_text.isascii() or not key_text.isalnum():
        raise ValueError("密钥必须正好是6个ASCII字母或数字")

    dart = _require_dict(messages["dart"], "dart")
    engineer = _require_dict(messages["engineer"], "engineer")
    sentry = _require_dict(messages["sentry"], "sentry")
    robot_multicast = _require_dict(
        messages["robot_multicast"], "robot_multicast"
    )
    health = _require_dict(messages.get("health"), "health")

    normalized_multicast_robots = {}
    for robot_key, robot_label in MULTICAST_ROBOT_KEYS:
        robot = _require_dict(
            robot_multicast.get(robot_key), f"机器人摘要.{robot_label}"
        )
        normalized_multicast_robots[robot_key] = {
            "hp_percent": _require_int(
                robot.get("hp_percent"), f"{robot_label}血量比例", 0, 100
            ),
            "ammo": _require_int(
                robot.get("ammo"), f"{robot_label}允许发弹量", 0, 0xFFFF
            ),
            "defense": _require_int(
                robot.get("defense"), f"{robot_label}防御增益", 0, 0xFF
            ),
            "negative_defense": _require_int(
                robot.get("negative_defense"),
                f"{robot_label}负防御增益",
                0,
                0xFF,
            ),
            "targetable": _require_bool(
                robot.get("targetable"), f"{robot_label}允许瞄准"
            ),
        }
    # 工程发弹量是协议保留字段，无论页面或旧配置提交什么都固定归零。
    normalized_multicast_robots["engineer"]["ammo"] = 0

    return {
        "side": side,
        "messages": {
            "radar_map": {
                "enabled": _require_bool(radar_map.get("enabled"), "地图坐标启用"),
                "frequency_hz": _frequency_hz(radar_map, "地图坐标"),
                "coordinates": coordinates,
            },
            "decision": {
                "enabled": _require_bool(decision.get("enabled"), "自主决策启用"),
                "frequency_hz": _frequency_hz(decision, "自主决策"),
                "radar_cmd": _require_int(
                    decision.get("radar_cmd"), "双倍易伤请求序号", 0, 0xFF
                ),
                "key_command": key_command,
                "key_text": key_text,
            },
            "dart": {
                "enabled": _require_bool(dart.get("enabled"), "无人机消息启用"),
                "frequency_hz": _frequency_hz(dart, "无人机消息"),
                "trigger_state": _require_int(
                    dart.get("trigger_state"), "双倍易伤阶段", 0, 4
                ),
                "remaining_count": _require_int(
                    dart.get("remaining_count"), "双倍易伤剩余次数", 0, 2
                ),
                "autoaim_stage": _require_int(
                    dart.get("autoaim_stage"), "自瞄反制阶段", 0, 6
                ),
            },
            "engineer": {
                "enabled": _require_bool(engineer.get("enabled"), "工程消息启用"),
                "frequency_hz": _frequency_hz(engineer, "工程消息"),
                "aerial_ammo": _require_int(
                    engineer.get("aerial_ammo"), "对方空中允许发弹量", 0, 0xFFFF
                ),
                "remaining_coins": _require_int(
                    engineer.get("remaining_coins"), "对方剩余金币", 0, 0xFFFF
                ),
            },
            "sentry": {
                "enabled": _require_bool(sentry.get("enabled"), "哨兵消息启用"),
                "frequency_hz": _frequency_hz(sentry, "哨兵消息"),
                "hero_targetable": _require_bool(
                    sentry.get("hero_targetable"), "英雄允许瞄准"
                ),
                "engineer_targetable": _require_bool(
                    sentry.get("engineer_targetable"), "工程允许瞄准"
                ),
                "standard_3_targetable": _require_bool(
                    sentry.get("standard_3_targetable"), "3号步兵允许瞄准"
                ),
                "standard_4_targetable": _require_bool(
                    sentry.get("standard_4_targetable"), "4号步兵允许瞄准"
                ),
                "sentry_targetable": _require_bool(
                    sentry.get("sentry_targetable"), "哨兵允许瞄准"
                ),
                "prefer_long_hole": _require_bool(
                    sentry.get("prefer_long_hole"), "建议前往 long_hole"
                ),
            },
            "robot_multicast": {
                "enabled": _require_bool(
                    robot_multicast.get("enabled"), "机器人摘要启用"
                ),
                "frequency_hz": _frequency_hz(
                    robot_multicast, "机器人摘要"
                ),
                **normalized_multicast_robots,
            },
            "health": {
                "enabled": _require_bool(health.get("enabled"), "血量信息启用"),
                "frequency_hz": _frequency_hz(health, "血量信息"),
                "hero": _require_int(health.get("hero"), "英雄血量", 0, 0xFFFF),
                "engineer": _require_int(
                    health.get("engineer"), "工程血量", 0, 0xFFFF
                ),
                "standard_3": _require_int(
                    health.get("standard_3"), "3号步兵血量", 0, 0xFFFF
                ),
                "standard_4": _require_int(
                    health.get("standard_4"), "4号步兵血量", 0, 0xFFFF
                ),
                "reserved": _require_int(
                    health.get("reserved"), "0x0A02保留字段", 0, 0xFFFF
                ),
                "sentry": _require_int(
                    health.get("sentry"), "哨兵血量", 0, 0xFFFF
                ),
            },
        },
    }


class PacketFactory:
    """只使用正式 serial_protocol 消息类生成测试帧。"""

    _lock = threading.RLock()

    @classmethod
    def build(cls, message_key: str, config: dict[str, Any]) -> bytes:
        with cls._lock:
            return cls._build_unlocked(message_key, config)

    @staticmethod
    def _build_unlocked(message_key: str, config: dict[str, Any]) -> bytes:
        is_blue = config["side"] == "blue"
        message_config = config["messages"][message_key]

        if message_key == "radar_map":
            return Radar2ClientMessage(**message_config["coordinates"]).pack()

        if message_key == "decision":
            key_bytes = bytes([message_config["key_command"]]) + message_config[
                "key_text"
            ].encode("ascii")
            key_array = (ctypes.c_uint8 * 7).from_buffer_copy(key_bytes)
            return RadarDecisionMessage(
                is_blue=is_blue,
                radar_cmd=message_config["radar_cmd"],
                key=key_array,
            ).pack()

        if message_key == "dart":
            return Radar2DartMessage(
                is_blue=is_blue,
                trigger_state=message_config["trigger_state"],
                double_vulnerability_count=message_config["remaining_count"],
                autoaim_stage=message_config["autoaim_stage"],
            ).pack()

        if message_key == "engineer":
            receiver_id = (
                OBJECT_ID.B_ENGINEER.value
                if is_blue
                else OBJECT_ID.R_ENGINEER.value
            )
            user_data = struct.pack(
                "<HH",
                message_config["aerial_ammo"],
                message_config["remaining_coins"],
            )
            return InfoWaveMessage(
                is_blue=is_blue,
                receiver_id=receiver_id,
                user_data=user_data,
            ).pack()

        if message_key == "sentry":
            flags = 0xC0
            semantic_fields = (
                "hero_targetable",
                "engineer_targetable",
                "standard_3_targetable",
                "standard_4_targetable",
                "sentry_targetable",
            )
            for bit_index, field_name in enumerate(semantic_fields):
                if message_config[field_name]:
                    flags |= 1 << bit_index
            if message_config["prefer_long_hole"]:
                flags |= 1 << 5
            return Radar2SentryMessage(
                is_blue=is_blue,
                targetable_flags=flags,
            ).pack()

        if message_key == "robot_multicast":
            user_data = bytearray()
            for robot_key, _robot_label in MULTICAST_ROBOT_KEYS:
                robot = message_config[robot_key]
                user_data.append(robot["hp_percent"])
                ammo = 0 if robot_key == "engineer" else robot["ammo"]
                user_data.extend(struct.pack("<H", ammo))
                user_data.append(robot["defense"])
                user_data.append(robot["negative_defense"])
                user_data.append(int(robot["targetable"]))
            receiver_id = (
                OBJECT_ID.B_HERO.value if is_blue else OBJECT_ID.R_HERO.value
            )
            return InfoWaveRobotMulticastMessage(
                is_blue=is_blue,
                receiver_id=receiver_id,
                user_data=bytes(user_data),
            ).pack()

        raise ValueError(f"未知消息类型：{message_key}")

    @staticmethod
    def preview(config: dict[str, Any]) -> dict[str, Any]:
        with PacketFactory._lock:
            with RefereeGenericMessage._seq_lock:
                saved_seq = RefereeGenericMessage._seq_counter
            try:
                result = {}
                for message_key in MESSAGE_KEYS:
                    frame = PacketFactory._build_unlocked(message_key, config)
                    result[message_key] = {
                        "enabled": config["messages"][message_key]["enabled"],
                        "cmd_id": f"0x{int.from_bytes(frame[5:7], 'little'):04X}",
                        "length": len(frame),
                        "hex": frame.hex().upper(),
                    }
                return result
            finally:
                with RefereeGenericMessage._seq_lock:
                    RefereeGenericMessage._seq_counter = saved_seq


class InfoWaveSnapshotFactory:
    """模拟 MATLAB 每次成功解析一个命令后生成的 InfoMsgBag v3 快照。"""

    FRESH_WINDOW_SEC = 3.0

    def __init__(self):
        self._lock = threading.RLock()
        self._payload_cache = {
            cmd_id: bytes(length)
            for cmd_id, length in zip(InfoMsgBag.CMD_IDS, InfoMsgBag.PAYLOAD_LENGTHS)
        }
        self._valid_mask = 0
        self._last_update = {cmd_id: None for cmd_id in InfoMsgBag.CMD_IDS}
        self._seq = 0

    @staticmethod
    def _payload_0a01(config: dict[str, Any]) -> bytes:
        coordinates = config["messages"]["radar_map"]["coordinates"]
        values = []
        for robot_key, _label in ROBOT_KEYS[:6]:
            values.extend(
                (
                    coordinates[f"{robot_key}_x"],
                    coordinates[f"{robot_key}_y"],
                )
            )
        return struct.pack("<12H", *values)

    @staticmethod
    def _payload_0a02(config: dict[str, Any]) -> bytes:
        health = config["messages"]["health"]
        return struct.pack(
            "<6H",
            health["hero"],
            health["engineer"],
            health["standard_3"],
            health["standard_4"],
            health["reserved"],
            health["sentry"],
        )

    @staticmethod
    def _payload_0a03(config: dict[str, Any]) -> bytes:
        robot = config["messages"]["robot_multicast"]
        aerial_ammo = config["messages"]["engineer"]["aerial_ammo"]
        return struct.pack(
            "<5H",
            robot["hero"]["ammo"],
            robot["standard_3"]["ammo"],
            robot["standard_4"]["ammo"],
            aerial_ammo,
            robot["sentry"]["ammo"],
        )

    @staticmethod
    def _payload_0a04(config: dict[str, Any]) -> bytes:
        remaining_coins = config["messages"]["engineer"]["remaining_coins"]
        return struct.pack("<HHI", remaining_coins, 0, 0)

    @staticmethod
    def _payload_0a05(config: dict[str, Any]) -> bytes:
        robot_summary = config["messages"]["robot_multicast"]
        payload = bytearray(41)
        for robot_key, defense_offset in (
            ("hero", 3),
            ("engineer", 10),
            ("standard_3", 17),
            ("standard_4", 24),
            ("sentry", 31),
        ):
            robot = robot_summary[robot_key]
            payload[defense_offset] = robot["defense"]
            payload[defense_offset + 1] = robot["negative_defense"]
        # 页面直接编辑最终“是否可瞄准”语义，再转换为五类机器人主要状态。
        payload[35] = 0
        payload[36] = 0 if robot_summary["hero"]["targetable"] else 1
        payload[37] = 0 if robot_summary["engineer"]["targetable"] else 1
        payload[38] = 0 if robot_summary["standard_3"]["targetable"] else 1
        payload[39] = 0 if robot_summary["standard_4"]["targetable"] else 1
        payload[40] = 0 if robot_summary["sentry"]["targetable"] else 1
        return bytes(payload)

    def _build_update(self, cmd_id: int, payload: bytes, now: float) -> bytes:
        index = InfoMsgBag.CMD_IDS.index(cmd_id)
        expected_length = InfoMsgBag.PAYLOAD_LENGTHS[index]
        if len(payload) != expected_length:
            raise ValueError(
                f"0x{cmd_id:04X} payload长度应为{expected_length}，实际为{len(payload)}"
            )

        self._payload_cache[cmd_id] = bytes(payload)
        self._valid_mask |= 1 << index
        self._last_update[cmd_id] = now
        self._seq = (self._seq + 1) & 0xFFFF

        fresh_mask = 0
        for fresh_index, fresh_cmd_id in enumerate(InfoMsgBag.CMD_IDS):
            updated_at = self._last_update[fresh_cmd_id]
            if updated_at is not None and now - updated_at <= self.FRESH_WINDOW_SEC:
                fresh_mask |= 1 << fresh_index

        header = bytes(
            [
                ord("I"),
                ord("F"),
                3,
                (self._valid_mask & 0x1F) | ((index + 1) << 5),
                fresh_mask & 0x1F,
            ]
        ) + self._seq.to_bytes(2, "little")
        snapshot = header + b"".join(
            self._payload_cache[cached_cmd_id]
            for cached_cmd_id in InfoMsgBag.CMD_IDS
        )
        # 用正式解析器反向校验，避免测试工具产生正式链路不接受的包。
        InfoMsgBag.from_bytes(snapshot)
        return snapshot

    def build(self, message_key: str, config: dict[str, Any]) -> list[tuple[str, bytes]]:
        """一次页面调度对应一次或两次独立的“解析成功”事件。"""
        with self._lock:
            now = time.monotonic()
            if message_key == "radar_map":
                updates = ((0x0A01, self._payload_0a01(config)),)
            elif message_key == "health":
                updates = ((0x0A02, self._payload_0a02(config)),)
            elif message_key == "engineer":
                updates = (
                    (0x0A03, self._payload_0a03(config)),
                    (0x0A04, self._payload_0a04(config)),
                )
            elif message_key == "sentry":
                updates = ((0x0A05, self._payload_0a05(config)),)
            else:
                raise ValueError(f"信息波解析注入不支持：{message_key}")

            return [
                (f"0x{cmd_id:04X}", self._build_update(cmd_id, payload, now))
                for cmd_id, payload in updates
            ]

    @classmethod
    def preview(cls, config: dict[str, Any]) -> dict[str, Any]:
        factory = cls()
        result = {}
        for message_key in INFO_WAVE_MESSAGE_KEYS:
            for label, frame in factory.build(message_key, config):
                info = InfoMsgBag.from_bytes(frame)
                result[f"{message_key}_{label}"] = {
                    "enabled": config["messages"][message_key]["enabled"],
                    "cmd_id": label,
                    "length": len(frame),
                    "hex": frame.hex().upper(),
                    "valid_mask": info.valid_mask,
                    "fresh_mask": info.fresh_mask,
                    "seq": info.seq,
                }
        return result


class DryRunTransport:
    def __init__(self):
        self.started = False
        self.frames: list[bytes] = []

    def start(self):
        self.started = True

    def send(self, frame: bytes) -> bool:
        if not self.started:
            return False
        self.frames.append(bytes(frame))
        return True

    def stop(self):
        self.started = False

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.started,
            "port": "DRY-RUN",
            "state": "模拟发送",
        }


class SerialTransport:
    """复用 RefereeSerialManager.tx() 的真实裁判串口发送链路。"""

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.manager = None
        self._owns_rclpy = False

    def start(self):
        import rclpy
        from driver.referee.serial_comm import RefereeSerialManager

        class InjectionSerialManager(RefereeSerialManager):
            """禁用业务 UDP，仅保留正式串口收发、日志和重连机制。"""

            def _open_udp_rx(self, force=False):
                return False

            def _close_udp_rx(self):
                self.udp_rx = None

            def _process_udp_rx_once(self):
                return

            def _decode_message(self, cmd_id, data):
                return RawMessage(cmd_id, data)

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self.manager = InjectionSerialManager(
            port=self.port,
            baudrate=self.baudrate,
            auto_scan=False,
            log=True,
        )
        self.manager.start()

    def send(self, frame: bytes) -> bool:
        return self.manager is not None and bool(self.manager.tx(bytes(frame)))

    def stop(self):
        if self.manager is not None:
            self.manager.close()
            self.manager.destroy_node()
            self.manager = None
        if self._owns_rclpy:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
            self._owns_rclpy = False

    def status(self) -> dict[str, Any]:
        if self.manager is None:
            return {"connected": False, "port": self.port, "state": "未启动"}
        return {
            "connected": bool(self.manager.is_connected()),
            "port": self.manager.current_port or self.port,
            "state": self.manager.state.name,
        }


class AttachedInfoWaveTransport:
    """只注入解析快照，复用已由测试态 start.sh 启动的正式后链路。"""

    def __init__(self, relay_host: str = "127.0.0.1", relay_port: int = 5010):
        self.relay_host = relay_host
        self.relay_port = int(relay_port)
        self.socket = None
        self.test_session = ""

    @staticmethod
    def _read_marker() -> dict[str, str]:
        try:
            lines = FULL_CHAIN_TEST_MARKER.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(
                "正式链路未授权早注入；请使用 ./start.sh --test-info-wave-injection 启动"
            ) from exc
        values = {}
        for line in lines:
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip()] = value.strip()
        if values.get("mode") != "full_chain_test":
            raise RuntimeError("全链路测试授权标记无效")
        return values

    def _formal_relay_running(self) -> bool:
        if not self.test_session:
            return False
        expected_port = str(self.relay_port).encode()
        expected_session = f"INFO_WAVE_TEST_SESSION={self.test_session}".encode()
        for proc_entry in Path("/proc").iterdir():
            if not proc_entry.name.isdigit():
                continue
            try:
                cmdline = (proc_entry / "cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if b"info_wave_udp_relay.py" not in cmdline:
                continue
            if b"--listen-port " + expected_port in cmdline:
                try:
                    environment = (proc_entry / "environ").read_bytes().split(b"\0")
                except OSError:
                    continue
                if expected_session in environment:
                    return True
        return False

    def start(self):
        if self.socket is not None:
            return
        marker = self._read_marker()
        if marker.get("relay_port") != str(self.relay_port):
            raise RuntimeError(
                f"测试态 relay 端口为 {marker.get('relay_port', '未知')}，"
                f"与注入目标 {self.relay_port} 不一致"
            )
        self.test_session = marker.get("session", "")
        if not self.test_session:
            raise RuntimeError("全链路测试授权缺少本次启动会话标识")
        if not self._formal_relay_running():
            raise RuntimeError(
                "没有发现 start.sh 启动的正式信息波 relay；禁止向未知 UDP 端口注入"
            )
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: bytes) -> bool:
        if self.socket is None:
            return False
        sent = self.socket.sendto(bytes(frame), (self.relay_host, self.relay_port))
        return sent == len(frame)

    def stop(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def status(self) -> dict[str, Any]:
        relay_running = self._formal_relay_running()
        return {
            "connected": relay_running,
            "port": f"UDP {self.relay_host}:{self.relay_port}",
            "state": "已附着正式 start.sh 后链路" if relay_running else "正式 relay 未运行",
            "relay_running": relay_running,
            "game_progress": None,
            "attached_to_start": True,
        }


class AttachedLateOverrideTransport:
    """向正式主程序注入组包前语义快照，不直接占用裁判串口。"""

    def __init__(self, host: str = "127.0.0.1", port: int = WEB_LATE_DEFAULT_PORT):
        self.host = host
        self.port = int(port)
        self.socket = None
        self.test_session = ""

    def _formal_main_running(self) -> bool:
        if not self.test_session:
            return False
        expected_session = f"INFO_WAVE_TEST_SESSION={self.test_session}".encode()
        for proc_entry in Path("/proc").iterdir():
            if not proc_entry.name.isdigit():
                continue
            try:
                cmdline = (proc_entry / "cmdline").read_bytes().replace(b"\0", b" ")
            except OSError:
                continue
            if b"main.py" not in cmdline or b"config/params.yaml" not in cmdline:
                continue
            try:
                environment = (proc_entry / "environ").read_bytes().split(b"\0")
            except OSError:
                continue
            if expected_session in environment:
                return True
        return False

    def start(self):
        if self.socket is not None:
            return
        marker = AttachedInfoWaveTransport._read_marker()
        self.test_session = marker.get("session", "")
        if not self.test_session:
            raise RuntimeError("全链路测试授权缺少本次启动会话标识")
        if not self._formal_main_running():
            raise RuntimeError(
                "没有发现同一测试会话的正式 Python 主程序；禁止晚注入"
            )
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_config(self, config: dict[str, Any]) -> bool:
        if self.socket is None:
            return False
        payload = encode_web_late_snapshot(self.test_session, config, active=True)
        return self.socket.sendto(payload, (self.host, self.port)) == len(payload)

    def send(self, _frame: bytes) -> bool:
        raise RuntimeError("正式链路晚注入只接受语义快照")

    def stop(self):
        if self.socket is not None:
            payload = encode_web_late_snapshot(self.test_session, {}, active=False)
            for _ in range(3):
                self.socket.sendto(payload, (self.host, self.port))
            self.socket.close()
            self.socket = None

    def status(self) -> dict[str, Any]:
        running = self._formal_main_running()
        return {
            "connected": running,
            "port": f"UDP {self.host}:{self.port}",
            "state": "已连接正式主程序晚注入口" if running else "正式主程序未运行",
            "attached_to_start": True,
            "late_override": True,
        }


class InjectionController:
    """保存页面配置，并按所选模式调度最终串口帧或信息波解析事件。"""

    def __init__(
        self,
        transport,
        max_active_sec: float,
        mode: str,
        transport_factory=None,
    ):
        self.transport = transport
        self.transport_factory = transport_factory
        self.max_active_sec = max(0.05, float(max_active_sec))
        self.mode = mode
        self.info_wave_mode = mode.startswith("info-wave")
        self.late_attach_mode = mode.startswith("late-attach")
        self.message_keys = (
            INFO_WAVE_MESSAGE_KEYS if self.info_wave_mode else MESSAGE_KEYS
        )
        self.info_wave_factory = InfoWaveSnapshotFactory()
        self.lock = threading.RLock()
        self.config = normalize_config(default_config())
        self.active = False
        self.started_monotonic = None
        self.expires_monotonic = None
        self.stop_event = threading.Event()
        self.worker = None
        self.send_counts = {
            key: 0 for key in dict.fromkeys(MESSAGE_KEYS + INFO_WAVE_MESSAGE_KEYS)
        }
        self.failed_count = 0
        self.last_frame = None
        self.last_error = ""

    def set_mode(self, requested_mode: str) -> str:
        """仅在停止状态切换早/晚注入，并完整释放旧传输资源。"""
        if self.transport_factory is None:
            raise ValueError("当前命令行模式不允许在 Web 页面切换")
        if requested_mode not in ("late-attach", "info-wave-attach"):
            raise ValueError("只允许选择正式链路早注入或晚注入")
        with self.lock:
            if self.active:
                raise ValueError("请先停止注入，再切换注入位置")
            old_transport = self.transport
        old_transport.stop()
        new_transport, controller_mode = self.transport_factory(requested_mode)
        with self.lock:
            self.transport = new_transport
            self.mode = controller_mode
            self.info_wave_mode = controller_mode.startswith("info-wave")
            self.late_attach_mode = controller_mode.startswith("late-attach")
            self.message_keys = (
                INFO_WAVE_MESSAGE_KEYS if self.info_wave_mode else MESSAGE_KEYS
            )
            self.info_wave_factory = InfoWaveSnapshotFactory()
            self.last_frame = None
            self.last_error = ""
        return controller_mode

    def update_config(self, raw_config: Any):
        normalized = normalize_config(raw_config)
        with self.lock:
            self.config = normalized
        return normalized

    def start(self, confirmation: str):
        if confirmation != PAGE_CONFIRMATION:
            raise ValueError(f"请输入确认文字：{PAGE_CONFIRMATION}")
        with self.lock:
            if self.active:
                return
            if self.info_wave_mode:
                # 每次点击开始都视为独立测试小局，禁止沿用上一次的valid缓存和seq。
                self.info_wave_factory = InfoWaveSnapshotFactory()
            self.transport.start()
            now = time.monotonic()
            self.active = True
            self.started_monotonic = now
            self.expires_monotonic = now + self.max_active_sec
            self.stop_event.clear()
            self.worker = threading.Thread(
                target=self._run_scheduler,
                name="referee-web-injector-scheduler",
                daemon=True,
            )
            self.worker.start()

    def stop(self):
        with self.lock:
            self.active = False
            self.stop_event.set()
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        self.transport.stop()
        with self.lock:
            self.worker = None
            self.started_monotonic = None
            self.expires_monotonic = None

    def _send_one(self, message_key: str, config: dict[str, Any]):
        if not config["messages"][message_key]["enabled"]:
            return False
        try:
            if self.info_wave_mode:
                frames = self.info_wave_factory.build(message_key, config)
            else:
                frames = [(message_key, PacketFactory.build(message_key, config))]

            all_success = True
            for frame_label, frame in frames:
                success = bool(self.transport.send(frame))
                all_success = all_success and success
                with self.lock:
                    if success:
                        self.send_counts[message_key] += 1
                        self.last_frame = {
                            "message": frame_label,
                            "time": time.strftime("%H:%M:%S"),
                            "hex": frame.hex().upper(),
                        }
                        self.last_error = ""
                    else:
                        self.failed_count += 1
                        self.last_error = "测试链路尚未连接或发送失败"
            return all_success
        except Exception as exc:
            with self.lock:
                self.failed_count += 1
                self.last_error = str(exc)
            return False

    def _run_scheduler(self):
        if self.info_wave_mode:
            self._run_info_wave_scheduler()
            return
        if self.late_attach_mode:
            self._run_late_attach_scheduler()
            return

        now = time.monotonic()
        next_due = {key: now for key in MESSAGE_KEYS}
        last_frequencies = None
        interactive_keys = (
            "decision",
            "dart",
            "engineer",
            "sentry",
            "robot_multicast",
        )
        interactive_cursor = 0
        interactive_next_allowed = now
        try:
            while not self.stop_event.wait(0.002):
                now = time.monotonic()
                with self.lock:
                    if not self.active:
                        break
                    if self.expires_monotonic is not None and now >= self.expires_monotonic:
                        self.active = False
                        self.last_error = (
                            f"已达到测试注入 {int(self.max_active_sec)} 秒安全时限，自动停止"
                        )
                        break
                    config = copy.deepcopy(self.config)

                frequencies = {
                    key: config["messages"][key]["frequency_hz"]
                    for key in MESSAGE_KEYS
                }
                if last_frequencies is not None:
                    for key in MESSAGE_KEYS:
                        if frequencies[key] != last_frequencies[key]:
                            # 频率热更新从修改时刻重新计时，避免降频后残留旧周期多发一帧。
                            next_due[key] = now + 1.0 / frequencies[key]
                last_frequencies = frequencies

                if now >= next_due["radar_map"]:
                    self._send_one("radar_map", config)
                    next_due["radar_map"] = now + 1.0 / frequencies["radar_map"]

                # 测试注入也统一串行调度 0x0301；即使各字段实时配置不同
                # 频率，任意两个实际启用的交互包之间仍至少间隔 34 ms。
                if now >= interactive_next_allowed:
                    for offset in range(len(interactive_keys)):
                        index = (interactive_cursor + offset) % len(interactive_keys)
                        message_key = interactive_keys[index]
                        if now < next_due[message_key]:
                            continue
                        next_due[message_key] = (
                            now + 1.0 / frequencies[message_key]
                        )
                        if not config["messages"][message_key]["enabled"]:
                            continue
                        self._send_one(message_key, config)
                        interactive_cursor = (index + 1) % len(interactive_keys)
                        interactive_next_allowed = (
                            now + RefereeTxPolicy.INTERACTIVE_MIN_GAP_SEC
                        )
                        break

        finally:
            self.transport.stop()
            with self.lock:
                self.active = False
                self.worker = None
                self.started_monotonic = None
                self.expires_monotonic = None

    def _run_late_attach_scheduler(self):
        """10Hz刷新正式主程序覆盖快照，实际串口仍按正式5/5/5/5/8槽位发送。"""
        try:
            while not self.stop_event.wait(0.1):
                now = time.monotonic()
                with self.lock:
                    if not self.active:
                        break
                    if self.expires_monotonic is not None and now >= self.expires_monotonic:
                        self.active = False
                        self.last_error = (
                            f"已达到测试注入 {int(self.max_active_sec)} 秒安全时限，自动停止"
                        )
                        break
                    config = copy.deepcopy(self.config)
                try:
                    success = bool(self.transport.send_config(config))
                    with self.lock:
                        if success:
                            for key in MESSAGE_KEYS:
                                if config["messages"][key]["enabled"]:
                                    self.send_counts[key] += 1
                            self.last_frame = {
                                "message": "晚注入语义快照",
                                "time": time.strftime("%H:%M:%S"),
                                "hex": "由正式主程序组包",
                            }
                            self.last_error = ""
                        else:
                            self.failed_count += 1
                            self.last_error = "正式主程序晚注入口发送失败"
                except Exception as exc:
                    with self.lock:
                        self.failed_count += 1
                        self.last_error = str(exc)
        finally:
            self.transport.stop()
            with self.lock:
                self.active = False
                self.worker = None
                self.started_monotonic = None
                self.expires_monotonic = None

    def _run_info_wave_scheduler(self):
        """按各信息波命令解析频率产生独立新代次，不套用最终串口调度。"""
        now = time.monotonic()
        next_due = {key: now for key in self.message_keys}
        last_frequencies = None
        try:
            while not self.stop_event.wait(0.002):
                now = time.monotonic()
                with self.lock:
                    if not self.active:
                        break
                    if self.expires_monotonic is not None and now >= self.expires_monotonic:
                        self.active = False
                        self.last_error = (
                            f"已达到测试注入 {int(self.max_active_sec)} 秒安全时限，自动停止"
                        )
                        break
                    config = copy.deepcopy(self.config)

                frequencies = {
                    key: config["messages"][key]["frequency_hz"]
                    for key in self.message_keys
                }
                if last_frequencies is not None:
                    for key in self.message_keys:
                        if frequencies[key] != last_frequencies[key]:
                            next_due[key] = now + 1.0 / frequencies[key]
                last_frequencies = frequencies

                for message_key in self.message_keys:
                    if now < next_due[message_key]:
                        continue
                    self._send_one(message_key, config)
                    next_due[message_key] = now + 1.0 / frequencies[message_key]
        finally:
            self.transport.stop()
            with self.lock:
                self.active = False
                self.worker = None
                self.started_monotonic = None
                self.expires_monotonic = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            remaining_sec = 0
            if self.active and self.expires_monotonic is not None:
                remaining_sec = max(0, int(self.expires_monotonic - time.monotonic()))
            config = copy.deepcopy(self.config)
            return {
                "active": self.active,
                "mode": self.mode,
                "remaining_sec": remaining_sec,
                "max_active_sec": int(self.max_active_sec),
                "config": config,
                "frequencies": {
                    key: config["messages"][key]["frequency_hz"]
                    for key in self.message_keys
                },
                "transport": self.transport.status(),
                "send_counts": dict(self.send_counts),
                "failed_count": self.failed_count,
                "last_frame": copy.deepcopy(self.last_frame),
                "last_error": self.last_error,
                "preview": (
                    InfoWaveSnapshotFactory.preview(config)
                    if self.info_wave_mode
                    else PacketFactory.preview(config)
                ),
            }


class InjectorHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, controller, api_token):
        super().__init__(address, InjectorRequestHandler)
        self.controller = controller
        self.api_token = api_token


class InjectorRequestHandler(BaseHTTPRequestHandler):
    server_version = "RM2026TestInjector/1.0"

    def log_message(self, format_string, *args):
        sys.stdout.write(
            f"[web] {self.address_string()} "
            f"{format_string % args}\n"
        )

    def _send_json(self, status: int, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Test-Token", ""),
            self.server.api_token,
        )

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 128 * 1024:
            raise ValueError("请求体长度无效")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            html = HTML_FILE.read_text(encoding="utf-8").replace(
                "__API_TOKEN__", self.server.api_token
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/state":
            if not self._authorized():
                self._send_json(403, {"ok": False, "error": "无效测试令牌"})
                return
            self._send_json(200, {"ok": True, "state": self.server.controller.snapshot()})
            return

        self._send_json(404, {"ok": False, "error": "接口不存在"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(403, {"ok": False, "error": "无效测试令牌"})
            return
        try:
            payload = self._read_json()
            if self.path == "/api/config":
                config = self.server.controller.update_config(payload.get("config"))
                self._send_json(200, {"ok": True, "config": config})
                return
            if self.path == "/api/mode":
                mode = self.server.controller.set_mode(str(payload.get("mode", "")))
                self._send_json(200, {"ok": True, "mode": mode})
                return
            if self.path == "/api/control/start":
                self.server.controller.start(str(payload.get("confirmation", "")))
                self._send_json(200, {"ok": True})
                return
            if self.path == "/api/control/stop":
                self.server.controller.stop()
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"ok": False, "error": "接口不存在"})
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def load_serial_defaults() -> tuple[str, int]:
    config_path = PROJECT_DIR / "config" / "params.yaml"
    try:
        import yaml

        root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        referee = root.get("referee", {})
        return str(referee.get("port", "/dev/referee")), int(
            referee.get("baudrate", 115200)
        )
    except Exception:
        return "/dev/referee", 115200


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _pid_is_injector(pid: int) -> bool:
    if not _pid_is_alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return b"referee_web_injector" in cmdline
    except OSError:
        return False


def claim_pid_file():
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = 0
        if old_pid > 0 and old_pid != os.getpid() and _pid_is_injector(old_pid):
            raise RuntimeError(f"测试注入器已经运行，PID={old_pid}")
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")


def release_pid_file():
    try:
        if PID_FILE.exists() and PID_FILE.read_text(encoding="utf-8").strip() == str(
            os.getpid()
        ):
            PID_FILE.unlink()
    except OSError:
        pass


def parse_args():
    default_port, default_baudrate = load_serial_defaults()
    parser = argparse.ArgumentParser(description="RM2026 裁判链路测试 Web 注入器")
    parser.add_argument("--enable-test-injection", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--mode",
        choices=("web", "serial", "info-wave-attach"),
        default="web",
        help=(
            "web=在页面切换早/后注入；serial=直接构造最终串口包；"
            "info-wave-attach=附着测试态 start.sh 的正式后链路"
        ),
    )
    parser.add_argument(
        "--attach-confirm",
        default="",
        help="附着正式链路时必须显式填写 FULL_CHAIN_TEST_ONLY",
    )
    parser.add_argument("--port", default=default_port)
    parser.add_argument("--baudrate", type=int, default=default_baudrate)
    parser.add_argument("--info-relay-listen-port", type=int, default=5010)
    parser.add_argument("--web-late-port", type=int, default=WEB_LATE_DEFAULT_PORT)
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--max-active-sec", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true", help="只构包，不写真实串口")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable_test_injection or args.confirm != TEST_CONFIRMATION:
        print("拒绝启动：测试注入器必须显式提供：")
        print("  --enable-test-injection --confirm TEST_ONLY")
        return 2
    if (
        args.mode == "info-wave-attach"
        and args.attach_confirm != "FULL_CHAIN_TEST_ONLY"
    ):
        print("拒绝附着正式链路：必须额外提供：")
        print("  --attach-confirm FULL_CHAIN_TEST_ONLY")
        return 2
    if not HTML_FILE.exists():
        print(f"页面文件不存在：{HTML_FILE}")
        return 2

    claim_pid_file()
    atexit.register(release_pid_file)
    controller = None
    server = None
    try:
        def build_selectable_transport(requested_mode: str):
            if args.dry_run:
                suffix = "-dry-run"
                return DryRunTransport(), requested_mode + suffix
            if requested_mode == "info-wave-attach":
                return (
                    AttachedInfoWaveTransport(
                        relay_port=args.info_relay_listen_port,
                    ),
                    "info-wave-attach",
                )
            if requested_mode == "late-attach":
                return (
                    AttachedLateOverrideTransport(port=args.web_late_port),
                    "late-attach",
                )
            raise ValueError(f"Web 不支持注入模式：{requested_mode}")

        transport_factory = None
        if args.mode == "web":
            # 页面默认停在正式主程序晚注入，早/晚两种方式都不抢裁判串口。
            transport_factory = build_selectable_transport
            transport, controller_mode = transport_factory("late-attach")
        elif args.dry_run:
            transport = DryRunTransport()
            controller_mode = (
                "info-wave-attach-dry-run"
                if args.mode == "info-wave-attach"
                else "serial-dry-run"
            )
        elif args.mode == "info-wave-attach":
            transport = AttachedInfoWaveTransport(
                relay_port=args.info_relay_listen_port,
            )
            controller_mode = "info-wave-attach"
        else:
            transport = SerialTransport(args.port, args.baudrate)
            controller_mode = "serial"
        controller = InjectionController(
            transport=transport,
            max_active_sec=args.max_active_sec,
            mode=controller_mode,
            transport_factory=transport_factory,
        )
        api_token = secrets.token_urlsafe(32)
        server = InjectorHttpServer(
            ("127.0.0.1", args.web_port), controller, api_token
        )
        url = f"http://127.0.0.1:{args.web_port}/"
        print("============================================================")
        print("RM2026 裁判链路测试注入器：仅限离线联调，正式比赛禁止运行")
        print(f"页面：{url}")
        if args.mode == "web":
            print("模式：在 Web 页面选择早注入或晚注入（均不抢裁判串口）")
        elif args.mode == "info-wave-attach":
            print(f"模式：{controller_mode}")
            print(
                "链路：InfoMsgBag v3 -> 已运行的正式 relay "
                f"{args.info_relay_listen_port} -> 正式主程序 -> 裁判串口"
            )
        else:
            print(f"模式：{controller_mode}")
        print(f"串口：{'DRY-RUN' if args.dry_run else args.port} @ {args.baudrate}")
        print(f"单次注入最长：{int(args.max_active_sec)} 秒")
        if args.mode == "web":
            print("早注入要求先用 --test-info-wave-injection 启动 start.sh。")
            print("晚注入会覆盖正式主程序组包前数据源，不直接占用裁判串口。")
        elif args.mode == "info-wave-attach":
            print("当前为全链路测试附着模式；必须先启动带测试参数的 start.sh。")
        else:
            print("正式 start.sh 检测到本进程后会拒绝启动。")
        print("============================================================")

        stop_requested = threading.Event()

        def request_stop(_signum=None, _frame=None):
            if not stop_requested.is_set():
                stop_requested.set()
                threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        if args.open_browser:
            webbrowser.open(url)
        server.serve_forever(poll_interval=0.2)
    finally:
        if controller is not None:
            controller.stop()
        if server is not None:
            server.server_close()
        release_pid_file()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
