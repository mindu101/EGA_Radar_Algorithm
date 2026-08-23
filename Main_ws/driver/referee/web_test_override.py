"""测试态 Web 晚注入快照协议；仅允许本机测试会话使用。"""

from __future__ import annotations

import json
from typing import Any


MAGIC = "RM2026_WEB_LATE_V1"
DEFAULT_PORT = 5011
MAX_PACKET_SIZE = 32 * 1024


def encode_snapshot(session: str, config: dict[str, Any], active: bool = True) -> bytes:
    return json.dumps(
        {
            "magic": MAGIC,
            "session": str(session),
            "active": bool(active),
            "config": config if active else {},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_snapshot(payload: bytes, expected_session: str) -> tuple[bool, dict[str, Any]]:
    if not payload or len(payload) > MAX_PACKET_SIZE:
        raise ValueError("Web 晚注入包长度无效")
    root = json.loads(payload.decode("utf-8"))
    if not isinstance(root, dict) or root.get("magic") != MAGIC:
        raise ValueError("Web 晚注入包标识无效")
    if root.get("session") != expected_session:
        raise ValueError("Web 晚注入测试会话不匹配")
    active = root.get("active") is True
    config = root.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Web 晚注入配置格式无效")
    return active, config
