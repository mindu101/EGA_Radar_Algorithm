from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from enum import Enum

INFO_MSG_MAGIC = b"IF"
INFO_MSG_VERSION = 3
INFO_MSG_LEGACY_VERSION = 2
INFO_MSG_HEADER_SIZE = 7
INFO_MSG_LEGACY_HEADER_SIZE = 6
INFO_MSG_PAYLOAD_LENGTHS = (24, 12, 10, 8, 41)
INFO_MSG_PAYLOAD_SIZE = sum(INFO_MSG_PAYLOAD_LENGTHS)
INFO_MSG_SIZE = INFO_MSG_HEADER_SIZE + INFO_MSG_PAYLOAD_SIZE
INFO_MSG_LEGACY_SIZE = INFO_MSG_LEGACY_HEADER_SIZE + INFO_MSG_PAYLOAD_SIZE
INFO_MSG_CACHE_VALID_MASK = 0x1F
INFO_MSG_UPDATE_INDEX_SHIFT = 5
INFO_MSG_UPDATE_INDEX_MASK = 0x07
INFO_MSG_LEGACY_0A01_RECENT_UPDATE_MASK = 0x20
INFO_MSG_LEGACY_0A05_RECENT_UPDATE_MASK = 0x40


# =========================
# UDP 数据类型
# =========================


@dataclass(frozen=True)
class XYZ:
    x: int
    y: int
    z: int


class SendData(ctypes.LittleEndianStructure):
    """
    UDP SendMsg 载荷。

    该结构把裁判系统中当前需要转发的少量状态压成一个固定小包：
    - Byte 0: 比赛类型 + 当前比赛阶段
    - Byte 1: 本机机器人 ID
    - Byte 2: 雷达状态信息
    """

    _pack_ = 1
    _fields_ = [
        ("game_progress", ctypes.c_uint8, 4),  # byte0 bit0-3：当前比赛阶段
        ("dart_selected_target", ctypes.c_uint8, 3),  # byte0 bit4-6：0x0105 bit6-8当前值
        ("reserve_0", ctypes.c_uint8, 1),  # byte0 bit7：保留
        ("robot_id", ctypes.c_uint8),  # byte1：本机机器人 ID
        ("own_encrypt_level", ctypes.c_uint8, 2),  # byte2 bit0-1：己方加密等级
        ("is_key_changeable", ctypes.c_uint8, 1),  # byte2 bit2：当前是否可以修改密钥
        ("enemy_drone_counteracted", ctypes.c_uint8, 1),  # byte2 bit3：对方空中是否被反制
        ("dart_4_to_1_event_seq", ctypes.c_uint8, 4),  # byte2 bit4-7：第二次机会4→…→1事件序号
    ]

class RecvData(ctypes.LittleEndianStructure):
    """
    UDP 接收数据载荷。

    bytes 0-6: 密钥，整体读入，不解析细节，直接转发给串口。
    """
    _pack_ = 1
    _fields_ = [
        ("key", ctypes.c_uint8 * 7),  # 7 字节密钥
    ]

class MsgBag:
    """
    UDP Msg 包基类，提供通用的 struct/bytes 转换功能。
    """

    def __init__(self, **kwargs):
        self.data = self.STRUCT_CLASS()

        for field_name, value in kwargs.items():
            if not self._has_field(field_name):
                raise ValueError(f"Field {field_name} not found in {self.STRUCT_CLASS.__name__}")
            self._set_field(field_name, value)

    def pack(self) -> bytes:
        return bytes(self.data)

    @classmethod
    def from_struct(cls, data):
        if not isinstance(data, cls.STRUCT_CLASS):
            raise TypeError(f"data must be {cls.STRUCT_CLASS.__name__}")

        msg = cls()
        msg.data = cls.STRUCT_CLASS.from_buffer_copy(bytes(data))
        return msg

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview):
        raw_bytes = bytes(raw)
        if len(raw_bytes) != cls.SIZE:
            raise ValueError(
                f"Invalid {cls.__name__} size: expected {cls.SIZE}, got {len(raw_bytes)}"
            )

        msg = cls()
        msg.data = cls.STRUCT_CLASS.from_buffer_copy(raw_bytes)
        return msg

    def __getattr__(self, name: str):
        if self._has_field(name):
            return getattr(self.data, name)
        raise AttributeError(f"{name} not found in {self.STRUCT_CLASS.__name__}")

    def __setattr__(self, name: str, value) -> None:
        if name == "data":
            super().__setattr__(name, value)
        elif "data" in self.__dict__ and self._has_field(name):
            self._set_field(name, value)
        else:
            super().__setattr__(name, value)

    def __str__(self) -> str:
        return "\n".join(
            f"{field[0]}: {getattr(self.data, field[0])}"
            for field in self.STRUCT_CLASS._fields_
        )

    @classmethod
    def _has_field(cls, name: str) -> bool:
        return any(field[0] == name for field in cls.STRUCT_CLASS._fields_)

    def _set_field(self, name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")

        min_value, max_value = self.FIELD_RANGES[name]
        if value < min_value or value > max_value:
            raise ValueError(f"{name}={value} out of range [{min_value}, {max_value}]")

        setattr(self.data, name, value)

class SendMsgBag(MsgBag):
    """
    UDP SendMsg 包。

    用法示例：
        SendMsgBag(game_progress=4, robot_id=7, own_encrypt_level=1)
        SendMsgBag.from_struct(msg_data)
        SendMsgBag.from_bytes(raw)
    """

    STRUCT_CLASS = SendData
    SIZE = ctypes.sizeof(STRUCT_CLASS)
    FIELD_RANGES = {
        "game_progress": (0, 0x0F),
        "dart_selected_target": (0, 0x07),
        "reserve_0": (0, 0x01),
        "robot_id": (0, 0xFF),
        "own_encrypt_level": (0, 0x03),
        "is_key_changeable": (0, 0x01),
        "enemy_drone_counteracted": (0, 0x01),
        "dart_4_to_1_event_seq": (0, 0x0F),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class RecvMsgBag(MsgBag):
    """
    UDP RecvMsg 包。

    当前只接收 7 字节密钥，不解析内部含义，后续可以直接转发给串口侧。
    """

    STRUCT_CLASS = RecvData
    SIZE = ctypes.sizeof(STRUCT_CLASS)

    def __init__(self, key: bytes | bytearray | memoryview | list[int] | tuple[int, ...] | None = None):
        super().__init__()
        if key is not None:
            self._set_field("key", key)

    @property
    def key_bytes(self) -> bytes:
        return bytes(self.data.key)

    @property
    def key_hex(self) -> str:
        return self.key_bytes.hex()

    def __str__(self) -> str:
        return f"key: {self.key_hex}"

    def _set_field(
        self,
        name: str,
        value: bytes | bytearray | memoryview | list[int] | tuple[int, ...],
    ) -> None:
        if name != "key":
            raise ValueError(f"Field {name} not found in {self.STRUCT_CLASS.__name__}")

        raw = bytes(value)
        if len(raw) != self.SIZE:
            raise ValueError(f"key must be {self.SIZE} bytes, got {len(raw)}")

        for index, byte in enumerate(raw):
            self.data.key[index] = byte


@dataclass(frozen=True)
class AutoAimStatusBag:
    """自瞄程序回传的单字节空中机器人反制阶段。"""

    stage: int

    SIZE = 1
    MIN_STAGE = 0x00
    MAX_STAGE = 0x06

    def __post_init__(self):
        stage = int(self.stage)
        if not self.MIN_STAGE <= stage <= self.MAX_STAGE:
            raise ValueError(
                f"autoaim stage={stage} out of range "
                f"[{self.MIN_STAGE}, {self.MAX_STAGE}]"
            )
        object.__setattr__(self, "stage", stage)

    def pack(self) -> bytes:
        return bytes([self.stage])

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview):
        raw_bytes = bytes(raw)
        if len(raw_bytes) != cls.SIZE:
            raise ValueError(
                f"Invalid {cls.__name__} size: expected {cls.SIZE}, got {len(raw_bytes)}"
            )
        return cls(stage=raw_bytes[0])


@dataclass(frozen=True)
class InfoMsgBag:
    """信息波五个命令码的最新快照。

    UDP v3 格式固定 102 字节：
    - byte 0-1: ASCII "IF"
    - byte 2: version，当前为 3
    - byte 3: bit0-4 为 valid_mask；bit5-7 为本快照更新的命令索引 1-5
    - byte 4: fresh_mask，bit0-4 分别表示对应命令是否在最近 3 秒内更新
    - byte 5-6: seq，小端递增序号
    - 后续 95 字节：依次拼接 0x0A01/0x0A02/0x0A03/0x0A04/0x0A05 的原始 dataBytes

    接收端兼容旧版 101 字节 v2 包，便于平滑升级和历史测试。
    """

    valid_mask: int
    fresh_mask: int
    seq: int
    updated_cmd_id: int | None
    payload: bytes
    raw: bytes

    SIZE = INFO_MSG_SIZE
    PAYLOAD_LENGTHS = INFO_MSG_PAYLOAD_LENGTHS
    CMD_IDS = (0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05)

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview) -> "InfoMsgBag":
        raw_bytes = bytes(raw)
        if len(raw_bytes) < 3:
            raise ValueError(f"Invalid InfoMsgBag size: got {len(raw_bytes)}")
        if raw_bytes[0:2] != INFO_MSG_MAGIC:
            raise ValueError(f"Invalid InfoMsgBag magic: {raw_bytes[0:2].hex()}")
        version = raw_bytes[2]
        if version == INFO_MSG_VERSION:
            if len(raw_bytes) != INFO_MSG_SIZE:
                raise ValueError(
                    f"Invalid InfoMsgBag v3 size: expected {INFO_MSG_SIZE}, got {len(raw_bytes)}"
                )
            valid_mask = raw_bytes[3] & INFO_MSG_CACHE_VALID_MASK
            update_index = (
                raw_bytes[3] >> INFO_MSG_UPDATE_INDEX_SHIFT
            ) & INFO_MSG_UPDATE_INDEX_MASK
            updated_cmd_id = (
                cls.CMD_IDS[update_index - 1]
                if 1 <= update_index <= len(cls.CMD_IDS)
                else None
            )
            fresh_mask = raw_bytes[4] & INFO_MSG_CACHE_VALID_MASK
            seq = int.from_bytes(raw_bytes[5:7], byteorder="little", signed=False)
            payload = raw_bytes[INFO_MSG_HEADER_SIZE:]
        elif version == INFO_MSG_LEGACY_VERSION:
            if len(raw_bytes) != INFO_MSG_LEGACY_SIZE:
                raise ValueError(
                    f"Invalid InfoMsgBag v2 size: expected {INFO_MSG_LEGACY_SIZE}, got {len(raw_bytes)}"
                )
            legacy_mask = raw_bytes[3]
            valid_mask = legacy_mask & INFO_MSG_CACHE_VALID_MASK
            updated_cmd_id = None
            fresh_mask = 0
            if legacy_mask & INFO_MSG_LEGACY_0A01_RECENT_UPDATE_MASK:
                fresh_mask |= 1 << cls.CMD_IDS.index(0x0A01)
            if legacy_mask & INFO_MSG_LEGACY_0A05_RECENT_UPDATE_MASK:
                fresh_mask |= 1 << cls.CMD_IDS.index(0x0A05)
            seq = int.from_bytes(raw_bytes[4:6], byteorder="little", signed=False)
            payload = raw_bytes[INFO_MSG_LEGACY_HEADER_SIZE:]
        else:
            raise ValueError(f"Unsupported InfoMsgBag version: {version}")

        return cls(
            valid_mask=valid_mask,
            fresh_mask=fresh_mask,
            seq=seq,
            updated_cmd_id=updated_cmd_id,
            payload=payload,
            raw=raw_bytes,
        )

    def pack(self) -> bytes:
        return self.raw

    def is_cmd_valid(self, cmd_id: int) -> bool:
        if cmd_id not in self.CMD_IDS:
            return False
        index = self.CMD_IDS.index(cmd_id)
        return bool(self.valid_mask & (1 << index))

    def is_cmd_recently_updated(self, cmd_id: int) -> bool:
        """判断该快照是否携带某个命令码的近期解析成功标志。"""
        if cmd_id not in self.CMD_IDS:
            return False
        return bool(self.fresh_mask & (1 << self.CMD_IDS.index(cmd_id)))

    def get_cmd_payload(self, cmd_id: int) -> bytes:
        if cmd_id not in self.CMD_IDS:
            raise ValueError(f"Unsupported info cmd_id: 0x{cmd_id:04X}")
        offset = 0
        for index, length in enumerate(self.PAYLOAD_LENGTHS):
            if self.CMD_IDS[index] == cmd_id:
                return self.payload[offset:offset + length]
            offset += length
        raise ValueError(f"Unsupported info cmd_id: 0x{cmd_id:04X}")

    @property
    def valid_cmd_ids(self) -> tuple[int, ...]:
        return tuple(cmd_id for cmd_id in self.CMD_IDS if self.is_cmd_valid(cmd_id))

    def __str__(self) -> str:
        cmds = ",".join(f"0x{cmd_id:04X}" for cmd_id in self.valid_cmd_ids)
        return (
            f"InfoMsg(seq={self.seq}, valid_mask=0x{self.valid_mask:02X}, "
            f"fresh_mask=0x{self.fresh_mask:02X}, valid=[{cmds}])"
        )


class UdpSendDataKind(Enum):
    XYZ = "XYZ"
    MSG = "SendMsg"

class XYZPacket:
    """
    XYZ UDP 载荷格式：
        int16 x
        int16 y
        int16 z
    总长度 6 字节。
    """

    STRUCT = struct.Struct("<hhh")
    SIZE = STRUCT.size

    INT16_MIN = -32768
    INT16_MAX = 32767

    @classmethod
    def pack(cls, x: int, y: int, z: int) -> bytes:
        cls._check_int16(x, "x")
        cls._check_int16(y, "y")
        cls._check_int16(z, "z")
        return cls.STRUCT.pack(x, y, z)

    @classmethod
    def unpack(cls, data: bytes | memoryview) -> XYZ:
        if len(data) != cls.SIZE:
            raise ValueError(f"Invalid XYZ size: expected {cls.SIZE}, got {len(data)}")

        x, y, z = cls.STRUCT.unpack(data)
        return XYZ(x=x, y=y, z=z)

    @classmethod
    def _check_int16(cls, value: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")

        if value < cls.INT16_MIN or value > cls.INT16_MAX:
            raise ValueError(
                f"{name}={value} out of int16 range "
                f"[{cls.INT16_MIN}, {cls.INT16_MAX}]"
            )

class SendMsgPacket:
    """固定的 UDP 载荷格式。"""

    SIZE = SendMsgBag.SIZE

    @classmethod
    def pack(cls, msg: SendMsgBag) -> bytes:
        if not isinstance(msg, SendMsgBag):
            raise TypeError("msg must be a SendMsgBag instance")
        return msg.pack()

    @classmethod
    def unpack(cls, data: bytes | bytearray | memoryview) -> SendMsgBag:
        return SendMsgBag.from_bytes(data)

class RecvMsgPacket:
    """固定的 UDP 接收载荷格式。"""

    SIZE = RecvMsgBag.SIZE

    @classmethod
    def pack(cls, msg: RecvMsgBag) -> bytes:
        if not isinstance(msg, RecvMsgBag):
            raise TypeError("msg must be a RecvMsgBag instance")
        return msg.pack()

    @classmethod
    def unpack(cls, data: bytes | bytearray | memoryview) -> RecvMsgBag:
        return RecvMsgBag.from_bytes(data)

class InfoMsgPacket:
    """信息波 UDP 快照载荷。"""

    SIZE = InfoMsgBag.SIZE
    SIZES = (InfoMsgBag.SIZE, INFO_MSG_LEGACY_SIZE)

    @classmethod
    def unpack(cls, data: bytes | bytearray | memoryview) -> InfoMsgBag:
        return InfoMsgBag.from_bytes(data)


class UdpPacket:
    """UDP 发送包分发器，仅支持当前工程明确使用的 XYZ 和 SendMsg 两类数据。"""

    @classmethod
    def pack(cls, data: XYZ | SendMsgBag) -> tuple[UdpSendDataKind, bytes]:
        if isinstance(data, XYZ):
            return UdpSendDataKind.XYZ, XYZPacket.pack(data.x, data.y, data.z)

        if isinstance(data, SendMsgBag):
            return UdpSendDataKind.MSG, SendMsgPacket.pack(data)

        raise TypeError("UDP data must be XYZ or SendMsgBag instance")
