import struct
import serial
from enum import Enum
from driver.referee.crc import Crc
import numpy as np
from abc import ABC, abstractmethod
import ctypes
import threading


# Command protocal
# ======================
# Start Bytes：uint8_t
# payloadLength : uint8_t
# id: uint8_t
# CRC16: uint16_t
# data field: x
# CRC16: uint16_t
# ======================
# Total size： 7 + x


class MsgID(Enum):
    """
    Message ID definition for RoboMaster protocol
    """

    # 比赛状态数据，固定以1Hz发
    GAME_STATUS = 0x0001
    # 比赛结果数据，比赛结束后触发
    GAME_RESULT = 0x0002
    # 机器人血量数据，固定以3Hz发送
    ROBOT_HP = 0x0003

    # 场地事件数据，固定以1Hz发送
    FIELD_EVENT = 0x0101
    # 裁判系统警告数据，己方判负/判罚时触发发送，其余时间以1Hz频率发送
    REFEREE_WARNING = 0x0104
    # 飞镖发射相关数据，固定以1Hz发送
    LAUNCHER_DATA = 0x0105

    # 机器人性能体系数据，固定以10Hz频率发送
    ROBOT_DATA = 0x0201
    # 实时底盘缓冲能量和射击热量数据，固定以10Hz频率发送
    ROBOT_POWER = 0x0202
    # 机器人位置数据，固定以1Hz频率发送
    SELF_ROBOT_POS = 0x0203
    # 机器人增益和底盘能量数据，固定以3Hz频率发送
    CHASSIS_AND_GAIN = 0x0204
    # 伤害状态数据，伤害发生后发送
    HURT_DATA = 0x0206

    # 实时射击数据，弹丸发射后发送
    SHOOT_DATA = 0x0207
    # 允许发弹量，固定以10Hz频率发送
    BULLET_ALLOWED = 0x0208
    # 机器人RFID模块状态，固定以3Hz频率发送
    RFID_STATUS = 0x0209

    # 飞镖选手端指令数据，固定以3Hz频率发送
    LAUNCHER_CMD = 0x020A
    # 地面机器人位置数据，固定以1Hz频率发送
    SENTRY_POS_DATA = 0x020B
    # 雷达标记进度数据，固定以1Hz频率发送
    RADAR_MARK_PROGRESS = 0x020C
    # 哨兵自主决策信息同步，固定以1Hz频率发送
    SENTRY_DECISION_SYNC = 0x020D

    # 雷达自主决策信息同步，固定以1Hz频率发送
    RADAR_DECISION_SYNC = 0x020E

    # 机器人交互数据，发送方触发发送，频率上限为30Hz
    INTERACTIVE_DATA = 0x0301
    # 自定义控制器与机器人交互数据，发送方触发发送，频率上限为30Hz
    CLIENT_CUSTOMIZED_CONTROLLER_DATA = 0x0302

    # 选手端小地图交互数据，选手端触发发送
    CLIENT_INTERACTIVE_DATA = 0x0303
    # 键鼠遥控数据，固定30Hz频率发送
    KEYBAORD_MOUSE_DATA = 0x0304

    # 选手端小地图接收雷达数据，频率上限为5Hz
    CLIENT_RADAR_DATA = 0x0305
    # 自定义控制器与选手端交互数据，发送方触发发送，频率上限为30Hz
    CUSTOMIZED_CONTROLLER_DATA = 0x0306

    # 选手端小地图接收路径数据，频率上限为1Hz
    CLIENT_PATH_DATA = 0x0307
    # 选手端小地图接收机器人数据，频率上限为3Hz
    CLIENT_ROBOT_DATA = 0x0308
    # 自定义控制器接收机器人数据，频率上限为10Hz
    CUSTOMIZED_CONTROLLER_ROBOT_DATA = 0x0309


class SubCmdID(Enum):
    DELETE_LAYER = 0x0100
    ADD_1_PATTERN = 0x0101
    ADD_2_PATTERN = 0x0102
    ADD_5_PATTERN = 0x0103
    ADD_7_PATTERN = 0x0104
    ADD_TEXT = 0x0110
    SENTRY_DECISION = 0x0120
    RADAR_DECISION = 0x0121

    SENTRY_SELF_POSITION = 0x02BB
    RADAR_2_SENTRY = 0x0233
    RADAR_2_DART = 0x0234
    INFO_WAVE = 0x02AA
    INFO_WAVE_ROBOT_MULTICAST = 0x02AB
    



class OBJECT_ID(Enum):
    R_HERO = 1
    R_ENGINEER = 2
    R_INFANTRY_3 = 3
    R_INFANTRY_4 = 4
    R_INFANTRY_5 = 5
    R_DRONE = 6
    R_SENTRY = 7
    R_LAUNCHER = 8
    R_RADAR = 9
    R_OUTPOST = 10
    R_BASE = 11

    B_HERO = 101
    B_ENGINEER = 102
    B_INFANTRY_3 = 103
    B_INFANTRY_4 = 104
    B_INFANTRY_5 = 105
    B_DRONE = 106
    B_SENTRY = 107
    B_LAUNCHER = 108
    B_RADAR = 109
    B_OUTPOST = 110
    B_BASE = 111

    R_HERO_CLIENT = 0x0101
    R_ENGINEER_CLIENT = 0x0102
    R_INFANTRY_3_CLIENT = 0x0103
    R_INFANTRY_4_CLIENT = 0x0104
    R_INFANTRY_5_CLIENT = 0x0105
    R_DRONE_CLIENT = 0x0106

    B_HERO_CLIENT = 0x0165
    B_ENGINEER_CLIENT = 0x0166
    B_INFANTRY_3_CLIENT = 0x0167
    B_INFANTRY_4_CLIENT = 0x0168
    B_INFANTRY_5_CLIENT = 0x0169
    B_DRONE_CLIENT = 0x016A

    SERVER = 0x8080


class BaseMsg(ABC):
    def __init__(self):
        self.packed_buffer = b""

    @abstractmethod
    def pack(self):
        """Pack the message into the binary format"""
        return self.packed_buffer

    def get_packed_buffer(self):
        return self.packed_buffer


class RefereeGenericMessage(BaseMsg):
    """
    RoboMaster protocol message format:
    Frame structure: frame_header(5-byte) + cmd_id(2-byte) + data(n-byte) + frame_tail(2-byte CRC16)
    Frame header: SOF(1-byte) + data_length(2-byte) + seq(1-byte) + CRC8(1-byte)
    """

    format_string = "BBBH B s H"
    SOF = 0xA5
    _seq_counter = 0
    _seq_lock = threading.Lock()

    def __init__(self, command_id: np.uint16, *data_fields):
        super().__init__()
        assert isinstance(command_id, np.uint16), "command_id must be np.uint16"
        self.command_id = command_id
        self.seq: int = 0
        self.data_fields = data_fields

    @classmethod
    def _next_seq(cls):
        # seq 属于整条裁判系统串口链路，而不是某一种消息类型。
        # 必须显式操作基类计数器；若使用 cls，Python 会在不同子类上分别
        # 创建计数器，造成 0x0305 与各类 0x0301 交错发送时序号重复或倒退。
        with RefereeGenericMessage._seq_lock:
            seq = RefereeGenericMessage._seq_counter & 0xFF
            RefereeGenericMessage._seq_counter = (seq + 1) & 0xFF
            return seq

    def pack(self):
        # 1. 准备 data 区，不包含 cmd_id。
        packed_data = b"".join(
            self._pack_one_field(field) for field in self.data_fields
        )

        # 2. 官方 data_length 只表示 data 区长度，不包含 2 字节 cmd_id。
        data_length = len(packed_data)
        self.seq = self._next_seq()

        # 3. 官方帧头：SOF(1B) + data_length(2B, LE) + seq(1B) + CRC8(1B)。
        header_without_crc = struct.pack("<BHB", self.SOF, data_length, self.seq)

        # 4. Calculate CRC8 for header
        crc8_header = Crc.get_crc8_check_sum(header_without_crc)

        # 5. Complete frame header
        frame_header = header_without_crc + struct.pack("B", crc8_header)

        # 6. Pack cmd_id
        cmd_id_packed = struct.pack("<H", int(self.command_id))

        # 7. Combine header + cmd_id + data
        message_without_tail = frame_header + cmd_id_packed + packed_data

        # 8. Calculate CRC16 for entire message
        crc16_tail = Crc.get_crc16_check_sum(message_without_tail)

        # 9. Pack final message
        self.packed_buffer = message_without_tail + struct.pack("<H", crc16_tail)

        return self.packed_buffer

    def _pack_one_field(self, data):
        """
        Pack data fields based on their type.
        This method can be extended to handle more types as needed. Numpy types are recommended.
        """
        if isinstance(data, bytes):
            return data
        elif isinstance(data, np.uint8):  # uint8
            return struct.pack("B", int(data))
        elif isinstance(data, np.uint16):  # uint16
            return struct.pack("<H", int(data))
        elif isinstance(data, np.uint32):  # uint32
            return struct.pack("<I", int(data))
        elif isinstance(data, np.int8):  # int8
            return struct.pack("b", int(data))
        elif isinstance(data, np.int16):  # int16
            return struct.pack("<h", int(data))
        elif isinstance(data, np.int32):  # int32
            return struct.pack("<i", int(data))
        elif isinstance(data, np.float32):  # float32
            return struct.pack("<f", float(data))
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Supported types are: uint8, uint16, uint32, int8, int16, int32, float32."
            )


class InteractiveMessage(RefereeGenericMessage):
    """
    Data field -> sub_cmd_id(2-byte) + sender_id(2-byte) + receiver_id(2-byte) + data(n-byte)
    Notice that n < 112 bytes
    """

    def __init__(
        self,
        sub_cmd_id: np.uint16,
        sender_id: np.uint16,
        receiver_id: np.uint16,
        *data_fields,
    ):
        assert isinstance(sub_cmd_id, np.uint16), "sub_cmd_id must be np.uint16"
        assert isinstance(sender_id, np.uint16), "sender_id must be np.uint16"
        assert isinstance(receiver_id, np.uint16), "receiver_id must be np.uint16"
        user_data_len = sum(
            len(field) if isinstance(field, bytes) else len(RefereeGenericMessage(np.uint16(0), field)._pack_one_field(field))
            for field in data_fields
        )
        if user_data_len > 112:
            raise ValueError(f"0x0301 user_data length must be <= 112 bytes, got {user_data_len}")

        super().__init__(
            np.uint16(MsgID.INTERACTIVE_DATA.value),
            np.uint16(sub_cmd_id),
            np.uint16(sender_id),
            np.uint16(receiver_id),
            *data_fields,
        )
        self.sub_cmd_id = sub_cmd_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id


class RadarSelfDecisionCmd(InteractiveMessage):

    def __init__(self, radar_cmd: np.uint8, is_blue: bool = True):
        assert isinstance(radar_cmd, np.uint8), "radar_cmd must be np.uint8"
        super().__init__(
            np.uint16(SubCmdID.RADAR_DECISION.value),
            np.uint16(OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value),
            np.uint16(OBJECT_ID.SERVER.value),
            np.uint8(radar_cmd),
        )


class InterRobotMessage(InteractiveMessage):
    """
    Inter-robot communication message format.
    This class can be used to send messages to different robot, such as sentry or hero
    """

    def __init__(
        self, sub_cmd_id: np.uint16, receiver_id: np.uint16, is_blue=True, *data_fields
    ):
        assert isinstance(sub_cmd_id, np.uint16), "sub_cmd_id must be np.uint16"
        assert isinstance(receiver_id, np.uint16), "receiver_id must be np.uint16"

        super().__init__(
            sub_cmd_id,
            np.uint16(
                OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
            ),
            receiver_id,
            *data_fields,
        )
        self.sub_cmd_id = sub_cmd_id
        self.receiver_id = receiver_id
class GameStatusData(ctypes.Structure):
    """
    Game status data structure.
    This structure is used to receive game status data from referee system.
    """

    _pack_ = 1
    _fields_ = [
        ("game_type", ctypes.c_uint8, 4),  # bit 0-3: game type
        ("game_progress", ctypes.c_uint8, 4),  # bit 4-7: game progress
        ("remaining_time", ctypes.c_uint16),  # 2 bytes: Remaining time in seconds
        ("UNIX_timestamp", ctypes.c_uint64),  # 8 bytes: UNIX timestamp of the data generation time
    ]

class SentrySelfPositionData(ctypes.Structure):
    """
    哨兵发给雷达的己方位置缓存。
    0x0301 / 0x02BB user_data：
    sentry(x,y), hero(x,y), engineer(x,y), infantry3(x,y), infantry4(x,y)，单位 m。
    """

    _pack_ = 1
    _fields_ = [
        ("sentry_x", ctypes.c_float),
        ("sentry_y", ctypes.c_float),
        ("hero_x", ctypes.c_float),
        ("hero_y", ctypes.c_float),
        ("engineer_x", ctypes.c_float),
        ("engineer_y", ctypes.c_float),
        ("standard_3_x", ctypes.c_float),
        ("standard_3_y", ctypes.c_float),
        ("standard_4_x", ctypes.c_float),
        ("standard_4_y", ctypes.c_float),
    ]


class Radar2SentryData(ctypes.Structure):
    """雷达发给哨兵的状态字节：bit0~4 可瞄准，bit5 区域建议，bit6~7 保留。"""

    _pack_ = 1
    _fields_ = [
        ("targetable_flags", ctypes.c_uint8),
    ]

class Radar2DartData(ctypes.Structure):
    """
    Radar to Dart data structure.
    This structure is used to send double vulnerability state (RadarTriggerState) to Dart
    """
    _pack_ = 1
    _fields_ = [
        ("trigger_state", ctypes.c_uint8),  # RadarTriggerState value
        ("double_vulnerability_count", ctypes.c_uint8),  # 剩余次数：0/1/2
        ("autoaim_stage", ctypes.c_uint8),  # 自瞄空中机器人反制阶段：0x00~0x06
    ]

class DartStatData(ctypes.LittleEndianStructure):
    """
    飞镖发射相关数据，官方 data 区长度为 3 字节。
    """

    _pack_ = 1
    _fields_ = [
        ("dart_remaining_time", ctypes.c_uint8),  # 飞镖发射剩余时间
        ("recent_hit_target", ctypes.c_uint16, 3),  # 最近击中的目标
        ("accumulated_hit_count", ctypes.c_uint16, 3),  # 累计命中次数
        ("selected_target", ctypes.c_uint16, 3),  # 当前选定目标
        ("reserve", ctypes.c_uint16, 7),  # 保留位
    ]


class RadarMarkProgressData(ctypes.LittleEndianStructure):
    """
    2026 雷达标记及空中机器人反制状态，官方 data 区长度为 2 字节。
    """

    _pack_ = 1
    _fields_ = [
        ("enemy_hero", ctypes.c_uint16, 1),  # bit0：对方英雄易伤
        ("enemy_engineer", ctypes.c_uint16, 1),  # bit1：对方工程易伤
        ("enemy_standard_3", ctypes.c_uint16, 1),  # bit2：对方3号步兵易伤
        ("enemy_standard_4", ctypes.c_uint16, 1),  # bit3：对方4号步兵易伤
        ("enemy_drone", ctypes.c_uint16, 1),  # bit4：对方空中特殊标识
        ("enemy_sentry", ctypes.c_uint16, 1),  # bit5：对方哨兵易伤
        ("self_hero", ctypes.c_uint16, 1),  # bit6：己方英雄特殊标识
        ("self_engineer", ctypes.c_uint16, 1),  # bit7：己方工程特殊标识
        ("self_standard_3", ctypes.c_uint16, 1),  # bit8：己方3号步兵特殊标识
        ("self_standard_4", ctypes.c_uint16, 1),  # bit9：己方4号步兵特殊标识
        ("self_drone", ctypes.c_uint16, 1),  # bit10：己方空中特殊标识
        ("self_sentry", ctypes.c_uint16, 1),  # bit11：己方哨兵特殊标识
        ("enemy_drone_aimed", ctypes.c_uint16, 1),  # bit12：对方空中被己方激光瞄准
        ("enemy_drone_counteracted", ctypes.c_uint16, 1),  # bit13：对方空中处于被反制
        ("self_drone_aimed", ctypes.c_uint16, 1),  # bit14：己方空中被对方激光瞄准
        ("self_drone_counteracted", ctypes.c_uint16, 1),  # bit15：己方空中处于被反制
    ]


class RadarInfoData(ctypes.Structure):
    """
    Radar Info data structure
    """

    _pack_ = 1
    _fields_ = [
        (
            "double_vulnerability_count",
            ctypes.c_uint8,
            2,
        ),  # Double vulnerability status (1 bit)
        (
            "is_double_vulnerability",
            ctypes.c_uint8,
            1,
        ),  # Is double vulnerability (1 bit)
        (
            "own_encrypt_level",
            ctypes.c_uint8,
            2,
        ),
        (
            "is_key_changeable",
            ctypes.c_uint8,
            1,
        ),
        ("reserve", ctypes.c_uint8, 2),  # Reserved bits for future use
    ]


class RobotStatusData(ctypes.Structure):
    """
    Robot status data structure. This status is to retrieve the faction of the robot.
    """

    _pack_ = 1
    _fields_ = [
        ("robot_id", ctypes.c_uint8),  # Robot ID (1 byte)
        ("robot_level", ctypes.c_uint8),  # Robot level (1 byte)
        ("current_hp", ctypes.c_uint16),  # Current HP (2 byte)
        ("max_hp", ctypes.c_uint16),  # Max HP (2 byte)
        (
            "shooter_barrel_cooling_value",
            ctypes.c_uint16,
        ),  # Shooter barrel cooling value (2 byte)
        (
            "shooter_barrel_heat_limit",
            ctypes.c_uint16,
        ),  # Shooter barrel heat limit (2 byte)
        ("chassis_power_limit", ctypes.c_uint16),  # Chassis power limit (1 bit)
        ("power_management_gimbal_output", ctypes.c_uint8, 1),
        (
            "power_management_chassis_output",
            ctypes.c_uint8,
            1,
        ),  # Power management chassis output (1 bit)
        ("power_management_shooter_output", ctypes.c_uint8, 1),  # Power
        ("reserve", ctypes.c_uint8, 5),  # Reserved bits for future use
    ]


class Radar2ClientData(ctypes.Structure):
    """
    Radar to client data structure
    """

    _pack_ = 1
    _fields_ = [
        ("enemy_hero_x", ctypes.c_uint16),  # Enemy Hero X position
        ("enemy_hero_y", ctypes.c_uint16),  # Enemy Hero Y position
        ("enemy_engineer_x", ctypes.c_uint16),  # Enemy Engineer X position
        ("enemy_engineer_y", ctypes.c_uint16),  # Enemy Engineer Y position
        ("enemy_standard_3_x", ctypes.c_uint16),  # Enemy Standard 3 X position
        ("enemy_standard_3_y", ctypes.c_uint16),  # Enemy Standard 3 Y position
        ("enemy_standard_4_x", ctypes.c_uint16),  # Enemy Standard 4 X position
        ("enemy_standard_4_y", ctypes.c_uint16),  # Enemy Standard 4 Y position
        ("enemy_drone_x", ctypes.c_uint16),  # Enemy Drone X position
        ("enemy_drone_y", ctypes.c_uint16),  # Enemy Drone Y position
        ("enemy_sentry_x", ctypes.c_uint16),  # Enemy Sentry X position
        ("enemy_sentry_y", ctypes.c_uint16),  # Enemy Sentry Y position

        ("self_hero_x", ctypes.c_uint16),  # Self Hero X position
        ("self_hero_y", ctypes.c_uint16),  # Self Hero Y position
        ("self_engineer_x", ctypes.c_uint16),  # Self Engineer X position
        ("self_engineer_y", ctypes.c_uint16),  # Self Engineer Y position
        ("self_standard_3_x", ctypes.c_uint16),  # Self Standard 3 X position
        ("self_standard_3_y", ctypes.c_uint16),  # Self Standard 3 Y position
        ("self_standard_4_x", ctypes.c_uint16),  # Self Standard 4 X position
        ("self_standard_4_y", ctypes.c_uint16),  # Self Standard 4 Y position
        ("self_drone_x", ctypes.c_uint16),  # Self Drone X position
        ("self_drone_y", ctypes.c_uint16),  # Self Drone Y position
        ("self_sentry_x", ctypes.c_uint16),  # Self Sentry X position
        ("self_sentry_y", ctypes.c_uint16),  # Self Sentry Y position
    ]


class RadarDecisionData(ctypes.Structure):
    """
    Radar double vulnerability decision data structure
    """

    _pack_ = 1
    _fields_ = [
        ("radar_cmd", ctypes.c_uint8),  # Radar command (1 byte)
        ("key", ctypes.c_uint8 * 7),  # Key for future use (7 bytes)
    ]


class StructureMessage(BaseMsg):
    STRUCT_CLASS = None

    def __init__(self, msg_id, **kwargs):
        if self.STRUCT_CLASS is None:
            raise NotImplementedError("struct class must be defined in subclass")

        super().__init__()
        self.struct_data = self.STRUCT_CLASS()
        self.msg_id = np.uint16(msg_id)

        for field, value in kwargs.items():
            if hasattr(self.struct_data, field):
                setattr(self.struct_data, field, value)
            else:
                raise ValueError(
                    f"Field {field} not found in {self.STRUCT_CLASS.__name__}"
                )

    def pack(self):
        msg = RefereeGenericMessage(self.msg_id, bytes(self.struct_data))
        msg.pack()
        self.packed_buffer = msg.get_packed_buffer()
        return self.packed_buffer

    def __getattr__(self, item):
        if hasattr(self.struct_data, item):
            return getattr(self.struct_data, item)
        raise AttributeError(f"{item} not found in {self.STRUCT_CLASS.__name__}")

    def __setattr__(self, name, value):
        if name in ["struct_data", "msg_id", "packed_buffer"]:
            super().__setattr__(name, value)
        elif hasattr(self, "struct_data") and hasattr(self.struct_data, name):
            setattr(self.struct_data, name, value)
        else:
            super().__setattr__(name, value)

    def __str__(self):
        text = ""
        if hasattr(self, "struct_data"):
            for field in self.STRUCT_CLASS._fields_:
                name = field[0]
                value = getattr(self.struct_data, name)
                text += f"{field}: {value}\n"
        return text

    @classmethod
    def from_bytes(cls, data: bytes):
        if cls.STRUCT_CLASS is None:
            raise NotImplementedError("struct class must be defined in subclass")

        if len(data) < ctypes.sizeof(cls.STRUCT_CLASS):
            raise ValueError(
                f"Data length {len(data)} is less than struct size {ctypes.sizeof(cls.STRUCT_CLASS)}"
            )

        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.struct_data = cls.STRUCT_CLASS.from_buffer_copy(data)
        instance.msg_id = cls._get_msg_id()
        return instance

    @classmethod
    def _get_msg_id(cls):
        """子类可以重写此方法返回对应的消息ID"""
        raise NotImplementedError("子类必须实现 _get_msg_id 方法")


class InteractiveStructMessage(BaseMsg):
    STRUCT_CLASS = None  # 子类需要定义
    SUB_CMD_ID = None  # 子类需要定义

    def __init__(self, sender_id, receiver_id, **kwargs):
        if self.STRUCT_CLASS is None or self.SUB_CMD_ID is None:
            raise NotImplementedError("子类必须定义 STRUCT_CLASS 和 SUB_CMD_ID")

        super().__init__()
        self.sub_cmd_id = self.SUB_CMD_ID
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.data = self.STRUCT_CLASS()

        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)

    def pack(self):
        msg = InteractiveMessage(
            np.uint16(self.sub_cmd_id),
            np.uint16(self.sender_id),
            np.uint16(self.receiver_id),
            bytes(self.data),
        )

        return msg.pack()

    def __getattr__(self, name):
        if hasattr(self.data, name):
            return getattr(self.data, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in ["sub_cmd_id", "sender_id", "receiver_id", "data", "packed_buffer"]:
            super().__setattr__(name, value)
        elif hasattr(self, "data") and hasattr(self.data, name):
            setattr(self.data, name, value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_bytes(cls, data_bytes: bytes):
        """从 0x0301 data 区创建消息，格式为 sub_cmd_id + sender_id + receiver_id + user_data。"""
        if cls.STRUCT_CLASS is None:
            raise NotImplementedError("子类必须定义 STRUCT_CLASS")

        header_size = 6
        struct_size = ctypes.sizeof(cls.STRUCT_CLASS)
        if len(data_bytes) < header_size + struct_size:
            raise ValueError(
                f"Data length {len(data_bytes)} is less than interactive header {header_size} + struct size {struct_size}"
            )

        # 创建实例但不调用__init__
        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.sub_cmd_id = cls.SUB_CMD_ID
        instance.sender_id = struct.unpack("<H", data_bytes[2:4])[0]
        instance.receiver_id = struct.unpack("<H", data_bytes[4:6])[0]
        instance.data = cls.STRUCT_CLASS.from_buffer_copy(data_bytes[header_size:])
        return instance

    def __str__(self):
        text = ""
        if hasattr(self, "data"):
            for field in self.STRUCT_CLASS._fields_:
                name = field[0]
                value = getattr(self.data, name)
                text += f"{field}: {value}\n"
        return text


##########
class UnknownMessage(BaseMsg):
    """未定义消息的占位对象，用于调试界面展示原始载荷。"""

    def __init__(
        self,
        cmd_id,
        raw_data: bytes,
        sub_cmd_id=None,
        sender_id=None,
        receiver_id=None,
        message_name: str = "UnknownMessage",
    ):
        super().__init__()
        self.cmd_id = int(cmd_id)
        self.raw_data = bytes(raw_data)
        self.sub_cmd_id = None if sub_cmd_id is None else int(sub_cmd_id)
        self.sender_id = None if sender_id is None else int(sender_id)
        self.receiver_id = None if receiver_id is None else int(receiver_id)
        self.message_name = message_name

    def pack(self):
        self.packed_buffer = self.raw_data
        return self.packed_buffer

    def __str__(self):
        return (
            f"cmd_id=0x{self.cmd_id:04x}, "
            f"sub_cmd_id={self.sub_cmd_id}, "
            f"sender_id={self.sender_id}, "
            f"receiver_id={self.receiver_id}, "
            f"raw_data={self.raw_data.hex()}"
        )


RawMessage = UnknownMessage
##########


# 具体的消息类
class GameStatusMessage(StructureMessage):
    STRUCT_CLASS = GameStatusData

    def __init__(self, **kwargs):
        super().__init__(MsgID.GAME_STATUS.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.GAME_STATUS.value)

class RobotStatusMessage(StructureMessage):
    STRUCT_CLASS = RobotStatusData

    def __init__(self, **kwargs):
        super().__init__(MsgID.ROBOT_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.ROBOT_DATA.value)


class DartStatusMessage(StructureMessage):
    STRUCT_CLASS = DartStatData

    def __init__(self, **kwargs):
        super().__init__(MsgID.LAUNCHER_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.LAUNCHER_DATA.value)


class RadarMarkMessage(StructureMessage):
    STRUCT_CLASS = RadarMarkProgressData

    def __init__(self, **kwargs):
        super().__init__(MsgID.RADAR_MARK_PROGRESS.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.RADAR_MARK_PROGRESS.value)


class RadarInfoMessage(StructureMessage):
    STRUCT_CLASS = RadarInfoData

    def __init__(self, **kwargs):
        super().__init__(MsgID.RADAR_DECISION_SYNC.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.RADAR_DECISION_SYNC.value)


class Radar2ClientMessage(StructureMessage):
    STRUCT_CLASS = Radar2ClientData

    def __init__(self, **kwargs):
        super().__init__(MsgID.CLIENT_RADAR_DATA.value, **kwargs)

    @classmethod
    def _get_msg_id(cls):
        return np.uint16(MsgID.CLIENT_RADAR_DATA.value)

class SentrySelfPositionMessage(InteractiveStructMessage):
    STRUCT_CLASS = SentrySelfPositionData
    SUB_CMD_ID = SubCmdID.SENTRY_SELF_POSITION.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_SENTRY.value if is_blue else OBJECT_ID.R_SENTRY.value
        receiver_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        super().__init__(sender_id, receiver_id, **kwargs)


class Radar2SentryMessage(InteractiveStructMessage):
    STRUCT_CLASS = Radar2SentryData
    SUB_CMD_ID = SubCmdID.RADAR_2_SENTRY.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        receiver_id = OBJECT_ID.B_SENTRY.value if is_blue else OBJECT_ID.R_SENTRY.value
        super().__init__(sender_id, receiver_id, **kwargs)

class Radar2DartMessage(InteractiveStructMessage):
    # receiver_id is waited to be tested, currently set to drone
    STRUCT_CLASS = Radar2DartData
    SUB_CMD_ID = SubCmdID.RADAR_2_DART.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        receiver_id = OBJECT_ID.B_DRONE.value if is_blue else OBJECT_ID.R_DRONE.value
        super().__init__(sender_id, receiver_id, **kwargs)

class RadarDecisionMessage(InteractiveStructMessage):
    STRUCT_CLASS = RadarDecisionData
    SUB_CMD_ID = SubCmdID.RADAR_DECISION.value

    def __init__(self, is_blue=True, **kwargs):
        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        receiver_id = OBJECT_ID.SERVER.value
        super().__init__(sender_id, receiver_id, **kwargs)


class InfoWaveMessage(InteractiveMessage):
    """向工程转发空中机器人弹量和对方剩余金币。

    0x0301 交互数据：
    - data_cmd_id: 0x02AA
    - sender_id: 己方雷达 ID，红方 9 / 蓝方 109
    - receiver_id: 己方工程机器人 ID，红方 2 / 蓝方 102
    - user_data byte0~1: 0x0A03 偏移 6 的对方空中机器人允许发弹量
    - user_data byte2~3: 0x0A04 偏移 0 的对方剩余金币数
    """

    SUB_CMD_ID = SubCmdID.INFO_WAVE.value
    USER_DATA_SIZE = 4

    def __init__(self, is_blue=True, receiver_id=None, user_data=None):
        if receiver_id is None:
            raise ValueError("receiver_id is required")
        if user_data is None:
            user_data = bytes(self.USER_DATA_SIZE)
        user_data = bytes(user_data)
        if len(user_data) != self.USER_DATA_SIZE:
            raise ValueError(
                f"InfoWaveMessage user_data must be {self.USER_DATA_SIZE} bytes, got {len(user_data)}"
            )

        sender_id = OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        super().__init__(
            np.uint16(self.SUB_CMD_ID),
            np.uint16(sender_id),
            np.uint16(receiver_id),
            user_data,
        )
        self.user_data = user_data

    @classmethod
    def from_bytes(cls, data_bytes: bytes):
        if len(data_bytes) < 6:
            raise ValueError("InfoWaveMessage data is shorter than interactive header")

        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.sub_cmd_id = cls.SUB_CMD_ID
        instance.sender_id = struct.unpack("<H", data_bytes[2:4])[0]
        instance.receiver_id = struct.unpack("<H", data_bytes[4:6])[0]
        instance.user_data = bytes(data_bytes[6:])
        return instance

    def __str__(self):
        return (
            f"sub_cmd_id=0x{self.sub_cmd_id:04x}, "
            f"sender_id={int(self.sender_id)}, "
            f"receiver_id={int(self.receiver_id)}, "
            f"user_data={self.user_data.hex()}"
        )


class InfoWaveRobotMulticastMessage(InteractiveMessage):
    """向己方英雄、3号和4号发送对方五类机器人的信息波摘要。"""

    SUB_CMD_ID = SubCmdID.INFO_WAVE_ROBOT_MULTICAST.value
    USER_DATA_SIZE = 30

    def __init__(self, is_blue=True, receiver_id=None, user_data=None):
        if receiver_id is None:
            raise ValueError("receiver_id is required")
        if user_data is None:
            user_data = bytes(self.USER_DATA_SIZE)
        user_data = bytes(user_data)
        if len(user_data) != self.USER_DATA_SIZE:
            raise ValueError(
                "InfoWaveRobotMulticastMessage user_data must be "
                f"{self.USER_DATA_SIZE} bytes, got {len(user_data)}"
            )

        sender_id = (
            OBJECT_ID.B_RADAR.value if is_blue else OBJECT_ID.R_RADAR.value
        )
        super().__init__(
            np.uint16(self.SUB_CMD_ID),
            np.uint16(sender_id),
            np.uint16(receiver_id),
            user_data,
        )
        self.user_data = user_data

    @classmethod
    def from_bytes(cls, data_bytes: bytes):
        if len(data_bytes) < 6:
            raise ValueError(
                "InfoWaveRobotMulticastMessage data is shorter than interactive header"
            )

        instance = cls.__new__(cls)
        BaseMsg.__init__(instance)
        instance.sub_cmd_id = cls.SUB_CMD_ID
        instance.sender_id = struct.unpack("<H", data_bytes[2:4])[0]
        instance.receiver_id = struct.unpack("<H", data_bytes[4:6])[0]
        instance.user_data = bytes(data_bytes[6:])
        return instance


if __name__ == "__main__":
    sentry_msg = SentrySelfPositionMessage(
        is_blue=True,
        sentry_x=9.0,
        sentry_y=10.0,
        hero_x=1.0,
        hero_y=2.0,
        engineer_x=3.0,
        engineer_y=4.0,
        standard_3_x=5.0,
        standard_3_y=6.0,
        standard_4_x=7.0,
        standard_4_y=8.0,
    )
    print(f"Hero position: ({sentry_msg.hero_x}, {sentry_msg.hero_y})")
    packed_data = sentry_msg.pack()
    print(f"Packed data length: {len(packed_data)} bytes")

    print(sentry_msg.sender_id)
    print(sentry_msg.receiver_id)
    print(sentry_msg.hero_x)
    print(sentry_msg.hero_y)
    print(sentry_msg.engineer_x)
    print(sentry_msg.engineer_y)
    print(sentry_msg.standard_3_x)
    print(sentry_msg.standard_3_y)
    print(sentry_msg.standard_4_x)
    print(sentry_msg.standard_4_y)
    print(sentry_msg.sentry_x)
    print(sentry_msg.sentry_y)

    # print(
    #     f"Pack data hex: {sentry_msg.pack()[7:].hex()}, length: {len(sentry_msg.pack())} bytes"
    # )

    sentry_msg_rx = SentrySelfPositionMessage.from_bytes(packed_data[7:])
    print(sentry_msg_rx.sender_id)
    print(sentry_msg_rx.receiver_id)
    print(sentry_msg_rx.hero_x)
    print(sentry_msg_rx.hero_y)
    print(sentry_msg_rx.engineer_x)
    print(sentry_msg_rx.engineer_y)
    print(sentry_msg_rx.standard_3_x)
    print(sentry_msg_rx.standard_3_y)
    print(sentry_msg_rx.standard_4_x)
    print(sentry_msg_rx.standard_4_y)
    print(sentry_msg_rx.sentry_x)
    print(sentry_msg_rx.sentry_y)
    # print(
    #     f"Pack data rx hex: {sentry_msg_rx.pack()[7:].hex()}, length: {len(sentry_msg_rx.pack())} bytes"
    # )
    sentry_msg_rx.pack()

    robot_status_msg = RobotStatusMessage(
        robot_id=OBJECT_ID.R_HERO.value,
        robot_level=1,
        current_hp=1000,
        max_hp=2000,
        shooter_barrel_cooling_value=500,
        shooter_barrel_heat_limit=1000,
        chassis_power_limit=1500,
        power_management_gimbal_output=1,
        power_management_chassis_output=1,
        power_management_shooter_output=1,
        reserve=0,
    )
    packed_data = robot_status_msg.pack()
    print(
        f"Robot Status Packed data length: {len(packed_data)} bytes, data: {packed_data}"
    )

    robot_status_msg_rx = RobotStatusMessage.from_bytes(packed_data[7:])
    print(robot_status_msg_rx.struct_data.robot_id)  # 打印机器人ID
    print(robot_status_msg_rx.struct_data.robot_level)  # 打印机器人等级
    print(robot_status_msg_rx.struct_data.current_hp)  # 打印当前HP
    print(robot_status_msg_rx.struct_data.max_hp)  # 打印最大HP
    print(
        robot_status_msg_rx.struct_data.shooter_barrel_cooling_value
    )  # 打印射手炮管冷却值
    print(robot_status_msg_rx.struct_data.shooter_barrel_heat_limit)  # 打印

    packed_data = robot_status_msg_rx.pack()
    print(
        f"Robot Status RX Packed data length: {len(packed_data)} bytes, data: {packed_data}"
    )
