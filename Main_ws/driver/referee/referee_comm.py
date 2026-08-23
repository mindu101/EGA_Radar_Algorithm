# referee_comm.py
# 管理与裁判系统的通信
# 1. 管理消息发送的种类以及发送频率
# 2. 自动接收裁判系统的消息
# 3. 管理裁判系统与哨兵的连接状态
# -*- coding: utf-8 -*-
from .serial_comm import RefereeSerialManager
from .serial_protocol import *
from .tx_policy import RefereeTxPolicy
from dataclasses import asdict, dataclass, field
# import ctypes
import os
import socket
import threading
import time
import math
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from utils.runtime_config import get_params_config_path
########################################################################
import ctypes
from rclpy.executors import SingleThreadedExecutor
from .udp_protocol import *
from .udp import UdpSender, UdpSenderConfig
from .web_test_override import DEFAULT_PORT as WEB_LATE_DEFAULT_PORT
from .web_test_override import MAX_PACKET_SIZE as WEB_LATE_MAX_PACKET_SIZE
from .web_test_override import decode_snapshot as decode_web_late_snapshot
########################################################################


class FACTION(Enum):
    RED = 0
    BLUE = 1
    UNKONWN = 2


@dataclass(frozen=True)
class AllyPositionSnapshot:
    positions_m: Mapping[str, tuple[float, float] | None]
    received_monotonic: float | None


class RadarTriggerState(Enum):
    IDLE = 0
    ARMED = 1
    REQUESTING = 2
    ACTIVE = 3
    COOLDOWN = 4
    # TRIGGERING = 5

REQUEST_TIMEOUT = 3  # seconds


class RefereeCommManager(RefereeSerialManager):
    _instance = None
    ACTIVE_GAME_PROGRESS = 4
    FIRST_DOUBLE_VULNERABILITY_TRIGGER_TIME_SEC = 350
    RADAR2CLIENT_TX_HZ = RefereeTxPolicy.RADAR2CLIENT_TX_HZ
    INTERACTIVE_ROUND_HZ = RefereeTxPolicy.INTERACTIVE_ROUND_HZ
    INTERACTIVE_ROUND_GAP_SEC = RefereeTxPolicy.INTERACTIVE_ROUND_GAP_SEC
    INTERACTIVE_MIN_GAP_SEC = RefereeTxPolicy.INTERACTIVE_MIN_GAP_SEC
    INFO_WAVE_ENGINEER_TX_HZ = RefereeTxPolicy.INFO_WAVE_ENGINEER_TX_HZ
    INFO_WAVE_SENTRY_TX_HZ = RefereeTxPolicy.INFO_WAVE_SENTRY_TX_HZ
    OPTIONAL_MULTICAST_SLOT_WEIGHTS = RefereeTxPolicy.OPTIONAL_MULTICAST_SLOT_WEIGHTS
    OPTIONAL_MULTICAST_SLOT_GAP_SEC = (
        RefereeTxPolicy.OPTIONAL_MULTICAST_SLOT_GAP_SEC
    )
    KEY_IDLE_CMD = 0
    KEY_VERIFY_CMD = 2
    INFO_WAVE_RECEIVERS_RED = (OBJECT_ID.R_ENGINEER.value,)
    INFO_WAVE_RECEIVERS_BLUE = (OBJECT_ID.B_ENGINEER.value,)
    INFO_WAVE_ROBOT_MULTICAST_RECEIVERS_RED = (
        OBJECT_ID.R_HERO.value,
        OBJECT_ID.R_INFANTRY_3.value,
        OBJECT_ID.R_INFANTRY_4.value,
    )
    INFO_WAVE_ROBOT_MULTICAST_RECEIVERS_BLUE = (
        OBJECT_ID.B_HERO.value,
        OBJECT_ID.B_INFANTRY_3.value,
        OBJECT_ID.B_INFANTRY_4.value,
    )
    # 英雄、3号、4号的摘要接收份额；英雄低频，两个步兵同频。
    INFO_WAVE_ROBOT_MULTICAST_RECEIVER_WEIGHTS = (1, 2, 2)
    INFO_WAVE_CMD_IDS = InfoMsgBag.CMD_IDS
    INFO_WAVE_ENGINEER_CMD_IDS = (0x0A03, 0x0A04)
    INFO_WAVE_ROBOT_MULTICAST_CMD_IDS = (0x0A02, 0x0A03, 0x0A05)
    INFO_WAVE_ROBOT_HEALTH_OFFSETS = (0, 2, 4, 6, 10)
    INFO_WAVE_POSITION_CMD_ID = 0x0A01
    INFO_WAVE_FRESH_TIMEOUT_SEC = 3.0
    FIELD_LENGTH_M = 28.0
    FIELD_WIDTH_M = 15.0
    # 长洞覆盖范围略大于短洞；保留旧常量供现有界面绘制逻辑兼容使用。
    SENTRY_HOLE_RADIUS_M = 3.0
    SENTRY_HOLE_RADII_M = {"long_hole": 4.2, "short_hole": 3.0}
    SENTRY_HOLE_SCORE_BASE = 2.0
    SENTRY_HOLE_SCORE_DECAY_PER_M = 0.4
    SENTRY_HOLE_ROBOTS = ("hero", "standard_3", "standard_4", "sentry")
    SENTRY_HOLES_RED_M = {
        "long_hole": (15.5, 1.56),
        "short_hole": (19.0, 12.0),
    }
    AUTOAIM_STAGE_TIMEOUT_SEC = 1.0
    WEB_LATE_OVERRIDE_TIMEOUT_SEC = 0.5

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, log=True):
        if self.__class__._instance is not None:
            return self.__class__._instance
        # 配置端口（通常是 /dev/referee）不存在时自动寻找裁判串口。
        # 底层扫描会排除 PlutoSDR 的虚拟串口，避免误占信息波/干扰波设备。
        super().__init__(port, baudrate, auto_scan=True, log=log)
        self.interactive_tx_min_gap_sec = self.INTERACTIVE_MIN_GAP_SEC
        self.bind(MsgID.ROBOT_DATA.value, self.status_message_decode_func)
        self.bind(MsgID.INTERACTIVE_DATA.value, self.interactive_message_decode_func)
        self.bind(MsgID.LAUNCHER_DATA.value, self.dart_status_message_decode_func)
        self.bind(MsgID.RADAR_MARK_PROGRESS.value, self.radar_mark_progress_message_decode_func)
        self.bind(MsgID.RADAR_DECISION_SYNC.value, self.radar_info_message_decode_func)
        self.bind(MsgID.GAME_STATUS.value, self.game_status_message_decode_func)

        # Status flag
        self.is_sentry_connected = False
        self.sentry_disconnect_counter = 0
        self.faction = FACTION.UNKONWN
        # self.faction = FACTION.RED
        self.sentry_received_flag = False


        self.__class__._instance = self

        # TX
        # 1. radar2sentrymsg
        self.radar2sentry_msg = Radar2SentryMessage(
            targetable_flags=0xDD,
        )

        # 2. radar2client_msg / cm
        self.radar2client_msg = Radar2ClientMessage(
            enemy_hero_x=0,
            enemy_hero_y=0,
            enemy_engineer_x=0,
            enemy_engineer_y=0,
            enemy_standard_3_x=0,
            enemy_standard_3_y=0,
            enemy_standard_4_x=0,
            enemy_standard_4_y=0,
            enemy_drone_x=0,
            enemy_drone_y=0,
            enemy_sentry_x=0,
            enemy_sentry_y=0,

            self_hero_x=0,
            self_hero_y=0,
            self_engineer_x=0,
            self_engineer_y=0,
            self_standard_3_x=0,
            self_standard_3_y=0,
            self_standard_4_x=0,
            self_standard_4_y=0,
            self_drone_x=0,
            self_drone_y=0,
            self_sentry_x=0,
            self_sentry_y=0,
        )
        self.radar2client_tracking_msg = self._copy_radar2client_msg(self.radar2client_msg)
        self.radar2client_enemy_source = "tracking"

        # 3. 0x02BB 哨兵坐标反馈 / m
        self.sentry_self_position_msg = SentrySelfPositionMessage(is_blue=False)

        # 4. 飞镖请求易伤指令
        self.dart_info = DartStatusMessage(
            dart_remaining_time=0,
            recent_hit_target=0,
            accumulated_hit_count=0,
            selected_target=0,
            reserve=0,
        )
        self.dart_selected_target = 0
        # 第二次双倍易伤机会就绪后，每完成一次“先见到4、之后见到1”便递增。
        # 中间允许经过其他值；该序号同时作为自瞄第三次放行事件的锁存量。
        self.dart_4_to_1_event_seq = 0
        self.second_opportunity_target_4_seen = False

        # - 给云台手双倍易伤的 state 和剩余触发机会
        self.radar2dart_msg = Radar2DartMessage(
            is_blue=False,
            trigger_state=RadarTriggerState.IDLE.value,
            double_vulnerability_count=0,
            autoaim_stage=0,
        )

        # 5. 标记进度反馈
        self.radar_mark_progress_msg = RadarMarkMessage()
        self.enemy_drone_counteracted = False
        self.autoaim_stage = 0
        self.autoaim_stage_received_monotonic = None

        # 6. 雷达双倍易伤标记
        self.radar_info_msg = RadarInfoMessage()
        self.double_vulnerability_count = 0
        self.is_double_vulnerability = 0
        # self.request_count = 0
        self.last_target = 0
        self.radar_cmd_counter = 0
        self.trigger_state = RadarTriggerState.IDLE
        self.start_request_time = None
        self.armed_double_vulnerability_opportunity = 0
        self.first_double_vulnerability_requested = False
        self.second_double_vulnerability_requested = False
        self.radar2client_tx_fps = 0.0
        self.last_radar2client_tx_time = None

        # 信息波密钥
        self.key = self._make_key_bytes(self.KEY_IDLE_CMD, b"fakeyh")
        self.key_verify_pending = False
        self.key_verify_level = None
        self.blocked_stale_key_text = None
        self.game_progress = 0
        self.game_remaining_time = None
        self.robot_id = 0
        self.own_encrypt_level = 0
        self.is_key_changeable = False
        self.latest_fly_points = []
        self.udp_sender = None
        self.info_msg = None
        self.info_msg_received_monotonic = None
        self.info_msg_valid_0a05_latest_time = None
        self.info_wave_payload_cache = self._make_empty_info_wave_payload_cache()
        self.info_0a01_last_update_time = None
        self.info_0a01_last_snapshot_seq = None
        self.info_0a01_payload = None
        self.info_enemy_positions_cm = None
        # bit5=1 表示建议哨兵前往 long_hole；新小局默认选择 short_hole。
        self.sentry_prefer_long_hole = False
        self.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 0.0}
        self._self_02bb_lock = threading.Lock()
        self.self_02bb_positions_m = {
            "hero": None,
            "engineer": None,
            "standard_3": None,
            "standard_4": None,
            "sentry": None,
        }
        self.self_02bb_last_update_time = None
        self.info_0a01_update_count = 0
        self.self_02bb_update_count = 0
        self.last_info_0a01_log_time = 0.0
        self.last_self_02bb_log_time = 0.0
        self.use_info_wave_0a01_for_referee = (
            self._load_use_info_wave_0a01_for_referee()
        )
        self.enable_info_wave_robot_multicast = (
            self._load_enable_info_wave_robot_multicast()
        )
        self.info_wave_robot_multicast_receiver_credits = [0, 0, 0]
        # 对方英雄、工程、3号、4号、哨兵在本小局内由信息波观测到的最大血量。
        self.info_wave_robot_max_health = [0, 0, 0, 0, 0]
        self.info_wave_position_timeout_sec = self._load_info_wave_position_timeout_sec()
        self.info_status_sock = None
        self.info_status_endpoint = self._load_info_status_endpoint()
        # 仅 start.sh 显式测试态会设置会话 ID；普通比赛启动不创建监听端口。
        self.web_late_test_session = os.environ.get("INFO_WAVE_TEST_SESSION", "")
        self.web_late_test_port = int(
            os.environ.get("WEB_LATE_INJECTION_PORT", WEB_LATE_DEFAULT_PORT)
        )
        self.web_late_override_lock = threading.Lock()
        self.web_late_override_config = None
        self.web_late_override_received_monotonic = None
        self.web_late_rx_socket = None
        self.web_late_rx_stop_event = None
        self.web_late_rx_thread = None

        ########################################################################
        # 四种 0x0301 交互包由一个专用线程按固定顺序错开发送。
        self.interactive_tx_stop_event = None
        self.interactive_tx_thread = None
        self.radar2client_timer = None
        self.udp_tx_timer = None

        self.timer_executor = None
        self.timer_executor_thread = None
        ########################################################################

    def pack_radar_decision_message(self) -> RadarDecisionMessage:
        """打包雷达自主决策信息"""
        return RadarDecisionMessage(
            is_blue=self._is_self_blue_for_comm(),
            radar_cmd=self.radar_cmd_counter,
            key=self._build_radar_decision_key(),
        )

    @staticmethod
    def _to_uint16_coord_cm(value) -> int:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(value) or value <= 0:
            return 0
        return max(0, min(0xFFFF, int(round(value))))

    def pack_info_wave_messages(self) -> list[InfoWaveMessage]:
        """打包信息波缓存转发消息，只发给己方工程机器人。"""
        is_blue = self._is_self_blue_for_comm()
        user_data = self._build_info_wave_user_data()
        receivers = (
            self.INFO_WAVE_RECEIVERS_BLUE if is_blue else self.INFO_WAVE_RECEIVERS_RED
        )
        return [
            InfoWaveMessage(
                is_blue=is_blue,
                receiver_id=receiver_id,
                user_data=user_data,
            )
            for receiver_id in receivers
        ]

    def pack_info_wave_robot_multicast_message(
        self, receiver_id: int
    ) -> InfoWaveRobotMulticastMessage:
        """打包一帧信息波机器人摘要，接收者由调度器轮询指定。"""
        return InfoWaveRobotMulticastMessage(
            is_blue=self._is_self_blue_for_comm(),
            receiver_id=receiver_id,
            user_data=self._build_info_wave_robot_multicast_user_data(),
        )

    ########################################################################
    def _build_radar_decision_key(self):
        """bytes -> 7 字节数组。bytes不能直接赋值ctypes"""
        self.key = self._normalize_key_length(self.key)
        key_command = (
            self.KEY_VERIFY_CMD if self.key_verify_pending else self.KEY_IDLE_CMD
        )
        self.key = bytes([key_command]) + self.key[1:7]
        return (ctypes.c_uint8 * 7).from_buffer_copy(self.key)

    def _normalize_key_length(self, key_bytes: bytes) -> bytes:
        """只修正长度，不改变密钥指令类型。"""
        raw = bytes(key_bytes)
        if len(raw) < 7:
            raw = raw + bytes(7 - len(raw))
        elif len(raw) > 7:
            raw = raw[:7]
        return raw

    def _make_key_bytes(self, command: int, key_text: bytes) -> bytes:
        """构造 7 字节密钥指令。command=0 表示不验证，command=2 表示验证。"""
        key_text = bytes(key_text)
        if len(key_text) < 6:
            key_text = key_text + bytes(6 - len(key_text))
        elif len(key_text) > 6:
            key_text = key_text[:6]
        return bytes([int(command) & 0xFF]) + key_text

    def _reset_key_verify_command(self, block_current_key: bool = False):
        """停止当前密钥验证；可同时屏蔽等级切换后的残留旧密钥。"""
        self.key = self._normalize_key_length(self.key)
        if block_current_key and self.key_verify_pending:
            self.blocked_stale_key_text = self.key[1:7]
        self.key = bytes([self.KEY_IDLE_CMD]) + self.key[1:7]
        self.key_verify_pending = False
        self.key_verify_level = None

    def _accept_parsed_key(self, key_bytes: bytes):
        """只接受当前 1/2 级的新密钥，并持续验证到裁判反馈升级。"""
        raw = self._normalize_key_length(key_bytes)
        key_text = raw[1:7]
        current_level = int(self.own_encrypt_level)
        if current_level not in (1, 2):
            return False
        if (
            self.blocked_stale_key_text is not None
            and key_text == self.blocked_stale_key_text
        ):
            return False
        self.key = bytes([self.KEY_VERIFY_CMD]) + raw[1:7]
        self.key_verify_pending = True
        self.key_verify_level = current_level
        return True

    def _handle_encrypt_level_change(self, old_level: int, new_level: int):
        """收到对应等级升级后停止验证，并屏蔽仍在途的旧密钥 UDP。"""
        old_level = int(old_level)
        new_level = int(new_level)
        is_expected_upgrade = (old_level, new_level) in ((1, 2), (2, 3))
        if is_expected_upgrade or new_level >= 3:
            self._reset_key_verify_command(block_current_key=True)

    def _should_send_radar_decision(self) -> bool:
        """仅在验证密钥或请求双倍易伤时发送 0x0301/0x0121。"""
        return self.key_verify_pending or self.trigger_state == RadarTriggerState.REQUESTING

    def _is_self_blue_for_comm(self) -> bool:
        """优先使用裁判系统下发的机器人 ID 判断己方颜色。"""
        if self.robot_id >= 100:
            return True
        if self.robot_id > 0:
            return False
        return self.faction == FACTION.BLUE

    def _build_info_wave_user_data(self) -> bytes:
        """提取空中机器人允许发弹量和对方剩余金币，均保持 uint16 小端。"""
        projectile_payload = self._get_info_wave_cmd_payload(0x0A03)
        economy_payload = self._get_info_wave_cmd_payload(0x0A04)
        return bytes(projectile_payload[6:8] + economy_payload[0:2])

    def _get_info_wave_cmd_payload(self, cmd_id: int) -> bytes:
        return self.info_wave_payload_cache[cmd_id]

    def _is_info_wave_cmd_valid(self, cmd_id: int) -> bool:
        """判断某命令是否在本小局内至少成功解析过一次。"""
        return self.info_msg is not None and self.info_msg.is_cmd_valid(cmd_id)

    def _is_info_wave_engineer_data_valid(self) -> bool:
        """0x0A03 或 0x0A04 任一来源有效后即可开始发送工程包。"""
        return any(
            self._is_info_wave_cmd_valid(cmd_id)
            for cmd_id in self.INFO_WAVE_ENGINEER_CMD_IDS
        )

    def _is_info_wave_robot_multicast_data_valid(self) -> bool:
        """三个来源均在本小局解析成功后才发送完整机器人摘要。"""
        return all(
            self._is_info_wave_cmd_valid(cmd_id)
            for cmd_id in self.INFO_WAVE_ROBOT_MULTICAST_CMD_IDS
        )

    def _build_info_wave_robot_multicast_user_data(self) -> bytes:
        """生成英雄、工程、3号、4号、哨兵五条固定6字节记录。"""
        health = self._get_info_wave_cmd_payload(0x0A02)
        projectile = self._get_info_wave_cmd_payload(0x0A03)
        buff = self._get_info_wave_cmd_payload(0x0A05)
        # 正式接收入口已逐包更新最大值；这里再更新一次，兼容直接注入缓存的测试工具。
        self._update_info_wave_robot_max_health(health)

        robot_layouts = (
            # 血量偏移、弹量偏移、防御、负防御、主状态、是否为工程、是否检查哨兵姿态
            (0, 0, 3, 4, 36, False, False),
            (2, None, 10, 11, 37, True, False),
            (4, 2, 17, 18, 38, False, False),
            (6, 4, 24, 25, 39, False, False),
            (10, 8, 31, 32, 40, False, True),
        )
        user_data = bytearray()
        for robot_index, (
            health_offset,
            projectile_offset,
            defense_offset,
            negative_defense_offset,
            status_offset,
            is_engineer,
            check_sentry_pose,
        ) in enumerate(robot_layouts):
            targetable = int(buff[status_offset]) == 0
            if check_sentry_pose and int(buff[35]) == 5:
                targetable = False
            current_health = int.from_bytes(
                health[health_offset : health_offset + 2], "little"
            )
            max_health = int(self.info_wave_robot_max_health[robot_index])
            health_percentage = (
                min(
                    100,
                    max(
                        0,
                        (current_health * 100 + max_health // 2) // max_health,
                    ),
                )
                if max_health > 0
                else 0
            )
            user_data.append(health_percentage)
            if is_engineer:
                # 信息波未提供工程允许发弹量，保留字段固定发0。
                user_data.extend(b"\x00\x00")
            else:
                user_data.extend(
                    projectile[projectile_offset : projectile_offset + 2]
                )
            user_data.append(int(buff[defense_offset]))
            user_data.append(int(buff[negative_defense_offset]))
            user_data.append(int(targetable))
        return bytes(user_data)

    def build_sentry_targetable_flags(self) -> int:
        """生成五个可瞄准位，并用 bit5 携带哨兵区域选择建议。"""
        payload = self._get_info_wave_cmd_payload(0x0A05)
        # bit6~7 保留为 1。无新鲜 0x0A05 时，工程默认不可瞄准，
        # 英雄、3号、4号和哨兵默认可瞄准。
        flags = 0xDD
        fresh_bit = 1 << self.INFO_WAVE_CMD_IDS.index(0x0A05)
        is_fresh = self._is_info_wave_cmd_valid(0x0A05) and bool(
            self._current_info_fresh_mask() & fresh_bit
        )
        if is_fresh and len(payload) >= 41:
            # 英雄、3号和4号仅在新鲜主状态非 0 时不可瞄准。
            for bit_index, payload_offset in ((0, 36), (2, 38), (3, 39)):
                if int(payload[payload_offset]) != 0:
                    flags &= ~(1 << bit_index)

            # 工程只有在新鲜主状态明确为 0 时才允许瞄准。
            if int(payload[37]) == 0:
                flags |= 1 << 1

            # 哨兵在新鲜主状态非 0 或强化姿态为 5 时不可瞄准。
            if int(payload[40]) != 0 or int(payload[35]) == 5:
                flags &= ~(1 << 4)

        if self._update_sentry_hole_choice():
            flags |= 1 << 5
        else:
            flags &= ~(1 << 5)
        return flags

    def get_sentry_hole_centers_m(self) -> dict[str, tuple[float, float]]:
        """返回当前阵营对应的两个敌方半场区域圆心。"""
        centers = dict(self.SENTRY_HOLES_RED_M)
        if self._is_self_blue_for_comm():
            return {
                name: (
                    self.FIELD_LENGTH_M - center_x,
                    self.FIELD_WIDTH_M - center_y,
                )
                for name, (center_x, center_y) in centers.items()
            }
        return centers

    def _calculate_sentry_hole_scores(
        self, positions_cm: Mapping[str, tuple[int, int]]
    ) -> dict[str, float]:
        """用信息波敌方 1/3/4/7 坐标按统一线性函数计算威胁分数。"""
        scores = {"long_hole": 0.0, "short_hole": 0.0}
        for hole_name, (center_x, center_y) in (
            self.get_sentry_hole_centers_m().items()
        ):
            for robot_name in self.SENTRY_HOLE_ROBOTS:
                pos_x_cm, pos_y_cm = positions_cm.get(robot_name, (0, 0))
                try:
                    pos_x = float(pos_x_cm) / 100.0
                    pos_y = float(pos_y_cm) / 100.0
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(pos_x) or not math.isfinite(pos_y):
                    continue
                if (pos_x == 0.0 and pos_y == 0.0) or not (
                    0.0 <= pos_x <= self.FIELD_LENGTH_M
                    and 0.0 <= pos_y <= self.FIELD_WIDTH_M
                ):
                    continue
                # if XXX: continue 这里可以加入其他过滤条件。
                distance = math.hypot(pos_x - center_x, pos_y - center_y)
                if distance <= self.SENTRY_HOLE_RADII_M[hole_name]:
                    # 两个区域使用同一条线性曲线，不按各自半径归一化。
                    scores[hole_name] += (
                        self.SENTRY_HOLE_SCORE_BASE
                        - self.SENTRY_HOLE_SCORE_DECAY_PER_M * distance
                    )
        return scores

    def _update_sentry_hole_choice(self) -> bool:
        """无新鲜信息波或两侧分数相等时选择 short_hole。"""
        if not self._info_enemy_positions_fresh():
            self.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 0.0}
            self.sentry_prefer_long_hole = False
            return False

        positions = dict(self.info_enemy_positions_cm or {})
        scores = self._calculate_sentry_hole_scores(positions)
        self.sentry_hole_scores = scores
        self.sentry_prefer_long_hole = (
            scores["long_hole"] < scores["short_hole"]
        )
        return self.sentry_prefer_long_hole

    def _make_empty_info_wave_payload_cache(self) -> dict[int, bytes]:
        return {
            cmd_id: bytes(payload_len)
            for cmd_id, payload_len in zip(
                InfoMsgBag.CMD_IDS,
                InfoMsgBag.PAYLOAD_LENGTHS,
            )
        }

    def _is_game_active_for_serial_tx(self) -> bool:
        """比赛阶段为 4 时才允许周期串口发送。"""
        return int(self.game_progress) == self.ACTIVE_GAME_PROGRESS

    def _reset_match_caches(self):
        """比赛阶段切换时清空发送侧缓存，避免上一局数据残留。"""
        self._reset_key_verify_command()
        self.blocked_stale_key_text = None
        self.info_msg = None
        self.info_msg_received_monotonic = None
        self.info_msg_valid_0a05_latest_time = None
        self.info_wave_payload_cache = self._make_empty_info_wave_payload_cache()
        self.info_wave_robot_multicast_receiver_credits = [0, 0, 0]
        self.info_wave_robot_max_health = [0, 0, 0, 0, 0]
        self._reset_info_wave_enemy_positions()

        self.radar2client_msg = Radar2ClientMessage(
            **{
                field_name: 0
                for field_name, *_ in Radar2ClientData._fields_
            }
        )
        self.radar2client_tracking_msg = self._copy_radar2client_msg(self.radar2client_msg)
        self.radar2client_enemy_source = "tracking"
        self.self_02bb_last_update_time = None
        self.latest_fly_points = []
        self.radar_mark_progress_msg = RadarMarkMessage()
        self.enemy_drone_counteracted = False
        self.autoaim_stage = 0
        self.autoaim_stage_received_monotonic = None
        self.dart_selected_target = 0
        self.dart_4_to_1_event_seq = 0
        self.second_opportunity_target_4_seen = False

        self.radar_cmd_counter = 0
        self.trigger_state = RadarTriggerState.IDLE
        self.start_request_time = None
        self.last_target = 0
        self.armed_double_vulnerability_opportunity = 0
        self.first_double_vulnerability_requested = False
        self.second_double_vulnerability_requested = False
        self.game_remaining_time = None
        self.double_vulnerability_count = 0
        self.is_double_vulnerability = 0
        self.radar2dart_msg = Radar2DartMessage(
            is_blue=self._is_self_blue_for_comm(),
            trigger_state=RadarTriggerState.IDLE.value,
            double_vulnerability_count=0,
            autoaim_stage=0,
        )

    def _update_info_wave_payload_cache(self, udp_msg: InfoMsgBag):
        """只用有效命令刷新缓存；未更新或无效命令继续保留上一次 payload。"""
        for cmd_id in self.INFO_WAVE_CMD_IDS:
            if udp_msg.is_cmd_valid(cmd_id):
                payload = bytes(udp_msg.get_cmd_payload(cmd_id))
                self.info_wave_payload_cache[cmd_id] = payload
                if cmd_id == 0x0A02:
                    self._update_info_wave_robot_max_health(payload)

    def _update_info_wave_robot_max_health(self, health_payload: bytes):
        """逐机器人维护本小局信息波观测到的最大血量。"""
        if len(health_payload) < 12:
            return
        max_health = list(
            getattr(self, "info_wave_robot_max_health", [0, 0, 0, 0, 0])
        )
        for robot_index, health_offset in enumerate(
            self.INFO_WAVE_ROBOT_HEALTH_OFFSETS
        ):
            current_health = int.from_bytes(
                health_payload[health_offset : health_offset + 2], "little"
            )
            if current_health > max_health[robot_index]:
                max_health[robot_index] = current_health
        self.info_wave_robot_max_health = max_health

    def _handle_udp_rx_message(self, udp_msg):
        """处理 rx_task_func 轮询到的 UDP 接收包。"""
        if isinstance(udp_msg, RecvMsgBag):
            self._accept_parsed_key(udp_msg.key_bytes)
        elif isinstance(udp_msg, AutoAimStatusBag):
            self.autoaim_stage = int(udp_msg.stage)
            self.autoaim_stage_received_monotonic = time.monotonic()
        elif isinstance(udp_msg, InfoMsgBag):
            self.info_msg = udp_msg
            self.info_msg_received_monotonic = time.monotonic()
            self.info_msg_valid_0a05_latest_time = (
                self.info_msg_received_monotonic
                if self.info_msg.is_cmd_recently_updated(0x0A05)
                else self.info_msg_valid_0a05_latest_time
            )
            self._update_info_wave_payload_cache(udp_msg)
            self._update_info_wave_enemy_positions(udp_msg)
    ########################################################################

    def _load_use_info_wave_0a01_for_referee(self) -> bool:
        """读取是否允许 0x0A01 覆盖发给裁判系统的对方坐标，默认开启。"""
        try:
            import yaml

            with open(get_params_config_path(), "r", encoding="utf-8") as f:
                root_config = yaml.safe_load(f) or {}
            value = root_config.get("referee", {}).get(
                "use_info_wave_0a01_for_referee", True
            )
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        except Exception:
            return True

    def _load_enable_info_wave_robot_multicast(self) -> bool:
        """读取是否向己方英雄、3号和4号转发信息波摘要，默认开启。"""
        try:
            import yaml

            with open(get_params_config_path(), "r", encoding="utf-8") as f:
                root_config = yaml.safe_load(f) or {}
            value = root_config.get("referee", {}).get(
                "enable_info_wave_robot_multicast", True
            )
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        except Exception:
            return True

    def _load_info_wave_position_timeout_sec(self) -> float:
        """读取 0x0A01 作为 0x0305 对方坐标数据源的有效时长。"""
        try:
            import yaml

            with open(get_params_config_path(), "r", encoding="utf-8") as f:
                root_config = yaml.safe_load(f) or {}
            referee_config = root_config.get("referee", {})
            return max(0.05, float(referee_config.get("info_wave_position_timeout_sec", 0.25)))
        except Exception:
            return 0.25

    def _update_info_wave_enemy_positions(self, udp_msg: InfoMsgBag):
        """只在0x0A01更新代次推进时刷新对方位置及250 ms看门狗。"""
        if not udp_msg.is_cmd_valid(self.INFO_WAVE_POSITION_CMD_ID):
            return

        payload = udp_msg.get_cmd_payload(self.INFO_WAVE_POSITION_CMD_ID)
        if len(payload) < 24:
            return

        payload = bytes(payload[:24])
        updated_cmd_id = getattr(udp_msg, "updated_cmd_id", None)
        marked_0a01_update = updated_cmd_id == self.INFO_WAVE_POSITION_CMD_ID
        if marked_0a01_update:
            if not self._is_newer_info_snapshot_seq(
                int(udp_msg.seq), self.info_0a01_last_snapshot_seq
            ):
                return
        elif self.info_0a01_payload == payload:
            # 心跳、重发和其他命令的更新都不能给 0x0A01 续期。
            return

        self.info_0a01_payload = payload
        self.info_enemy_positions_cm = self._parse_info_0a01_positions_cm(self.info_0a01_payload)
        self.info_0a01_last_update_time = time.monotonic()
        if marked_0a01_update:
            self.info_0a01_last_snapshot_seq = int(udp_msg.seq)
        self.info_0a01_update_count += 1
        if self._rate_limited_log_ready("last_info_0a01_log_time"):
            hero = self.info_enemy_positions_cm.get("hero", (0, 0))
            sentry = self.info_enemy_positions_cm.get("sentry", (0, 0))
            self.get_logger().info(
                "[INFO_WAVE] 0x0A01 updated count=%d valid_mask=0x%02x fresh_mask=0x%02x "
                "hero_cm=(%d,%d) sentry_cm=(%d,%d)"
                % (
                    self.info_0a01_update_count,
                    int(udp_msg.valid_mask),
                    int(udp_msg.fresh_mask),
                    hero[0],
                    hero[1],
                    sentry[0],
                    sentry[1],
                )
            )

    def _reset_info_wave_enemy_positions(self):
        """新小局或机器人 ID 变化时丢弃信息波对方位置缓存。"""
        self.info_0a01_last_update_time = None
        self.info_0a01_last_snapshot_seq = None
        self.info_0a01_payload = None
        self.info_enemy_positions_cm = None
        self.sentry_prefer_long_hole = False
        self.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 0.0}

    def _rate_limited_log_ready(self, attr_name: str, interval_sec: float = 1.0) -> bool:
        now = time.monotonic()
        last_time = float(getattr(self, attr_name, 0.0))
        if now - last_time < interval_sec:
            return False
        setattr(self, attr_name, now)
        return True

    def _parse_info_0a01_positions_cm(self, payload: bytes) -> dict[str, tuple[int, int]]:
        """解析 0x0A01 对方机器人坐标，单位 cm，顺序与 0x0305 对方字段一致。"""
        values = [
            int.from_bytes(payload[index:index + 2], byteorder="little", signed=False)
            for index in range(0, 24, 2)
        ]
        names = ("hero", "engineer", "standard_3", "standard_4", "drone", "sentry")
        return {
            name: (values[2 * idx], values[2 * idx + 1])
            for idx, name in enumerate(names)
        }

    def _info_enemy_positions_fresh(self, now: float | None = None) -> bool:
        if (
            getattr(self, "info_enemy_positions_cm", None) is None
            or getattr(self, "info_0a01_last_update_time", None) is None
        ):
            return False
        if now is None:
            now = time.monotonic()
        return (
            now - self.info_0a01_last_update_time
            <= self.info_wave_position_timeout_sec
        )

    def get_info_wave_enemy_positions_snapshot(
        self, require_fresh: bool = True
    ) -> dict[str, tuple[int, int]]:
        """为 UI 返回独立副本，避免绘制过程中读到被替换的坐标字典。"""
        if require_fresh and not self._info_enemy_positions_fresh():
            return {}
        return dict(getattr(self, "info_enemy_positions_cm", None) or {})

    def get_sentry_hole_scores_snapshot(self) -> dict[str, float]:
        """为 UI 返回当前两个哨兵区域分数的独立副本。"""
        scores = getattr(self, "sentry_hole_scores", None) or {}
        return {
            "long_hole": float(scores.get("long_hole", 0.0)),
            "short_hole": float(scores.get("short_hole", 0.0)),
        }

    @staticmethod
    def _is_newer_info_snapshot_seq(seq: int, last_seq: int | None) -> bool:
        """按 uint16 回绕规则判断快照序号，拒绝重复包和乱序旧包。"""
        if last_seq is None:
            return True
        delta = (int(seq) - int(last_seq)) & 0xFFFF
        return 0 < delta < 0x8000

    def _current_info_fresh_mask(self, now: float | None = None) -> int:
        """返回当前仍可信的新鲜度；UDP 中断超过 3 秒时强制全部过期。"""
        if self.info_msg is None or self.info_msg_received_monotonic is None:
            return 0
        if now is None:
            now = time.monotonic()
        if now - self.info_msg_received_monotonic > self.INFO_WAVE_FRESH_TIMEOUT_SEC:
            return 0
        return int(self.info_msg.fresh_mask) & 0x1F

    def update_tracking_radar2client_msg(self, tracking_msg: Radar2ClientMessage):
        """更新视觉/激光跟踪得到的 0x0305 备用数据源。"""
        self.radar2client_tracking_msg = self._copy_radar2client_msg(tracking_msg)

    def _build_effective_radar2client_msg(self) -> Radar2ClientMessage:
        """组装发送时刻的0x0305快照：0A01新帧未超时则覆盖对方坐标。"""
        msg = self._copy_radar2client_msg(self.radar2client_tracking_msg)
        if (
            self.use_info_wave_0a01_for_referee
            and self._info_enemy_positions_fresh()
        ):
            self._apply_info_enemy_positions(msg)
            self.radar2client_enemy_source = "info_wave"
        else:
            self.radar2client_enemy_source = "tracking"

        self._apply_cached_self_positions(msg)
        return msg

    def _copy_radar2client_msg(self, msg: Radar2ClientMessage) -> Radar2ClientMessage:
        """复制 0x0305 消息，避免 ctypes 包装类被 deepcopy 递归访问。"""
        return Radar2ClientMessage(
            **{
                field_name: int(getattr(msg, field_name))
                for field_name, *_ in Radar2ClientData._fields_
            }
        )

    def _apply_info_enemy_positions(self, msg: Radar2ClientMessage):
        positions = self.info_enemy_positions_cm or {}
        field_map = {
            "hero": ("enemy_hero_x", "enemy_hero_y"),
            "engineer": ("enemy_engineer_x", "enemy_engineer_y"),
            "standard_3": ("enemy_standard_3_x", "enemy_standard_3_y"),
            "standard_4": ("enemy_standard_4_x", "enemy_standard_4_y"),
            "drone": ("enemy_drone_x", "enemy_drone_y"),
            "sentry": ("enemy_sentry_x", "enemy_sentry_y"),
        }
        for name, (x_field, y_field) in field_map.items():
            x, y = positions.get(name, (0, 0))
            setattr(msg, x_field, self._to_uint16_coord_cm(x))
            setattr(msg, y_field, self._to_uint16_coord_cm(y))

    def _apply_cached_self_positions(self, msg: Radar2ClientMessage):
        """用 0x0301/0x02BB 刷新的己方坐标覆盖对应字段。

        0x02BB 只包含哨兵、英雄、工程、3 号、4 号，不包含己方空中机器人。
        因此这里只覆盖 0x02BB 实际提供的字段，己方空中等缺失目标继续保留
        tracking_msg 中的激光雷达/视觉数据源。
        """
        with self._self_02bb_lock:
            if self.self_02bb_last_update_time is None:
                return

            self_fields = (
                "self_hero_x", "self_hero_y",
                "self_engineer_x", "self_engineer_y",
                "self_standard_3_x", "self_standard_3_y",
                "self_standard_4_x", "self_standard_4_y",
                "self_sentry_x", "self_sentry_y",
            )
            for field_name in self_fields:
                setattr(msg, field_name, getattr(self.radar2client_msg, field_name))

    def get_sentry_self_position_snapshot(self) -> AllyPositionSnapshot:
        """返回不可变的 0x02BB 米单位坐标及本机单调接收时间。"""
        with self._self_02bb_lock:
            return AllyPositionSnapshot(
                positions_m=MappingProxyType(dict(self.self_02bb_positions_m)),
                received_monotonic=self.self_02bb_last_update_time,
            )

    def _apply_sentry_self_positions_m(self, msg):
        """把哨兵发来的己方 m 单位坐标写入 0x0305 己方 cm 缓存。"""
        positions_m = {
            "sentry": (float(msg.sentry_x), float(msg.sentry_y)),
            "hero": (float(msg.hero_x), float(msg.hero_y)),
            "engineer": (float(msg.engineer_x), float(msg.engineer_y)),
            "standard_3": (float(msg.standard_3_x), float(msg.standard_3_y)),
            "standard_4": (float(msg.standard_4_x), float(msg.standard_4_y)),
        }
        with self._self_02bb_lock:
            self.self_02bb_positions_m = positions_m
            self.radar2client_msg.self_sentry_x = self._to_uint16_coord_cm(msg.sentry_x * 100)
            self.radar2client_msg.self_sentry_y = self._to_uint16_coord_cm(msg.sentry_y * 100)
            self.radar2client_msg.self_hero_x = self._to_uint16_coord_cm(msg.hero_x * 100)
            self.radar2client_msg.self_hero_y = self._to_uint16_coord_cm(msg.hero_y * 100)
            self.radar2client_msg.self_engineer_x = self._to_uint16_coord_cm(msg.engineer_x * 100)
            self.radar2client_msg.self_engineer_y = self._to_uint16_coord_cm(msg.engineer_y * 100)
            self.radar2client_msg.self_standard_3_x = self._to_uint16_coord_cm(msg.standard_3_x * 100)
            self.radar2client_msg.self_standard_3_y = self._to_uint16_coord_cm(msg.standard_3_y * 100)
            self.radar2client_msg.self_standard_4_x = self._to_uint16_coord_cm(msg.standard_4_x * 100)
            self.radar2client_msg.self_standard_4_y = self._to_uint16_coord_cm(msg.standard_4_y * 100)
            self.self_02bb_last_update_time = time.monotonic()
        self.self_02bb_update_count += 1
        if self._rate_limited_log_ready("last_self_02bb_log_time"):
            self.get_logger().info(
                "[REFEREE] 0x0301/0x02BB updated count=%d "
                "sentry_cm=(%d,%d) hero_cm=(%d,%d) engineer_cm=(%d,%d) "
                "standard3_cm=(%d,%d) standard4_cm=(%d,%d)"
                % (
                    self.self_02bb_update_count,
                    self.radar2client_msg.self_sentry_x,
                    self.radar2client_msg.self_sentry_y,
                    self.radar2client_msg.self_hero_x,
                    self.radar2client_msg.self_hero_y,
                    self.radar2client_msg.self_engineer_x,
                    self.radar2client_msg.self_engineer_y,
                    self.radar2client_msg.self_standard_3_x,
                    self.radar2client_msg.self_standard_3_y,
                    self.radar2client_msg.self_standard_4_x,
                    self.radar2client_msg.self_standard_4_y,
                )
            )
    
    # def reset_double_trigger_state(self):
        # print("Before resetting: ", self.request_count)
        # self.request_count = 0
        # print("After resetting: ", self.request_count)
        # self.trigger_state = RadarTriggerState.IDLE


    def get_faction(self):
        return self.faction

    def _update_radar2client_tx_fps(self, now: float):
        """更新发往裁判系统的定位消息发送帧率。"""
        if self.last_radar2client_tx_time is not None:
            dt = now - self.last_radar2client_tx_time
            if dt > 0:
                instant_fps = 1.0 / dt
                if self.radar2client_tx_fps <= 0:
                    self.radar2client_tx_fps = instant_fps
                else:
                    self.radar2client_tx_fps = 0.8 * self.radar2client_tx_fps + 0.2 * instant_fps
        self.last_radar2client_tx_time = now

    def status_message_decode_func(self, cmd_id, data):
        # 此处有修改阵营
        if cmd_id == MsgID.ROBOT_DATA.value:
            message = RobotStatusMessage.from_bytes(data)
            old_robot_id = self.robot_id
            self.id = message.robot_id
            self.robot_id = message.robot_id
            if old_robot_id != 0 and old_robot_id != self.robot_id:
                self._reset_key_verify_command()
                self.blocked_stale_key_text = None
                self._reset_info_wave_enemy_positions()
            if self.id < 100:
                self.faction = FACTION.RED
            else:
                self.faction = FACTION.BLUE
            # print(f"[STATUS] Robot Status: {message}")

    def dart_status_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.LAUNCHER_DATA.value:
            self.dart_info = DartStatusMessage.from_bytes(data)
            selected_target = int(self.dart_info.selected_target)
            self.dart_selected_target = selected_target
            self.last_target = selected_target

            if (
                self.trigger_state == RadarTriggerState.ARMED
                and self.armed_double_vulnerability_opportunity == 2
                and self.double_vulnerability_count == 1
                and self.is_double_vulnerability == 0
                and not self.second_double_vulnerability_requested
            ):
                if selected_target == 4:
                    self.second_opportunity_target_4_seen = True
                elif selected_target == 1 and self.second_opportunity_target_4_seen:
                    self._start_double_vulnerability_request(
                        opportunity=2,
                        reason="dart selected_target sequence 4->...->1",
                    )

    def _start_double_vulnerability_request(
        self, opportunity: int, reason: str
    ) -> bool:
        """开始一次双倍易伤请求；第二次请求同时锁存自瞄放行事件。"""
        if self.trigger_state in (
            RadarTriggerState.REQUESTING,
            RadarTriggerState.ACTIVE,
        ):
            return False
        if opportunity == 1:
            if self.first_double_vulnerability_requested:
                return False
            self.first_double_vulnerability_requested = True
        elif opportunity == 2:
            if self.second_double_vulnerability_requested:
                return False
            self.second_double_vulnerability_requested = True
            self.dart_4_to_1_event_seq = (
                int(self.dart_4_to_1_event_seq) + 1
            ) & 0x0F
        else:
            raise ValueError(f"Invalid double vulnerability opportunity: {opportunity}")

        self.trigger_state = RadarTriggerState.REQUESTING
        self.start_request_time = time.time()
        self.radar_cmd_counter += 1
        self.armed_double_vulnerability_opportunity = 0
        self.second_opportunity_target_4_seen = False
        self.get_logger().warning(
            f"[RefereeCommLogic] Requesting double vulnerability opportunity "
            f"{opportunity}: {reason}."
        )
        return True

    def _evaluate_double_vulnerability_opportunity(self):
        """按剩余次数和比赛时间布置第一次自动请求或第二次条件请求。"""
        if not self._is_game_active_for_serial_tx():
            return
        if self.is_double_vulnerability != 0:
            return
        if self.trigger_state in (
            RadarTriggerState.REQUESTING,
            RadarTriggerState.ACTIVE,
        ):
            return

        remaining_count = int(self.double_vulnerability_count)
        if remaining_count >= 2 and not self.first_double_vulnerability_requested:
            self.trigger_state = RadarTriggerState.ARMED
            self.armed_double_vulnerability_opportunity = 1
            if (
                self.game_remaining_time is not None
                and int(self.game_remaining_time)
                <= self.FIRST_DOUBLE_VULNERABILITY_TRIGGER_TIME_SEC
            ):
                self._start_double_vulnerability_request(
                    opportunity=1,
                    reason=f"game remaining time {int(self.game_remaining_time)}s",
                )
            return

        if remaining_count == 1 and not self.second_double_vulnerability_requested:
            if self.armed_double_vulnerability_opportunity != 2:
                # 第二次机会出现之前见到的目标4不算，必须从此刻之后重新观察。
                self.second_opportunity_target_4_seen = False
            self.trigger_state = RadarTriggerState.ARMED
            self.armed_double_vulnerability_opportunity = 2
            return

        if self.trigger_state == RadarTriggerState.ARMED:
            self.trigger_state = RadarTriggerState.IDLE
            self.armed_double_vulnerability_opportunity = 0
            self.second_opportunity_target_4_seen = False

    def radar_mark_progress_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.RADAR_MARK_PROGRESS.value:
            self.radar_mark_progress_msg = RadarMarkMessage.from_bytes(data)
            # 0x020C data 是 uint16_t 小端；显式掩码 bit13，避免关键控制
            # 信号依赖 Python/平台的位域布局实现细节。bit13=0x2000，
            # 因此线上原始两字节为 00 20 时该值为 True。
            mark_progress = int.from_bytes(data[:2], byteorder="little", signed=False)
            self.enemy_drone_counteracted = bool(mark_progress & (1 << 13))

    def radar_info_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.RADAR_DECISION_SYNC.value:
            previous_count = int(self.double_vulnerability_count)
            old_encrypt_level = self.own_encrypt_level
            self.radar_info_msg = RadarInfoMessage.from_bytes(data)
            current_count = int(self.radar_info_msg.double_vulnerability_count)
            self.is_double_vulnerability = self.radar_info_msg.is_double_vulnerability
            self.own_encrypt_level = self.radar_info_msg.own_encrypt_level
            self.is_key_changeable = self.radar_info_msg.is_key_changeable
            if old_encrypt_level != self.own_encrypt_level:
                self._handle_encrypt_level_change(
                    old_encrypt_level, self.own_encrypt_level
                )

            if (
                self.trigger_state == RadarTriggerState.REQUESTING
                and (
                    self.is_double_vulnerability == 1
                    or previous_count > current_count
                )
            ):
                self.start_request_time = None
                self.trigger_state = RadarTriggerState.ACTIVE
                self.get_logger().warning(
                    "[RefereeCommLogic] Double vulnerability ACTIVE, "
                    "waiting for cooldown."
                )
            if self.is_double_vulnerability == 0 and self.trigger_state == RadarTriggerState.ACTIVE:
                # self.trigger_state = RadarTriggerState.COOLDOWN

                # TODO: 暂时直接回到 IDLE 状态
                self.trigger_state = RadarTriggerState.IDLE

                self.get_logger().warning("[RefereeCommLogic] Double vulnerability COOLING DOWN.")
            # if self.start_request_time is not None and time.time() - self.start_request_time > REQUEST_TIMEOUT :
            #     # assert self.trigger_state == RadarTriggerState.REQUESTING
            #     self.start_request_time = None
            #     self.trigger_state = RadarTriggerState.IDLE
            #     self.get_logger().warning("[RefereeCommLogic] Double vulnerability request timed out, resetting trigger state.")

            self.double_vulnerability_count = current_count
            self._evaluate_double_vulnerability_opportunity()
            # print(f"[RADAR INFO] Radar Info: {self.radar_info_msg}")

    def game_status_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.GAME_STATUS.value:
            message = GameStatusMessage.from_bytes(data)
            old_game_progress = self.game_progress
            self.game_progress = message.game_progress
            if old_game_progress != self.game_progress:
                self._reset_match_caches()
            self.game_remaining_time = int(message.remaining_time)
            self._evaluate_double_vulnerability_opportunity()
            # print(f"[GAME STATUS] Game Status: {message}")

    def interactive_message_decode_func(self, cmd_id, data):
        if cmd_id == MsgID.INTERACTIVE_DATA.value:
            import struct

            sub_cmd_id = struct.unpack("<H", data[0:2])[0]
            match sub_cmd_id:
                case SubCmdID.SENTRY_SELF_POSITION.value:
                    # 0x02BB user_data：哨兵/英雄/工程/3号/4号的 x,y，float32，单位 m。
                    self.is_sentry_connected = True
                    self.sentry_disconnect_counter = 0
                    self.sentry_self_position_msg = SentrySelfPositionMessage.from_bytes(data)
                    self.sentry_received_flag = True
                    self._apply_sentry_self_positions_m(self.sentry_self_position_msg)

    def _start_web_late_override_receiver(self):
        """测试态监听 Web 晚注入快照；正式启动因无会话 ID 永远不会开启。"""
        if not self.web_late_test_session or self.web_late_rx_thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self.web_late_test_port))
        sock.settimeout(0.2)
        self.web_late_rx_socket = sock
        self.web_late_rx_stop_event = threading.Event()

        def receive_loop():
            while not self.web_late_rx_stop_event.is_set():
                try:
                    payload, _address = sock.recvfrom(WEB_LATE_MAX_PACKET_SIZE)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    active, config = decode_web_late_snapshot(
                        payload, self.web_late_test_session
                    )
                except (ValueError, UnicodeDecodeError):
                    continue
                with self.web_late_override_lock:
                    self.web_late_override_config = config if active else None
                    self.web_late_override_received_monotonic = (
                        time.monotonic() if active else None
                    )

        self.web_late_rx_thread = threading.Thread(
            target=receive_loop,
            name="referee_web_late_override_rx",
            daemon=True,
        )
        self.web_late_rx_thread.start()
        self.get_logger().warning(
            f"[TEST ONLY] Web晚注入监听已开启：127.0.0.1:{self.web_late_test_port}"
        )

    def _stop_web_late_override_receiver(self):
        if self.web_late_rx_stop_event is not None:
            self.web_late_rx_stop_event.set()
        if self.web_late_rx_socket is not None:
            self.web_late_rx_socket.close()
            self.web_late_rx_socket = None
        if (
            self.web_late_rx_thread is not None
            and self.web_late_rx_thread is not threading.current_thread()
        ):
            self.web_late_rx_thread.join(timeout=1.0)
        self.web_late_rx_thread = None
        self.web_late_rx_stop_event = None
        with self.web_late_override_lock:
            self.web_late_override_config = None
            self.web_late_override_received_monotonic = None

    def _get_web_late_override(self, message_key: str):
        """返回仍有心跳且页面已启用的消息配置，否则恢复正式数据源。"""
        lock = getattr(self, "web_late_override_lock", None)
        if lock is None:
            return None
        with lock:
            received = self.web_late_override_received_monotonic
            config = self.web_late_override_config
            if (
                received is None
                or config is None
                or time.monotonic() - received > self.WEB_LATE_OVERRIDE_TIMEOUT_SEC
            ):
                return None
            messages = config.get("messages", {})
            message = messages.get(message_key)
            if not isinstance(message, dict) or message.get("enabled") is not True:
                return None
            return dict(message)

    def _build_web_late_message(self, message_key: str, config: dict):
        """使用 Web 语义字段构造消息，身份、CRC和序号仍由正式管理器决定。"""
        is_blue = self._is_self_blue_for_comm()
        if message_key == "radar_map":
            coordinates = config.get("coordinates", {})
            return Radar2ClientMessage(
                **{
                    field_name: self._to_uint16_coord_cm(coordinates.get(field_name, 0))
                    for field_name, *_ in Radar2ClientData._fields_
                }
            )
        if message_key == "decision":
            key_text = str(config.get("key_text", ""))[:6].encode("ascii", "replace")
            key_text = key_text.ljust(6, b"?")
            key = bytes([max(0, min(2, int(config.get("key_command", 0))))]) + key_text
            return RadarDecisionMessage(
                is_blue=is_blue,
                radar_cmd=max(0, min(2, int(config.get("radar_cmd", 0)))),
                key=(ctypes.c_uint8 * 7).from_buffer_copy(key),
            )
        if message_key == "dart":
            return Radar2DartMessage(
                is_blue=is_blue,
                trigger_state=max(0, min(3, int(config.get("trigger_state", 0)))),
                double_vulnerability_count=max(
                    0, min(2, int(config.get("remaining_count", 0)))
                ),
                autoaim_stage=max(0, min(6, int(config.get("autoaim_stage", 0)))),
            )
        if message_key == "engineer":
            receiver_id = (
                OBJECT_ID.B_ENGINEER.value if is_blue else OBJECT_ID.R_ENGINEER.value
            )
            aerial_ammo = max(0, min(0xFFFF, int(config.get("aerial_ammo", 0))))
            remaining_coins = max(
                0, min(0xFFFF, int(config.get("remaining_coins", 0)))
            )
            user_data = aerial_ammo.to_bytes(2, "little")
            user_data += remaining_coins.to_bytes(2, "little")
            return InfoWaveMessage(
                is_blue=is_blue,
                receiver_id=receiver_id,
                user_data=user_data,
            )
        if message_key == "robot_multicast":
            user_data = bytearray()
            for robot_key in (
                "hero",
                "engineer",
                "standard_3",
                "standard_4",
                "sentry",
            ):
                robot = config[robot_key]
                user_data.append(max(0, min(100, int(robot["hp_percent"]))))
                ammo = 0 if robot_key == "engineer" else int(robot["ammo"])
                user_data.extend(max(0, min(0xFFFF, ammo)).to_bytes(2, "little"))
                user_data.append(max(0, min(0xFF, int(robot["defense"]))))
                user_data.append(
                    max(0, min(0xFF, int(robot["negative_defense"])))
                )
                user_data.append(int(robot["targetable"] is True))
            return InfoWaveRobotMulticastMessage(
                is_blue=is_blue,
                receiver_id=int(config["receiver_id"]),
                user_data=bytes(user_data),
            )
        if message_key == "sentry":
            flags = 0xC0
            for bit_index, field_name in enumerate(
                (
                    "hero_targetable",
                    "engineer_targetable",
                    "standard_3_targetable",
                    "standard_4_targetable",
                    "sentry_targetable",
                )
            ):
                if config.get(field_name) is True:
                    flags |= 1 << bit_index
            if config.get("prefer_long_hole") is True:
                flags |= 1 << 5
            return Radar2SentryMessage(is_blue=is_blue, targetable_flags=flags)
        raise ValueError(f"未知 Web 晚注入消息：{message_key}")

    def start(self):
        """启动裁判系统通信管理器"""
        status = super().start()
        if not status:
            self.get_logger().warning(
                "Failed to start RefereeComm Main Handling Logic."
            )
            return False
        self._start_web_late_override_receiver()
        # self.message_daemon_stop_event = threading.Event()
        # self.message_daemon_thread = threading.Thread(
        #     target=self.message_daemon, daemon=True
        # )
        # self.message_daemon_thread.start()

        ########################################################################
        # 0x0301 不再使用会互相阻塞的多个 ROS2 timer，而由单线程轮询。
        self.interactive_tx_stop_event = threading.Event()
        self.interactive_tx_thread = threading.Thread(
            target=self._interactive_tx_loop,
            name="referee_interactive_tx",
            daemon=True,
        )
        self.interactive_tx_thread.start()
        self.radar2client_timer = self.create_timer(
            1.0 / self.RADAR2CLIENT_TX_HZ, self._send_radar2client
        )
        self.udp_sender = UdpSender(UdpSenderConfig.from_yaml())
        self.udp_sender.open()
        self.udp_tx_timer = self.create_timer(
            1.0 / 5.0, self._send_udp_message
        )

        self.timer_executor = SingleThreadedExecutor()
        self.timer_executor.add_node(self)
        self.timer_executor_thread = threading.Thread(
            target=self.timer_executor.spin, daemon=True
        )
        self.timer_executor_thread.start()
        ########################################################################
        return True

    def close(self):
        """停止裁判系统通信管理器"""
        # self.message_daemon_stop_event.set()
        ########################################################################
        self._stop_web_late_override_receiver()
        # 先停止 0x0301 轮询线程，避免关闭串口期间仍有交互包写入。
        if self.interactive_tx_stop_event is not None:
            self.interactive_tx_stop_event.set()
        if (
            self.interactive_tx_thread is not None
            and self.interactive_tx_thread is not threading.current_thread()
        ):
            self.interactive_tx_thread.join(timeout=1.0)
        self.interactive_tx_thread = None
        self.interactive_tx_stop_event = None

        # 关闭其余 ROS2 定时发送资源。
        for timer in (self.radar2client_timer, self.udp_tx_timer):
            if timer is not None:
                timer.cancel()
                self.destroy_timer(timer)

        if self.udp_sender is not None:
            self.udp_sender.close()
            self.udp_sender = None

        if self.info_status_sock is not None:
            self.info_status_sock.close()
            self.info_status_sock = None

        if self.timer_executor is not None:
            self.timer_executor.shutdown()

        if self.timer_executor_thread is not None:
            self.timer_executor_thread.join(timeout=1.0)
        ########################################################################
        super().close()

    ########################################################################
    def _send_radar2client(self):
        """按 RADAR2CLIENT_TX_HZ 发送雷达到客户端消息。"""
        if not self._is_game_active_for_serial_tx():
            return
        web_config = self._get_web_late_override("radar_map")
        self.radar2client_msg = (
            self._build_web_late_message("radar_map", web_config)
            if web_config is not None
            else self._build_effective_radar2client_msg()
        )
        self.tx(self.radar2client_msg.pack())
        self._update_radar2client_tx_fps(time.time())

    def _send_info_wave(self):
        """通过 0x0301/0x02AA 向工程持续转发最近一次有效缓存。"""
        if not self._is_game_active_for_serial_tx():
            return
        web_config = self._get_web_late_override("engineer")
        if web_config is not None:
            self.tx(self._build_web_late_message("engineer", web_config).pack())
            return
        if not self._is_info_wave_engineer_data_valid():
            return
        for info_wave_msg in self.pack_info_wave_messages():
            self.tx(info_wave_msg.pack())
    
    def _send_radar2sentry(self):
        """比赛进行时通过 0x0301/0x0233 持续发送可瞄准状态。"""
        if not self._is_game_active_for_serial_tx():
            return
        web_config = self._get_web_late_override("sentry")
        if web_config is not None:
            self.tx(self._build_web_late_message("sentry", web_config).pack())
            return
        is_blue = self._is_self_blue_for_comm()
        targetable_flags = self.build_sentry_targetable_flags()
        self.radar2sentry_msg = Radar2SentryMessage(
            is_blue=is_blue,
            targetable_flags=targetable_flags,
        )
        self.tx(self.radar2sentry_msg.pack())

    def _send_radar2dart(self):
        """通过 0x0301/0x0234 向无人机发送雷达双倍易伤状态。"""
        if not self._is_game_active_for_serial_tx():
            return
        web_config = self._get_web_late_override("dart")
        if web_config is not None:
            self.tx(self._build_web_late_message("dart", web_config).pack())
            return
        self.get_logger().debug(f"Current trigger state: {self.trigger_state.name}")
        is_blue = self._is_self_blue_for_comm()
        raw_count = int(self.double_vulnerability_count)
        remaining_count = max(0, min(2, raw_count))
        if remaining_count != raw_count:
            self.get_logger().warning(
                f"[RefereeCommLogic] Invalid double vulnerability count {raw_count}, clamped to {remaining_count}."
            )
        autoaim_stage = 0
        if (
            self.autoaim_stage_received_monotonic is not None
            and time.monotonic() - self.autoaim_stage_received_monotonic
            <= self.AUTOAIM_STAGE_TIMEOUT_SEC
        ):
            autoaim_stage = max(0, min(6, int(self.autoaim_stage)))
        self.radar2dart_msg = Radar2DartMessage(
            is_blue=is_blue,
            trigger_state=self.trigger_state.value,
            double_vulnerability_count=remaining_count,
            autoaim_stage=autoaim_stage,
        )
        self.tx(self.radar2dart_msg.pack())

    def _send_radar_decision(self):
        """仅在有密钥验证或双倍易伤请求时发送 0x0301/0x0121。"""
        if not self._is_game_active_for_serial_tx():
            return False
        web_config = self._get_web_late_override("decision")
        if web_config is not None:
            return bool(
                self.tx(self._build_web_late_message("decision", web_config).pack())
            )
        if not self._should_send_radar_decision():
            return False
        return bool(self.tx(self.pack_radar_decision_message().pack()))

    def _send_info_wave_robot_multicast_one(self):
        """在空闲决策槽中向一个己方接收者发送信息波机器人摘要。"""
        if not self.enable_info_wave_robot_multicast:
            return False
        if not self._is_game_active_for_serial_tx():
            return False
        web_config = self._get_web_late_override("robot_multicast")
        if web_config is None and not self._is_info_wave_robot_multicast_data_valid():
            return False

        receivers = (
            self.INFO_WAVE_ROBOT_MULTICAST_RECEIVERS_BLUE
            if self._is_self_blue_for_comm()
            else self.INFO_WAVE_ROBOT_MULTICAST_RECEIVERS_RED
        )
        receiver_weights = self.INFO_WAVE_ROBOT_MULTICAST_RECEIVER_WEIGHTS
        next_credits = [
            int(credit) + int(weight)
            for credit, weight in zip(
                self.info_wave_robot_multicast_receiver_credits,
                receiver_weights,
            )
        ]
        receiver_index = max(
            range(len(receivers)), key=lambda index: next_credits[index]
        )
        next_credits[receiver_index] -= sum(receiver_weights)
        receiver_id = receivers[receiver_index]
        if web_config is not None:
            web_config["receiver_id"] = receiver_id
            message = self._build_web_late_message("robot_multicast", web_config)
        else:
            message = self.pack_info_wave_robot_multicast_message(receiver_id)
        sent = bool(self.tx(message.pack()))
        if sent:
            self.info_wave_robot_multicast_receiver_credits = next_credits
        return sent

    def _send_radar_decision_or_info_wave_robot_multicast(self):
        """决策消息优先；决策空闲时复用该槽发送一帧可选摘要。"""
        if self._send_radar_decision():
            return True
        return self._send_info_wave_robot_multicast_one()

    def _interactive_tx_loop(self):
        """单线程调度0x0301：两种模式的总槽位上限均为28 Hz。"""
        optional_multicast_enabled = bool(
            getattr(self, "enable_info_wave_robot_multicast", False)
        )
        if optional_multicast_enabled:
            senders = (
                self._send_radar_decision_or_info_wave_robot_multicast,
                self._send_radar2dart,
                self._send_info_wave,
                self._send_radar2sentry,
                self._send_info_wave_robot_multicast_one,
            )
            slot_weights = self.OPTIONAL_MULTICAST_SLOT_WEIGHTS
            slot_credits = [0] * len(senders)
            slot_weight_total = sum(slot_weights)
            slot_period = self.OPTIONAL_MULTICAST_SLOT_GAP_SEC
        else:
            # 默认关闭时保持原行为：四种逻辑槽各7 Hz。
            senders = (
                self._send_radar_decision,
                self._send_radar2dart,
                self._send_info_wave,
                self._send_radar2sentry,
            )
            slot_weights = None
            slot_credits = None
            slot_weight_total = 0
            slot_period = self.INTERACTIVE_ROUND_GAP_SEC
        next_slot = time.monotonic()
        sender_index = 0

        while self.interactive_tx_stop_event is not None:
            wait_time = max(0.0, next_slot - time.monotonic())
            if self.interactive_tx_stop_event.wait(wait_time):
                break

            if slot_weights is None:
                senders[sender_index]()
                sender_index = (sender_index + 1) % len(senders)
            else:
                # 平滑加权轮询：每28个槽严格得到5/5/5/5/8次调度，
                # 避免把同类消息集中连发，同时保持每个槽最多发送一帧。
                for index, weight in enumerate(slot_weights):
                    slot_credits[index] += weight
                sender_index = max(
                    range(len(senders)), key=lambda index: slot_credits[index]
                )
                slot_credits[sender_index] -= slot_weight_total
                senders[sender_index]()

            # 使用绝对时间保持当前总槽频率；执行超时后从当前时刻重新排槽，
            # 禁止追赶式连发导致相邻 0x0301 小于安全间隔。
            next_slot += slot_period
            now = time.monotonic()
            if next_slot <= now:
                next_slot = now + slot_period

    def _send_udp_message(self):
        """按固定频率发送 UDP 状态数据，供信息波进程和 autoaim 同步颜色。"""
        if self.udp_sender is None:
            return

        try:
            send_msg = SendMsgBag(
                game_progress=int(self.game_progress) & 0x0F,
                dart_selected_target=int(self.dart_selected_target) & 0x07,
                robot_id=int(self.robot_id) & 0xFF,
                own_encrypt_level=int(self.own_encrypt_level) & 0x03,
                is_key_changeable=int(bool(self.is_key_changeable)),
                enemy_drone_counteracted=int(bool(self.enemy_drone_counteracted)),
                dart_4_to_1_event_seq=int(self.dart_4_to_1_event_seq) & 0x0F,
            )
            self.udp_sender.send_msg(send_msg)
            _, send_payload = UdpPacket.pack(send_msg)
            self.record_udp_tx_debug_message("SendMsg", send_payload, send_msg)
            # 同一份 3 字节包发到 autoaim：同步颜色及对方空中机器人反制状态。
            self.udp_sender.send_xyz_raw(send_payload)
            self._send_info_status_copy(send_payload)
        except Exception as e:
            self.get_logger().warning(f"[UDP] SendMsg failed: {e}")

    def _load_info_status_endpoint(self):
        """读取信息波 MATLAB 进程的状态包复制目标。"""
        host = os.environ.get("INFO_STATUS_REMOTE_HOST", "127.0.0.1").strip()
        port_text = os.environ.get("INFO_STATUS_REMOTE_PORT", "").strip()
        if not port_text:
            return None
        try:
            port = int(port_text)
        except ValueError:
            self.get_logger().warning(f"[UDP] Invalid INFO_STATUS_REMOTE_PORT: {port_text}")
            return None
        if port <= 0 or port > 65535:
            self.get_logger().warning(f"[UDP] INFO_STATUS_REMOTE_PORT out of range: {port}")
            return None
        return (host or "127.0.0.1", port)

    def _send_info_status_copy(self, payload: bytes):
        """把比赛状态额外复制给信息波接收进程，不影响原 5006 干扰波链路。"""
        if self.info_status_endpoint is None:
            return
        try:
            if self.info_status_sock is None:
                self.info_status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.info_status_sock.sendto(payload, self.info_status_endpoint)
        except Exception as e:
            self.get_logger().debug(f"[UDP] Info status copy failed: {e}")

    def _select_udp_fly_point(self):
        """选择需要通过 UDP 发送的飞坡点，当前取 latest_fly_points 中的第一个点。"""
        if not self.latest_fly_points:
            return (0, 0, 0)

        x, y, z = self.latest_fly_points[0]
        return int(round(x)), int(round(y)), int(round(z))

    ########################################################################

    def tx_double_vulnerability_trigger(self):
        """兼容旧调用入口；0x0301 实际发送统一由专用轮询线程负责。"""
        # 当前暂不需要把信息波缓存转发给英雄、3 号、4 号。
        # 暂停这 3 帧 0x0301，避免本函数 10Hz 运行时总交互消息频率超过 30Hz。
        # for info_wave_msg in self.pack_info_wave_messages():
        #     self.tx(info_wave_msg.pack())
        if self.trigger_state == RadarTriggerState.IDLE:
            pass # 不可触发的消息
        if self.trigger_state == RadarTriggerState.ARMED:
            pass # 兼容旧版状态，当前不会主动进入
        if self.trigger_state == RadarTriggerState.REQUESTING:
            pass # 已经请求触发，等待裁判系统反馈
        if self.trigger_state == RadarTriggerState.ACTIVE:
            pass # 已经触发，等待裁判系统反馈
        if self.trigger_state == RadarTriggerState.COOLDOWN:
            pass # 冷却中，不可触发


    def message_daemon(self):
        """消息处理守护线程"""
        ########################################################################
        pass
        ########################################################################
        # tick = 0
        # radar2xx_counter = 0
        # last, now = time.time(), time.time()
        # while not self.message_daemon_stop_event.is_set():
        #     self.sentry_disconnect_counter += 1
            # 检测哨兵的连接状态
            # if self.sentry_disconnect_counter > 3:
            #     self.is_sentry_connected = False
            #     self.sentry_disconnect_counter = 0
                # self.get_logger().warning(
                #     "[RefereeCommLogic] Sentry disconnected, resetting connection status."
                # )

            # 发送雷达标记信息
            # 1 HZ -> Triggering double 
            # 8 HZ -> Referee
            # if tick % 10 == 0:
            # radar2xx_counter += 1
            # if radar2xx_counter % 10 == 5:
            #     self.tx_double_vulnerability_trigger()
                 # 历史飞镖目标触发逻辑已移除；当前触发在 radar_info_message_decode_func 中完成。
            # else:
            #     self.tx(self.radar2client_msg.pack())
            #     self._update_radar2client_tx_fps(time.time())

            # time.sleep(0.08)
    
    def _decode_message(self, cmd_id, data):
        """解码裁判系统消息，未知消息保留原始载荷供 UI 调试。"""
        decoder_map = {
            ########################################################################
            MsgID.GAME_STATUS.value: GameStatusMessage,
            ########################################################################
            MsgID.ROBOT_DATA.value: RobotStatusMessage,
            MsgID.LAUNCHER_DATA.value: DartStatusMessage,
            MsgID.RADAR_MARK_PROGRESS.value: RadarMarkMessage,
            MsgID.RADAR_DECISION_SYNC.value: RadarInfoMessage,
            ########################################################################
            MsgID.CLIENT_RADAR_DATA.value: Radar2ClientMessage,
            ########################################################################
        }
        try:
            if cmd_id == MsgID.INTERACTIVE_DATA.value:
                return self._decode_interactive_message(cmd_id, data)

            decoder_cls = decoder_map.get(cmd_id)
            if decoder_cls is None:
                # 未注册解码器的裁判系统消息保持原始载荷记录，避免常规帧刷屏。
                # self.get_logger().warning(f"Unknown message ID: 0x{cmd_id:04x}")
                return UnknownMessage(cmd_id=cmd_id, raw_data=data)
            return decoder_cls.from_bytes(data)
        except Exception as e:
            self.get_logger().warning(
                f"Decode failed for cmd_id 0x{cmd_id:04x}: {e}"
            )
            return UnknownMessage(
                cmd_id=cmd_id,
                raw_data=data,
                message_name="DecodeFailedMessage",
            )

    def _decode_interactive_message(self, cmd_id, data):
        """解码交互消息，未知 sub_cmd_id 返回原始消息占位对象。"""
        decoder_map = {
            SubCmdID.SENTRY_SELF_POSITION.value: SentrySelfPositionMessage,
            ########################################################################
            SubCmdID.RADAR_2_SENTRY.value: Radar2SentryMessage,
            SubCmdID.RADAR_2_DART.value: Radar2DartMessage,
            SubCmdID.RADAR_DECISION.value: RadarDecisionMessage,
            SubCmdID.INFO_WAVE.value: InfoWaveMessage,
            SubCmdID.INFO_WAVE_ROBOT_MULTICAST.value: InfoWaveRobotMulticastMessage,
            ########################################################################
        }
        sub_cmd_id, sender_id, receiver_id = self._extract_interactive_header(data)
        decoder_cls = decoder_map.get(sub_cmd_id)
        if decoder_cls is None:
            self.get_logger().warning(
                f"Unknown interactive message sub ID: 0x{sub_cmd_id:04x}"
            )
            return UnknownMessage(
                cmd_id=cmd_id,
                raw_data=data,
                sub_cmd_id=sub_cmd_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
            )
        try:
            return decoder_cls.from_bytes(data)
        except Exception as e:
            self.get_logger().warning(
                f"Decode failed for interactive sub_cmd_id 0x{sub_cmd_id:04x}: {e}"
            )
            return UnknownMessage(
                cmd_id=cmd_id,
                raw_data=data,
                sub_cmd_id=sub_cmd_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_name="DecodeFailedInteractiveMessage",
            )

    def _extract_interactive_header(self, data):
        """提取交互消息公共头，兼容长度不足的异常输入。"""
        """后续移植到 serial_protocol.py 中"""
        import struct

        sub_cmd_id = 0
        sender_id = None
        receiver_id = None
        if len(data) >= 2:
            sub_cmd_id = struct.unpack("<H", data[0:2])[0]
        if len(data) >= 4:
            sender_id = struct.unpack("<H", data[2:4])[0]
        if len(data) >= 6:
            receiver_id = struct.unpack("<H", data[4:6])[0]
        return sub_cmd_id, sender_id, receiver_id
    ##########
    #######  TODO: wait to be tested  #####

if __name__ == "__main__":
    # Example usage
    import rclpy

    port = "/dev/ttyUSB0"  # Replace with your actual port
    baudrate = 115200

    rclpy.init()

    referee_manager = RefereeCommManager(port, baudrate)
    if referee_manager.start():
        print("RefereeCommManager started successfully.")
    else:
        print("Failed to start RefereeCommManager.")

    try:
        while True:
            referee_manager.summarize()
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        referee_manager.close()
        print("RefereeCommManager stopped.")
