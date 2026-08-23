import struct
import threading
import time
from unittest.mock import patch

from driver.referee.referee_comm import RefereeCommManager
from driver.referee.serial_protocol import Radar2ClientMessage, RefereeGenericMessage
from driver.referee.udp_protocol import InfoMsgBag


def _make_position_manager(use_info_wave: bool) -> RefereeCommManager:
    """构造只验证 0x0305 数据源选择的最小通信管理器。"""
    manager = object.__new__(RefereeCommManager)
    manager.use_info_wave_0a01_for_referee = use_info_wave
    manager.radar2client_tracking_msg = Radar2ClientMessage(
        enemy_hero_x=111,
        enemy_hero_y=222,
    )
    manager.info_enemy_positions_cm = {
        "hero": (1001, 1002),
        "engineer": (2001, 2002),
        "standard_3": (3001, 3002),
        "standard_4": (4001, 4002),
        "drone": (5001, 5002),
        "sentry": (6001, 6002),
    }
    manager.info_msg = type("_InfoMsg", (), {"fresh_mask": 0x01})()
    manager.info_msg_received_monotonic = time.monotonic()
    manager.info_0a01_last_update_time = time.monotonic()
    manager.info_wave_position_timeout_sec = 0.25
    manager._self_02bb_lock = threading.Lock()
    manager.self_02bb_last_update_time = None
    return manager


def test_info_wave_0a01_switch_defaults_to_tracking_when_disabled():
    """关闭开关时，即使 0x0A01 新鲜也必须使用视觉/激光结果。"""
    manager = _make_position_manager(use_info_wave=False)

    message = manager._build_effective_radar2client_msg()

    assert manager.radar2client_enemy_source == "tracking"
    assert (message.enemy_hero_x, message.enemy_hero_y) == (111, 222)


def test_info_wave_0a01_switch_allows_fresh_override_when_enabled():
    """打开开关后，新鲜 0x0A01 可以覆盖全部对方坐标。"""
    manager = _make_position_manager(use_info_wave=True)

    message = manager._build_effective_radar2client_msg()

    assert manager.radar2client_enemy_source == "info_wave"
    assert (message.enemy_hero_x, message.enemy_hero_y) == (1001, 1002)
    assert (message.enemy_drone_x, message.enemy_drone_y) == (5001, 5002)
    assert (message.enemy_sentry_x, message.enemy_sentry_y) == (6001, 6002)


def test_info_wave_only_overrides_enemy_and_keeps_sentry_self_positions():
    """0x0A01只替换对方十二个坐标，己方仍取哨兵0x02BB缓存。"""
    manager = _make_position_manager(use_info_wave=True)
    manager.radar2client_msg = Radar2ClientMessage(
        self_hero_x=7001,
        self_hero_y=7002,
        self_engineer_x=7101,
        self_engineer_y=7102,
        self_sentry_x=7201,
        self_sentry_y=7202,
    )
    manager.self_02bb_last_update_time = time.monotonic()

    message = manager._build_effective_radar2client_msg()

    assert (message.enemy_hero_x, message.enemy_hero_y) == (1001, 1002)
    assert (message.self_hero_x, message.self_hero_y) == (7001, 7002)
    assert (message.self_engineer_x, message.self_engineer_y) == (7101, 7102)
    assert (message.self_sentry_x, message.self_sentry_y) == (7201, 7202)


def test_info_wave_0a01_falls_back_after_250_ms_without_new_frame():
    """超过250 ms没有新的0x0A01代次时，下一班0x0305必须改用跟踪坐标。"""
    manager = _make_position_manager(use_info_wave=True)
    manager.info_0a01_last_update_time = 100.0

    with patch("driver.referee.referee_comm.time.monotonic", return_value=100.25):
        message = manager._build_effective_radar2client_msg()
    assert manager.radar2client_enemy_source == "info_wave"
    assert (message.enemy_hero_x, message.enemy_hero_y) == (1001, 1002)

    with patch("driver.referee.referee_comm.time.monotonic", return_value=100.251):
        message = manager._build_effective_radar2client_msg()

    assert manager.radar2client_enemy_source == "tracking"
    assert (message.enemy_hero_x, message.enemy_hero_y) == (111, 222)


def _make_info_packet(seq: int, update_index: int, positions: tuple[int, ...]) -> InfoMsgBag:
    payload_0a01 = struct.pack("<12H", *positions)
    payload = payload_0a01 + bytes(sum(InfoMsgBag.PAYLOAD_LENGTHS[1:]))
    valid_and_update = 0x01 | ((update_index & 0x07) << 5)
    raw = (
        b"IF"
        + bytes((3, valid_and_update, 0x01))
        + int(seq).to_bytes(2, "little")
        + payload
    )
    return InfoMsgBag.from_bytes(raw)


def test_only_new_0a01_generation_refreshes_watchdog_even_if_payload_unchanged():
    """心跳不续期；官方序号变化时即使坐标完全相同也必须续期。"""
    manager = object.__new__(RefereeCommManager)
    manager.info_0a01_payload = None
    manager.info_enemy_positions_cm = None
    manager.info_0a01_last_update_time = None
    manager.info_0a01_last_snapshot_seq = None
    manager.info_0a01_update_count = 0
    manager.last_info_0a01_log_time = 100.0
    positions = tuple(range(1, 13))
    first = _make_info_packet(seq=10, update_index=1, positions=positions)
    heartbeat = _make_info_packet(seq=10, update_index=1, positions=positions)
    next_official = _make_info_packet(seq=15, update_index=1, positions=positions)
    other_cmd = _make_info_packet(seq=16, update_index=2, positions=positions)

    with patch(
        "driver.referee.referee_comm.time.monotonic",
        side_effect=(100.0, 100.2, 100.3, 100.35),
    ):
        manager._update_info_wave_enemy_positions(first)
        manager._update_info_wave_enemy_positions(heartbeat)
        assert manager.info_0a01_last_update_time == 100.0
        manager._update_info_wave_enemy_positions(next_official)
        assert manager.info_0a01_last_update_time == 100.3
        manager._update_info_wave_enemy_positions(other_cmd)

    assert manager.info_0a01_last_update_time == 100.3
    assert manager.info_0a01_update_count == 2
    assert manager.info_0a01_last_snapshot_seq == 15


def test_radar2client_0305_matches_official_48_byte_layout():
    """0x0305 应按官方顺序发送24个小端 uint16 坐标。"""
    RefereeGenericMessage._seq_counter = 0
    field_names = [field[0] for field in Radar2ClientMessage.STRUCT_CLASS._fields_]
    values = list(range(1, 25))
    message = Radar2ClientMessage(**dict(zip(field_names, values)))

    frame = message.pack()

    assert int.from_bytes(frame[1:3], byteorder="little") == 48
    assert frame[5:7] == bytes.fromhex("05 03")
    assert frame[7:55] == struct.pack("<24H", *values)
    assert len(frame) == 57
