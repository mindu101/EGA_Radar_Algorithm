from driver.referee.referee_comm import (
    FACTION,
    RadarTriggerState,
    RefereeCommManager,
)


def _make_manager() -> RefereeCommManager:
    """构造不连接 ROS2/串口的最小通信管理器。"""
    manager = object.__new__(RefereeCommManager)
    manager.key = manager._make_key_bytes(manager.KEY_IDLE_CMD, b"fakeyh")
    manager.key_verify_pending = False
    manager.key_verify_level = None
    manager.blocked_stale_key_text = None
    manager.own_encrypt_level = 1
    manager.game_progress = manager.ACTIVE_GAME_PROGRESS
    manager.trigger_state = RadarTriggerState.IDLE
    manager.radar_cmd_counter = 0
    manager.robot_id = 9
    manager.faction = FACTION.RED
    manager.sent_frames = []
    manager.tx = lambda frame: manager.sent_frames.append(frame) or True
    return manager


def _decision_fields(frame: bytes) -> tuple[int, int, bytes]:
    """提取 0x0121 的 radar_cmd、key_cmd 和 6 字节密钥。"""
    data_length = int.from_bytes(frame[1:3], "little")
    data = frame[7 : 7 + data_length]
    assert int.from_bytes(data[0:2], "little") == 0x0121
    user_data = data[6:]
    return user_data[0], user_data[1], user_data[2:8]


def test_key_repeats_until_level_upgrade_then_stale_key_is_ignored():
    manager = _make_manager()

    assert manager._accept_parsed_key(b"\x02ABCDEF")
    assert manager._send_radar_decision()
    assert manager._send_radar_decision()
    assert [_decision_fields(frame)[1:] for frame in manager.sent_frames] == [
        (2, b"ABCDEF"),
        (2, b"ABCDEF"),
    ]

    manager.own_encrypt_level = 2
    manager._handle_encrypt_level_change(1, 2)
    assert not manager._send_radar_decision()
    assert not manager._accept_parsed_key(b"\x02ABCDEF")
    assert not manager._send_radar_decision()

    assert manager._accept_parsed_key(b"\x02UVWXYZ")
    assert manager._send_radar_decision()
    assert _decision_fields(manager.sent_frames[-1])[1:] == (2, b"UVWXYZ")


def test_key_keeps_sending_without_corresponding_upgrade():
    manager = _make_manager()
    assert manager._accept_parsed_key(b"\x02ABCDEF")

    manager._handle_encrypt_level_change(1, 0)

    assert manager._send_radar_decision()
    assert _decision_fields(manager.sent_frames[-1])[1:] == (2, b"ABCDEF")


def test_level_three_stops_key_packet_and_ignores_all_late_keys():
    manager = _make_manager()
    assert manager._accept_parsed_key(b"\x02ABCDEF")

    manager.own_encrypt_level = 3
    manager._handle_encrypt_level_change(2, 3)

    assert not manager._send_radar_decision()
    assert not manager._accept_parsed_key(b"\x02UVWXYZ")
    assert not manager._send_radar_decision()


def test_double_vulnerability_request_sends_key_command_zero():
    manager = _make_manager()
    manager.trigger_state = RadarTriggerState.REQUESTING
    manager.radar_cmd_counter = 1

    assert manager._send_radar_decision()
    radar_cmd, key_cmd, _ = _decision_fields(manager.sent_frames[-1])
    assert radar_cmd == 1
    assert key_cmd == 0

    manager.trigger_state = RadarTriggerState.ACTIVE
    assert not manager._send_radar_decision()


def test_radar_decision_is_not_sent_before_game_starts():
    manager = _make_manager()
    manager.game_progress = 3
    manager.trigger_state = RadarTriggerState.REQUESTING

    assert not manager._send_radar_decision()
    assert manager.sent_frames == []


def test_key_and_double_vulnerability_can_share_one_packet():
    manager = _make_manager()
    manager.trigger_state = RadarTriggerState.REQUESTING
    manager.radar_cmd_counter = 1
    assert manager._accept_parsed_key(b"\x02ABCDEF")

    assert manager._send_radar_decision()
    radar_cmd, key_cmd, key_text = _decision_fields(manager.sent_frames[-1])
    assert radar_cmd == 1
    assert key_cmd == 2
    assert key_text == b"ABCDEF"

    manager.own_encrypt_level = 2
    manager._handle_encrypt_level_change(1, 2)
    assert manager._send_radar_decision()
    radar_cmd, key_cmd, key_text = _decision_fields(manager.sent_frames[-1])
    assert radar_cmd == 1
    assert key_cmd == 0
    assert key_text == b"ABCDEF"
