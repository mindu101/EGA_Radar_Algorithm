import ctypes

from driver.referee.referee_comm import RefereeCommManager
from driver.referee.serial_protocol import (
    MsgID,
    RadarMarkMessage,
    RadarMarkProgressData,
)
from driver.referee.udp_protocol import SendMsgBag


def test_radar_mark_bit13_decodes_enemy_drone_counteracted():
    assert ctypes.sizeof(RadarMarkProgressData) == 2
    raw = (1 << 13).to_bytes(2, "little")

    message = RadarMarkMessage.from_bytes(raw)

    assert message.enemy_drone_counteracted == 1
    assert message.enemy_drone_aimed == 0
    assert message.self_drone_aimed == 0


def test_referee_manager_caches_radar_mark_bit13():
    manager = object.__new__(RefereeCommManager)
    manager.radar_mark_progress_msg = RadarMarkMessage()
    manager.enemy_drone_counteracted = False

    manager.radar_mark_progress_message_decode_func(
        MsgID.RADAR_MARK_PROGRESS.value,
        (1 << 13).to_bytes(2, "little"),
    )

    assert manager.enemy_drone_counteracted is True


def test_referee_manager_does_not_confuse_neighboring_high_bits():
    manager = object.__new__(RefereeCommManager)
    manager.radar_mark_progress_msg = RadarMarkMessage()
    manager.enemy_drone_counteracted = False

    for bit in (12, 14, 15):
        manager.radar_mark_progress_message_decode_func(
            MsgID.RADAR_MARK_PROGRESS.value,
            (1 << bit).to_bytes(2, "little"),
        )
        assert manager.enemy_drone_counteracted is False


def test_send_msg_byte2_bit3_carries_enemy_drone_counteracted():
    message = SendMsgBag(
        game_progress=4,
        dart_selected_target=4,
        robot_id=9,
        own_encrypt_level=2,
        is_key_changeable=1,
        enemy_drone_counteracted=1,
        dart_4_to_1_event_seq=1,
    )

    raw = message.pack()

    assert len(raw) == 3
    assert raw == bytes([0x44, 0x09, 0x1E])
    decoded = SendMsgBag.from_bytes(raw)
    assert decoded.dart_selected_target == 4
    assert decoded.enemy_drone_counteracted == 1
    assert decoded.dart_4_to_1_event_seq == 1
