import ctypes
import socket
import time

import pytest

from driver.referee.referee_comm import FACTION, RadarTriggerState, RefereeCommManager
from driver.referee.serial_protocol import (
    Radar2DartData,
    RefereeGenericMessage,
)
from driver.referee.udp import UdpRecver, UdpRecverConfig
from driver.referee.udp_protocol import AutoAimStatusBag, RecvMsgBag


def _make_manager() -> RefereeCommManager:
    """构造不连接串口和 ROS2 的最小通信管理器。"""
    manager = object.__new__(RefereeCommManager)
    manager.game_progress = manager.ACTIVE_GAME_PROGRESS
    manager.robot_id = 9
    manager.faction = FACTION.RED
    manager.trigger_state = RadarTriggerState.ARMED
    manager.double_vulnerability_count = 2
    manager.autoaim_stage = 5
    manager.autoaim_stage_received_monotonic = time.monotonic()
    manager.sent_frames = []
    manager.tx = lambda frame: manager.sent_frames.append(frame) or True
    manager.get_logger = lambda: type(
        "_Logger",
        (),
        {"debug": lambda *_args: None, "warning": lambda *_args: None},
    )()
    return manager


def test_autoaim_status_udp_packet_is_exactly_one_byte():
    packet = AutoAimStatusBag(stage=6)

    assert packet.pack() == b"\x06"
    assert AutoAimStatusBag.from_bytes(packet.pack()).stage == 6
    with pytest.raises(ValueError):
        AutoAimStatusBag(stage=7)


def test_radar2dart_packet_appends_autoaim_stage_as_third_user_byte():
    manager = _make_manager()
    RefereeGenericMessage._seq_counter = 0

    manager._send_radar2dart()

    frame = manager.sent_frames[-1]
    data_length = int.from_bytes(frame[1:3], byteorder="little")
    assert ctypes.sizeof(Radar2DartData) == 3
    assert data_length == 9
    assert len(frame) == 18
    assert frame[7:13] == bytes.fromhex("34 02 09 00 06 00")
    assert frame[13:16] == bytes(
        [
            RadarTriggerState.ARMED.value,
            2,
            5,
        ]
    )


def test_autoaim_udp_status_updates_next_radar2dart_packet():
    manager = _make_manager()

    manager._handle_udp_rx_message(AutoAimStatusBag(stage=3))
    manager._send_radar2dart()

    assert manager.autoaim_stage == 3
    assert manager.autoaim_stage_received_monotonic is not None
    assert manager.sent_frames[-1][15] == 3


def test_stale_autoaim_status_falls_back_to_first_aiming():
    manager = _make_manager()
    manager.autoaim_stage = 6
    manager.autoaim_stage_received_monotonic = (
        time.monotonic() - manager.AUTOAIM_STAGE_TIMEOUT_SEC - 0.1
    )

    manager._send_radar2dart()

    assert manager.sent_frames[-1][15] == 0


def test_radar2dart_is_not_sent_before_game_starts():
    manager = _make_manager()
    manager.game_progress = 3

    manager._send_radar2dart()

    assert manager.sent_frames == []


def test_udp_receiver_keeps_autoaim_status_and_key_packets_in_order():
    config = UdpRecverConfig(bind_host="127.0.0.1", bind_port=0)
    with UdpRecver(config) as receiver, socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM
    ) as sender:
        port = receiver._sock.getsockname()[1]
        sender.sendto(b"\x04", ("127.0.0.1", port))
        sender.sendto(b"\x02ABCDEF", ("127.0.0.1", port))

        messages = receiver.read_available()

    assert [type(message) for message in messages] == [
        AutoAimStatusBag,
        RecvMsgBag,
    ]
    assert messages[0].stage == 4
    assert messages[1].key_bytes == b"\x02ABCDEF"


def test_serial_udp_dispatch_accepts_autoaim_status_packet():
    manager = _make_manager()
    manager.udp_rx_latest_msg = None
    manager.udp_rx_autoaim_status = None
    manager._record_udp_rx_debug_message = lambda _message: None
    manager._write_udp_rx_log = lambda _message: None

    manager._handle_udp_rx_packet(AutoAimStatusBag(stage=6))

    assert manager.udp_rx_autoaim_status.stage == 6
    assert manager.autoaim_stage == 6
