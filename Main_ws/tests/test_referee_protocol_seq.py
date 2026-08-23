import collections
import threading
import time

import numpy as np

from driver.referee.serial_comm import RefereeSerialManager, SerialState
from driver.referee.serial_protocol import (
    InfoWaveMessage,
    InteractiveMessage,
    MsgID,
    Radar2ClientMessage,
    Radar2SentryMessage,
    RefereeGenericMessage,
)


def _frame_seq(frame: bytes) -> int:
    """裁判系统帧头 byte3 为全局发送序号。"""
    return frame[3]


def test_seq_is_global_across_message_classes():
    """0x0305 和不同实现路径的 0x0301 必须共用同一个 seq。"""
    RefereeGenericMessage._seq_counter = 0

    frames = [
        Radar2ClientMessage().pack(),
        Radar2SentryMessage(is_blue=False).pack(),
        InfoWaveMessage(
            is_blue=False,
            receiver_id=2,
            user_data=bytes(InfoWaveMessage.USER_DATA_SIZE),
        ).pack(),
        InteractiveMessage(
            np.uint16(0x02FE),
            np.uint16(9),
            np.uint16(2),
            bytes([0x01]),
        ).pack(),
        RefereeGenericMessage(
            np.uint16(MsgID.CLIENT_RADAR_DATA.value),
            bytes(),
        ).pack(),
    ]

    assert [_frame_seq(frame) for frame in frames] == [0, 1, 2, 3, 4]


def test_seq_wraps_after_255():
    """全局 seq 按 uint8 从 255 回绕到 0。"""
    RefereeGenericMessage._seq_counter = 254

    frames = [
        Radar2ClientMessage().pack(),
        Radar2SentryMessage(is_blue=True).pack(),
        Radar2ClientMessage().pack(),
    ]

    assert [_frame_seq(frame) for frame in frames] == [254, 255, 0]


def test_serial_tx_enforces_interactive_minimum_gap():
    """串口最终写入口保证任意两个 0x0301 实际写入间隔大于 33 ms。"""

    class _TimedSerial:
        is_open = True

        def __init__(self):
            self.write_times = []

        def write(self, data):
            self.write_times.append(time.monotonic())
            return len(data)

    manager = object.__new__(RefereeSerialManager)
    manager.state = SerialState.OPENNED
    manager.ser = _TimedSerial()
    manager.tx_lock = threading.Lock()
    manager.lock = threading.Lock()
    manager.last_interactive_tx_monotonic = None
    manager.interactive_tx_min_gap_sec = 0.034
    manager.log_file = None
    manager.tx_frame_count = 0
    manager.msg_queue_len = 3
    manager.tx_msg_message_queue = {}
    manager.tx_msg_hit_count = {}
    manager.tx_interactive_sub_hit_count = {}
    manager.get_formatted_time = lambda: "test"

    frame = Radar2SentryMessage(is_blue=False).pack()
    assert manager.tx(frame)
    assert manager.tx(frame)

    assert manager.ser.write_times[1] - manager.ser.write_times[0] > 0.033
