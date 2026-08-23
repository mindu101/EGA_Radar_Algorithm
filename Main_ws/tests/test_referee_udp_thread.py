"""裁判串口与5007 UDP接收线程隔离回归测试。"""

from __future__ import annotations

import threading
import time
import socket

from driver.referee.serial_comm import RefereeSerialManager, SerialState
from driver.referee.udp import UdpRecver, UdpRecverConfig
from driver.referee.udp_protocol import AutoAimStatusBag, InfoMsgBag, RecvMsgBag


def make_info_snapshot(seq: int) -> bytes:
    """构造一个合法的102字节InfoMsgBag v3。"""
    payload = bytearray(102)
    payload[:3] = b"IF\x03"
    payload[3] = 0x01
    payload[4] = 0x01
    payload[5:7] = int(seq).to_bytes(2, "little")
    return bytes(payload)


def test_udp_rx_keeps_running_while_serial_recovery_is_blocked():
    """串口恢复sleep期间，独立UDP线程必须继续接收真实数据报。"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        udp_port = probe.getsockname()[1]

    manager = object.__new__(RefereeSerialManager)
    manager.rx_task_stop_event = threading.Event()
    manager.udp_rx_task_stop_event = threading.Event()
    manager.state = SerialState.RECOVERING
    manager.reconnect_interval = 0.25
    manager.error_counter = 0
    manager.last_scan_time = 0.0
    manager.scan_interval = 2.0
    manager._close_port = lambda: None
    manager._open_port = lambda: False
    manager.close_log_file = lambda: None
    manager.state_lock = threading.RLock()
    manager.udp_rx_config = UdpRecverConfig(
        bind_host="127.0.0.1",
        bind_port=udp_port,
        receive_buffer_bytes=65536,
    )
    manager.udp_rx = UdpRecver(manager.udp_rx_config)
    manager.udp_rx_retry_interval = 0.02
    manager.last_udp_rx_retry_time = 0.0
    manager.udp_rx_unavailable_since = None
    manager.udp_rx_latest_msg = None
    manager.udp_rx_key = None
    manager.udp_rx_info = None
    manager.udp_rx_autoaim_status = None
    manager.last_udp_rx_info_print_time = time.time()
    manager._record_udp_rx_debug_message = lambda _message: None
    manager._write_udp_rx_log = lambda _message: None
    received_events: list[tuple[str, int | bytes]] = []
    received_lock = threading.Lock()
    all_received = threading.Event()

    def handle_udp_message(message):
        with received_lock:
            if isinstance(message, AutoAimStatusBag):
                received_events.append(("stage", int(message.stage)))
            elif isinstance(message, RecvMsgBag):
                received_events.append(("key", message.key_bytes))
            elif isinstance(message, InfoMsgBag):
                received_events.append(("info", int(message.seq)))
            if len(received_events) == 21:
                all_received.set()

    manager._handle_udp_rx_message = handle_udp_message

    serial_thread = threading.Thread(target=manager.rx_task_func)
    udp_thread = threading.Thread(target=manager.udp_rx_task_func)
    serial_thread.start()
    udp_thread.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 等待独立线程完成bind，然后在串口线程固定阻塞的0.25秒窗口内
        # 交错发送实际共用5007的1B状态、7B密钥和102B信息快照。
        # 同一发送socket的数据报顺序必须被完整保留。
        deadline = time.monotonic() + 0.2
        while manager.udp_rx._sock is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert manager.udp_rx._sock is not None
        datagrams: list[bytes] = []
        expected_events: list[tuple[str, int | bytes]] = []
        for index in range(7):
            stage = index % 7
            key = bytes((2,)) + f"K{index:05d}".encode("ascii")
            info = make_info_snapshot(100 + index)
            datagrams.extend((bytes((stage,)), key, info))
            expected_events.extend(
                (("stage", stage), ("key", key), ("info", 100 + index))
            )
        for datagram in datagrams:
            sender.sendto(datagram, ("127.0.0.1", udp_port))
        assert all_received.wait(timeout=0.15), (
            "串口恢复阻塞时真实UDP数据报没有被独立线程及时排空"
        )
        with received_lock:
            assert received_events == expected_events
        assert serial_thread.is_alive()
        assert udp_thread.is_alive()
    finally:
        sender.close()
        manager.udp_rx_task_stop_event.set()
        manager.rx_task_stop_event.set()
        udp_thread.join(timeout=1.0)
        serial_thread.join(timeout=1.0)
        manager._close_udp_rx()

    assert not udp_thread.is_alive()
    assert not serial_thread.is_alive()


def test_udp_thread_stop_event_interrupts_wait_immediately():
    """关闭时不等待串口重连周期，UDP线程应立即退出。"""
    manager = object.__new__(RefereeSerialManager)
    manager.udp_rx_task_stop_event = threading.Event()
    manager._open_udp_rx = lambda force=False: True
    manager._process_udp_rx_once = lambda: None

    udp_thread = threading.Thread(target=manager.udp_rx_task_func)
    udp_thread.start()
    time.sleep(0.01)
    stop_started = time.monotonic()
    manager.udp_rx_task_stop_event.set()
    udp_thread.join(timeout=0.2)

    assert not udp_thread.is_alive()
    assert time.monotonic() - stop_started < 0.1
