from __future__ import annotations

import argparse
import logging
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .udp_protocol import (
    AutoAimStatusBag,
    InfoMsgBag,
    InfoMsgPacket,
    RecvMsgBag,
    RecvMsgPacket,
    SendMsgBag,
    SendMsgPacket,
    UdpPacket,
    UdpSendDataKind,
    XYZ,
    XYZPacket,
)


DEFAULT_XYZ_PORT = 5005
DEFAULT_MSG_PORT = 5006
DEFAULT_RECV_PORT = 5007


@dataclass(frozen=True)
class UdpEndpointConfig:
    remote_host: str = "127.0.0.1"
    remote_port: int = DEFAULT_XYZ_PORT
    send_buffer_bytes: int = 4096


def _endpoint_from_dict(config: dict, default_port: int) -> UdpEndpointConfig:
    return UdpEndpointConfig(
        remote_host=config.get("remote_host", "127.0.0.1"),
        remote_port=int(config.get("remote_port", default_port)),
        send_buffer_bytes=int(config.get("send_buffer_bytes", 4096)),
    )


@dataclass(frozen=True)
class UdpSenderConfig:
    xyz: UdpEndpointConfig = field(default_factory=UdpEndpointConfig)
    msg: UdpEndpointConfig = field(
        default_factory=lambda: UdpEndpointConfig(remote_port=DEFAULT_MSG_PORT)
    )

    @classmethod
    def from_yaml(cls, path: str | Path = "config/params.yaml") -> "UdpSenderConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as f:
            root_config = yaml.safe_load(f) or {}

        udp_config = root_config.get("udp", {})
        return cls(
            xyz=_endpoint_from_dict(
                udp_config.get("XYZ", {}),
                default_port=DEFAULT_XYZ_PORT,
            ),
            msg=_endpoint_from_dict(
                udp_config.get("Msg", {}),
                default_port=DEFAULT_MSG_PORT,
            ),
        )

@dataclass(frozen=True)
class UdpRecverConfig:
    bind_host: str = "127.0.0.1"
    bind_port: int = DEFAULT_RECV_PORT
    receive_buffer_bytes: int = 4096
    max_datagram_bytes: int = max(RecvMsgPacket.SIZE, InfoMsgPacket.SIZE)

    @classmethod
    def from_yaml(cls, path: str | Path = "config/params.yaml") -> "UdpRecverConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as f:
            root_config = yaml.safe_load(f) or {}

        msg_config = root_config.get("udp", {}).get("Msg", {})
        return cls(
            bind_host=msg_config.get("recv_host", "127.0.0.1"),
            bind_port=int(msg_config.get("recv_port", DEFAULT_RECV_PORT)),
            receive_buffer_bytes=int(msg_config.get("recv_buffer_bytes", 4096)),
        )

class UdpSender:
    """
    UDP 发送端。
    - send_xyz(x, y, z): 发送 XYZ 坐标包
    - send_msg(msg): 发送固定结构的 Msg 状态包
    """

    def __init__(self, config: UdpSenderConfig):
        self._config = config
        self._socks: dict[UdpSendDataKind, socket.socket] = {}
        self._endpoints: dict[UdpSendDataKind, tuple[str, int]] = {}

    def open(self) -> None:
        if self._socks:
            return

        self._socks[UdpSendDataKind.XYZ] = self._open_socket(self._config.xyz)
        self._socks[UdpSendDataKind.MSG] = self._open_socket(self._config.msg)
        self._endpoints[UdpSendDataKind.XYZ] = (
            self._config.xyz.remote_host,
            self._config.xyz.remote_port,
        )
        self._endpoints[UdpSendDataKind.MSG] = (
            self._config.msg.remote_host,
            self._config.msg.remote_port,
        )

    def close(self) -> None:
        for sock in self._socks.values():
            sock.close()
        self._socks.clear()
        self._endpoints.clear()

    def send(self, data: XYZ | SendMsgBag) -> None:
        if not self._socks:
            raise RuntimeError("UdpSender is not opened")

        data_kind, payload = UdpPacket.pack(data)
        self._socks[data_kind].sendto(payload, self._endpoints[data_kind])

    def send_xyz(self, x: int, y: int, z: int) -> None:
        self.send(XYZ(x=x, y=y, z=z))

    def send_msg(self, msg: SendMsgBag) -> None:
        self.send(msg)

    def send_xyz_raw(self, payload: bytes | bytearray | memoryview) -> None:
        """向 XYZ 通道发送原始载荷，用于复用 5005 端口转发轻量状态包。"""
        if not self._socks:
            raise RuntimeError("UdpSender is not opened")
        self._socks[UdpSendDataKind.XYZ].sendto(
            bytes(payload), self._endpoints[UdpSendDataKind.XYZ]
        )

    def __enter__(self) -> "UdpSender":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _open_socket(config: UdpEndpointConfig) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.send_buffer_bytes)
        return sock

class UdpRecver:
    """UDP 回传接收端，逐包处理自瞄阶段、7字节密钥和信息波快照。"""

    def __init__(self, config: UdpRecverConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._buffer = bytearray(config.max_datagram_bytes)

    def open(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._config.receive_buffer_bytes)
        sock.bind((self._config.bind_host, self._config.bind_port))
        sock.setblocking(False)

        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read_available(self) -> list[RecvMsgBag | InfoMsgBag | AutoAimStatusBag]:
        """读出当前 socket 中所有合法回传包，保持接收顺序。"""
        if self._sock is None:
            raise RuntimeError("UdpRecver is not opened")

        messages: list[RecvMsgBag | InfoMsgBag | AutoAimStatusBag] = []

        while True:
            try:
                nbytes, _addr = self._sock.recvfrom_into(self._buffer)
            except BlockingIOError:
                break

            view = memoryview(self._buffer)[:nbytes]
            if nbytes == AutoAimStatusBag.SIZE:
                try:
                    messages.append(AutoAimStatusBag.from_bytes(view))
                except ValueError as e:
                    logging.warning("Ignore invalid UDP autoaim status packet: %s", e)
                continue

            if nbytes == RecvMsgPacket.SIZE:
                messages.append(RecvMsgPacket.unpack(view))
                continue

            if nbytes in InfoMsgPacket.SIZES:
                try:
                    messages.append(InfoMsgPacket.unpack(view))
                except ValueError as e:
                    logging.warning("Ignore invalid UDP InfoData packet: %s", e)
                continue

            else:
                logging.warning(
                    "Ignore invalid UDP return packet: expected %d, %d, %d or %d bytes, got %d bytes",
                    AutoAimStatusBag.SIZE,
                    RecvMsgPacket.SIZE,
                    InfoMsgPacket.SIZE,
                    min(InfoMsgPacket.SIZES),
                    nbytes,
                )

        return messages

    def read_latest(self) -> Optional[RecvMsgBag | InfoMsgBag | AutoAimStatusBag]:
        messages = self.read_available()
        if not messages:
            return None
        return messages[-1]

        return latest

    def __enter__(self) -> "UdpRecver":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()




##########################################
######## below is for testing ############
##########################################

@dataclass(frozen=True)
class UdpXYZReceiverConfig:
    bind_host: str = "127.0.0.1"
    bind_port: int = DEFAULT_XYZ_PORT
    receive_buffer_bytes: int = 4096
    max_datagram_bytes: int = XYZPacket.SIZE


class UdpXYZReceiver:
    """UDP XYZ 接收端，只保留最新一帧坐标。"""

    def __init__(self, config: UdpXYZReceiverConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._buffer = bytearray(config.max_datagram_bytes)

    def open(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._config.receive_buffer_bytes)
        sock.bind((self._config.bind_host, self._config.bind_port))
        sock.setblocking(False)

        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read_latest(self) -> Optional[XYZ]:
        if self._sock is None:
            raise RuntimeError("UdpXYZReceiver is not opened")

        latest: Optional[XYZ] = None

        while True:
            try:
                nbytes, _addr = self._sock.recvfrom_into(self._buffer)
            except BlockingIOError:
                break

            if nbytes != XYZPacket.SIZE:
                logging.warning(
                    "Ignore invalid UDP XYZ packet: expected %d bytes, got %d bytes",
                    XYZPacket.SIZE,
                    nbytes,
                )
                continue

            latest = XYZPacket.unpack(memoryview(self._buffer)[:nbytes])

        return latest

    def __enter__(self) -> "UdpXYZReceiver":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class UdpMsgReceiverConfig:
    bind_host: str = "127.0.0.1"
    bind_port: int = DEFAULT_MSG_PORT
    receive_buffer_bytes: int = 4096
    max_datagram_bytes: int = SendMsgBag.SIZE


class UdpMsgReceiver:
    """UDP Msg 接收端，只保留最新一帧状态包。"""

    def __init__(self, config: UdpMsgReceiverConfig):
        self._config = config
        self._sock: Optional[socket.socket] = None
        self._buffer = bytearray(config.max_datagram_bytes)

    def open(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._config.receive_buffer_bytes)
        sock.bind((self._config.bind_host, self._config.bind_port))
        sock.setblocking(False)

        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read_latest(self) -> Optional[SendMsgBag]:
        if self._sock is None:
            raise RuntimeError("UdpMsgReceiver is not opened")

        latest: Optional[SendMsgBag] = None

        while True:
            try:
                nbytes, _addr = self._sock.recvfrom_into(self._buffer)
            except BlockingIOError:
                break

            if nbytes != SendMsgPacket.SIZE:
                logging.warning(
                    "Ignore invalid UDP SendMsg packet: expected %d bytes, got %d bytes",
                    SendMsgPacket.SIZE,
                    nbytes,
                )
                continue

            latest = SendMsgPacket.unpack(memoryview(self._buffer)[:nbytes])

        return latest

    def __enter__(self) -> "UdpMsgReceiver":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_loopback_test(args: argparse.Namespace) -> None:
    """在本机同时测试 XYZ 和 Msg 两个 UDP 通道。"""

    sender_config = UdpSenderConfig(
        xyz=UdpEndpointConfig(
            remote_host=args.host,
            remote_port=args.xyz_port,
        ),
        msg=UdpEndpointConfig(
            remote_host=args.host,
            remote_port=args.msg_port,
        ),
    )
    xyz_receiver_config = UdpXYZReceiverConfig(
        bind_host=args.host,
        bind_port=args.xyz_port,
    )
    msg_receiver_config = UdpMsgReceiverConfig(
        bind_host=args.host,
        bind_port=args.msg_port,
    )

    period = 1.0 / args.hz

    logging.info(
        "UDP loopback test: XYZ %s:%d, Msg %s:%d, count=%d, hz=%.2f",
        args.host,
        args.xyz_port,
        args.host,
        args.msg_port,
        args.count,
        args.hz,
    )

    with UdpXYZReceiver(xyz_receiver_config) as xyz_receiver, UdpMsgReceiver(
        msg_receiver_config
    ) as msg_receiver, UdpSender(sender_config) as sender:
        next_time = time.perf_counter()

        for i in range(args.count):
            xyz = XYZ(x=i, y=i * 2, z=-i)
            msg = SendMsgBag(
                game_type=args.game_type,
                game_progress=i % 6,
                robot_id=args.robot_id,
                double_vulnerability_count=i % 4,
                is_double_vulnerability=(i // 2) % 2,
                own_encrypt_level=args.own_encrypt_level,
                is_key_changeable=i % 2,
            )

            sender.send(xyz)
            sender.send(msg)

            latest_xyz: Optional[XYZ] = None
            latest_msg: Optional[SendMsgBag] = None
            deadline = time.perf_counter() + min(period, 0.05)

            while time.perf_counter() < deadline:
                latest_xyz = xyz_receiver.read_latest() or latest_xyz
                latest_msg = msg_receiver.read_latest() or latest_msg
                if latest_xyz is not None and latest_msg is not None:
                    break
                time.sleep(0.001)

            if latest_xyz is None or latest_msg is None:
                raise RuntimeError(
                    f"Loopback packet lost at index {i}: "
                    f"xyz={latest_xyz is not None}, msg={latest_msg is not None}"
                )

            print(
                f"[{i:03d}] XYZ=({latest_xyz.x}, {latest_xyz.y}, {latest_xyz.z}) "
                f"Msg(raw={latest_msg.pack().hex()}, "
                f"game_progress={latest_msg.game_progress}, "
                f"robot_id={latest_msg.robot_id}, "
                f"double_count={latest_msg.double_vulnerability_count}, "
                f"double={latest_msg.is_double_vulnerability}, "
                f"encrypt={latest_msg.own_encrypt_level}, "
                f"key_changeable={latest_msg.is_key_changeable})"
            )

            next_time += period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)


def run_recv_test(args: argparse.Namespace) -> None:
    """独立测试 RecvData 接收链路，不经过 UdpSender。"""

    config = UdpRecverConfig(
        bind_host=args.host,
        bind_port=args.port,
        receive_buffer_bytes=args.recv_buffer_bytes,
    )
    send_host = args.send_host
    if send_host is None:
        send_host = "127.0.0.1" if args.host in {"", "0.0.0.0"} else args.host

    period = 1.0 / args.hz

    logging.info(
        "UDP RecvData test: bind %s:%d, sendto %s:%d, count=%d, hz=%.2f",
        args.host,
        args.port,
        send_host,
        args.port,
        args.count,
        args.hz,
    )

    with UdpRecver(config) as recver, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        next_time = time.perf_counter()

        for i in range(args.count):
            key = bytes(
                (args.key_seed + i + offset) & 0xFF
                for offset in range(RecvMsgPacket.SIZE)
            )
            payload = RecvMsgPacket.pack(RecvMsgBag(key=key))
            sock.sendto(payload, (send_host, args.port))

            latest: Optional[RecvMsgBag] = None
            deadline = time.perf_counter() + min(period, 0.05)

            while time.perf_counter() < deadline:
                latest = recver.read_latest() or latest
                if latest is not None:
                    break
                time.sleep(0.001)

            if latest is None:
                raise RuntimeError(f"RecvData packet lost at index {i}")

            if latest.key_bytes != key:
                raise RuntimeError(
                    f"RecvData mismatch at index {i}: "
                    f"expected={key.hex()}, got={latest.key_hex}"
                )

            print(f"[{i:03d}] RecvData.key={latest.key_hex}")

            next_time += period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)


def run_sender(args: argparse.Namespace) -> None:
    base_config = UdpSenderConfig.from_yaml()
    config = UdpSenderConfig(
        xyz=UdpEndpointConfig(
            remote_host=args.host or base_config.xyz.remote_host,
            remote_port=args.port if args.port is not None else base_config.xyz.remote_port,
            send_buffer_bytes=base_config.xyz.send_buffer_bytes,
        ),
        msg=UdpEndpointConfig(
            remote_host=args.msg_host or base_config.msg.remote_host,
            remote_port=args.msg_port if args.msg_port is not None else base_config.msg.remote_port,
            send_buffer_bytes=base_config.msg.send_buffer_bytes,
        ),
    )

    period = 1.0 / args.hz

    logging.info(
        "UDP sender XYZ -> %s:%d, Msg -> %s:%d, hz=%.2f",
        config.xyz.remote_host,
        config.xyz.remote_port,
        config.msg.remote_host,
        config.msg.remote_port,
        args.hz,
    )

    with UdpSender(config) as sender:
        next_time = time.perf_counter()
        i = 0

        while True:
            sender.send_msg(
                SendMsgBag(
                    game_type=args.game_type,
                    game_progress=args.game_progress,
                    robot_id=args.robot_id,
                    double_vulnerability_count=args.double_vulnerability_count,
                    is_double_vulnerability=args.is_double_vulnerability,
                    own_encrypt_level=args.own_encrypt_level,
                    is_key_changeable=args.is_key_changeable,
                )
            )
            if args.demo:
                t = i / args.hz
                x = int(10000 * math.sin(2.0 * math.pi * 0.5 * t))
                y = int(10000 * math.sin(2.0 * math.pi * 0.7 * t))
                z = int(10000 * math.sin(2.0 * math.pi * 1.0 * t))
            else:
                x = args.x
                y = args.y
                z = args.z

            sender.send_xyz(x, y, z)

            i += 1
            next_time += period

            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()


def run_receiver(args: argparse.Namespace) -> None:
    xyz_receiver_config = UdpXYZReceiverConfig(
        bind_host=args.host,
        bind_port=args.xyz_port,
    )
    msg_receiver_config = UdpMsgReceiverConfig(
        bind_host=args.host,
        bind_port=args.msg_port,
    )

    logging.info("UDP XYZ receiver <- %s:%d", args.host, args.xyz_port)
    logging.info("UDP Msg receiver <- %s:%d", args.host, args.msg_port)

    print_period = 1.0 / args.print_hz
    last_print_time = 0.0

    with UdpXYZReceiver(xyz_receiver_config) as receiver, UdpMsgReceiver(msg_receiver_config) as msg_receiver:
        while True:
            xyz = receiver.read_latest()
            msg = msg_receiver.read_latest()

            if xyz is None and msg is None:
                time.sleep(args.idle_sleep)
                continue

            now = time.perf_counter()
            if now - last_print_time >= print_period:
                if xyz is not None:
                    print(f"x={xyz.x:6d}, y={xyz.y:6d}, z={xyz.z:6d}")
                if msg is not None:
                    print(f"msg: {msg}")
                last_print_time = now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime local UDP communication for XYZ and Msg data."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    recv_parser = subparsers.add_parser("recv", help="Run UDP XYZ receiver")
    recv_parser.add_argument("--host", default="127.0.0.1")
    recv_parser.add_argument("--xyz-port", type=int, default=DEFAULT_XYZ_PORT)
    recv_parser.add_argument("--msg-port", type=int, default=DEFAULT_MSG_PORT)
    recv_parser.add_argument("--print-hz", type=float, default=20.0)
    recv_parser.add_argument("--idle-sleep", type=float, default=0.001)

    send_parser = subparsers.add_parser("send", help="Run UDP sender")
    send_parser.add_argument("--host", default=None)
    send_parser.add_argument("--port", type=int, default=None)
    send_parser.add_argument("--msg-host", default=None)
    send_parser.add_argument("--msg-port", type=int, default=None)
    send_parser.add_argument("--hz", type=float, default=1000.0)
    send_parser.add_argument("--demo", action="store_true", help="Send demo sine-wave XYZ values.")
    send_parser.add_argument("--send-msg", action="store_true", help="Send Msg instead of XYZ.")
    send_parser.add_argument("--x", type=int, default=0)
    send_parser.add_argument("--y", type=int, default=0)
    send_parser.add_argument("--z", type=int, default=0)
    send_parser.add_argument("--game-type", type=int, default=0)
    send_parser.add_argument("--game-progress", type=int, default=0)
    send_parser.add_argument("--robot-id", type=int, default=0)
    send_parser.add_argument("--double-vulnerability-count", type=int, default=0)
    send_parser.add_argument("--is-double-vulnerability", type=int, choices=[0, 1], default=0)
    send_parser.add_argument("--own-encrypt-level", type=int, default=0)
    send_parser.add_argument("--is-key-changeable", type=int, choices=[0, 1], default=0)

    test_parser = subparsers.add_parser("test", help="Run local XYZ and Msg loopback test")
    test_parser.add_argument("--host", default="127.0.0.1")
    test_parser.add_argument("--xyz-port", type=int, default=DEFAULT_XYZ_PORT)
    test_parser.add_argument("--msg-port", type=int, default=DEFAULT_MSG_PORT)
    test_parser.add_argument("--count", type=int, default=20)
    test_parser.add_argument("--hz", type=float, default=20.0)
    test_parser.add_argument("--game-type", type=int, default=0)
    test_parser.add_argument("--robot-id", type=int, default=7)
    test_parser.add_argument("--own-encrypt-level", type=int, default=1)

    recv_test_parser = subparsers.add_parser(
        "recv-test",
        help="Run local RecvData test without UdpSender",
    )
    recv_test_parser.add_argument("--host", default="127.0.0.1")
    recv_test_parser.add_argument("--send-host", default=None)
    recv_test_parser.add_argument("--port", type=int, default=DEFAULT_RECV_PORT)
    recv_test_parser.add_argument("--recv-buffer-bytes", type=int, default=4096)
    recv_test_parser.add_argument("--count", type=int, default=10)
    recv_test_parser.add_argument("--hz", type=float, default=20.0)
    recv_test_parser.add_argument("--key-seed", type=int, default=0x10)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "send":
        if args.hz <= 0:
            raise ValueError("--hz must be positive")
        run_sender(args)
    elif args.mode == "recv":
        if args.print_hz <= 0:
            raise ValueError("--print-hz must be positive")
        if args.idle_sleep < 0:
            raise ValueError("--idle-sleep must be non-negative")
        run_receiver(args)
    elif args.mode == "test":
        if args.count <= 0:
            raise ValueError("--count must be positive")
        if args.hz <= 0:
            raise ValueError("--hz must be positive")
        run_loopback_test(args)
    elif args.mode == "recv-test":
        if args.count <= 0:
            raise ValueError("--count must be positive")
        if args.hz <= 0:
            raise ValueError("--hz must be positive")
        if args.recv_buffer_bytes <= 0:
            raise ValueError("--recv-buffer-bytes must be positive")
        if args.key_seed < 0 or args.key_seed > 0xFF:
            raise ValueError("--key-seed must be in [0, 255]")
        run_recv_test(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit.")

'''
cd Main_ws
python -m driver.referee.udp test
'''
