import serial
import struct
import collections
import time
import os
import fcntl
import re
import threading
from contextlib import nullcontext
from driver.referee.crc import Crc
from driver.referee.serial_protocol import RefereeGenericMessage
from driver.referee.serial_protocol import MsgID
from driver.referee.usb import USBDeviceManager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
import rclpy
from rclpy.node import Node

from .udp_protocol import *
from .udp import *


# Constants
RX_BUF_SIZE = 1024
MAX_PACKAGE_SIZE = 128
START_BYTE = 0xA5
CRC16_SIZE = 2
FRAME_HEADER_SIZE = 5
########################################################################
# DEBUG_UDP_ID = -1
DEBUG_UDP_NAME = "udp"
DEBUG_UDP_RX_ID = -1
DEBUG_UDP_TX_XYZ_ID = -3
DEBUG_UDP_TX_MSG_ID = -2
DEBUG_UDP_DISPLAY_NAME = {
    DEBUG_UDP_RX_ID: "udp_recv",
    DEBUG_UDP_TX_XYZ_ID: "udp_xyz",
    DEBUG_UDP_TX_MSG_ID: "udp_msg",
}
INTERACTIVE_DEBUG_ID_BASE = MsgID.INTERACTIVE_DATA.value << 16
########################################################################

class SerialState(Enum):
    CLOSED = 0
    SCANNING = 3
    OPENNED = 1
    RECOVERING = 2

class PackageStatus:
    HEADER_INCOMPLETE = 1
    HEADER_CRC_ERROR = 2
    PAYLOAD_INCOMPLETE = 3
    CRC_ERROR = 4
    PACKAGE_COMPLETE = 5

class CircularBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)

    def enqueue_rear(self, data):
        self.buffer.extend(data)

    def enqueue_front(self, data):
        self.buffer.extendleft(reversed(data))

    def dequeue_front(self, count):
        data = bytearray()
        for _ in range(count):
            if self.buffer:
                data.append(self.buffer.popleft())
            else:
                break
        return data

    def available(self):
        return len(self.buffer)

class RefereeSerialManager(Node):
    def __init__(self, port=None, baudrate=115200, auto_scan=False, log=True):
        super().__init__("referee_serial_manager")
        self.initial_port = port
        self.baudrate = baudrate
        self.auto_scan = auto_scan
        # self.log = log
        self.current_port = None
        self.current_port_info = None
        self.ser = None
        self.serial_lock_file = None
        self.serial_lock_path = None
        self.rx_task_stop_event = None
        self.udp_rx_task_stop_event = None
        self.udp_rx_task = None

        self.state = SerialState.CLOSED
        self.error_counter = 0
        self.max_error_count = 10
        self.reconnect_interval = 2.0
        self.scan_interval = 2.0
        self.last_scan_time = 0

        self.rx_buffer = CircularBuffer(RX_BUF_SIZE)
        self.rx_frame_buffer = bytearray(FRAME_HEADER_SIZE + MAX_PACKAGE_SIZE + CRC16_SIZE)
        self.rx_count = {
            "Package_Received": 0,
            "Bytes_Received": 0,
            "Header_Incomplete": 0,
            "Header_CRC_Error": 0,
            "Payload_Incomplete": 0,
            "CRC_Error": 0,
            "Package_Complete": 0,
        }
        self.rx_rate = {"CRC_Error_Rate": 0}
        self.package_rx_count = {}
        self.cb_funcs = {}

        # Logging
        self.recording_save_root_dir = str(
            Path(__file__).resolve().parents[2] / "logs" / "referee"
        )
        self.tx_frame_count = 0
        self.rx_frame_count = 0
        self.lock = threading.Lock()
        # 串口与UDP现在由两个独立线程接收。业务回调必须按到达顺序互斥，
        # 避免等级变化和迟到密钥同时改写同一份裁判状态。
        self.state_lock = threading.RLock()
        # 串口写入统一串行化，并对全部 0x0301 帧实施最终间隔保护。
        self.tx_lock = threading.Lock()
        self.last_interactive_tx_monotonic = None
        self.interactive_tx_min_gap_sec = 0.034
        self.log_file = None
        timestamp = self.get_formatted_time()
        self.log_file_path = os.path.join(self.recording_save_root_dir, f"referee_logs_{timestamp}.txt")
        Path(self.recording_save_root_dir).mkdir(parents=True, exist_ok=True)
        # if log: 
        self.open_log_file()
        self.rx_task = None

        ########################################################################
        # UDP 5007由独立线程接收，不受串口扫描、重连和解析积压影响。
        # 该端口同时承载自瞄阶段、MATLAB密钥和信息波包，绑定失败时持续重试。
        self.udp_rx_config = UdpRecverConfig.from_yaml()
        self.udp_rx = UdpRecver(self.udp_rx_config)
        self.udp_rx_retry_interval = 2.0
        self.last_udp_rx_retry_time = 0.0
        self.udp_rx_unavailable_since = None
        self.udp_rx_latest_msg = None
        self.udp_rx_key = None
        self.udp_rx_info = None
        self.udp_rx_autoaim_status = None
        self.last_udp_rx_info_print_time = 0.0
        ########################################################################
        
        # self.msg_queue_len = 3 # TODO: 这个长度应当提供接口让用户设置
        # self.msg_message_queue = {msg_id: collections.deque(maxlen=self.msg_queue_len) for msg_id in MsgID}
        ##########
        self.msg_queue_len = 3  # TODO: 这个长度应当提供接口让用户设置
        self.msg_message_queue = {
            msg_id.value: collections.deque(maxlen=self.msg_queue_len)
            for msg_id in MsgID
        }
        self.msg_message_queue[DEBUG_UDP_RX_ID] = collections.deque(maxlen=self.msg_queue_len)
        self.msg_hit_count = {msg_id.value: 0 for msg_id in MsgID}
        self.msg_hit_count[DEBUG_UDP_RX_ID] = 0
        self.interactive_sub_hit_count = {}
        self.tx_msg_message_queue = {
            msg_id.value: collections.deque(maxlen=self.msg_queue_len)
            for msg_id in MsgID
        }
        self.tx_msg_message_queue[DEBUG_UDP_TX_XYZ_ID] = collections.deque(maxlen=self.msg_queue_len)
        self.tx_msg_message_queue[DEBUG_UDP_TX_MSG_ID] = collections.deque(maxlen=self.msg_queue_len)
        self.tx_msg_hit_count = {msg_id.value: 0 for msg_id in MsgID}
        self.tx_msg_hit_count[DEBUG_UDP_TX_XYZ_ID] = 0
        self.tx_msg_hit_count[DEBUG_UDP_TX_MSG_ID] = 0
        self.tx_interactive_sub_hit_count = {}
        ##########

    ##########
    def _ensure_msg_slot(self, cmd_id: int, message_queue=None, hit_count=None):
        """确保消息缓存与命中计数存在，兼容未在枚举中声明的命令。"""
        message_queue = self.msg_message_queue if message_queue is None else message_queue
        hit_count = self.msg_hit_count if hit_count is None else hit_count
        if cmd_id not in message_queue:
            message_queue[cmd_id] = collections.deque(maxlen=self.msg_queue_len)
        if cmd_id not in hit_count:
            hit_count[cmd_id] = 0

    def _build_rx_message_record(
        self, cmd_id: int, timestamp: str, raw_data: bytes, decoded_msg
    ):
        """构造供 UI 调试窗口消费的接收记录。"""
        return self._build_debug_message_record(cmd_id, timestamp, raw_data, decoded_msg, "serial")

    def _build_debug_message_record(
        self, cmd_id: int, timestamp: str, raw_data: bytes, decoded_msg, transport: str, packet_name: str = ""
    ):
        """构造供 UI 调试窗口消费的通用记录。"""
        return {
            "timestamp": timestamp,
            "cmd_id": int(cmd_id),
            "raw_data": bytes(raw_data),
            "message": decoded_msg,
            "transport": transport,
            "packet_name": packet_name,
        }

    def _record_interactive_hit(self, decoded_msg, interactive_sub_hit_count=None):
        """记录交互消息的 sub_cmd_id 命中次数，供后续细分界面使用。"""
        interactive_sub_hit_count = (
            self.interactive_sub_hit_count
            if interactive_sub_hit_count is None
            else interactive_sub_hit_count
        )
        sub_cmd_id = getattr(decoded_msg, "sub_cmd_id", None)
        if sub_cmd_id is None:
            return
        sub_cmd_id = int(sub_cmd_id)
        interactive_sub_hit_count[sub_cmd_id] = (
            interactive_sub_hit_count.get(sub_cmd_id, 0) + 1
        )

    @staticmethod
    def _interactive_debug_id(sub_cmd_id: int) -> int:
        """把 0x0301 的 sub_cmd_id 映射为 UI 独立显示用虚拟 ID。"""
        return INTERACTIVE_DEBUG_ID_BASE | (int(sub_cmd_id) & 0xFFFF)

    @staticmethod
    def _is_interactive_debug_id(cmd_id: int) -> bool:
        return (int(cmd_id) & 0xFFFF0000) == INTERACTIVE_DEBUG_ID_BASE

    @staticmethod
    def _interactive_debug_display_name(cmd_id: int) -> str:
        sub_cmd_id = int(cmd_id) & 0xFFFF
        return f"0x0301/0x{sub_cmd_id:04x}"

    def _record_interactive_detail_message(
        self,
        decoded_msg,
        record,
        message_queue=None,
        hit_count=None,
    ):
        """为 0x0301 交互消息按 sub_cmd_id 额外记录一份 UI 调试行。"""
        sub_cmd_id = getattr(decoded_msg, "sub_cmd_id", None)
        if sub_cmd_id is None:
            return

        message_queue = self.msg_message_queue if message_queue is None else message_queue
        hit_count = self.msg_hit_count if hit_count is None else hit_count
        detail_id = self._interactive_debug_id(sub_cmd_id)
        self._ensure_msg_slot(detail_id, message_queue, hit_count)
        message_queue[detail_id].append(record)
        hit_count[detail_id] += 1

    def _build_msg_queue_snapshot(self, message_queue, hit_count, interactive_sub_hit_count):
        """获取消息缓存快照，避免 UI 线程直接遍历正在写入的队列。"""
        return {
            "msg_queue_len": self.msg_queue_len,
            "msg_message_queue": {
                cmd_id: list(queue)
                for cmd_id, queue in message_queue.items()
            },
            "msg_hit_count": dict(hit_count),
            "interactive_sub_hit_count": dict(interactive_sub_hit_count),
            ########################################################################
            "msg_display_name": {
                cmd_id: DEBUG_UDP_DISPLAY_NAME[cmd_id]
                for cmd_id in message_queue
                if cmd_id in DEBUG_UDP_DISPLAY_NAME
            } | {
                cmd_id: self._interactive_debug_display_name(cmd_id)
                for cmd_id in message_queue
                if self._is_interactive_debug_id(cmd_id)
            },
            "msg_filter_group": {
                cmd_id: DEBUG_UDP_NAME
                for cmd_id in message_queue
                if cmd_id in DEBUG_UDP_DISPLAY_NAME
            },
            "msg_filter_display_name": {DEBUG_UDP_NAME: DEBUG_UDP_NAME},
            ########################################################################
        }

    def get_msg_queue_snapshot(self):
        """获取接收消息缓存快照，避免 UI 线程直接遍历正在写入的队列。"""
        with self.lock:
            return self._build_msg_queue_snapshot(
                self.msg_message_queue,
                self.msg_hit_count,
                self.interactive_sub_hit_count,
            )

    def get_tx_msg_queue_snapshot(self):
        """获取发送消息缓存快照，避免 UI 线程直接遍历正在写入的队列。"""
        with self.lock:
            return self._build_msg_queue_snapshot(
                self.tx_msg_message_queue,
                self.tx_msg_hit_count,
                self.tx_interactive_sub_hit_count,
            )

    def _record_udp_rx_debug_message(self, udp_msg):
        """记录 UDP 接收消息，UI 中统一显示为 udp。"""
        timestamp = self.get_formatted_time()
        record = self._build_debug_message_record(
            DEBUG_UDP_RX_ID,
            timestamp,
            udp_msg.pack(),
            udp_msg,
            "udp",
            udp_msg.__class__.__name__,
        )
        with self.lock:
            self._ensure_msg_slot(DEBUG_UDP_RX_ID)
            self.msg_message_queue[DEBUG_UDP_RX_ID].append(record)
            self.msg_hit_count[DEBUG_UDP_RX_ID] += 1

    def record_udp_tx_debug_message(self, packet_name: str, raw_data: bytes, udp_msg):
        """记录 UDP 发送消息，UI 中统一显示为 udp。"""
        ########################################################################
        udp_debug_id = {
            "XYZ": DEBUG_UDP_TX_XYZ_ID,
            "SendMsg": DEBUG_UDP_TX_MSG_ID,
        }.get(packet_name, DEBUG_UDP_TX_MSG_ID)
        ########################################################################
        timestamp = self.get_formatted_time()
        record = self._build_debug_message_record(
            udp_debug_id,
            timestamp,
            raw_data,
            udp_msg,
            "udp",
            packet_name,
        )
        with self.lock:
            self._ensure_msg_slot(udp_debug_id, self.tx_msg_message_queue, self.tx_msg_hit_count)
            self.tx_msg_message_queue[udp_debug_id].append(record)
            self.tx_msg_hit_count[udp_debug_id] += 1
    ##########

    def _write_udp_rx_log(self, udp_msg):
        """把 UDP 接收包的原始数据和解析摘要写入裁判日志。"""
        if not self.log_file:
            return

        try:
            timestamp = self.get_formatted_time()
            raw_data = udp_msg.pack()
            packet_name = udp_msg.__class__.__name__
            summary = self._format_udp_rx_summary(udp_msg)
            log_entry = (
                f"udp_rx {timestamp} packet={packet_name} "
                f"len={len(raw_data)} raw={raw_data.hex()} parsed={summary}\n"
            )
            with self.lock:
                self.log_file.write(log_entry)
                self.log_file.flush()
        except Exception as e:
            print(f"UDP rx log failed: {e}")

    def _format_udp_rx_summary(self, udp_msg):
        """生成 UDP 回传包解析摘要，便于赛后从日志直接排查。"""
        if isinstance(udp_msg, RecvMsgBag):
            raw_key = udp_msg.key_bytes
            key_text = self._bytes_to_printable_ascii(raw_key[1:])
            return f"密钥包 prefix=0x{raw_key[0]:02X} key={key_text}"

        if isinstance(udp_msg, InfoMsgBag):
            cmd_parts = []
            for cmd_id in udp_msg.valid_cmd_ids:
                cmd_parts.append(self._format_info_cmd_payload(udp_msg, cmd_id))
            if not cmd_parts:
                cmd_parts.append("无有效命令缓存")
            recent_position = int(udp_msg.is_cmd_recently_updated(0x0A01))
            recent_buff = int(udp_msg.is_cmd_recently_updated(0x0A05))
            return (
                f"信息波 seq={udp_msg.seq} valid_mask=0x{udp_msg.valid_mask:02X} "
                f"fresh_mask=0x{udp_msg.fresh_mask:02X} "
                f"updated_cmd="
                f"{'none' if udp_msg.updated_cmd_id is None else f'0x{udp_msg.updated_cmd_id:04X}'} "
                f"pos_recent={recent_position} buff_recent={recent_buff} "
                f"cmds=[{'; '.join(cmd_parts)}]"
            )

        if isinstance(udp_msg, AutoAimStatusBag):
            return f"自瞄反制阶段=0x{udp_msg.stage:02X}"

        return str(udp_msg)

    def _format_info_cmd_payload(self, info_msg, cmd_id: int) -> str:
        """解析 InfoMsgBag 中单个命令码的 payload。"""
        payload = info_msg.get_cmd_payload(cmd_id)
        if cmd_id == 0x0A01:
            vals = struct.unpack("<12H", payload)
            names = ("英雄", "工程", "步兵3", "步兵4", "空中", "哨兵")
            pairs = [
                f"{names[i]}=({vals[2 * i]},{vals[2 * i + 1]})"
                for i in range(len(names))
            ]
            return "0x0A01位置 " + ",".join(pairs)

        if cmd_id == 0x0A02:
            hero, engineer, infantry3, infantry4, reserved, sentry = struct.unpack("<6H", payload)
            return (
                "0x0A02血量 "
                f"英雄={hero},工程={engineer},步兵3={infantry3},"
                f"步兵4={infantry4},保留={reserved},哨兵={sentry}"
            )

        if cmd_id == 0x0A03:
            hero, infantry3, infantry4, aerial, sentry = struct.unpack("<5H", payload)
            return (
                "0x0A03弹量 "
                f"英雄={hero},步兵3={infantry3},步兵4={infantry4},"
                f"空中={aerial},哨兵={sentry}"
            )

        if cmd_id == 0x0A04:
            remain_coins, total_coins, buff_status_bits = struct.unpack("<HHI", payload)
            return (
                "0x0A04经济 "
                f"剩余金币={remain_coins},总金币={total_coins},"
                f"增益点状态=0x{buff_status_bits:08X}"
            )

        if cmd_id == 0x0A05:
            vals = struct.unpack("<B H B B H B H B B H B H B B H B H B B H B H B B H B 5B", payload)
            names = ("英雄", "工程", "步兵3", "步兵4", "哨兵")
            buff_parts = []
            for index, name in enumerate(names):
                base = index * 5
                buff_parts.append(
                    f"{name}(恢复={vals[base]},冷却={vals[base + 1]},"
                    f"防御={vals[base + 2]},负防御={vals[base + 3]},"
                    f"攻击={vals[base + 4]})"
                )
            return (
                "0x0A05增益 "
                + ",".join(buff_parts)
                + f",哨兵姿态={vals[25]}"
                + ",主要状态="
                + ",".join(
                    f"{name}:{vals[26 + index]}" for index, name in enumerate(names)
                )
            )

        return f"0x{cmd_id:04X} raw={payload.hex()}"

    @staticmethod
    def _bytes_to_printable_ascii(data: bytes) -> str:
        """把 ASCII 字节转成日志文本，不可打印内容保留 hex。"""
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            return "0x" + data.hex()
        if all(32 <= ord(ch) <= 126 for ch in text):
            return text
        return "0x" + data.hex()

    def get_formatted_time(self):
        """获取UTC+8时间戳"""
        tz = timezone(timedelta(hours=8))
        return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def open_log_file(self):
        """打开日志文件"""
        try:
            self.log_file = open(self.log_file_path, 'a')
            print(f"Opened log file: {self.log_file_path}")
        except Exception as e:
            print(f"Error opening log file: {e}")

    def close_log_file(self):
        """关闭日志文件"""
        try:
            if self.log_file:
                self.log_file.close()
                print(f"Closed log file: {self.log_file_path}")
        except Exception as e:
            print(f"Error closing log file: {e}")
        finally:
            self.log_file = None

    def _find_available_port(self):
        """查找可用串口"""
        if self.initial_port and os.path.exists(self.initial_port):
            if not self._acquire_serial_lock(self.initial_port):
                print(f"Port {self.initial_port} is locked by another radar process")
                return None
            if USBDeviceManager.test_serial_port(self.initial_port, self.baudrate):
                return {
                    "device": self.initial_port,
                    "description": "Specified Port",
                    "manufacturer": "Unknown",
                    "vid": None,
                    "pid": None,
                    "serial_number": None,
                }
            self._release_serial_lock()

        if self.auto_scan:
            print("Scanning for USB serial ports...")
            usb_ports = USBDeviceManager.scan_usb_serial_ports()
            candidate_ports = [
                port_info
                for port_info in usb_ports
                if self._is_referee_serial_candidate(port_info)
            ]
            print(
                f"Found {len(usb_ports)} USB serial ports, "
                f"{len(candidate_ports)} referee candidates:"
            )
            for port_info in candidate_ports:
                print(f"  - {port_info['device']}: {port_info['description']}")

            for port_info in candidate_ports:
                port = port_info["device"]
                print(f"Testing port: {port}")
                if not self._acquire_serial_lock(port):
                    print(f"Port {port} is locked by another radar process")
                    continue
                if USBDeviceManager.test_serial_port(port, self.baudrate):
                    print(f"Port {port} is available")
                    return port_info
                else:
                    print(f"Port {port} test failed")
                    self._release_serial_lock()

        return None

    @staticmethod
    def _is_referee_serial_candidate(port_info):
        """排除 Pluto 和板载串口，只保留可能连接裁判系统的 USB 串口。"""
        device = str(port_info.get("device", ""))
        if not device.startswith(("/dev/ttyUSB", "/dev/ttyACM", "/dev/ttyCH341USB")):
            return False

        description = str(port_info.get("description", "")).lower()
        manufacturer = str(port_info.get("manufacturer", "")).lower()
        vid = port_info.get("vid")
        pid = port_info.get("pid")
        is_pluto = (
            (vid == 0x0456 and pid == 0xB673)
            or "pluto" in description
            or "pluto" in manufacturer
        )
        if is_pluto:
            print(f"Skip PlutoSDR serial port: {device}")
            return False
        return True

    def _open_port(self):
        """打开串口"""
        try:
            port_info = self._find_available_port()
            if not port_info:
                print("No available USB serial ports found")
                return False

            port = port_info["device"]
            if not self._acquire_serial_lock(port):
                print(f"Port {port} is locked by another radar process")
                return False
            self.ser = serial.Serial(port, self.baudrate, timeout=1)
            self.current_port = port
            self.current_port_info = port_info
            print(f"Port {port} opened successfully")
            return True

        except serial.SerialException as e:
            self._release_serial_lock()
            print(f"Failed to open port: {e}")
            return False
        except Exception as e:
            self._release_serial_lock()
            print(f"Unexpected error opening port: {e}")
            return False

    def _acquire_serial_lock(self, port):
        """同一裁判串口只允许一个雷达进程持有，避免正式程序与测试注入器混发。"""
        if self.serial_lock_file is not None:
            return True

        resolved_port = os.path.realpath(str(port))
        safe_port = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved_port).strip("_")
        lock_path = f"/tmp/rm2026_referee_serial_{safe_port or 'unknown'}.lock"
        lock_file = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        self.serial_lock_file = lock_file
        self.serial_lock_path = lock_path
        return True

    def _release_serial_lock(self):
        """释放裁判串口进程互斥锁。"""
        if self.serial_lock_file is None:
            return
        try:
            fcntl.flock(self.serial_lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self.serial_lock_file.close()
            self.serial_lock_file = None
            self.serial_lock_path = None

    def _close_port(self):
        """关闭串口"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(f"Port {self.current_port} closed")
        except Exception as e:
            print(f"Error closing port: {e}")
        finally:
            self.ser = None
            self.current_port = None
            self._release_serial_lock()

    def is_connected(self):
        """检查连接状态"""
        return self.state == SerialState.OPENNED and self.ser and self.ser.is_open

    def start(self):
        """启动串口管理器"""
        if self.rx_task and self.rx_task.is_alive():
            print("Serial manager is already running")
            return False

        self.rx_task_stop_event, self.rx_task = self._create_thread()
        self.rx_task.daemon = True
        self.udp_rx_task_stop_event, self.udp_rx_task = self._create_udp_thread()
        self.udp_rx_task.daemon = True

        # UDP线程先启动；即使串口打开或恢复耗时，5007也能立即独立排空。
        self.udp_rx_task.start()
        self.rx_task.start()
        print("Serial manager started")
        return True

    def close(self):
        """关闭串口管理器"""
        if self.rx_task_stop_event:
            self.rx_task_stop_event.set()
        if self.udp_rx_task_stop_event:
            self.udp_rx_task_stop_event.set()

        if self.udp_rx_task and self.udp_rx_task.is_alive():
            self.udp_rx_task.join(timeout=2.0)
        if self.rx_task and self.rx_task.is_alive():
            self.rx_task.join(timeout=2.0)

        
        # UDP线程已先停止；资源由close统一释放，串口线程不再拥有UDP生命周期。
        self._close_udp_rx()
        
        self._close_port()
        self.close_log_file()
        print("Serial manager stopped")

    def _create_thread(self):
        return threading.Event(), threading.Thread(target=self.rx_task_func)

    def _create_udp_thread(self):
        return threading.Event(), threading.Thread(
            target=self.udp_rx_task_func,
            name="referee-udp-rx",
        )

    ########################################################################
    def _open_udp_rx(self, force=False):
        """打开 UDP 接收端。"""
        now = time.time()
        if not force and now - self.last_udp_rx_retry_time < self.udp_rx_retry_interval:
            return False
        self.last_udp_rx_retry_time = now

        if self.udp_rx is None:
            self.udp_rx = UdpRecver(self.udp_rx_config)

        try:
            self.udp_rx.open()
            if self.udp_rx_unavailable_since is not None:
                print("UDP rx recovered")
            self.udp_rx_unavailable_since = None
            return True
        except Exception as e:
            print(f"UDP rx open failed: {e}")
            try:
                self.udp_rx.close()
            except Exception:
                pass
            self.udp_rx = None
            if self.udp_rx_unavailable_since is None:
                self.udp_rx_unavailable_since = now
            return False

    def _close_udp_rx(self):
        """关闭 UDP 接收端。"""
        if self.udp_rx is None:
            return

        try:
            self.udp_rx.close()
        except Exception as e:
            print(f"UDP rx close failed: {e}")
        finally:
            self.udp_rx = None

    def _process_udp_rx_once(self):
        """非阻塞读取一次 UDP 接收端，UDP 失效时周期性尝试恢复。"""
        if self.udp_rx is None:
            self._open_udp_rx()
            return

        try:
            udp_messages = self.udp_rx.read_available()
        except Exception as e:
            print(f"UDP rx read failed: {e}")
            self._close_udp_rx()
            return

        for latest_msg in udp_messages:
            self._handle_udp_rx_packet(latest_msg)

    def udp_rx_task_func(self):
        """独立轮询5007；串口重连或阻塞不得影响该循环。"""
        self._open_udp_rx(force=True)
        while not self.udp_rx_task_stop_event.is_set():
            self._process_udp_rx_once()
            # Event.wait可在close时立即唤醒，比time.sleep更利于可靠退出。
            self.udp_rx_task_stop_event.wait(0.001)
        print("UDP RX task terminated")

    def _handle_udp_rx_packet(self, latest_msg):
        """处理一个 UDP 回传包；5007 上的三类消息必须保持顺序逐包处理。"""
        self.udp_rx_latest_msg = latest_msg
        if isinstance(latest_msg, RecvMsgBag):
            self.udp_rx_key = latest_msg.key_bytes
            print(f"UDP RX KEY: {self.udp_rx_key}")
        elif isinstance(latest_msg, AutoAimStatusBag):
            self.udp_rx_autoaim_status = latest_msg
        elif isinstance(latest_msg, InfoMsgBag):
            self.udp_rx_info = latest_msg
            now = time.time()
            if now - self.last_udp_rx_info_print_time >= 1.0:
                self.last_udp_rx_info_print_time = now
                print(
                    f"UDP RX INFO: seq={latest_msg.seq} "
                    f"valid_mask=0x{latest_msg.valid_mask:02X} "
                    f"fresh_mask=0x{latest_msg.fresh_mask:02X} "
                    f"valid_cmds={[hex(cmd_id) for cmd_id in latest_msg.valid_cmd_ids]}"
                )
        else:
            print(f"UDP RX UNKNOWN: {latest_msg}")
            return

        ########################################################################
        self._record_udp_rx_debug_message(latest_msg)
        self._write_udp_rx_log(latest_msg)
        ########################################################################

        if hasattr(self, "_handle_udp_rx_message"):
            try:
                # 少量离线测试桩通过object.__new__绕过__init__；正式实例
                # 始终持有RLock，测试桩则安全降级为无操作上下文。
                with getattr(self, "state_lock", nullcontext()):
                    self._handle_udp_rx_message(latest_msg)
            except Exception as e:
                print(f"UDP rx callback failed: {e}")
    ########################################################################

    def rx_task_func(self):
        """接收任务主循环"""
        while not self.rx_task_stop_event.is_set():
            current_time = time.time()

            if self.state == SerialState.CLOSED or self.state == SerialState.SCANNING:
                if current_time - self.last_scan_time > self.scan_interval:
                    self.state = SerialState.SCANNING
                    print("Attempting to connect to serial device...")
                    if self._open_port():
                        self.state = SerialState.OPENNED
                        self.error_counter = 0
                        print("Successfully connected")
                    else:
                        print("Failed to connect, retrying...")
                        time.sleep(self.reconnect_interval)
                    self.last_scan_time = current_time
                else:
                    time.sleep(0.1)

            elif self.state == SerialState.OPENNED:
                try:
                    if self.ser.in_waiting > 0:
                        data = self.ser.read(self.ser.in_waiting)
                        self.rx_buffer.enqueue_rear(data)

                    status = PackageStatus.PACKAGE_COMPLETE
                    while self.rx_buffer.available() and (
                        status != PackageStatus.HEADER_INCOMPLETE
                        and status != PackageStatus.PAYLOAD_INCOMPLETE
                    ):
                        status = self.update_and_get_next_frame()
                        if status == PackageStatus.PACKAGE_COMPLETE:
                            data_length = struct.unpack("<H", self.rx_frame_buffer[1:3])[0]
                            cmd_id, data = self._get_cmd_id_and_data_from_buffer(data_length)
                            # 直接记录rx消息
                            with self.lock:
                                timestamp = self.get_formatted_time()
                                data_hex = data.hex()
                                log_entry = f"rx {timestamp} frame_{self.rx_frame_count:06d} cmd_id=0x{cmd_id:04x} data={data_hex}\n"
                                if self.log_file:
                                    self.log_file.write(log_entry)
                                    self.log_file.flush()
                                self.rx_frame_count += 1
                                # print(f"\033[91m Logged rx message: frame_{self.rx_frame_count-1:06d} \033[0m")

                            # 这里加一个 decode 函数，把 data 转成具体的消息对象，传 msg 给回调函数
                            # 而非在回调函数里再去解析一次。(暂时解析两次，等 decoder 测试完了再优化)
                            assert hasattr(self, "_decode_message"), "RefereeSerialManager must implement _decode_message method"
                            decoded_msg = self._decode_message(cmd_id, data)

                            # 维护一个以 cmd_id 为 key 的 msg 历史记录队列，每个队列长度限制为 N, 供 debug 和 ui 查询使用
                            # self.msg_message_queue[cmd_id].append(decoded_msg)
                            ##########
                            record = self._build_rx_message_record(
                                cmd_id, timestamp, data, decoded_msg
                            )
                            with self.lock:
                                self._ensure_msg_slot(cmd_id)
                                self.msg_message_queue[cmd_id].append(record)
                                self.msg_hit_count[cmd_id] += 1
                                self._record_interactive_hit(decoded_msg)
                                if cmd_id == MsgID.INTERACTIVE_DATA.value:
                                    self._record_interactive_detail_message(
                                        decoded_msg,
                                        record,
                                    )
                            ##########

                            key_id = f"0x{cmd_id:04x}"
                            if key_id in self.cb_funcs:
                                for cb_func in self.cb_funcs[key_id]:
                                    try:
                                        with getattr(self, "state_lock", nullcontext()):
                                            cb_func(cmd_id, data)
                                    except Exception as e:
                                        print(f"Error in callback for cmd_id {key_id}: {e}")

                except serial.SerialException as e:
                    print(f"Serial read error: {e}")
                    self.error_counter += 1
                    self._handle_connection_error()

                except Exception as e:
                    print(f"Unexpected error in rx_task: {e}")
                    self.error_counter += 1
                    self._handle_connection_error()

            elif self.state == SerialState.RECOVERING:
                self._close_port()
                time.sleep(self.reconnect_interval)
                if self._open_port():
                    self.state = SerialState.OPENNED
                    self.error_counter = 0
                    print("Connection recovered")
                else:
                    print("Recovery failed, entering scan mode...")
                    self.state = SerialState.SCANNING
                    self.last_scan_time = 0

            time.sleep(0.001)

        self._close_port()
        self.close_log_file()
        self.state = SerialState.CLOSED
        print("RX task terminated")

    def _handle_connection_error(self):
        """处理连接错误"""
        if self.error_counter >= self.max_error_count:
            print(f"Too many errors ({self.error_counter}), entering recovery mode...")
            self.state = SerialState.RECOVERING
        else:
            time.sleep(0.1)

    def _log_rx_package(self, cmd_id):
        cmd_id_str = f"0x{cmd_id:04x}"
        if cmd_id_str not in self.package_rx_count:
            self.package_rx_count[cmd_id_str] = 1
        else:
            self.package_rx_count[cmd_id_str] += 1

    def update_and_get_next_frame(self):
        size = self.rx_buffer.available()

        for _ in range(size):
            if self.rx_buffer.buffer[0] == START_BYTE:
                break
            self.rx_buffer.dequeue_front(1)
            size -= 1

        if size < FRAME_HEADER_SIZE:
            self.rx_count["Header_Incomplete"] += 1
            return PackageStatus.HEADER_INCOMPLETE

        self.rx_frame_buffer[:FRAME_HEADER_SIZE] = self.rx_buffer.dequeue_front(FRAME_HEADER_SIZE)
        header_bytes = self.rx_frame_buffer[:FRAME_HEADER_SIZE]
        data_length = struct.unpack("<H", header_bytes[1:3])[0]
        crc8_received = header_bytes[4]

        header_for_crc = header_bytes[:4]
        crc8_calculated = Crc.get_crc8_check_sum(header_for_crc)
        if crc8_received != crc8_calculated:
            self.rx_count["Header_CRC_Error"] += 1
            return PackageStatus.HEADER_CRC_ERROR

        if data_length > MAX_PACKAGE_SIZE:
            self.rx_count["Header_CRC_Error"] += 1
            return PackageStatus.HEADER_CRC_ERROR

        total_needed = FRAME_HEADER_SIZE + 2 + data_length + CRC16_SIZE
        if size < total_needed:
            self.rx_buffer.enqueue_front(header_bytes)
            self.rx_count["Payload_Incomplete"] += 1
            return PackageStatus.PAYLOAD_INCOMPLETE

        remaining_bytes = self.rx_buffer.dequeue_front(2 + data_length + CRC16_SIZE)
        self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + data_length + CRC16_SIZE + 2] = remaining_bytes

        message_without_tail_crc = self.rx_frame_buffer[: FRAME_HEADER_SIZE + data_length + 2]
        crc16_received = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE + 2 + data_length : FRAME_HEADER_SIZE + 2 + data_length + CRC16_SIZE])[0]
        crc16_calculated = Crc.get_crc16_check_sum(message_without_tail_crc)

        if crc16_received != crc16_calculated:
            self.rx_count["CRC_Error"] += 1
            return PackageStatus.CRC_ERROR

        self.rx_count["Package_Complete"] += 1
        self.rx_count["Bytes_Received"] += total_needed
        cmd_id = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]
        self._log_rx_package(cmd_id)
        return PackageStatus.PACKAGE_COMPLETE

    def _get_cmd_id_and_data_from_buffer(self, data_length):
        """提取cmd_id和数据"""
        cmd_id = struct.unpack("<H", self.rx_frame_buffer[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]
        data = self.rx_frame_buffer[FRAME_HEADER_SIZE + 2 : FRAME_HEADER_SIZE + data_length + 2]
        return cmd_id, data

    def bind(self, msg_id, cb_func):
        if isinstance(msg_id, MsgID):
            msg_id = msg_id.value
        key_id = f"0x{msg_id:04x}"
        if key_id not in self.cb_funcs:
            self.cb_funcs[key_id] = []
        self.cb_funcs[key_id].append(cb_func)

    def unbind(self, msg_id, cb_func):
        if isinstance(msg_id, MsgID):
            msg_id = msg_id.value
        key_id = f"0x{msg_id:04x}"
        if key_id not in self.cb_funcs:
            print(f"Message ID {key_id} not registered")
            return
        if cb_func not in self.cb_funcs[key_id]:
            print(f"Callback function not found for ID {key_id}")
            return
        self.cb_funcs[key_id].remove(cb_func)

    def tx(self, data: bytes):
        """发送消息并记录"""
        if self.state != SerialState.OPENNED:
            return False
        if not isinstance(data, bytes):
            print("Data must be of type bytes")
            return False

        try:
            cmd_id = None
            if len(data) >= FRAME_HEADER_SIZE + 2:
                cmd_id = struct.unpack("<H", data[FRAME_HEADER_SIZE : FRAME_HEADER_SIZE + 2])[0]

            with self.tx_lock:
                if (
                    cmd_id == MsgID.INTERACTIVE_DATA.value
                    and self.last_interactive_tx_monotonic is not None
                ):
                    elapsed = (
                        time.monotonic() - self.last_interactive_tx_monotonic
                    )
                    wait_time = self.interactive_tx_min_gap_sec - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

                if cmd_id is not None:
                ########################################################################
                    data_length = struct.unpack("<H", data[1:3])[0]
                    payload = data[
                        FRAME_HEADER_SIZE + 2 : FRAME_HEADER_SIZE + 2 + data_length
                    ]
                    decoded_msg = (
                        self._decode_message(cmd_id, payload)
                        if hasattr(self, "_decode_message")
                        else None
                    )
                ########################################################################
                    with self.lock:
                        timestamp = self.get_formatted_time()
                        data_hex = data.hex()
                        log_entry = f"tx {timestamp} frame_{self.tx_frame_count:06d} cmd_id=0x{cmd_id:04x} data={data_hex}\n"
                        if self.log_file:
                            self.log_file.write(log_entry)
                            self.log_file.flush()
                        self.tx_frame_count += 1
                    ########################################################################
                        record = self._build_debug_message_record(
                            cmd_id, timestamp, payload, decoded_msg, "serial"
                        )
                        self._ensure_msg_slot(
                            cmd_id, self.tx_msg_message_queue, self.tx_msg_hit_count
                        )
                        self.tx_msg_message_queue[cmd_id].append(record)
                        self.tx_msg_hit_count[cmd_id] += 1
                        self._record_interactive_hit(
                            decoded_msg, self.tx_interactive_sub_hit_count
                        )
                        if cmd_id == MsgID.INTERACTIVE_DATA.value:
                            self._record_interactive_detail_message(
                                decoded_msg,
                                record,
                                self.tx_msg_message_queue,
                                self.tx_msg_hit_count,
                            )
                    ########################################################################
                        # print(f"Logged tx message: frame_{self.tx_frame_count-1:06d}")
                self.ser.write(data)
                if cmd_id == MsgID.INTERACTIVE_DATA.value:
                    self.last_interactive_tx_monotonic = time.monotonic()
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            self.error_counter += 1
            self._handle_connection_error()
            return False

    def summarize(self):
        total_packages = (
            self.rx_count["Header_CRC_Error"]
            + self.rx_count["CRC_Error"]
            + self.rx_count["Package_Complete"]
        )
        if total_packages > 0:
            self.rx_rate["CRC_Error_Rate"] = (
                self.rx_count["Header_CRC_Error"] + self.rx_count["CRC_Error"]
            ) / total_packages

        print(f"Current port: {self.current_port}")
        print(f"Connection state: {self.state}")
        print(f"Error count: {self.error_counter}")
        print(f"Rx Count: {self.rx_count}")
        print(f"Rx Rate: {self.rx_rate}")
        print(f"Package count: {self.package_rx_count}")
        print(f"Tx frames logged: {self.tx_frame_count}")
        print(f"Rx frames logged: {self.rx_frame_count}")


def main(port, baudrate):

    def msg_cb_func(cmd_id, data):
        print("In Callback")
        print("cmd_id: 0x{:04x}; data length: {}".format(cmd_id, len(data)))

    referee_manager = RefereeSerialManager(port, baudrate, auto_scan=True)

    # Bind some test callbacks
    referee_manager.bind(MsgID.GAME_STATUS, msg_cb_func)
    referee_manager.bind(MsgID.ROBOT_HP, msg_cb_func)

    referee_manager.start()
    count = 0

    try:
        while True:
            count += 1
            if count % 100 == 0:
                referee_manager.summarize()

            # Send test message (you can customize this)
            from .serial_protocol import Radar2ClientMessage
            radar2client_msg = Radar2ClientMessage(
                hero_x=100,
                hero_y=200,
                engineer_x=300,
                engineer_y=400,
                standard_3_x=500,
                standard_3_y=600,
                standard_4_x=700,
                standard_4_y=800,
                sentry_x=900,
                sentry_y=1000,

            )
            referee_manager.tx(radar2client_msg.pack())
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        referee_manager.close()


if __name__ == "__main__":
    SERIAL_PORT = "/dev/ttyUSB0"  # Change to your serial port
    BAUD_RATE = 115200  # RoboMaster referee system baud rate

    main(SERIAL_PORT, BAUD_RATE)
