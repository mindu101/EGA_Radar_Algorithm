import os
import re
import struct
import sys
from html import escape
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from driver.referee.serial_protocol import (  # noqa: E402
    InfoWaveMessage,
    InteractiveStructMessage,
    MsgID,
    OBJECT_ID,
    RadarDecisionMessage,
    Radar2SentryMessage,
    StructureMessage,
    SubCmdID,
    UnknownMessage,
)
from driver.referee.referee_comm import RefereeCommManager  # noqa: E402


LOG_LINE_RE = re.compile(
    r"^(?P<direction>rx|tx)\s+"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"frame_(?P<frame_index>\d+)\s+"
    r"cmd_id=0x(?P<cmd_id>[0-9a-fA-F]+)\s+"
    r"data=(?P<data>[0-9a-fA-F]*)\s*$"
)


INTERACTIVE_DETAIL_ID_BASE = MsgID.INTERACTIVE_DATA.value << 16


@dataclass
class ParsedRecord:
    """日志中的单包解析结果。"""

    direction: str
    timestamp: str
    frame_index: int
    cmd_id: int
    log_hex: str
    payload: bytes
    message: object
    message_name: str
    parse_error: str = ""
    frame_meta: dict = field(default_factory=dict)


class LogParseError(Exception):
    """日志文件解析失败。"""


class DecodeWarningLogger:
    """接收 referee_comm 解码告警，避免日志工具依赖 ROS logger。"""

    def warning(self, message):
        pass


class RefereeCommDecodeProxy:
    """只复用 RefereeCommManager 的解码函数，不初始化串口和主程序状态。"""

    def get_logger(self):
        return DecodeWarningLogger()

    def _decode_message(self, cmd_id, data):
        return RefereeCommManager._decode_message(self, cmd_id, data)

    def _decode_interactive_message(self, cmd_id, data):
        return RefereeCommManager._decode_interactive_message(self, cmd_id, data)

    def _extract_interactive_header(self, data):
        return RefereeCommManager._extract_interactive_header(self, data)


REFEREE_DECODER = RefereeCommDecodeProxy()


def cmd_id_name(cmd_id: int) -> str:
    """返回 cmd_id 的枚举名，未知命令保留 UNKNOWN_CMD。"""
    try:
        return MsgID(cmd_id).name
    except ValueError:
        return "UNKNOWN_CMD"


def sub_cmd_id_name(sub_cmd_id: int) -> str:
    """返回 sub_cmd_id 的枚举名，未知子命令保留 UNKNOWN_SUB_CMD。"""
    try:
        return SubCmdID(sub_cmd_id).name
    except ValueError:
        return "UNKNOWN_SUB_CMD"


def interactive_detail_id(sub_cmd_id: int) -> int:
    """把 0x0301 的 sub_cmd_id 映射成离线查看用虚拟消息 ID。"""
    return INTERACTIVE_DETAIL_ID_BASE | (int(sub_cmd_id) & 0xFFFF)


def is_interactive_detail_id(message_id: int) -> bool:
    """判断是否为 0x0301/sub_cmd_id 虚拟消息 ID。"""
    return (int(message_id) & 0xFFFF0000) == INTERACTIVE_DETAIL_ID_BASE


def format_message_id(message_id: int) -> str:
    """格式化真实 cmd_id 或 0x0301/sub_cmd_id 虚拟消息 ID。"""
    if is_interactive_detail_id(message_id):
        sub_cmd_id = int(message_id) & 0xFFFF
        return f"0x0301/0x{sub_cmd_id:04x} ({sub_cmd_id_name(sub_cmd_id)})"
    return f"0x{message_id:04x} ({cmd_id_name(message_id)})"


def get_record_group_id(record) -> int:
    """返回摘要分组 ID，0x0301 按 sub_cmd_id 独立分组。"""
    if record.cmd_id == MsgID.INTERACTIVE_DATA.value:
        sub_cmd_id = getattr(record.message, "sub_cmd_id", None)
        if sub_cmd_id is not None:
            return interactive_detail_id(sub_cmd_id)
    return record.cmd_id


def object_id_name(object_id):
    """把机器人 ID 补充成人可读名称。"""
    if object_id is None:
        return "None"
    try:
        return OBJECT_ID(int(object_id)).name
    except ValueError:
        return "UNKNOWN_OBJECT"


def maybe_extract_full_frame(cmd_id: int, raw: bytes):
    """tx 日志记录完整串口帧，rx 日志通常只记录 data 区；这里自动拆分。"""
    if len(raw) < 9 or raw[0] != 0xA5:
        return raw, {}

    data_length = struct.unpack("<H", raw[1:3])[0]
    total_length = 5 + 2 + data_length + 2
    if len(raw) < total_length:
        return raw, {"frame_parse_warning": f"完整帧长度不足: expected={total_length}, actual={len(raw)}"}

    frame_cmd_id = struct.unpack("<H", raw[5:7])[0]
    if frame_cmd_id != cmd_id:
        return raw, {
            "frame_parse_warning": (
                f"帧内 cmd_id=0x{frame_cmd_id:04x} 与日志 cmd_id=0x{cmd_id:04x} 不一致"
            )
        }

    tail_start = 7 + data_length
    return raw[7:tail_start], {
        "sof": raw[0],
        "data_length": data_length,
        "seq": raw[3],
        "header_crc8": raw[4],
        "frame_cmd_id": frame_cmd_id,
        "frame_crc16": struct.unpack("<H", raw[tail_start : tail_start + 2])[0],
        "frame_total_length": total_length,
    }


def extract_interactive_header(data: bytes):
    """提取 0x0301 交互数据公共头。"""
    sub_cmd_id = struct.unpack("<H", data[0:2])[0] if len(data) >= 2 else None
    sender_id = struct.unpack("<H", data[2:4])[0] if len(data) >= 4 else None
    receiver_id = struct.unpack("<H", data[4:6])[0] if len(data) >= 6 else None
    return sub_cmd_id, sender_id, receiver_id


def decode_message(cmd_id: int, data: bytes):
    """直接复用 referee_comm.py 中主程序使用的解码函数。"""
    return REFEREE_DECODER._decode_message(cmd_id, data)


def parse_log_file(path: str):
    """读取日志文件并返回所有可解析记录。"""
    if not os.path.isabs(path):
        raise LogParseError("请输入 txt 日志文件的绝对路径。")
    if not os.path.exists(path):
        raise LogParseError(f"文件不存在: {path}")

    records = []
    skipped_count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            match = LOG_LINE_RE.match(line)
            if not match:
                skipped_count += 1
                continue

            cmd_id = int(match.group("cmd_id"), 16)
            log_hex = match.group("data")
            parse_error = ""
            frame_meta = {}
            raw = b""
            payload = b""
            try:
                raw = bytes.fromhex(log_hex)
                payload, frame_meta = maybe_extract_full_frame(cmd_id, raw)
                message = decode_message(cmd_id, payload)
            except Exception as exc:
                if not payload:
                    payload = raw if raw else (bytes.fromhex(log_hex) if len(log_hex) % 2 == 0 else b"")
                sub_cmd_id, sender_id, receiver_id = (None, None, None)
                if cmd_id == MsgID.INTERACTIVE_DATA.value:
                    sub_cmd_id, sender_id, receiver_id = extract_interactive_header(payload)
                message = UnknownMessage(
                    cmd_id=cmd_id,
                    raw_data=payload,
                    sub_cmd_id=sub_cmd_id,
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                )
                parse_error = f"line {line_number}: {exc}"

            records.append(
                ParsedRecord(
                    direction=match.group("direction"),
                    timestamp=match.group("timestamp"),
                    frame_index=int(match.group("frame_index")),
                    cmd_id=cmd_id,
                    log_hex=log_hex,
                    payload=payload,
                    message=message,
                    message_name=get_message_name(message),
                    parse_error=parse_error,
                    frame_meta=frame_meta,
                )
            )

    records.sort(key=lambda record: (record.timestamp, record.frame_index))
    return records, skipped_count


def get_message_name(message) -> str:
    """获取消息类名。"""
    if isinstance(message, UnknownMessage):
        return getattr(message, "message_name", "UnknownMessage")
    return message.__class__.__name__


def format_value(value):
    """统一格式化 ctypes、bytes 与基础类型。"""
    if hasattr(value, "_length_") and hasattr(value, "_type_"):
        try:
            value_bytes = bytes(value)
            ascii_text = bytes_to_ascii(value_bytes)
            if ascii_text:
                return f"{value_bytes.hex()} ({ascii_text})"
            return value_bytes.hex()
        except Exception:
            return str(list(value))
    if isinstance(value, (bytes, bytearray)):
        ascii_text = bytes_to_ascii(value)
        if ascii_text:
            return f"{bytes(value).hex()} ({ascii_text})"
        return bytes(value).hex()
    return str(value)


def bytes_to_ascii(value: bytes) -> str:
    """仅在内容可读时显示 ASCII。"""
    if not value:
        return ""
    if all(32 <= byte <= 126 for byte in value):
        return value.decode("ascii", errors="ignore")
    return ""


def extract_struct_lines(struct_obj, struct_cls):
    """展开 ctypes 结构体字段。"""
    lines = []
    for field_desc in getattr(struct_cls, "_fields_", []):
        field_name = field_desc[0]
        field_value = getattr(struct_obj, field_name)
        lines.append(f"{field_name}: {format_value(field_value)}")
    return lines


def format_info_wave_user_data(user_data: bytes):
    """展示发给工程的空中允许发弹量和对方剩余金币。"""
    lines = [f"user_data_len: {len(user_data)}"]
    if len(user_data) != InfoWaveMessage.USER_DATA_SIZE:
        lines.append(
            f"warning: 长度错误，期望 {InfoWaveMessage.USER_DATA_SIZE} 字节"
        )
        lines.append(f"raw: {user_data.hex()}")
        return lines
    aerial_ammo, remaining_coins = struct.unpack("<HH", user_data)
    lines.append(f"enemy_aerial_allowed_ammo: {aerial_ammo}")
    lines.append(f"enemy_remaining_coins: {remaining_coins}")
    return lines


def format_radar_decision_extra(message):
    """补充密钥指令块的人可读含义。"""
    try:
        key_bytes = bytes(message.data.key)
    except Exception:
        return []

    key_cmd = key_bytes[0] if key_bytes else None
    key_text = bytes_to_ascii(key_bytes[1:7]) if len(key_bytes) >= 7 else ""
    key_cmd_name = {
        0: "未解析成功/重置",
        1: "修改密钥指令",
        2: "验证密钥",
    }.get(key_cmd, "未知密钥指令")
    return [
        f"key_cmd: {key_cmd} ({key_cmd_name})",
        f"key_ascii: {key_text or key_bytes[1:7].hex()}",
    ]


def format_message_lines(message, payload: bytes):
    """把消息对象转成详情窗口显示行。"""
    if isinstance(message, InteractiveStructMessage):
        lines = [
            f"sub_cmd_id: 0x{int(message.sub_cmd_id):04x} ({sub_cmd_id_name(message.sub_cmd_id)})",
            f"sender_id: 0x{int(message.sender_id):04x} ({object_id_name(message.sender_id)})",
            f"receiver_id: 0x{int(message.receiver_id):04x} ({object_id_name(message.receiver_id)})",
        ]
        lines.extend(extract_struct_lines(message.data, message.STRUCT_CLASS))
        if isinstance(message, RadarDecisionMessage):
            lines.extend(format_radar_decision_extra(message))
        elif isinstance(message, Radar2SentryMessage):
            flags = int(message.targetable_flags)
            labels = ("hero", "engineer", "standard_3", "standard_4", "sentry")
            lines.extend(
                f"{label}_targetable: {(flags >> bit) & 1}"
                for bit, label in enumerate(labels)
            )
            lines.append(f"prefer_long_hole: {(flags >> 5) & 1}")
            lines.append(f"reserved: 0b{(flags >> 6) & 0x03:02b}")
        return lines

    if isinstance(message, StructureMessage):
        return extract_struct_lines(message.struct_data, message.STRUCT_CLASS)

    if isinstance(message, InfoWaveMessage):
        lines = [
            f"sub_cmd_id: 0x{int(message.sub_cmd_id):04x} ({sub_cmd_id_name(message.sub_cmd_id)})",
            f"sender_id: 0x{int(message.sender_id):04x} ({object_id_name(message.sender_id)})",
            f"receiver_id: 0x{int(message.receiver_id):04x} ({object_id_name(message.receiver_id)})",
        ]
        lines.extend(format_info_wave_user_data(message.user_data))
        return lines

    if isinstance(message, UnknownMessage):
        lines = []
        if message.sub_cmd_id is not None:
            lines.append(f"sub_cmd_id: 0x{int(message.sub_cmd_id):04x} ({sub_cmd_id_name(message.sub_cmd_id)})")
        if message.sender_id is not None:
            lines.append(f"sender_id: 0x{int(message.sender_id):04x} ({object_id_name(message.sender_id)})")
        if message.receiver_id is not None:
            lines.append(f"receiver_id: 0x{int(message.receiver_id):04x} ({object_id_name(message.receiver_id)})")
        lines.append(f"raw_data_len: {len(payload)}")
        lines.append(f"raw_data: {payload.hex()}")
        return lines

    return [f"raw_data_len: {len(payload)}", f"raw_data: {payload.hex()}"]


def format_record_text(record: ParsedRecord):
    """生成单个包的完整文本。"""
    lines = [
        f"direction: {record.direction}",
        f"timestamp: {record.timestamp}",
        f"frame_id: frame_{record.frame_index:06d}",
        f"cmd_id: 0x{record.cmd_id:04x} ({cmd_id_name(record.cmd_id)})",
        f"class: {record.message_name}",
    ]
    if record.frame_meta:
        for key, value in record.frame_meta.items():
            if isinstance(value, int):
                if key in {"sof", "header_crc8"}:
                    lines.append(f"{key}: 0x{value:02x}")
                elif "crc16" in key or key == "frame_cmd_id":
                    lines.append(f"{key}: 0x{value:04x}")
                else:
                    lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
    if record.parse_error:
        lines.append(f"parse_error: {record.parse_error}")
    lines.extend(format_message_lines(record.message, record.payload))
    return "\n".join(lines)


def normalize_search_text(text: str) -> str:
    """归一化搜索文本，兼容 = / : / 大小写 差异。"""
    normalized = text.lower()
    normalized = normalized.replace("=", ":")
    normalized = re.sub(r"\s*:\s*", ":", normalized)
    return " ".join(normalized.split())


def build_record_search_text(record: ParsedRecord) -> str:
    """构造单条记录的可搜索文本。"""
    text = format_record_text(record)
    aliases = [
        record.message_name,
        cmd_id_name(record.cmd_id),
        f"0x{record.cmd_id:04x}",
        f"frame_{record.frame_index:06d}",
    ]
    return "\n".join([text, *aliases])


def build_record_search_fields(record: ParsedRecord) -> dict[str, list[str]]:
    """提取记录中的字段键值，供 field:value 查询精确过滤。"""
    fields = defaultdict(list)
    for line in build_record_search_text(record).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        normalized_key = normalize_search_text(key)
        normalized_value = normalize_search_text(value)
        if normalized_key:
            fields[normalized_key].append(normalized_value)
    return fields


def record_matches_query(record: ParsedRecord, query: str) -> bool:
    """判断单条记录是否匹配搜索关键字，空查询匹配全部。"""
    tokens = query_tokens(query)
    if not tokens:
        return True

    search_fields = build_record_search_fields(record)
    searchable_text = normalize_search_text(build_record_search_text(record))
    for token in tokens:
        if ":" in token:
            field_name, _, expected_value = token.partition(":")
            field_values = search_fields.get(field_name, [])
            if not field_values:
                return False
            if expected_value and not any(expected_value in field_value for field_value in field_values):
                return False
            continue
        if token not in searchable_text:
            return False
    return True


def query_tokens(query: str):
    """拆分搜索 token。"""
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return []
    return normalized_query.split()


def build_highlight_patterns(query: str):
    """把搜索 token 转成原始文本高亮用正则。"""
    patterns = []
    for token in query_tokens(query):
        if ":" in token:
            parts = [re.escape(part) for part in token.split(":") if part]
            if not parts:
                continue
            pattern = r"\s*[:=]\s*".join(parts)
        else:
            pattern = re.escape(token)
        patterns.append(re.compile(pattern, flags=re.IGNORECASE))
    return patterns


def highlight_text(text: str, query: str, is_current: bool = False) -> str:
    """生成带高亮的 HTML 文本。"""
    patterns = build_highlight_patterns(query)
    if not patterns:
        return f"<pre style='margin:0; font-family:Consolas; font-size:10pt;'>{escape(text)}</pre>"

    ranges = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            ranges.append((match.start(), match.end()))

    if not ranges:
        return f"<pre style='margin:0; font-family:Consolas; font-size:10pt;'>{escape(text)}</pre>"

    ranges.sort()
    merged = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)

    highlight_color = "#ffb74d" if is_current else "#fff59d"
    parts = []
    last_index = 0
    for start, end in merged:
        parts.append(escape(text[last_index:start]))
        matched_text = escape(text[start:end])
        parts.append(f"<span style='background:{highlight_color};'>{matched_text}</span>")
        last_index = end
    parts.append(escape(text[last_index:]))
    html = "".join(parts)
    return f"<pre style='margin:0; font-family:Consolas; font-size:10pt;'>{html}</pre>"


def clear_layout(layout):
    """清空布局。"""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


class PacketBlock(QFrame):
    """详情窗口中的单包分块。"""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.raw_text = text
        self.setFrameShape(QFrame.StyledPanel)
        self._set_frame_style(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        self.label = QLabel()
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setFont(QFont("Consolas", 10))
        self.label.setWordWrap(False)
        self.label.setTextFormat(Qt.RichText)
        layout.addWidget(self.label)
        self.update_display("")

    def _set_frame_style(self, is_current: bool):
        border_color = "#fb8c00" if is_current else "#bdbdbd"
        background = "#fff3e0" if is_current else "#fafafa"
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {border_color}; border-radius: 6px; background: {background}; }}"
        )

    def update_display(self, query: str, is_current: bool = False):
        """刷新文本高亮和当前项样式。"""
        self.label.setText(highlight_text(self.raw_text, query, is_current=is_current))
        self._set_frame_style(is_current)


class DetailWindow(QMainWindow):
    """按 timestamp 升序展示某类消息的所有包。"""

    def __init__(self, title: str, records, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 860)
        self.records = sorted(records, key=lambda item: (item.timestamp, item.frame_index))
        self.packet_blocks = []
        self.matched_blocks = []
        self.current_match_index = -1

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索当前详情，例如: game_progress=4 或 sender_id=109")
        self.search_edit.textChanged.connect(self.apply_filter)
        self.search_edit.returnPressed.connect(self.goto_next_match)
        self.search_summary_label = QLabel("显示全部记录")
        self.prev_btn = QPushButton("上一条")
        self.prev_btn.clicked.connect(self.goto_prev_match)
        self.next_btn = QPushButton("下一条")
        self.next_btn.clicked.connect(self.goto_next_match)
        search_layout.addWidget(QLabel("搜索"))
        search_layout.addWidget(self.search_edit, 1)
        search_layout.addWidget(self.prev_btn)
        search_layout.addWidget(self.next_btn)
        search_layout.addWidget(self.search_summary_label)
        root_layout.addLayout(search_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        root_layout.addWidget(self.scroll_area, 1)

        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)

        for record in self.records:
            block = PacketBlock(format_record_text(record), self)
            self.packet_blocks.append((record, block))
            self.layout.addWidget(block)
        self.layout.addStretch()
        self.scroll_area.setWidget(container)
        self.apply_filter("")

    def apply_filter(self, query: str):
        """按搜索关键字高亮详情内的记录，但不隐藏任何记录。"""
        self.matched_blocks = []
        for record, block in self.packet_blocks:
            matched = record_matches_query(record, query)
            block.update_display(query, is_current=False)
            if matched:
                self.matched_blocks.append(block)

        if query.strip():
            if self.matched_blocks:
                self.current_match_index = min(self.current_match_index, len(self.matched_blocks) - 1)
                if self.current_match_index < 0:
                    self.current_match_index = 0
                self._focus_current_match()
                self.search_summary_label.setText(
                    f"匹配 {len(self.matched_blocks)} / {len(self.records)} | "
                    f"第 {self.current_match_index + 1} 条"
                )
            else:
                self.current_match_index = -1
                self.search_summary_label.setText(f"匹配 0 / {len(self.records)}")
        else:
            self.current_match_index = -1
            self.search_summary_label.setText(f"显示全部 {len(self.records)}")
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        """刷新上一条/下一条按钮状态。"""
        enabled = len(self.matched_blocks) > 0 and bool(self.search_edit.text().strip())
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)

    def _focus_current_match(self):
        """把当前匹配项滚动到可视区域，并更新样式。"""
        for _, block in self.packet_blocks:
            block.update_display(self.search_edit.text(), is_current=False)
        for index, block in enumerate(self.matched_blocks):
            block.update_display(self.search_edit.text(), is_current=index == self.current_match_index)
        if 0 <= self.current_match_index < len(self.matched_blocks):
            self.scroll_area.ensureWidgetVisible(self.matched_blocks[self.current_match_index], 20, 20)

    def goto_next_match(self):
        """跳到下一条匹配记录。"""
        if not self.matched_blocks:
            return
        self.current_match_index = (self.current_match_index + 1) % len(self.matched_blocks)
        self._focus_current_match()
        self.search_summary_label.setText(
            f"匹配 {len(self.matched_blocks)} / {len(self.records)} | 第 {self.current_match_index + 1} 条"
        )

    def goto_prev_match(self):
        """跳到上一条匹配记录。"""
        if not self.matched_blocks:
            return
        self.current_match_index = (self.current_match_index - 1) % len(self.matched_blocks)
        self._focus_current_match()
        self.search_summary_label.setText(
            f"匹配 {len(self.matched_blocks)} / {len(self.records)} | 第 {self.current_match_index + 1} 条"
        )


class SummaryRow(QFrame):
    """主窗口中的单个消息类型摘要行。"""

    def __init__(self, message_id: int, records, detail_callback, match_count=0, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        border_color = "#fb8c00" if match_count > 0 else "#9e9e9e"
        background = "#fff8e1" if match_count > 0 else "white"
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {border_color}; border-radius: 6px; background: {background}; }}"
        )
        latest = records[-1]
        class_counter = Counter(record.message_name for record in records)
        common_class = class_counter.most_common(1)[0][0]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        text = (
            f"message: {format_message_id(message_id)} | "
            f"count: {len(records)} | "
            f"class: {common_class} | "
            f"latest: {latest.timestamp}"
        )
        if match_count > 0:
            text += f" | matched: {match_count}"
        label = QLabel(text)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setFont(QFont("Consolas", 10))
        layout.addWidget(label, 1)

        detail_btn = QPushButton("详情")
        detail_btn.clicked.connect(lambda: detail_callback(message_id, records))
        layout.addWidget(detail_btn)


class LogsParserWindow(QMainWindow):
    """独立日志解析 UI。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("裁判串口日志解析")
        self.resize(1280, 860)
        self.records = []
        self.skipped_count = 0
        self.current_direction = "rx"
        self.detail_windows = []

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("输入 referee log txt 文件绝对路径")
        self.path_edit.returnPressed.connect(self.load_path)

        browse_btn = QPushButton("选择")
        browse_btn.clicked.connect(self.browse_path)

        load_btn = QPushButton("读取")
        load_btn.clicked.connect(self.load_path)

        self.direction_btn = QPushButton("当前: 接收数据")
        self.direction_btn.clicked.connect(self.toggle_direction)

        path_layout.addWidget(QLabel("日志路径"))
        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(browse_btn)
        path_layout.addWidget(load_btn)
        path_layout.addWidget(self.direction_btn)
        root_layout.addLayout(path_layout)

        filter_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索消息内容，例如: GAME_STATUS game_progress=4")
        self.search_edit.textChanged.connect(lambda _text: self.render_summary())
        self.search_status_label = QLabel("未启用筛选")
        filter_layout.addWidget(QLabel("内容搜索"))
        filter_layout.addWidget(self.search_edit, 1)
        filter_layout.addWidget(self.search_status_label)
        root_layout.addLayout(filter_layout)

        self.summary_label = QLabel("未加载日志")
        self.summary_label.setStyleSheet("font-weight: bold;")
        root_layout.addWidget(self.summary_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        root_layout.addWidget(self.scroll_area, 1)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(8)
        self.scroll_area.setWidget(self.rows_container)

    def browse_path(self):
        """选择日志文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择裁判日志",
            os.path.join(PROJECT_ROOT, "logs", "referee"),
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)
            self.load_path()

    def load_path(self):
        """读取并解析当前路径。"""
        path = self.path_edit.text().strip()
        try:
            self.records, self.skipped_count = parse_log_file(path)
        except LogParseError as exc:
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "解析失败", str(exc))
            return
        self.render_summary()

    def toggle_direction(self):
        """切换接收/发送数据视图。"""
        self.current_direction = "tx" if self.current_direction == "rx" else "rx"
        self.direction_btn.setText("当前: 发送数据" if self.current_direction == "tx" else "当前: 接收数据")
        self.render_summary()

    def render_summary(self):
        """按消息类型分组刷新摘要区，搜索仅高亮命中分组，不筛选。"""
        clear_layout(self.rows_layout)

        query = self.search_edit.text().strip()
        direction_records = [
            record for record in self.records if record.direction == self.current_direction
        ]
        grouped = defaultdict(list)
        match_counts = defaultdict(int)
        for record in direction_records:
            grouped[get_record_group_id(record)].append(record)
            if query and record_matches_query(record, query):
                match_counts[get_record_group_id(record)] += 1

        total = len(self.records)
        rx_count = sum(1 for record in self.records if record.direction == "rx")
        tx_count = sum(1 for record in self.records if record.direction == "tx")
        matched_records = sum(match_counts.values())
        matched_groups = sum(1 for count in match_counts.values() if count > 0)
        self.summary_label.setText(
            f"总包数: {total} | rx: {rx_count} | tx: {tx_count} | "
            f"当前方向: {self.current_direction} {len(direction_records)} 包 | "
            f"跳过行: {self.skipped_count}"
        )
        if query:
            self.search_status_label.setText(
                f"关键字: {query} | 匹配 {matched_records} 包 | 分组 {matched_groups}"
            )
        else:
            self.search_status_label.setText("未启用搜索")

        if not grouped:
            empty_label = QLabel("当前方向没有可展示数据")
            empty_label.setAlignment(Qt.AlignCenter)
            self.rows_layout.addWidget(empty_label)
            self.rows_layout.addStretch()
            return

        for message_id in sorted(grouped, key=lambda item: (-len(grouped[item]), item)):
            self.rows_layout.addWidget(
                SummaryRow(
                    message_id,
                    grouped[message_id],
                    self.open_detail_window,
                    match_count=match_counts.get(message_id, 0),
                    parent=self,
                )
            )
        self.rows_layout.addStretch()

    def open_detail_window(self, message_id: int, records):
        """打开某个消息类型的详情窗口。"""
        title = f"{self.current_direction} {format_message_id(message_id)} 详情"
        window = DetailWindow(title, records, self)
        self.detail_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()


def main():
    """程序入口。"""
    app = QApplication(sys.argv)
    window = LogsParserWindow()
    if len(sys.argv) >= 2:
        window.path_edit.setText(os.path.abspath(sys.argv[1]))
        window.load_path()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
