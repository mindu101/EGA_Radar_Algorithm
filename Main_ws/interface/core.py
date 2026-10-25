import sys
import os
import signal
import re
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .interfactive_display import ZoomableDraggableLabel
import numpy as np
import cv2
import logging
import yaml
from driver.referee.referee_comm import RefereeCommManager
from driver.referee.serial_protocol import (
    MsgID,
    StructureMessage,
    InteractiveStructMessage,
    UnknownMessage,
)
from utils.runtime_config import (
    get_extrinsics_import_path,
    get_params_config_path,
    load_params_config,
)
from utils.async_camera_recorder import AsyncCameraRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _scrollable_panel(widget):
    """让长控制面板可以滚动，避免子控件最小尺寸撑大整个窗口。"""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    scroll.setMinimumSize(0, 0)
    return scroll


def _fit_window_to_screen(window, preferred_size):
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        window.resize(preferred_size)
        return
    available = screen.availableGeometry()
    size = preferred_size.boundedTo(available.size() * 0.9)
    window.resize(size)
    window.move(available.center() - QPoint(size.width() // 2, size.height() // 2))


def get_extrinsics_record_dir():
    """返回 UI 外参记录目录。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "recorder", "extrinstics")


def get_latest_extrinsics_path():
    """返回“导入最近外参”实际读取路径，测试模式可由启动器覆盖。"""
    return get_extrinsics_import_path(get_extrinsics_record_dir())


def format_extrinsics_record(rotation_matrix, translation_vector, residual=None, source="pnp", timestamp=None):
    """生成外参记录文本。"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))

    lines = [
        f"timestamp: {timestamp}",
        f"source: {source}",
    ]
    if residual is not None:
        lines.append(f"residual: {float(residual):.6f}")
    lines.append("")
    lines.append("rotation_matrix:")
    lines.extend(" ".join(f"{value:.10f}" for value in row) for row in rotation_matrix)
    lines.append("")
    lines.append("translation_vector:")
    lines.extend(" ".join(f"{value:.10f}" for value in row) for row in translation_vector)
    lines.append("")
    lines.append("extrinsic_3x4:")
    lines.extend(" ".join(f"{value:.10f}" for value in row) for row in extrinsic_matrix)
    lines.append("")
    return "\n".join(lines)


def load_extrinsics_record(path):
    """从保存的外参文本中读取 R 和 t。"""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    def extract_section(section_name, next_section_names):
        pattern = rf"{re.escape(section_name)}:\s*\n(.*?)(?=\n(?:{'|'.join(re.escape(name) for name in next_section_names)}):|\Z)"
        match = re.search(pattern, text, flags=re.S)
        if not match:
            return []
        return [
            float(item)
            for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))
        ]

    rotation_numbers = extract_section("rotation_matrix", ["translation_vector", "extrinsic_3x4"])
    translation_numbers = extract_section("translation_vector", ["extrinsic_3x4"])
    if len(rotation_numbers) != 9:
        raise ValueError("最近外参记录中的 rotation_matrix 不完整。")
    if len(translation_numbers) != 3:
        raise ValueError("最近外参记录中的 translation_vector 不完整。")

    rotation_matrix = np.array(rotation_numbers, dtype=np.float64).reshape(3, 3)
    translation_vector = np.array(translation_numbers, dtype=np.float64).reshape(3, 1)
    return rotation_matrix, translation_vector, text


##########
class SerialMessageColumnWidget(QFrame):
    """单条历史消息列。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(260)
        self.setStyleSheet(
            "QFrame { border: 1px solid #bdbdbd; border-radius: 4px; background: #fafafa; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.content_label = QLabel("")
        self.content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.content_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content_label.setFont(QFont("Consolas", 10))
        self.content_label.setWordWrap(False)
        layout.addWidget(self.content_label)

    def set_text(self, text: str):
        self.content_label.setText(text or "")


class SerialMessageRowWidget(QFrame):
    """单个 cmd_id 的历史分块。"""

    def __init__(self, history_len: int, parent=None):
        super().__init__(parent)
        self.history_len = max(1, int(history_len))
        self.column_widgets = []
        self.header_text = ""
        self.column_texts = []

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { border: 1px solid #9e9e9e; border-radius: 6px; background: white; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        header_layout = QHBoxLayout()
        self.header_label = QLabel("")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 20px;")
        self.copy_btn = QPushButton("拷贝")
        self.copy_btn.clicked.connect(self.copy_row_content)
        header_layout.addWidget(self.header_label)
        header_layout.addWidget(self.copy_btn)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self.column_layout = QHBoxLayout()
        self.column_layout.setSpacing(8)
        layout.addLayout(self.column_layout)
        self.set_history_len(self.history_len)

    def set_history_len(self, history_len: int):
        history_len = max(1, int(history_len))
        if history_len == self.history_len and self.column_widgets:
            return
        self.history_len = history_len

        while self.column_layout.count():
            item = self.column_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.column_widgets.clear()

        for _ in range(self.history_len):
            column_widget = SerialMessageColumnWidget(self)
            self.column_layout.addWidget(column_widget, 1)
            self.column_widgets.append(column_widget)

    def update_content(self, header_text: str, column_texts):
        self.header_text = header_text or ""
        self.column_texts = list(column_texts or [])
        self.header_label.setText(header_text)
        self.set_history_len(max(1, len(column_texts)))
        for index, column_widget in enumerate(self.column_widgets):
            text = column_texts[index] if index < len(column_texts) else ""
            column_widget.set_text(text)

    def copy_row_content(self):
        """按约定格式拷贝当前行分块内容。"""
        sections = [self.header_text]
        for text in self.column_texts:
            if not text:
                continue
            sections.append("")
            sections.append(text)
        QApplication.clipboard().setText("\n".join(sections))


class SerialReceiveDebugWindow(QMainWindow):
    """串口收发数据调试窗口。"""

    def __init__(
        self,
        referee,
        parent=None,
        window_title: str = "串口接收数据",
        snapshot_method_name: str = "get_msg_queue_snapshot",
    ):
        super().__init__(parent)
        self.referee = referee
        ########################################################################
        self.snapshot_method_name = snapshot_method_name
        self.window_title = window_title
        ########################################################################
        self.visible_cmd_limit = 8
        self.main_cmd_expanded = False
        self._syncing_checkbox_state = False
        self._manual_cmd_selection = set()
        self._cmd_checkboxes = {}
        self._row_widgets = {}
        self._last_main_display_ids = []
        self._last_interactive_display_ids = []
        self._last_selected_cmd_ids = []
        self.current_history_len = 3
        self.current_snapshot = None

        self.setWindowTitle(self.window_title)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(1480, 920)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_view)

        self._init_ui()
        self.refresh_view()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        filter_group = QGroupBox("筛选")
        filter_layout = QVBoxLayout(filter_group)

        control_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.clear_all_btn = QPushButton("全不选")
        self.main_expand_btn = QPushButton("展开全部")
        self.snapshot_hint_label = QLabel("刷新方式: 10Hz 快照")
        self.snapshot_hint_label.setStyleSheet("color: #666666;")

        self.select_all_btn.clicked.connect(self.select_all_cmd_ids)
        self.clear_all_btn.clicked.connect(self.clear_all_cmd_ids)
        self.main_expand_btn.clicked.connect(self.toggle_main_cmd_expand)

        control_layout.addWidget(self.select_all_btn)
        control_layout.addWidget(self.clear_all_btn)
        control_layout.addWidget(self.main_expand_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.snapshot_hint_label)
        filter_layout.addLayout(control_layout)

        main_header_layout = QHBoxLayout()
        self.main_cmd_summary_label = QLabel("")
        self.main_cmd_summary_label.setStyleSheet("font-weight: bold;")
        main_header_layout.addWidget(QLabel("消息类型"))
        main_header_layout.addStretch()
        main_header_layout.addWidget(self.main_cmd_summary_label)
        filter_layout.addLayout(main_header_layout)

        self.main_cmd_container = QWidget()
        self.main_cmd_layout = QGridLayout(self.main_cmd_container)
        self.main_cmd_layout.setContentsMargins(0, 0, 0, 0)
        self.main_cmd_layout.setHorizontalSpacing(8)
        self.main_cmd_layout.setVerticalSpacing(6)
        filter_layout.addWidget(self.main_cmd_container)

        interactive_header_layout = QHBoxLayout()
        self.interactive_summary_label = QLabel(
            "INTERACTIVE_DATA 独立筛选区（sub_cmd_id 细分接口预留）"
        )
        self.interactive_summary_label.setStyleSheet("font-weight: bold;")
        interactive_header_layout.addWidget(self.interactive_summary_label)
        interactive_header_layout.addStretch()
        filter_layout.addLayout(interactive_header_layout)

        self.interactive_cmd_container = QWidget()
        self.interactive_cmd_layout = QHBoxLayout(self.interactive_cmd_container)
        self.interactive_cmd_layout.setContentsMargins(0, 0, 0, 0)
        self.interactive_cmd_layout.setSpacing(8)
        filter_layout.addWidget(self.interactive_cmd_container)

        layout.addWidget(filter_group)

        self.content_scroll_area = QScrollArea()
        self.content_scroll_area.setWidgetResizable(True)
        self.content_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.content_placeholder = QLabel("当前没有勾选任何消息。")
        self.content_placeholder.setStyleSheet("color: #666666; padding: 12px;")
        self.content_scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.content_scroll_area, 1)

    def start_updates(self):
        """启动窗口刷新。"""
        self.refresh_view()
        if not self.refresh_timer.isActive():
            self.refresh_timer.start()

    def stop_updates(self):
        """停止窗口刷新。"""
        if self.refresh_timer.isActive():
            self.refresh_timer.stop()

    def closeEvent(self, event):
        """关闭窗口时停止刷新，但保留界面状态与历史。"""
        self.stop_updates()
        super().closeEvent(event)

    def refresh_view(self):
        """拉取串口缓存快照并刷新筛选区与消息分块。"""
        snapshot = self._get_snapshot()
        self.current_snapshot = snapshot
        self.current_history_len = max(1, snapshot.get("msg_queue_len", 3))
        self._sync_checkboxes(snapshot)
        self._render_filter_sections(snapshot)
        self._render_message_rows(snapshot)

    def _get_snapshot(self):
        """优先使用串口管理器的快照接口，避免 UI 线程直接遍历共享队列。"""
        ########################################################################
        snapshot_func = getattr(self.referee, self.snapshot_method_name, None)
        if snapshot_func is not None:
            return snapshot_func()
        if self.snapshot_method_name != "get_msg_queue_snapshot":
            return {
                "msg_queue_len": 3,
                "msg_message_queue": {},
                "msg_hit_count": {},
                "interactive_sub_hit_count": {},
                "msg_display_name": {},
                ########################################################################
                "msg_filter_group": {},
                "msg_filter_display_name": {},
                ########################################################################
            }
        ########################################################################

        with self.referee.lock:
            return {
                "msg_queue_len": getattr(self.referee, "msg_queue_len", 3),
                "msg_message_queue": {
                    cmd_id: list(queue)
                    for cmd_id, queue in getattr(
                        self.referee, "msg_message_queue", {}
                    ).items()
                },
                "msg_hit_count": dict(getattr(self.referee, "msg_hit_count", {})),
                "interactive_sub_hit_count": dict(
                    getattr(self.referee, "interactive_sub_hit_count", {})
                ),
                ########################################################################
                "msg_display_name": {},
                "msg_filter_group": {},
                "msg_filter_display_name": {},
                ########################################################################
            }

    def _sync_checkboxes(self, snapshot):
        """同步复选框对象、文本与默认勾选状态。"""
        ########################################################################
        all_cmd_ids = self._get_filter_ids(snapshot)
        ########################################################################
        self._syncing_checkbox_state = True
        try:
            for cmd_id in all_cmd_ids:
                checkbox = self._ensure_checkbox(cmd_id)
                ########################################################################
                hit_count = self._get_filter_hit_count(snapshot, cmd_id)
                ########################################################################
                checkbox.setText(self._build_cmd_checkbox_text(cmd_id, hit_count))
                if cmd_id not in self._manual_cmd_selection:
                    checkbox.setChecked(hit_count > 0)
        finally:
            self._syncing_checkbox_state = False

    def _render_filter_sections(self, snapshot):
        """按折叠/展开状态刷新筛选区，仅在显示集合变化时重排布局。"""
        main_cmd_ids = self._get_main_cmd_ids(snapshot)
        interactive_cmd_ids = self._get_interactive_cmd_ids(snapshot)
        main_display_ids = self._get_main_display_ids(snapshot)

        if main_display_ids != self._last_main_display_ids:
            self._rebuild_main_filter_layout(main_display_ids)
            self._last_main_display_ids = list(main_display_ids)
        if interactive_cmd_ids != self._last_interactive_display_ids:
            self._rebuild_interactive_filter_layout(interactive_cmd_ids)
            self._last_interactive_display_ids = list(interactive_cmd_ids)

        self.main_expand_btn.setText("收起" if self.main_cmd_expanded else "展开全部")
        self.main_cmd_summary_label.setText(
            f"显示 {len(main_display_ids)}/{len(main_cmd_ids)} 项"
        )
        interactive_hit = snapshot.get("msg_hit_count", {}).get(
            MsgID.INTERACTIVE_DATA.value, 0
        )
        sub_hit_count = len(snapshot.get("interactive_sub_hit_count", {}))
        self.interactive_summary_label.setText(
            "INTERACTIVE_DATA 独立筛选区（sub_cmd_id 细分接口预留）"
            f" | hit: {interactive_hit:<5} | sub_cmd_id: {sub_hit_count}"
        )

    def _rebuild_main_filter_layout(self, main_display_ids):
        """重建主命令筛选区布局。"""
        self._clear_layout(self.main_cmd_layout)
        for index, cmd_id in enumerate(main_display_ids):
            self.main_cmd_layout.addWidget(
                self._cmd_checkboxes[cmd_id], index // 4, index % 4
            )

    def _rebuild_interactive_filter_layout(self, interactive_cmd_ids):
        """重建交互命令筛选区布局。"""
        self._clear_layout(self.interactive_cmd_layout)
        for cmd_id in interactive_cmd_ids:
            self.interactive_cmd_layout.addWidget(self._cmd_checkboxes[cmd_id])
        self.interactive_cmd_layout.addStretch()

    def _render_message_rows(self, snapshot):
        """根据勾选项刷新消息展示分块，仅在集合变化时重排行布局。"""
        selected_cmd_ids = self._get_sorted_selected_cmd_ids()
        if selected_cmd_ids != self._last_selected_cmd_ids:
            self._rebuild_message_rows_layout(selected_cmd_ids)
            self._last_selected_cmd_ids = list(selected_cmd_ids)

        history_map = snapshot.get("msg_message_queue", {})
        hit_count_map = snapshot.get("msg_hit_count", {})
        for cmd_id in selected_cmd_ids:
            row_widget = self._row_widgets.get(cmd_id)
            if row_widget is None:
                row_widget = SerialMessageRowWidget(self.current_history_len, self)
                self._row_widgets[cmd_id] = row_widget
            records = history_map.get(cmd_id, [])
            row_widget.update_content(
                self._build_row_header(cmd_id, records, hit_count_map.get(cmd_id, 0)),
                self._build_column_texts(records),
            )

    def _get_sorted_selected_cmd_ids(self):
        """获取排序后的勾选 cmd_id 列表。"""
        return sorted(
            self._get_selected_cmd_ids(),
            key=lambda cmd_id: (
                1 if cmd_id == MsgID.INTERACTIVE_DATA.value else 0,
                cmd_id,
            ),
        )

    def _rebuild_message_rows_layout(self, selected_cmd_ids):
        """仅在筛选集合变化时重建内容布局，并尽量保留滚动位置。"""
        v_scroll = self.content_scroll_area.verticalScrollBar()
        h_scroll = self.content_scroll_area.horizontalScrollBar()
        old_v_value = v_scroll.value()
        old_h_value = h_scroll.value()

        self._clear_layout(self.content_layout)
        if not selected_cmd_ids:
            self.content_layout.addWidget(self.content_placeholder)
            self.content_layout.addStretch()
        else:
            for cmd_id in selected_cmd_ids:
                row_widget = self._row_widgets.get(cmd_id)
                if row_widget is None:
                    row_widget = SerialMessageRowWidget(self.current_history_len, self)
                    self._row_widgets[cmd_id] = row_widget
                self.content_layout.addWidget(row_widget)
            self.content_layout.addStretch()

        QTimer.singleShot(
            0,
            lambda: self._restore_scroll_position(old_v_value, old_h_value),
        )

    def _restore_scroll_position(self, vertical_value: int, horizontal_value: int):
        """布局重排后恢复滚动条位置，减少视觉跳动。"""
        self.content_scroll_area.verticalScrollBar().setValue(vertical_value)
        self.content_scroll_area.horizontalScrollBar().setValue(horizontal_value)

    def _build_row_header(self, cmd_id: int, records, hit_count: int):
        """生成每个 cmd_id 分块的标题。"""
        latest_message = records[-1]["message"] if records else None
        class_name = self._get_message_class_name(latest_message)
        ########################################################################
        title = self._format_msg_id(cmd_id)
        return f"{title} | hit: {hit_count:<5} | class: {class_name}"
        ########################################################################

    def _build_column_texts(self, records):
        """按最新优先生成固定列数的显示文本。"""
        ordered_records = list(reversed(records))
        column_texts = []
        for index in range(self.current_history_len):
            if index < len(ordered_records):
                column_texts.append(self._format_record_text(ordered_records[index]))
            else:
                column_texts.append("")
        return column_texts

    def _format_record_text(self, record):
        """格式化单条历史记录。"""
        timestamp = record.get("timestamp", "")
        message = record.get("message")
        raw_data = record.get("raw_data", b"")
        lines = [f"timestamp: {timestamp}"]
        ########################################################################
        transport = record.get("transport", "")
        packet_name = record.get("packet_name", "")
        if transport:
            lines.append(f"transport: {transport}")
        if packet_name:
            lines.append(f"packet: {packet_name}")
        ########################################################################
        lines.extend(self._format_message_lines(message, raw_data))
        return "\n".join(lines)

    def _format_message_lines(self, message, raw_data: bytes):
        """按消息类型提取可视化字段。"""
        if isinstance(message, InteractiveStructMessage):
            lines = [
                f"sub_cmd_id: {self._format_hex_value(message.sub_cmd_id)}",
                f"sender_id: {self._format_hex_value(message.sender_id)}",
                f"receiver_id: {self._format_hex_value(message.receiver_id)}",
            ]
            lines.extend(self._extract_struct_lines(message.data, message.STRUCT_CLASS))
            return lines

        if isinstance(message, StructureMessage):
            return self._extract_struct_lines(message.struct_data, message.STRUCT_CLASS)

        if isinstance(message, UnknownMessage):
            lines = []
            if message.sub_cmd_id is not None:
                lines.append(f"sub_cmd_id: {self._format_hex_value(message.sub_cmd_id)}")
            if message.sender_id is not None:
                lines.append(f"sender_id: {self._format_hex_value(message.sender_id)}")
            if message.receiver_id is not None:
                lines.append(f"receiver_id: {self._format_hex_value(message.receiver_id)}")
            lines.append(f"raw_data: {message.raw_data.hex()}")
            return lines

        ########################################################################
        if hasattr(message, "STRUCT_CLASS") and hasattr(message, "data"):
            return self._extract_struct_lines(message.data, message.STRUCT_CLASS)

        if hasattr(message, "__dataclass_fields__"):
            return [
                f"{field_name}: {self._format_value(getattr(message, field_name))}"
                for field_name in message.__dataclass_fields__
            ]
        ########################################################################

        return [f"raw_data: {bytes(raw_data).hex()}"]

    def _extract_struct_lines(self, struct_obj, struct_cls):
        """展开 ctypes 结构体字段。"""
        if struct_obj is None or struct_cls is None:
            return []
        lines = []
        for field in getattr(struct_cls, "_fields_", []):
            field_name = field[0]
            field_value = getattr(struct_obj, field_name)
            lines.append(f"{field_name}: {self._format_value(field_value)}")
        return lines

    def _format_value(self, value):
        """统一格式化字段值。"""
        if isinstance(value, np.generic):
            value = value.item()
        ########################################################################
        if hasattr(value, "_length_") and hasattr(value, "_type_"):
            try:
                return bytes(value).hex()
            except Exception:
                pass
        ########################################################################
        if isinstance(value, (bytes, bytearray)):
            return value.hex()
        if isinstance(value, (int, float, str)):
            return str(value)
        return str(value)

    def _format_hex_value(self, value):
        """将命令与对象 ID 统一格式化为十六进制。"""
        if value is None:
            return "None"
        if isinstance(value, np.generic):
            value = value.item()
        return f"0x{int(value):04x}"

    def _get_message_class_name(self, message):
        """获取消息类名，未知消息保留占位名称。"""
        if message is None:
            return "-"
        if isinstance(message, UnknownMessage):
            return getattr(message, "message_name", "UnknownMessage")
        return message.__class__.__name__

    def _ensure_checkbox(self, cmd_id: int):
        """按需创建筛选复选框。"""
        checkbox = self._cmd_checkboxes.get(cmd_id)
        if checkbox is not None:
            return checkbox

        checkbox = QCheckBox()
        checkbox.toggled.connect(
            lambda checked, current_cmd_id=cmd_id: self._on_checkbox_toggled(
                current_cmd_id, checked
            )
        )
        self._cmd_checkboxes[cmd_id] = checkbox
        return checkbox

    def _on_checkbox_toggled(self, cmd_id: int, checked: bool):
        """记录用户手动筛选行为。"""
        if self._syncing_checkbox_state:
            return
        self._manual_cmd_selection.add(cmd_id)
        self._render_message_rows(self.current_snapshot or self._get_snapshot())

    def select_all_cmd_ids(self):
        """全选所有筛选项。"""
        self._manual_cmd_selection.update(self._cmd_checkboxes.keys())
        self._syncing_checkbox_state = True
        try:
            for checkbox in self._cmd_checkboxes.values():
                checkbox.setChecked(True)
        finally:
            self._syncing_checkbox_state = False
        self.refresh_view()

    def clear_all_cmd_ids(self):
        """取消勾选所有筛选项。"""
        self._manual_cmd_selection.update(self._cmd_checkboxes.keys())
        self._syncing_checkbox_state = True
        try:
            for checkbox in self._cmd_checkboxes.values():
                checkbox.setChecked(False)
        finally:
            self._syncing_checkbox_state = False
        self.refresh_view()

    def toggle_main_cmd_expand(self):
        """切换主命令筛选区的折叠与展开。"""
        self.main_cmd_expanded = not self.main_cmd_expanded
        self.refresh_view()

    def _get_selected_cmd_ids(self):
        """读取当前勾选的 cmd_id。"""
        ########################################################################
        selected_cmd_ids = []
        for cmd_id, checkbox in self._cmd_checkboxes.items():
            if checkbox.isChecked():
                selected_cmd_ids.extend(self._get_filter_members(cmd_id))
        return list(dict.fromkeys(selected_cmd_ids))
        ########################################################################

    def _get_main_display_ids(self, snapshot):
        """返回主命令筛选区当前需要展示的 cmd_id。"""
        main_cmd_ids = self._get_main_cmd_ids(snapshot)
        if self.main_cmd_expanded or len(main_cmd_ids) <= self.visible_cmd_limit:
            return sorted(main_cmd_ids)
        hit_count_map = snapshot.get("msg_hit_count", {})
        ########################################################################
        display_ids = sorted(
            main_cmd_ids,
            key=lambda cmd_id: (-self._get_filter_hit_count(snapshot, cmd_id), cmd_id),
        )[: self.visible_cmd_limit]
        for cmd_id in snapshot.get("msg_display_name", {}):
            if cmd_id in main_cmd_ids and cmd_id not in display_ids:
                display_ids = [cmd_id] + display_ids[: self.visible_cmd_limit - 1]
        return sorted(display_ids)
        ########################################################################

    def _get_main_cmd_ids(self, snapshot):
        """获取除 INTERACTIVE_DATA 之外的主命令列表。"""
        official_cmd_ids = [
            msg_id.value for msg_id in MsgID if msg_id != MsgID.INTERACTIVE_DATA
        ]
        ########################################################################
        filter_ids = self._get_filter_ids(snapshot)
        ########################################################################
        extra_cmd_ids = sorted(
            cmd_id
            for cmd_id in filter_ids
            if cmd_id not in official_cmd_ids
            and cmd_id != MsgID.INTERACTIVE_DATA.value
            and not self._is_interactive_detail_id(cmd_id)
        )
        return official_cmd_ids + extra_cmd_ids

    def _get_interactive_cmd_ids(self, snapshot):
        """获取交互命令筛选区当前使用的 cmd_id。"""
        interactive_cmd_ids = []
        all_cmd_ids = self._get_all_cmd_ids(snapshot)
        if MsgID.INTERACTIVE_DATA.value in all_cmd_ids:
            interactive_cmd_ids.append(MsgID.INTERACTIVE_DATA.value)
        interactive_cmd_ids.extend(
            sorted(
                cmd_id
                for cmd_id in all_cmd_ids
                if self._is_interactive_detail_id(cmd_id)
            )
        )
        return interactive_cmd_ids

    def _get_all_cmd_ids(self, snapshot):
        """汇总所有需要展示的 cmd_id，包括运行期出现的未知命令。"""
        official_cmd_ids = [msg_id.value for msg_id in MsgID]
        dynamic_cmd_ids = sorted(
            set(snapshot.get("msg_message_queue", {}).keys())
            | set(snapshot.get("msg_hit_count", {}).keys())
        )
        return list(dict.fromkeys(official_cmd_ids + dynamic_cmd_ids))

    def _build_cmd_checkbox_text(self, cmd_id: int, hit_count: int):
        """构造筛选区复选框文本。"""
        ########################################################################
        display_name = self._get_filter_display_name(cmd_id)
        if display_name:
            return f"{display_name} | hit: {hit_count:<5}"
        ########################################################################
        try:
            cmd_name = MsgID(cmd_id).name
        except ValueError:
            cmd_name = "UNKNOWN_CMD"
        return f"0x{cmd_id:04x} | hit: {hit_count:<5} | {cmd_name}"

    ########################################################################
    def _get_msg_display_name(self, cmd_id: int):
        """读取虚拟消息类型显示名，例如 udp。"""
        snapshot = self.current_snapshot or {}
        return snapshot.get("msg_display_name", {}).get(cmd_id, "")

    def _get_filter_ids(self, snapshot):
        """获取筛选项 ID，属于同一组的 UDP 消息只保留一个筛选项。"""
        all_cmd_ids = self._get_all_cmd_ids(snapshot)
        group_map = snapshot.get("msg_filter_group", {})
        group_representative = {}
        for cmd_id in all_cmd_ids:
            group_name = group_map.get(cmd_id)
            if group_name:
                group_representative[group_name] = min(
                    cmd_id, group_representative.get(group_name, cmd_id)
                )

        filter_ids = []
        for cmd_id in all_cmd_ids:
            group_name = group_map.get(cmd_id)
            if group_name and group_representative.get(group_name) != cmd_id:
                continue
            filter_ids.append(cmd_id)
        return filter_ids

    def _get_filter_members(self, cmd_id: int):
        """把一个筛选项展开成实际需要显示的消息行。"""
        snapshot = self.current_snapshot or {}
        group_map = snapshot.get("msg_filter_group", {})
        group_name = group_map.get(cmd_id)
        if not group_name:
            return [cmd_id]
        return [
            member_id
            for member_id in self._get_all_cmd_ids(snapshot)
            if group_map.get(member_id) == group_name
        ]

    def _get_filter_hit_count(self, snapshot, cmd_id: int):
        """统计筛选项命中次数；UDP 组显示组内总数。"""
        hit_count_map = snapshot.get("msg_hit_count", {})
        group_map = snapshot.get("msg_filter_group", {})
        group_name = group_map.get(cmd_id)
        if not group_name:
            return hit_count_map.get(cmd_id, 0)
        return sum(
            hit_count_map.get(member_id, 0)
            for member_id in self._get_all_cmd_ids(snapshot)
            if group_map.get(member_id) == group_name
        )

    def _get_filter_display_name(self, cmd_id: int):
        """读取筛选项显示名；UDP 子消息在筛选区统一显示为 udp。"""
        snapshot = self.current_snapshot or {}
        group_name = snapshot.get("msg_filter_group", {}).get(cmd_id)
        if group_name:
            return snapshot.get("msg_filter_display_name", {}).get(group_name, group_name)
        return self._get_msg_display_name(cmd_id)

    def _format_msg_id(self, cmd_id: int):
        """格式化真实 cmd_id 或虚拟消息类型。"""
        display_name = self._get_msg_display_name(cmd_id)
        if display_name:
            return display_name
        return f"cmd_id: 0x{cmd_id:04x}"

    @staticmethod
    def _is_interactive_detail_id(cmd_id: int) -> bool:
        return (int(cmd_id) & 0xFFFF0000) == (MsgID.INTERACTIVE_DATA.value << 16)
    ########################################################################

    def _clear_layout(self, layout):
        """清空布局中的控件引用，不直接销毁复用控件。"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
            elif child_layout is not None:
                self._clear_layout(child_layout)


class TimingDebugDetailWindow(QMainWindow):
    """跟踪定位耗时调试详情窗口。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("跟踪定位耗时调试详情")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(980, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        self.copy_btn = QPushButton("拷贝到剪贴板")
        self.copy_btn.clicked.connect(self.copy_detail_text)
        control_layout.addWidget(self.copy_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        pipeline_group = QGroupBox("详细管线流程（模块编号与计时覆盖）")
        pipeline_layout = QVBoxLayout(pipeline_group)
        self.pipeline_text = QTextEdit()
        self.pipeline_text.setReadOnly(True)
        self.pipeline_text.setFont(QFont("Consolas", 10))
        self.pipeline_text.setMinimumHeight(260)
        pipeline_layout.addWidget(self.pipeline_text)
        layout.addWidget(pipeline_group)

        timing_group = QGroupBox("实时耗时统计（last / EMA平均 / 历史最大）")
        timing_layout = QVBoxLayout(timing_group)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Consolas", 10))
        timing_layout.addWidget(self.detail_text)
        layout.addWidget(timing_group)

    def set_pipeline_text(self, text: str):
        """更新编号管线区域，同时保留用户当前的滚动阅读位置。"""
        if self.pipeline_text.toPlainText() == text:
            return

        vertical_bar = self.pipeline_text.verticalScrollBar()
        horizontal_bar = self.pipeline_text.horizontalScrollBar()
        old_vertical_value = vertical_bar.value()
        old_horizontal_value = horizontal_bar.value()
        had_content = bool(self.pipeline_text.toPlainText())
        was_at_bottom = had_content and old_vertical_value >= max(
            0, vertical_bar.maximum() - 2
        )

        self.pipeline_text.blockSignals(True)
        self.pipeline_text.setPlainText(text)
        self.pipeline_text.blockSignals(False)

        if was_at_bottom:
            vertical_bar.setValue(vertical_bar.maximum())
        else:
            vertical_bar.setValue(min(old_vertical_value, vertical_bar.maximum()))
        horizontal_bar.setValue(min(old_horizontal_value, horizontal_bar.maximum()))

    def set_detail_text(self, text: str):
        """更新详情文本并尽量保留滚动位置。"""
        if self.detail_text.toPlainText() == text:
            return

        vertical_bar = self.detail_text.verticalScrollBar()
        horizontal_bar = self.detail_text.horizontalScrollBar()
        old_vertical_value = vertical_bar.value()
        old_horizontal_value = horizontal_bar.value()
        had_content = bool(self.detail_text.toPlainText())
        was_at_bottom = had_content and old_vertical_value >= max(
            0, vertical_bar.maximum() - 2
        )

        self.detail_text.blockSignals(True)
        self.detail_text.setPlainText(text)
        self.detail_text.blockSignals(False)

        if was_at_bottom:
            vertical_bar.setValue(vertical_bar.maximum())
        else:
            vertical_bar.setValue(min(old_vertical_value, vertical_bar.maximum()))
        horizontal_bar.setValue(min(old_horizontal_value, horizontal_bar.maximum()))

    def copy_detail_text(self):
        """拷贝当前详情文本到系统剪贴板。"""
        QApplication.clipboard().setText(
            self.pipeline_text.toPlainText()
            + "\n\n"
            + self.detail_text.toPlainText()
        )
##########


class RadarStationMainWindow(QMainWindow):
    """雷达站主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboMaster 2026 雷达站控制界面")
        ########################################################################
        # 显式保留最大化按钮，并按当前屏幕限制初始窗口尺寸，避免窗口初始高度超过桌面。
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )
        # self.setGeometry(100, 100, 2048, 1534)
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_rect = screen.availableGeometry()
            initial_width = min(2048, max(1280, int(screen_rect.width() * 0.9)))
            initial_height = min(1534, max(850, int(screen_rect.height() * 0.9)))
            initial_width = min(initial_width, screen_rect.width())
            initial_height = min(initial_height, screen_rect.height())
            initial_x = screen_rect.x() + max(0, (screen_rect.width() - initial_width) // 2)
            initial_y = screen_rect.y() + max(0, (screen_rect.height() - initial_height) // 2)
            self.setGeometry(initial_x, initial_y, initial_width, initial_height)
        else:
            self.setGeometry(100, 100, 1600, 1000)
        ########################################################################
        self._window_aspect_ratio = 2048.0 / 1534.0
        self._is_adjusting_aspect = False
        self._pending_aspect_size = None
        self._aspect_resize_timer = QTimer(self)
        self._aspect_resize_timer.setSingleShot(True)
        self._aspect_resize_timer.timeout.connect(self._apply_pending_aspect_resize)
        self.timing_debug_enabled = True

        ########################################################################
        # init_ui 会创建前哨站阈值控件，控件初始化需要先读取 params.yaml。
        self.config = load_params_config()
        ########################################################################

        # 初始化UI组件
        self.init_ui()
        self.logger = logging.getLogger("RadarStation UI")
        # import yaml
        # with open("config/params.yaml", "r") as file:
        #     config = yaml.safe_load(file)
        #
        # self.config = config
        config = self.config
        debug_cfg = config.get("debug", {})
        self.timing_debug_enabled = bool(debug_cfg.get("timing_debug_enabled", True))
        if hasattr(self, "timing_debug_cb"):
            self.timing_debug_cb.blockSignals(True)
            self.timing_debug_cb.setChecked(self.timing_debug_enabled)
            self.timing_debug_cb.blockSignals(False)
        self._apply_timing_debug_visibility()
        camera_mode = debug_cfg.get("camera_mode")
        if not camera_mode:
            camera_mode = (
                "image" if debug_cfg.get("inference_video", False) else "camera"
            )
        camera_mode = camera_mode.lower()
        self.camera_color_order = debug_cfg.get("camera_color_order", "rgb").lower()
        if self.camera_color_order not in ("rgb", "bgr"):
            self.logger.warning(
                "Unknown camera_color_order: %s, defaulting to rgb",
                self.camera_color_order,
            )
            self.camera_color_order = "rgb"
        if camera_mode == "replay":
            from driver.replay_camera import TimestampReplayCamera

            self.camera = TimestampReplayCamera._instance
        elif camera_mode == "ros":
            from driver.ros_camera.ros_image_camera import RosImageCamera

            self.camera = RosImageCamera._instance
        elif camera_mode == "image":
            from driver.hik_camera.mock_hik import SimpleHikCamera

            self.camera = SimpleHikCamera._instance
        else:
            from driver.hik_camera.hik import SimpleHikCamera

            self.camera = SimpleHikCamera._instance
        
        self.referee = RefereeCommManager._instance
        assert (
            self.referee is not None
        ), "Referee Comm Manager instance is not initialized."
        assert self.camera is not None, "Hik Camera instance is not initialized."
        ##########
        self.serial_receive_window = None
        self.serial_send_window = None
        self.timing_debug_detail_window = None
        ##########
        self.camera.register_group("radar_station")
        self.camera.register_group("sensor_recorder")
        # 相机启动参数来自 config/device.yaml；绑定相机后立即同步到 UI，避免显示硬编码兜底值。
        self.get_current_exposure()
        self.get_current_gain()
        self.is_setting_exposure = False
        self.sensor_recording_active = False
        self.sensor_record_camera_enabled = True
        self.sensor_record_lidar_enabled = False
        camera_args = getattr(self.camera, "args", None)
        self.sensor_record_camera_fps = max(
            1, int(round(float(getattr(camera_args, "acquisition_rate", 30.0))))
        )
        self.sensor_record_camera_queue_size = max(
            1, int(getattr(camera_args, "recording_queue_size", 8))
        )
        self.sensor_record_camera_encoder_preset = str(
            getattr(camera_args, "recording_encoder_preset", "p3")
        )
        self.sensor_record_camera_encoder_bitrate_mbps = float(
            getattr(camera_args, "recording_encoder_bitrate_mbps", 60.0)
        )
        self.sensor_record_camera_encoder_max_bitrate_mbps = float(
            getattr(camera_args, "recording_encoder_max_bitrate_mbps", 100.0)
        )
        self.sensor_record_camera_ffmpeg_path = str(
            getattr(camera_args, "recording_ffmpeg_path", "ffmpeg")
        )
        self.sensor_record_save_root_dir = os.path.abspath("recorder")
        self.sensor_record_session_dir = ""
        self.sensor_record_camera_dir = ""
        self.sensor_record_camera_video_path = ""
        self.sensor_record_camera_timestamp_path = ""
        self.sensor_record_camera_metrics_path = ""
        self.sensor_record_camera_recorder = None
        self.sensor_record_camera_error_logged = False
        self.sensor_record_camera_frame_count = 0
        self.sensor_record_lidar_dir = ""
        self.sensor_record_yolo_enabled = False
        self.sensor_record_yolo_dataset_dir = ""
        self.sensor_record_yolo_images_dir = ""
        self.sensor_record_yolo_labels_dir = ""
        self.sensor_record_yolo_frame_count = 0
        self.sensor_record_last_yolo_stamp = None
        self.sensor_record_last_yolo_write_time = None
        self.sensor_record_yolo_wait_logged = False
        self.sensor_record_armor_yolo_enabled = False
        self.sensor_record_armor_yolo_dataset_dir = ""
        self.sensor_record_armor_yolo_images_dir = ""
        self.sensor_record_armor_yolo_labels_dir = ""
        self.sensor_record_armor_yolo_frame_count = 0
        self.sensor_record_last_armor_yolo_stamp = None
        self.sensor_record_last_armor_yolo_write_time = None
        self.sensor_record_armor_yolo_wait_logged = False
        # UI 线程取图默认使用非阻塞，避免相机超时阻塞导致界面卡顿
        self.image_fetch_timeout_sec = 0.0
        # 状态刷新无需高频，降低主线程压力
        self.status_update_interval_ms = 500
        self._camera_frame_received = False
        self._tracker_frame_received = False
        self.show()  # Show the window first
        QApplication.processEvents()  # Process all pending events
        self.repaint()  # Force repaint
        QApplication.processEvents()  # Process all pending events
        # 初始化数据更新定时器
        self.setup_timers()

    def resizeEvent(self, event):
        # """保持主窗口长宽比例，避免布局在不同尺寸下被拉伸变形。"""
        """允许主窗口自由缩放，图像控件自行适配可用空间。"""
        super().resizeEvent(event)

        ########################################################################
        # 主窗口允许自由缩放和最大化，图像内容在 display_image 中按 QLabel 当前尺寸缩放。
        return
        ########################################################################
        # if self._is_adjusting_aspect:
        #     return
        # if self.isMaximized() or self.isFullScreen():
        #     return
        #
        # old_size = event.oldSize()
        # new_size = event.size()
        # if old_size.width() <= 0 or old_size.height() <= 0:
        #     return
        #
        # delta_w = new_size.width() - old_size.width()
        # delta_h = new_size.height() - old_size.height()
        #
        # target_w = new_size.width()
        # target_h = new_size.height()
        # if abs(delta_w) >= abs(delta_h):
        #     target_h = int(round(target_w / self._window_aspect_ratio))
        # else:
        #     target_w = int(round(target_h * self._window_aspect_ratio))
        #
        # target_w = max(self.minimumWidth(), target_w)
        # target_h = max(self.minimumHeight(), target_h)
        #
        # if target_w == new_size.width() and target_h == new_size.height():
        #     return
        #
        # # 合并高频 resize 事件，避免在 resizeEvent 中递归触发重排导致卡顿。
        # self._pending_aspect_size = QSize(target_w, target_h)
        # # 拖拽期间进行防抖，停止拖拽后再统一修正比例，避免交互卡顿
        # self._aspect_resize_timer.start(60)

    def _apply_pending_aspect_resize(self):
        """异步应用等比尺寸，减少窗口拖拽时的重绘抖动。"""
        if self._pending_aspect_size is None:
            return
        target_size = self._pending_aspect_size
        self._pending_aspect_size = None
        if self.size() == target_size:
            return

        self._is_adjusting_aspect = True
        self.resize(target_size)
        self._is_adjusting_aspect = False

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧区域：相机图像和雷达点云
        left_panel = self.create_image_panel()

        # 中间区域：场地投影和车辆坐标
        center_panel = self.create_field_projection_panel()

        # 右侧区域：控制面板和状态监控
        right_panel = self.create_control_panel()

        # 三栏可拖动分配空间；控制区独立滚动，不再撑大主窗口。
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.control_scroll = _scrollable_panel(right_panel)
        self.control_scroll.setMinimumWidth(260)
        self.main_splitter.addWidget(self.control_scroll)
        control_width = right_panel.minimumSizeHint().width() + self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        self.main_splitter.setSizes([750, 650, max(450, control_width)])
        for index, stretch in enumerate((4, 4, 3)):
            self.main_splitter.setStretchFactor(index, stretch)
        main_layout.addWidget(self.main_splitter)

    # 修改原有的RadarStationMainWindow类中的create_image_panel方法
    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QGroupBox("图像显示")
        layout = QVBoxLayout(panel)

        # 检测与追踪已经在后台合并，只保留一个可扩展的大画面。
        ########################################################################
        # 图像显示区保留目标尺寸作为缩放 fallback，但不再把它作为最小尺寸。
        self.target_camera_size = QSize(819, 700)  # 2/5 of 2048px width
        self.target_tracker_size = QSize(819, 700)
        # self.target_camera_size = QSize(576, 432)  # 2/5 of 1440px width
        # self.target_tracker_size = QSize(576, 432)
        self.image_display_min_size = QSize(240, 180)
        self.visualization_label = QLabel("检测与追踪画面")
        self.visualization_label.setMinimumSize(self.image_display_min_size)
        self.visualization_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        self.visualization_label.setStyleSheet("border: 1px solid black")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_label, 2)

        # 保留旧属性名供截图、标定和外部调试代码访问，二者指向同一个 QLabel。
        self.camera_label = self.visualization_label
        self.tracker_label = self.visualization_label
        ########################################################################

        ########################################################################
        # 前哨站二值化图像与判断参数。
        # self.outpost_binary_size = QSize(
        #     self.target_camera_size.width() // 2,
        #     self.target_camera_size.height() // 2,
        # )
        # outpost_row = QWidget()
        # outpost_layout = QHBoxLayout(outpost_row)
        # outpost_layout.setContentsMargins(0, 0, 0, 0)
        # self.outpost_binary_label = QLabel("前哨站二值图")
        # self.outpost_binary_label.setMinimumSize(self.outpost_binary_size)
        # self.outpost_binary_label.setMaximumSize(self.outpost_binary_size)
        # self.outpost_binary_label.setStyleSheet("border: 1px solid black")
        # self.outpost_binary_label.setAlignment(Qt.AlignCenter)
        # outpost_layout.addWidget(self.outpost_binary_label)
        #
        # outpost_cfg = self._load_outpost_light_ui_config()
        # outpost_controls = QGridLayout()
        # self.outpost_color_combo = QComboBox()
        # self.outpost_color_combo.addItems(["blue", "red"])
        # self.outpost_color_combo.setCurrentText(outpost_cfg["color"])
        # outpost_controls.addWidget(QLabel("颜色"), 0, 0)
        # outpost_controls.addWidget(self.outpost_color_combo, 0, 1)
        #
        # self.outpost_threshold_inputs = {}
        # threshold_fields = [
        #     ("h_min", "H下限", 0, 179),
        #     ("h_max", "H上限", 0, 179),
        #     ("s_min", "S下限", 0, 255),
        #     ("s_max", "S上限", 0, 255),
        #     ("v_min", "V下限", 0, 255),
        #     ("v_max", "V上限", 0, 255),
        # ]
        # for row, (key, label, min_value, max_value) in enumerate(threshold_fields, start=1):
        #     spin = QSpinBox()
        #     spin.setRange(min_value, max_value)
        #     spin.setValue(int(outpost_cfg[key]))
        #     self.outpost_threshold_inputs[key] = spin
        #     outpost_controls.addWidget(QLabel(label), row, 0)
        #     outpost_controls.addWidget(spin, row, 1)
        #     spin.valueChanged.connect(self.apply_outpost_light_config)
        #
        # self.outpost_ratio_spin = QDoubleSpinBox()
        # self.outpost_ratio_spin.setRange(0.0, 1.0)
        # self.outpost_ratio_spin.setSingleStep(0.01)
        # self.outpost_ratio_spin.setDecimals(3)
        # self.outpost_ratio_spin.setValue(float(outpost_cfg["alive_ratio_threshold"]))
        # outpost_controls.addWidget(QLabel("存活比例"), 7, 0)
        # outpost_controls.addWidget(self.outpost_ratio_spin, 7, 1)
        # # self.outpost_color_combo.currentTextChanged.connect(self.apply_outpost_light_config)
        # self.outpost_color_combo.currentTextChanged.connect(self._on_outpost_color_changed)
        # self.outpost_ratio_spin.valueChanged.connect(self.apply_outpost_light_config)
        #
        # outpost_layout.addLayout(outpost_controls)
        # outpost_layout.addStretch()
        # layout.addWidget(outpost_row)
        self.outpost_binary_size = QSize(
            self.target_camera_size.width() // 2,
            self.target_camera_size.height() // 2,
        )
        outpost_row = QWidget()
        outpost_layout = QVBoxLayout(outpost_row)
        outpost_layout.setContentsMargins(0, 0, 0, 0)

        self.outpost_binary_label = QLabel("前哨站二值图")
        self.outpost_binary_label.setMinimumSize(160, 100)
        self.outpost_binary_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.outpost_binary_label.setStyleSheet("border: 1px solid black")
        self.outpost_binary_label.setAlignment(Qt.AlignCenter)
        outpost_layout.addWidget(self.outpost_binary_label, 1)

        outpost_cfg = self._load_outpost_light_ui_config()
        outpost_controls_widget = QWidget()
        outpost_controls_layout = QHBoxLayout(outpost_controls_widget)
        outpost_controls_layout.setContentsMargins(0, 0, 0, 0)

        hsv_group = QGroupBox("HSV阈值")
        hsv_layout = QGridLayout(hsv_group)
        self.outpost_threshold_inputs = {}
        threshold_fields = [
            ("h_min", "H下限", 0, 179),
            ("h_max", "H上限", 0, 179),
            ("s_min", "S下限", 0, 255),
            ("s_max", "S上限", 0, 255),
            ("v_min", "V下限", 0, 255),
            ("v_max", "V上限", 0, 255),
        ]
        for row, (key, label, min_value, max_value) in enumerate(threshold_fields):
            spin = QSpinBox()
            spin.setRange(min_value, max_value)
            spin.setValue(int(outpost_cfg[key]))
            self.outpost_threshold_inputs[key] = spin
            hsv_layout.addWidget(QLabel(label), row, 0)
            hsv_layout.addWidget(spin, row, 1)
            spin.valueChanged.connect(self.apply_outpost_light_config)

        state_group = QGroupBox("颜色与存活")
        state_layout = QGridLayout(state_group)
        self.outpost_color_combo = QComboBox()
        ########################################################################
        # self.outpost_color_combo.addItems(["blue", "red"])
        # self.outpost_color_combo.addItems(["blue", "red", "green", "anti-white"])
        self.outpost_color_combo.addItems(["blue", "red", "green", "white"])
        ########################################################################
        self.outpost_color_combo.setCurrentText(outpost_cfg["color"])
        state_layout.addWidget(QLabel("颜色"), 0, 0)
        ########################################################################
        # state_layout.addWidget(self.outpost_color_combo, 0, 1)
        color_control_widget = QWidget()
        color_control_layout = QHBoxLayout(color_control_widget)
        color_control_layout.setContentsMargins(0, 0, 0, 0)
        color_control_layout.addWidget(self.outpost_color_combo)
        self.outpost_reverse_checkbox = QCheckBox("reverse")
        self.outpost_reverse_checkbox.setChecked(bool(outpost_cfg["reverse"]))
        color_control_layout.addWidget(self.outpost_reverse_checkbox)
        state_layout.addWidget(color_control_widget, 0, 1)
        ########################################################################

        self.outpost_ratio_spin = QDoubleSpinBox()
        self.outpost_ratio_spin.setRange(0.0, 1.0)
        self.outpost_ratio_spin.setSingleStep(0.01)
        self.outpost_ratio_spin.setDecimals(3)
        self.outpost_ratio_spin.setValue(float(outpost_cfg["alive_ratio_threshold"]))
        state_layout.addWidget(QLabel("存活比例"), 1, 0)
        state_layout.addWidget(self.outpost_ratio_spin, 1, 1)

        ########################################################################
        self.outpost_roi_inputs = {}
        roi = getattr(self, "outpost_roi", None)
        roi_values = roi if roi is not None else (0, 0, 0, 0)
        roi_fields = [
            ("x1", "ROI x1", int(roi_values[0])),
            ("y1", "ROI y1", int(roi_values[1])),
            ("x2", "ROI x2", int(roi_values[2])),
            ("y2", "ROI y2", int(roi_values[3])),
        ]
        for row, (key, label, value) in enumerate(roi_fields, start=2):
            spin = QSpinBox()
            spin.setRange(0, 99999)
            spin.setValue(value)
            self.outpost_roi_inputs[key] = spin
            state_layout.addWidget(QLabel(label), row, 0)
            state_layout.addWidget(spin, row, 1)
            spin.valueChanged.connect(self._on_outpost_roi_input_changed)
        ########################################################################

        self.outpost_alive_status_label = QLabel("当前：等待")
        self.outpost_alive_status_label.setAlignment(Qt.AlignCenter)
        self.outpost_alive_status_label.setStyleSheet("color: gray; font-weight: bold;")
        # state_layout.addWidget(self.outpost_alive_status_label, 2, 0, 1, 2)
        # state_layout.setRowStretch(3, 1)
        ########################################################################
        # state_layout.setRowStretch(2, 1)
        # state_layout.addWidget(self.outpost_alive_status_label, 3, 0, 1, 2)
        state_layout.addWidget(self.outpost_alive_status_label, 6, 0, 1, 2)
        ########################################################################

        # self.outpost_color_combo.currentTextChanged.connect(self.apply_outpost_light_config)
        self.outpost_color_combo.currentTextChanged.connect(self._on_outpost_color_changed)
        self.outpost_ratio_spin.valueChanged.connect(self.apply_outpost_light_config)
        ########################################################################
        self.outpost_reverse_checkbox.stateChanged.connect(self.apply_outpost_light_config)
        ########################################################################
        outpost_controls_layout.addWidget(hsv_group)
        outpost_controls_layout.addWidget(state_group)
        # 预览与参数分为标签页，保持相机画面有足够空间。
        self.image_controls_tabs = QTabWidget()
        self.image_controls_tabs.setMinimumSize(260, 180)
        self.image_controls_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.image_controls_tabs.setMaximumHeight(460)
        layout.addWidget(self.image_controls_tabs, 1)
        ########################################################################


        # 相机控制按钮
        camera_controls_widget = QWidget()
        camera_controls = QVBoxLayout(camera_controls_widget)

        # Basic control of camera
        basic_controls = QHBoxLayout()
      
        self.keypoint_btn = QPushButton("标关键点")
        self.sensor_record_btn = QPushButton("录制传感器数据")

        basic_controls.addWidget(self.keypoint_btn)
        basic_controls.addWidget(self.sensor_record_btn)
        camera_controls.addLayout(basic_controls)

        # 整车检测置信度运行时调节
        car_detect_controls = QHBoxLayout()
        car_conf_label = QLabel("整车置信度:")
        car_detect_controls.addWidget(car_conf_label)
        self.car_conf_input = QDoubleSpinBox()
        self.car_conf_input.setRange(0.0, 1.0)
        self.car_conf_input.setSingleStep(0.05)
        self.car_conf_input.setDecimals(2)
        self.car_conf_input.setKeyboardTracking(False)
        car_conf_value = float(
            self.config.get("car_detector", {}).get("conf_thres", 0.3)
        )
        self.car_conf_input.setValue(car_conf_value)
        car_detect_controls.addWidget(self.car_conf_input)
        self.current_car_conf_label = QLabel(f"当前: {car_conf_value:.2f}")
        self.current_car_conf_label.setStyleSheet("color: gray; font-size: 12px;")
        car_detect_controls.addWidget(self.current_car_conf_label)
        car_detect_controls.addStretch()
        camera_controls.addLayout(car_detect_controls)

        # Exposure controls
        exposure_controls = QHBoxLayout()
        exposure_label = QLabel("曝光值:")
        exposure_controls.addWidget(exposure_label)
        # 曝光值输入框
        self.exposure_input = QSpinBox()
        self.exposure_input.setRange(1, 100000)  # 曝光时间范围，单位微秒
        self.exposure_input.setValue(10000)  # 相机绑定后会同步真实启动值
        self.exposure_input.setSuffix(" μs")
        exposure_controls.addWidget(self.exposure_input)
        # 曝光确认按钮
        self.exposure_confirm_btn = QPushButton("设置曝光")
        self.exposure_confirm_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; }"
        )
        exposure_controls.addWidget(self.exposure_confirm_btn)
        # 当前曝光显示
        self.current_exposure_label = QLabel("当前: 10000μs")
        self.current_exposure_label.setStyleSheet("color: gray; font-size: 12px;")
        exposure_controls.addWidget(self.current_exposure_label)

        exposure_controls.addStretch()  # 添加弹性空间
        camera_controls.addLayout(exposure_controls)

        # Gain controls
        gain_controls = QHBoxLayout()
        gain_label = QLabel("增益值:")
        gain_controls.addWidget(gain_label)
        # 增益值输入框
        self.gain_input = QSpinBox()
        self.gain_input.setRange(0, 100)  # 增益范围，单位dB
        self.gain_input.setValue(20)  # 相机绑定后会同步真实启动值
        self.gain_input.setSuffix(" dB")
        gain_controls.addWidget(self.gain_input)
        # 增益确认按钮
        self.gain_confirm_btn = QPushButton("设置增益")
        self.gain_confirm_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; }"
        )
        gain_controls.addWidget(self.gain_confirm_btn)
        # 当前增益显示
        self.current_gain_label = QLabel("当前: 20dB")
        self.current_gain_label.setStyleSheet("color: gray; font-size: 12px;")
        gain_controls.addWidget(self.current_gain_label)

        gain_controls.addStretch()  # 添加弹性空间
        camera_controls.addLayout(gain_controls)

        camera_controls.addStretch()
        self.image_controls_tabs.addTab(_scrollable_panel(camera_controls_widget), "相机控制")
        self.image_controls_tabs.addTab(outpost_row, "前哨站二值图")
        self.image_controls_tabs.addTab(_scrollable_panel(outpost_controls_widget), "前哨站参数")

        # 连接信号槽
        self.exposure_confirm_btn.clicked.connect(self.on_exposure_confirm)
        self.exposure_input.valueChanged.connect(self.on_exposure_value_changed)
        self.gain_confirm_btn.clicked.connect(self.on_gain_confirm)
        self.gain_input.valueChanged.connect(self.on_gain_value_changed)
        self.car_conf_input.valueChanged.connect(self.on_car_conf_value_changed)
        self.keypoint_btn.clicked.connect(self.open_keypoint_calibration)  # 新增
        self.sensor_record_btn.clicked.connect(self.open_sensor_recording_dialog)

        return panel

    def create_field_projection_panel(self):
        """创建场地投影面板"""
        panel = QGroupBox("场地实时投影")
        layout = QVBoxLayout(panel)

        # 场地投影显示区域
        self.field_view = FieldProjectionWidget()
        layout.addWidget(self.field_view)

        # 投影控制选项
        controls = QGridLayout()
        self.show_enemy_cb = QCheckBox("显示敌方车辆")
        self.show_ally_cb = QCheckBox("显示己方车辆")
        self.show_guessed_cb = QCheckBox("显示猜点")
        self.show_lidar_cb = QCheckBox("显示雷达点")
        self.show_projection_cb = QCheckBox("显示投影点")
        self.show_info_wave_cb = QCheckBox("显示信息波坐标")
        self.show_tracking_cb = QCheckBox("显示追踪坐标")

        self.show_enemy_cb.setChecked(True)
        self.show_ally_cb.setChecked(True)
        self.show_guessed_cb.setChecked(False)
        self.show_lidar_cb.setChecked(False)
        self.show_projection_cb.setChecked(False)
        self.show_info_wave_cb.setChecked(True)
        self.show_tracking_cb.setChecked(True)

        for index, checkbox in enumerate((
            self.show_enemy_cb, self.show_ally_cb, self.show_guessed_cb,
            self.show_lidar_cb, self.show_projection_cb,
            self.show_info_wave_cb, self.show_tracking_cb,
        )):
            controls.addWidget(checkbox, index // 2, index % 2)
        layout.addLayout(controls)

        self.show_enemy_cb.stateChanged.connect(self.update_display_settings)
        self.show_ally_cb.stateChanged.connect(self.update_display_settings)
        self.show_guessed_cb.stateChanged.connect(self.update_display_settings)
        self.show_lidar_cb.stateChanged.connect(self.update_display_settings)
        self.show_projection_cb.stateChanged.connect(self.update_display_settings)
        self.show_info_wave_cb.stateChanged.connect(self.update_display_settings)
        self.show_tracking_cb.stateChanged.connect(self.update_display_settings)

        return panel

    def create_control_panel(self):
        """创建控制和状态面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 状态监控面板
        status_panel = self.create_status_panel()
        layout.addWidget(status_panel)

        # 校准控制面板
        calibration_panel = self.create_calibration_panel()
        layout.addWidget(calibration_panel)

        # 系统信息面板
        system_panel = self.create_system_panel()
        layout.addWidget(system_panel)

        return panel

    def update_display_settings(self):
        """更新显示设置"""
        self.field_view.show_enemy = self.show_enemy_cb.isChecked()
        self.field_view.show_ally = self.show_ally_cb.isChecked()
        if hasattr(self, "show_guessed_cb"):
            self.field_view.show_guessed = self.show_guessed_cb.isChecked()
        if hasattr(self, "show_lidar_cb"):
            self.field_view.show_lidar = self.show_lidar_cb.isChecked()
        if hasattr(self, "show_projection_cb"):
            self.field_view.show_projection = self.show_projection_cb.isChecked()
        if hasattr(self, "show_info_wave_cb"):
            self.field_view.show_info_wave = self.show_info_wave_cb.isChecked()
        if hasattr(self, "show_tracking_cb"):
            self.field_view.show_tracking = self.show_tracking_cb.isChecked()
        self.field_view.update()

    def create_status_panel(self):
        """创建设备状态监控面板"""
        panel = QGroupBox("设备状态")
        layout = QVBoxLayout(panel)

        # 连接状态指示器
        self.referee_status = StatusIndicator("裁判系统连接")
        self.sentry_received = StatusIndicator("哨兵接收雷达数据")  # 存疑
        self.sentry_status = StatusIndicator("雷达接收哨兵数据")
        self.camera_status = StatusIndicator("相机连接")

        layout.addWidget(self.referee_status)
        layout.addWidget(self.sentry_received)
        layout.addWidget(self.sentry_status)
        layout.addWidget(self.camera_status)

        # 相机帧率显示
        self.fps_label = QLabel("相机帧率: 0 FPS")
        layout.addWidget(self.fps_label)

        self.localization_fps_label = QLabel("定位帧率: 0.00 FPS")
        layout.addWidget(self.localization_fps_label)

        self.referee_tx_fps_label = QLabel("裁判系统定位发送帧率: 0.00 FPS")
        layout.addWidget(self.referee_tx_fps_label)

        # 当前阵营显示
        self.team_label = QLabel("当前阵营: 未知")
        self.team_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.team_label)

        return panel

    def create_calibration_panel(self):
        """创建校准控制面板"""
        panel = QGroupBox("校准控制")
        layout = QVBoxLayout(panel)

        # 校准按钮
        self.calibrate_btn = QPushButton("开始外参校准")
        self.calibrate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )
        layout.addWidget(self.calibrate_btn)
        self.calibrate_btn.clicked.connect(self.open_keypoint_calibration)

        self.manual_extrinsics_btn = QPushButton("直接输入外参")
        self.manual_extrinsics_btn.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; font-weight: bold; }"
        )
        self.manual_extrinsics_btn.clicked.connect(self.open_manual_extrinsics_dialog)
        layout.addWidget(self.manual_extrinsics_btn)

        # 校准状态显示
        self.calibration_status = QLabel("雷达标定状态: 未校准")
        layout.addWidget(self.calibration_status)

        self.xy_maps_checkbox = QCheckBox("xy maps")
        self.xy_maps_checkbox.setChecked(
            bool(self.config.get("ray_projection", {}).get("xy_maps_default_enabled", False))
        )
        self.xy_maps_checkbox.stateChanged.connect(self.on_xy_maps_changed)
        layout.addWidget(self.xy_maps_checkbox)
        self.xy_maps_status = QLabel(
            "XY Map 状态: 等待标定" if self.xy_maps_checkbox.isChecked() else "XY Map 状态: 未启用"
        )
        layout.addWidget(self.xy_maps_status)

        # 校准结果显示
        self.calibration_result = QTextEdit()
        self.calibration_result.setMaximumHeight(100)
        self.calibration_result.setPlaceholderText("校准结果将在此显示...")
        layout.addWidget(self.calibration_result)

        # 开始启动主进程
        self.start_tracking_btn = QPushButton("启动雷达站")
        self.start_tracking_btn.setEnabled(False)
        self.start_tracking_btn.clicked.connect(self.start_tracking)
        layout.addWidget(self.start_tracking_btn)
        self.main_event_loop = None
        self.calibrated_R = None
        self.calibrated_T = None
        ########################################################################
        # 前哨站 ROI，格式为原图坐标 (x1, y1, x2, y2)。
        self.outpost_roi = None
        ########################################################################

        return panel

    def create_system_panel(self):
        """创建系统信息面板"""
        panel = QGroupBox("系统状态")
        layout = QVBoxLayout(panel)

        # 双倍易伤状态机状态
        self.double_damage_status = QLabel("双倍易伤状态机: IDLE")
        self.double_damage_status.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.double_damage_status)

        # 双倍易伤触发次数
        self.double_damage_count = QLabel("可触发次数:")
        self.double_damage_count.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.double_damage_count)

        # 机器人标记状态面板
        robot_mark_panel = QGroupBox("机器人标记状态")
        robot_mark_layout = QGridLayout(robot_mark_panel)
        
        # 创建机器人状态指示器字典
        self.robot_status_indicators = {}
        robot_types = [
            ("1Hero", "1英雄"),
            ("2Engineer", "2工程"),
            ("3Standard", "3步兵"),
            ("4Standard", "4步兵"),
            ("7Sentry", "7哨兵")
        ]
        
        # 按2列布局排列机器人状态
        for i, (robot_id, robot_name) in enumerate(robot_types):
            row = i // 2
            col = i % 2
            
            # 创建状态指示器容器
            robot_widget = QWidget()
            robot_layout = QHBoxLayout(robot_widget)
            robot_layout.setContentsMargins(5, 2, 5, 2)
            
            # 状态指示圆点
            status_dot = QLabel("●")
            status_dot.setFixedSize(20, 20)
            status_dot.setAlignment(Qt.AlignCenter)
            robot_layout.addWidget(status_dot)
            
            # 机器人名称标签
            name_label = QLabel(f"{robot_name}")
            name_label.setFixedWidth(80)
            robot_layout.addWidget(name_label)
            
            # 存储到字典中以便后续更新
            self.robot_status_indicators[robot_id] = {
                'dot': status_dot,
                'label': name_label,
                'is_marked': False
            }
            
            # 设置初始状态（未标记 = 红色）
            self.update_robot_indicator_status(robot_id, False)
            
            robot_mark_layout.addWidget(robot_widget, row, col)
        
        layout.addWidget(robot_mark_panel)

        # self.timing_debug_cb = QCheckBox("启用跟踪定位耗时调试")
        # self.timing_debug_cb.setChecked(self.timing_debug_enabled)
        # self.timing_debug_cb.toggled.connect(self.on_timing_debug_toggled)
        # layout.addWidget(self.timing_debug_cb)
        ##########
        timing_debug_control_layout = QHBoxLayout()
        self.timing_debug_cb = QCheckBox("启用跟踪定位耗时调试")
        self.timing_debug_cb.setChecked(self.timing_debug_enabled)
        self.timing_debug_cb.toggled.connect(self.on_timing_debug_toggled)
        self.timing_debug_detail_btn = QPushButton("详情")
        self.timing_debug_detail_btn.clicked.connect(self.open_timing_debug_detail_window)
        timing_debug_control_layout.addWidget(self.timing_debug_cb)
        timing_debug_control_layout.addWidget(self.timing_debug_detail_btn)
        timing_debug_control_layout.addStretch()
        layout.addLayout(timing_debug_control_layout)
        ##########

        self.timing_debug_text = QTextEdit()
        self.timing_debug_text.setReadOnly(True)
        self.timing_debug_text.setMaximumHeight(220)
        self.timing_debug_text.setPlaceholderText("跟踪定位耗时调试...")
        layout.addWidget(self.timing_debug_text)
        self._apply_timing_debug_visibility()

        # 系统日志
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setPlaceholderText("系统日志...")
        layout.addWidget(self.log_text)
        
        self.reset_referee_btn = QPushButton("重置裁判状态")
        self.reset_referee_btn.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; font-weight: bold; }"
        )
        self.reset_referee_btn.clicked.connect(self.on_reset_referee)
        layout.addWidget(self.reset_referee_btn)

        self.send_sentry2radar_btn = QPushButton("Test")
        self.send_sentry2radar_btn.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; font-weight: bold; }"
        )
        self.send_sentry2radar_btn.clicked.connect(self.test)
        layout.addWidget(self.send_sentry2radar_btn)

        ##########
        # self.serial_receive_btn = QPushButton("串口接收数据")
        # self.serial_receive_btn.setStyleSheet(
        #     "QPushButton { background-color: #455A64; color: white; font-weight: bold; }"
        # )
        # self.serial_receive_btn.clicked.connect(self.open_serial_receive_window)
        # layout.addWidget(self.serial_receive_btn)
        ########################################################################
        serial_debug_btn_layout = QHBoxLayout()
        self.serial_receive_btn = QPushButton("接收数据")
        self.serial_send_btn = QPushButton("发送数据")
        for button in (self.serial_receive_btn, self.serial_send_btn):
            button.setStyleSheet(
                "QPushButton { background-color: #455A64; color: white; font-weight: bold; }"
            )
            serial_debug_btn_layout.addWidget(button)
        self.serial_receive_btn.clicked.connect(self.open_serial_receive_window)
        self.serial_send_btn.clicked.connect(self.open_serial_send_window)
        layout.addLayout(serial_debug_btn_layout)
        ########################################################################
        ##########

        return panel

    def _apply_timing_debug_visibility(self):
        """根据开关状态更新耗时调试控件显示。"""
        if not hasattr(self, "timing_debug_text"):
            return
        enabled = bool(getattr(self, "timing_debug_enabled", False))
        self.timing_debug_text.setVisible(enabled)
        if enabled:
            self.timing_debug_text.setPlaceholderText("跟踪定位耗时调试...")
        else:
            # self.timing_debug_text.setPlainText("跟踪定位耗时调试已关闭。")
            ##########
            self._set_textedit_plain_text_preserve_scroll(
                self.timing_debug_text,
                "跟踪定位耗时调试已关闭。",
            )
            ##########

    ##########
    def _set_textedit_plain_text_preserve_scroll(self, text_edit: QTextEdit, text: str):
        """更新文本框内容时尽量保留滚动条位置，避免高频刷新打断阅读。"""
        if text_edit is None:
            return
        if text_edit.toPlainText() == text:
            return

        vertical_bar = text_edit.verticalScrollBar()
        horizontal_bar = text_edit.horizontalScrollBar()
        old_vertical_value = vertical_bar.value()
        old_horizontal_value = horizontal_bar.value()
        was_at_bottom = old_vertical_value >= max(0, vertical_bar.maximum() - 2)

        text_edit.blockSignals(True)
        text_edit.setPlainText(text)
        text_edit.blockSignals(False)

        if was_at_bottom:
            vertical_bar.setValue(vertical_bar.maximum())
        else:
            vertical_bar.setValue(min(old_vertical_value, vertical_bar.maximum()))
        horizontal_bar.setValue(min(old_horizontal_value, horizontal_bar.maximum()))
    ##########

    def on_timing_debug_toggled(self, checked):
        """运行时切换耗时调试统计与显示。"""
        self.timing_debug_enabled = bool(checked)
        self._apply_timing_debug_visibility()

        event_loop = getattr(self, "main_event_loop", None)
        if event_loop is None:
            try:
                from main_event_loop import MainEventLoop

                event_loop = MainEventLoop._instance
            except Exception:
                event_loop = None
        if event_loop is not None and hasattr(event_loop, "set_timing_debug_enabled"):
            event_loop.set_timing_debug_enabled(self.timing_debug_enabled)

    ##########
    def open_timing_debug_detail_window(self):
        """打开跟踪定位耗时调试详情窗口。"""
        if self.timing_debug_detail_window is None:
            self.timing_debug_detail_window = TimingDebugDetailWindow(self)
        if self.timing_debug_detail_window.isMinimized():
            self.timing_debug_detail_window.showNormal()
        self.timing_debug_detail_window.show()
        self.timing_debug_detail_window.raise_()
        self.timing_debug_detail_window.activateWindow()
        self.timing_debug_detail_window.set_detail_text(
            self.timing_debug_text.toPlainText()
        )
        event_loop = getattr(self, "main_event_loop", None)
        if event_loop is not None and hasattr(
            event_loop, "get_pipeline_module_text"
        ):
            self.timing_debug_detail_window.set_pipeline_text(
                event_loop.get_pipeline_module_text()
            )
    ##########

    def setup_timers(self):
        """设置定时器用于数据更新"""
        visual_cfg = self.config.get("visualization", {})
        self.single_visualization_view = bool(visual_cfg.get("single_view", True))
        visualization_update_hz = max(
            0.1, float(visual_cfg.get("update_hz", 30.0))
        )
        # 图像更新定时器
        self.image_timer = QTimer()
        self.image_timer.timeout.connect(self.update_images)
        self.image_timer.start(max(1, int(round(1000.0 / visualization_update_hz))))

        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(getattr(self, "status_update_interval_ms", 500))

        # 场地投影与裁判系统定位消息使用同一个频率来源。
        field_projection_update_hz = max(
            0.1, float(self.referee.RADAR2CLIENT_TX_HZ)
        )
        self.field_projection_timer = QTimer()
        self.field_projection_timer.timeout.connect(self.update_field_projection)
        self.field_projection_timer.start(
            max(1, int(round(1000.0 / field_projection_update_hz)))
        )

        self.sensor_recording_timer = QTimer()
        self.sensor_recording_timer.timeout.connect(self.process_sensor_recording)
        self.sensor_recording_timer.setInterval(10)

    ########################################################################
    def _get_main_event_loop_instance(self):
        """获取当前 MainEventLoop 实例，允许雷达站启动前后共用同一条更新路径。"""
        event_loop = getattr(self, "main_event_loop", None)
        if event_loop is not None:
            return event_loop

        try:
            from main_event_loop import MainEventLoop
        except Exception:
            return None
        # return MainEventLoop._instance
        return getattr(MainEventLoop, "_instance", None)

    def on_car_conf_value_changed(self, value):
        """UI 调整整车检测置信度阈值，运行中立即同步到检测器。"""
        value = max(0.0, min(1.0, float(value)))
        self.config.setdefault("car_detector", {})["conf_thres"] = value

        applied_value = value
        event_loop = self._get_main_event_loop_instance()
        if event_loop is not None and hasattr(event_loop, "set_car_detector_conf_thres"):
            applied_value = event_loop.set_car_detector_conf_thres(value)

        if hasattr(self, "current_car_conf_label"):
            self.current_car_conf_label.setText(f"当前: {applied_value:.2f}")

    def on_xy_maps_changed(self, _state=None):
        """切换像素二维映射；实际生成由主循环后台管理器执行。"""
        enabled = bool(self.xy_maps_checkbox.isChecked())
        event_loop = self._get_main_event_loop_instance()
        if event_loop is not None and hasattr(event_loop, "set_xy_maps_enabled"):
            event_loop.set_xy_maps_enabled(enabled)
            status = event_loop.get_xy_maps_status()
        else:
            status = "等待标定" if enabled else "未启用"
        self.xy_maps_status.setText(f"XY Map 状态: {status}")

    def _refresh_xy_maps_status(self):
        if not hasattr(self, "xy_maps_status"):
            return
        event_loop = self._get_main_event_loop_instance()
        if event_loop is None or not hasattr(event_loop, "get_xy_maps_status"):
            return
        status = event_loop.get_xy_maps_status()
        enabled = bool(event_loop.get_xy_maps_enabled())
        if self.xy_maps_checkbox.isChecked() != enabled:
            self.xy_maps_checkbox.blockSignals(True)
            self.xy_maps_checkbox.setChecked(enabled)
            self.xy_maps_checkbox.blockSignals(False)
        self.xy_maps_status.setText(f"XY Map 状态: {status}")
        if "失败" in status:
            self.xy_maps_status.setStyleSheet("color: red; font-weight: bold;")
        elif "检查" in status:
            self.xy_maps_status.setStyleSheet("color: orange; font-weight: bold;")
        elif "已就绪" in status:
            self.xy_maps_status.setStyleSheet("color: green;")
        else:
            self.xy_maps_status.setStyleSheet("")

        if hasattr(event_loop, "consume_xy_map_consistency_events"):
            for event in event_loop.consume_xy_map_consistency_events():
                self._append_system_log(str(event.get("message", "")))

    def save_outpost_roi(self, roi, update_result_text=True, log_message=True):
        """确认保存前哨站 ROI，并在主循环存在时立即更新裁剪区域。"""
        if roi is None:
            return False

        try:
            x1, y1, x2, y2 = [int(round(value)) for value in roi]
        except Exception:
            QMessageBox.warning(self, "ROI 无效", "前哨站 ROI 坐标无效，请重新框选。")
            return False

        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "ROI 无效", "前哨站 ROI 宽高必须大于 0。")
            return False

        self.outpost_roi = (x1, y1, x2, y2)
        ########################################################################
        self._set_outpost_roi_inputs(self.outpost_roi)
        ########################################################################
        event_loop = self._get_main_event_loop_instance()
        if event_loop is not None and hasattr(event_loop, "set_outpost_roi"):
            event_loop.set_outpost_roi(self.outpost_roi)

        if update_result_text:
            self._sync_calibration_result_outpost_roi_text()
        if log_message:
            self._append_system_log(f"前哨站 ROI 已保存: {self.outpost_roi}")
        return True

    def _format_outpost_roi_result_text(self):
        """生成随外参一起发布/显示的前哨站 ROI 文本。"""
        if self.outpost_roi is None:
            return ""
        x1, y1, x2, y2 = self.outpost_roi
        return (
            "\n\n前哨站 ROI:\n"
            f"左上点: ({x1}, {y1})\n"
            f"右下点: ({x2}, {y2})"
        )

    def _sync_calibration_result_outpost_roi_text(self):
        """在外参结果显示框中同步当前前哨站 ROI。"""
        if not hasattr(self, "calibration_result"):
            return
        text = self.calibration_result.toPlainText()
        marker = "\n\n前哨站 ROI:"
        if marker in text:
            text = text.split(marker, 1)[0]
        text += self._format_outpost_roi_result_text()
        self.calibration_result.setText(text)

    def _set_outpost_roi_inputs(self, roi):
        """将当前前哨站 ROI 同步到主界面的四个微调框。"""
        if roi is None or not hasattr(self, "outpost_roi_inputs"):
            return

        keys = ("x1", "y1", "x2", "y2")
        self._updating_outpost_roi_inputs = True
        try:
            for key, value in zip(keys, roi):
                spin = self.outpost_roi_inputs.get(key)
                if spin is None:
                    continue
                spin.blockSignals(True)
                spin.setValue(int(round(value)))
                spin.blockSignals(False)
        finally:
            self._updating_outpost_roi_inputs = False

    def _get_outpost_roi_from_inputs(self):
        """读取主界面四个 ROI 微调框的当前值。"""
        if not hasattr(self, "outpost_roi_inputs"):
            return None
        try:
            return tuple(
                int(self.outpost_roi_inputs[key].value())
                for key in ("x1", "y1", "x2", "y2")
            )
        except Exception:
            return None

    def _on_outpost_roi_input_changed(self):
        """ROI 微调框变化时，若 ROI 合法则立即同步到主循环。"""
        if getattr(self, "_updating_outpost_roi_inputs", False):
            return
        roi = self._get_outpost_roi_from_inputs()
        if roi is None:
            return
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            return
        self.save_outpost_roi(roi, update_result_text=True, log_message=False)
    ########################################################################

    def start_tracking(self):
        # if self.calibrated_R is not None and self.calibrated_T is not None:
        from main_event_loop import MainEventLoop
        self.main_event_loop: MainEventLoop = MainEventLoop._instance
        self.main_event_loop.set_calibrated_param(
            self.calibrated_R, self.calibrated_T
        )
        ########################################################################
        # 将前哨站 ROI 传入主循环，用于每帧裁图判断前哨站状态。
        if hasattr(self.main_event_loop, "set_outpost_roi"):
            self.main_event_loop.set_outpost_roi(self.outpost_roi)
        ########################################################################
        ########################################################################
        # 将前哨站二值化配置传入主循环。
        if hasattr(self.main_event_loop, "set_outpost_light_config"):
            self.main_event_loop.set_outpost_light_config(self.get_outpost_light_config())
        ########################################################################
        if hasattr(self.main_event_loop, "set_timing_debug_enabled"):
            self.main_event_loop.set_timing_debug_enabled(self.timing_debug_enabled)
        ########################################################################
        # 启动前在 UI 中调整过的整车检测阈值，需要在主循环启动时同步到 tracker。
        if hasattr(self.main_event_loop, "set_car_detector_conf_thres") and hasattr(
            self, "car_conf_input"
        ):
            applied_value = self.main_event_loop.set_car_detector_conf_thres(
                self.car_conf_input.value()
            )
            if hasattr(self, "current_car_conf_label"):
                self.current_car_conf_label.setText(f"当前: {applied_value:.2f}")
        ########################################################################

        self.main_event_loop.run()
        # 禁用按钮，避免重复启动
        self.start_tracking_btn.setEnabled(False)
        self.start_tracking_btn.setText("雷达站运行中...")
        # else:
        #     QMessageBox.warning(
        #         self,
        #         "校准未完成",
        #         "请先完成外参校准后再启动雷达站。",
        #     )

    def display_image(self, image, label_widget):
        """显示图像到指定的标签控件"""
        if image is not None:
            image = self._prepare_display_image(image)
            if image is None:
                self._set_label_message(label_widget, "图像获取失败")
                return
            # 将OpenCV图像转换为QImage
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)
            ########################################################################
            # 按当前 QLabel 尺寸缩放，避免主窗口高度变化时图像区域仍保持固定高度。
            target_size = label_widget.contentsRect().size()
            if target_size.width() <= 1 or target_size.height() <= 1:
                target_size = self.target_camera_size
            # pixmap = pixmap.scaled(
            #     self.target_camera_size, Qt.IgnoreAspectRatio, Qt.FastTransformation
            # )
            pixmap = pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.FastTransformation
            )
            ########################################################################
            label_widget.setPixmap(pixmap)
        else:
            self._set_label_message(label_widget, "图像获取失败")

    def _set_label_message(self, label_widget: QLabel, message: str):
        """仅在状态变化时更新文本，避免高频 setText 造成重绘卡顿。"""
        has_pixmap = label_widget.pixmap() is not None
        if (not has_pixmap) and label_widget.text() == message:
            return
        label_widget.clear()
        label_widget.setText(message)   

    def _prepare_display_image(self, image):
        if image is None:
            return None
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        if self.camera_color_order == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    ########################################################################
    def _default_outpost_light_ui_config(self):
        """默认前哨站灯光判断参数。"""
        return {
            "color": "blue",
            ########################################################################
            "reverse": False,
            ########################################################################
            "h_min": 95,
            "h_max": 130,
            "s_min": 80,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255,
            "alive_ratio_threshold": 0.03,
        }

    def _default_outpost_light_thresholds(self, color):
        """给不同颜色提供 params.yaml 缺省时的 HSV 阈值。"""
        defaults = {
            "blue": {"h_min": 95, "h_max": 130, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            "red": {"h_min": 170, "h_max": 10, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            "green": {"h_min": 35, "h_max": 85, "s_min": 60, "s_max": 255, "v_min": 60, "v_max": 255},
            ########################################################################
            # "anti-white": {"h_min": 0, "h_max": 179, "s_min": 0, "s_max": 70, "v_min": 160, "v_max": 255},
            "white": {"h_min": 0, "h_max": 179, "s_min": 0, "s_max": 70, "v_min": 160, "v_max": 255},
            ########################################################################
        }
        return defaults.get(color, defaults["blue"]).copy()

    def _load_outpost_light_ui_config(self, color=None):
        """从 params.yaml 读取前哨站灯光判断参数。"""
        result = self._default_outpost_light_ui_config()
        cfg = self.config.get("outpost_light", {}) if isinstance(self.config, dict) else {}
        cfg = cfg if isinstance(cfg, dict) else {}
        selected_color = str(color or cfg.get("color", result["color"])).lower()
        ########################################################################
        result.update(self._default_outpost_light_thresholds(selected_color))
        ########################################################################
        thresholds = cfg.get("hsv_thresholds", {})
        color_thresholds = thresholds.get(selected_color, {}) if isinstance(thresholds, dict) else {}
        result.update({k: cfg[k] for k in result.keys() & cfg.keys() if k != "color"})
        result.update({k: color_thresholds[k] for k in result.keys() & color_thresholds.keys()})
        ########################################################################
        # result["color"] = selected_color if selected_color in ("blue", "red") else "blue"
        result["color"] = (
            selected_color
            ########################################################################
            # if selected_color in ("blue", "red", "green", "anti-white")
            if selected_color in ("blue", "red", "green", "white")
            ########################################################################
            else "blue"
        )
        ########################################################################
        return result

    def _on_outpost_color_changed(self, color):
        # """切换红/蓝阈值时，用配置默认值回填阈值框。"""
        """切换颜色类型时，用配置默认值回填阈值框。"""
        cfg = self._load_outpost_light_ui_config(color)
        for key, spin in self.outpost_threshold_inputs.items():
            spin.blockSignals(True)
            spin.setValue(int(cfg[key]))
            spin.blockSignals(False)
        self.apply_outpost_light_config()

    def get_outpost_light_config(self):
        """读取 UI 当前前哨站灯光判断参数。"""
        config = {
            "color": self.outpost_color_combo.currentText(),
            ########################################################################
            "reverse": bool(self.outpost_reverse_checkbox.isChecked()),
            ########################################################################
            "alive_ratio_threshold": float(self.outpost_ratio_spin.value()),
        }
        for key, spin in self.outpost_threshold_inputs.items():
            config[key] = int(spin.value())
        return config

    def apply_outpost_light_config(self):
        """将 UI 前哨站灯光判断参数应用到主循环。"""
        ########################################################################
        # if getattr(self, "main_event_loop", None) is None:
        #     return
        # if hasattr(self.main_event_loop, "set_outpost_light_config"):
        #     self.main_event_loop.set_outpost_light_config(self.get_outpost_light_config())
        event_loop = self._get_main_event_loop_instance()
        if event_loop is None:
            return
        if hasattr(event_loop, "set_outpost_light_config"):
            event_loop.set_outpost_light_config(self.get_outpost_light_config())
        ########################################################################

    def display_outpost_binary_image(self, image):
        """显示前哨站二值化图像。"""
        if image is None:
            self._set_label_message(self.outpost_binary_label, "等待前哨站二值图")
            return

        image = self._prepare_display_image(image)
        if image is None:
            self._set_label_message(self.outpost_binary_label, "前哨站二值图无效")
            return

        height, width, channel = image.shape
        q_image = QImage(image.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.outpost_binary_label.contentsRect().size(),
            Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.outpost_binary_label.setPixmap(pixmap)

    def update_outpost_alive_status(self, alive=None, ratio=None):
        """刷新前哨站当前存活判断显示。"""
        if not hasattr(self, "outpost_alive_status_label"):
            return

        if alive is None:
            self.outpost_alive_status_label.setText("当前：等待")
            self.outpost_alive_status_label.setStyleSheet("color: gray; font-weight: bold;")
            return

        state_text = "存活" if bool(alive) else "死亡"
        if ratio is None:
            text = f"当前：{state_text}"
        else:
            text = f"当前：{state_text}  比例：{float(ratio):.3f}"
        color = "green" if bool(alive) else "red"
        self.outpost_alive_status_label.setText(text)
        self.outpost_alive_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    ########################################################################

    def update_images(self):
        """更新图像显示"""
        merged_track_image = None
        if (
            hasattr(self, "main_event_loop")
            and self.main_event_loop is not None
            and hasattr(self.main_event_loop, "track_vis_img")
        ):
            merged_track_image = self.main_event_loop.track_vis_img

        single_view_active = bool(
            getattr(self, "single_visualization_view", True)
            and merged_track_image is not None
        )
        visualization_rendered = False
        if single_view_active:
            # 合并结果只转换、缩放和绘制一次，并占用原来的两个图像区域。
            self.display_image(merged_track_image, self.visualization_label)
            visualization_rendered = True
        else:
            image, time_stamp = self.camera.get_image_latest(
                "radar_station", timeout=getattr(self, "image_fetch_timeout_sec", 0.0)
            )

        if not single_view_active and image is not None:
            if (
                hasattr(self, "main_event_loop")
                and self.main_event_loop is not None
                and hasattr(self.main_event_loop, "detect_vis_img")
            ):
                if (detect_img := self.main_event_loop.detect_vis_img) is not None:
                    self.display_image(detect_img, self.visualization_label)
                    visualization_rendered = True
                ########################################################################
                # 雷达站刚启动时 detect_vis_img 可能尚未产出，此时回退显示原始相机帧。
                else:
                    self.display_image(image, self.visualization_label)
                    visualization_rendered = True
                ########################################################################
            else:
                self.display_image(image, self.visualization_label)
                visualization_rendered = True

        if visualization_rendered:
            self._camera_frame_received = True
            self._tracker_frame_received = True
        elif not self._camera_frame_received:
            self._set_label_message(self.visualization_label, "等待检测与追踪画面")
        ########################################################################
        # 显示主循环输出的前哨站二值化调试图。
        if (
            hasattr(self, "main_event_loop")
            and self.main_event_loop is not None
            and hasattr(self.main_event_loop, "outpost_binary_image")
        ):
            ########################################################################
            # self.display_outpost_binary_image(self.main_event_loop.outpost_binary_image)
            # self.update_outpost_alive_status(
            #     getattr(self.main_event_loop, "outpost_alive", None),
            #     getattr(self.main_event_loop, "outpost_alive_ratio", None),
            # )
            binary_image = self.main_event_loop.outpost_binary_image
            self.display_outpost_binary_image(binary_image)
            if binary_image is None:
                self.update_outpost_alive_status()
            else:
                self.update_outpost_alive_status(
                    getattr(self.main_event_loop, "outpost_alive", None),
                    getattr(self.main_event_loop, "outpost_alive_ratio", None),
                )
            ########################################################################
        else:
            self._set_label_message(self.outpost_binary_label, "等待前哨站二值图")
            self.update_outpost_alive_status()
        ########################################################################
        # # TODO: 实现相机图像获取和显示
        # # TODO: 实现雷达点云图像显示
        pass

    def update_robot_indicator_status(self, robot_id, is_marked):
        """更新机器人标记状态指示器"""
        if robot_id not in self.robot_status_indicators:
            return
        
        indicator = self.robot_status_indicators[robot_id]
        indicator['is_marked'] = is_marked
        
        if is_marked:
            # 已标记 - 绿色
            indicator['dot'].setStyleSheet("color: #4CAF50; font-size: 16px;")
            indicator['label'].setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            # 未标记 - 红色
            indicator['dot'].setStyleSheet("color: #F44336; font-size: 16px;")
            indicator['label'].setStyleSheet("color: #F44336; font-weight: normal;")

    def update_double_damage_status(self, trigger_state):
        """更新双倍易伤状态机状态显示"""
        state_name = getattr(trigger_state, "name", str(trigger_state))
        self.double_damage_status.setText(f"双倍易伤状态机: {state_name}")

        if state_name == "ACTIVE":
            self.double_damage_status.setStyleSheet("color: red; font-weight: bold; background-color: #ffeeee;")
        elif state_name == "REQUESTING":
            self.double_damage_status.setStyleSheet("color: #ff8c00; font-weight: bold; background-color: #fff4e6;")
        elif state_name == "ARMED":
            self.double_damage_status.setStyleSheet("color: blue; font-weight: bold;")
        elif state_name == "COOLDOWN":
            self.double_damage_status.setStyleSheet("color: #8a2be2; font-weight: bold;")
        else:
            self.double_damage_status.setStyleSheet("color: green; font-weight: bold;")

    def update_double_damage_count(self, used_count, total_count):
        """更新双倍易伤触发次数显示"""
        remaining_count = total_count - used_count
        self.double_damage_count.setText(f"可触发次数: {used_count}/{total_count}")
        
        # 根据剩余次数设置颜色
        if remaining_count <= 0:
            self.double_damage_count.setStyleSheet("color: red; font-weight: bold")
        elif remaining_count <= 1:
            self.double_damage_count.setStyleSheet("color: orange; font-weight: bold")
        else:
            self.double_damage_count.setStyleSheet("color: blue; font-weight: bold")

    def get_robot_mark_status(self, robot_id):
        """
        获取指定机器人的标记状态
        
        Args:
            robot_id (str): 机器人ID，如 "1Hero", "2Engineer" 等
        
        Returns:
            bool: True表示已标记，False表示未标记
        """
        if hasattr(self, 'referee') and self.referee is not None:
            try:
                # 获取雷达标记进度消息
                mark_progress = self.referee.radar_mark_progress_msg
                
                # 根据机器人ID映射到对应的标记状态字段
                robot_mark_mapping = {
                    "1Hero": mark_progress.enemy_hero,
                    "2Engineer": mark_progress.enemy_engineer, 
                    "3Standard": mark_progress.enemy_standard_3,
                    "4Standard": mark_progress.enemy_standard_4,
                    "7Sentry": mark_progress.enemy_sentry
                }
                
                # 获取对应机器人的标记状态
                if robot_id in robot_mark_mapping:
                    return bool(robot_mark_mapping[robot_id])
                else:
                    # 如果robot_id不在映射中，返回False
                    return False
                    
            except AttributeError as e:
                # 如果访问字段失败，返回False
                return False
        
        # 默认返回未标记状态
        return False

    def update_robot_mark_status(self):
        """更新所有机器人标记状态"""
        if not hasattr(self, 'robot_status_indicators'):
            return
        
        # 遍历所有机器人类型，获取其标记状态
        for robot_id in self.robot_status_indicators.keys():
            # 获取标记状态
            is_marked = self.get_robot_mark_status(robot_id)
            # 更新显示
            self.update_robot_indicator_status(robot_id, is_marked)
    def update_status(self):
        """更新设备状态"""
        if hasattr(self.camera, "is_connected"):
            camera_connected = self.camera.is_connected()
            self.camera_status.update_status(camera_connected)
            if camera_connected:
                self.get_camera_fps()

        if hasattr(self.referee, "is_connected"):
            referee_connected = self.referee.is_connected()
            self.referee_status.update_status(referee_connected)

        if hasattr(self.referee, "is_sentry_connected"):
            sentry_connected = self.referee.is_sentry_connected
            self.sentry_status.update_status(sentry_connected)

        if hasattr(self.referee, "sentry_received_flag"):
            sentry_received = self.referee.sentry_received_flag
            self.sentry_received.update_status(sentry_received)

        self._refresh_xy_maps_status()

        if hasattr(self, "main_event_loop") and self.main_event_loop is not None:
            localization_fps = float(getattr(self.main_event_loop, "localization_fps", 0.0))
            self.localization_fps_label.setText(f"定位帧率: {localization_fps:.2f} FPS")
        else:
            self.localization_fps_label.setText("定位帧率: 0.00 FPS")

        referee_tx_fps = float(getattr(self.referee, "radar2client_tx_fps", 0.0))
        self.referee_tx_fps_label.setText(f"裁判系统定位发送帧率: {referee_tx_fps:.2f} FPS")

        if self.timing_debug_enabled:
            timing_debug_summary = "跟踪定位耗时调试: 等待主循环启动。"
            if hasattr(self, "main_event_loop") and self.main_event_loop is not None and hasattr(self.main_event_loop, "get_timing_debug_text"):
                timing_debug_summary = self.main_event_loop.get_timing_debug_text()
            # if self.timing_debug_text.toPlainText() != timing_debug_summary:
            #     self.timing_debug_text.setPlainText(timing_debug_summary)
            ##########
            self._set_textedit_plain_text_preserve_scroll(
                self.timing_debug_text,
                timing_debug_summary,
            )
            ##########
        # elif self.timing_debug_text.toPlainText() != "跟踪定位耗时调试已关闭。":
        #     self.timing_debug_text.setPlainText("跟踪定位耗时调试已关闭。")
        elif self.timing_debug_text.toPlainText() != "跟踪定位耗时调试已关闭。":
            ##########
            self._set_textedit_plain_text_preserve_scroll(
                self.timing_debug_text,
                "跟踪定位耗时调试已关闭。",
            )
            ##########

        ##########
        if self.timing_debug_detail_window is not None:
            self.timing_debug_detail_window.set_detail_text(
                self.timing_debug_text.toPlainText()
            )
            if (
                hasattr(self, "main_event_loop")
                and self.main_event_loop is not None
                and hasattr(self.main_event_loop, "get_pipeline_module_text")
            ):
                self.timing_debug_detail_window.set_pipeline_text(
                    self.main_event_loop.get_pipeline_module_text()
                )
        ##########

        from driver.referee.referee_comm import FACTION

        if hasattr(self.referee, "get_faction"):
            match self.referee.get_faction():
                case FACTION.BLUE:
                    self.team_label.setStyleSheet("color: blue; font-size: 12px")
                    self.team_label.setText("当前阵营：蓝方")
                case FACTION.RED:
                    self.team_label.setStyleSheet("color: red; font-size: 12px")
                    self.team_label.setText("当前阵营：红方")

        # 新增：更新双倍易伤状态机状态
        if hasattr(self.referee, "trigger_state"):
            self.update_double_damage_status(self.referee.trigger_state)

        # 新增：更新双倍易伤触发次数
        if hasattr(self.referee, "double_vulnerability_count"):
            total_count = 2
            used_count = self.referee.double_vulnerability_count
            self.update_double_damage_count(used_count, total_count)
        
        # 新增：更新机器人标记状态
        self.update_robot_mark_status()

    def update_field_projection(self):
        """按裁判系统定位消息发送频率刷新场地投影。"""
        if (
            hasattr(self, "main_event_loop")
            and self.main_event_loop is not None
            and hasattr(self.main_event_loop, "divisions_pos")
            and hasattr(self.main_event_loop, "faction")
            and hasattr(self.field_view, "update_vehicle_data")
        ):
            completed = None
            if hasattr(self.main_event_loop, "get_latest_pipeline_output"):
                completed = self.main_event_loop.get_latest_pipeline_output()
            if completed is not None:
                divisions_pos = completed.divisions_pos
                faction = completed.faction
                lidar_points_cm = [
                    (float(point[0]) * 100.0, float(point[1]) * 100.0)
                    for point in completed.tracker_lidar_points
                ]
                lidar_point_owners = dict(completed.lidar_point_owners)
                projection_points_cm = completed.projection_points_cm
            else:
                divisions_pos = self.main_event_loop.divisions_pos
                faction = self.main_event_loop.faction
                lidar_points_cm = self.main_event_loop.lidar_points_cm
                lidar_point_owners = getattr(
                    self.main_event_loop, "lidar_point_owners", {}
                )
                projection_points_cm = self.main_event_loop.projection_points_cm
            field_need_update = False
            self.field_view.update_vehicle_data(
                divisions_pos,
                faction,
                trigger_update=False,
            )
            field_need_update = True
            if hasattr(self.referee, "get_info_wave_enemy_positions_snapshot"):
                info_wave_positions_cm = (
                    self.referee.get_info_wave_enemy_positions_snapshot(
                        require_fresh=True
                    )
                )
                self.field_view.update_info_wave_data(
                    info_wave_positions_cm,
                    trigger_update=False,
                )
            if hasattr(self.referee, "get_sentry_hole_scores_snapshot"):
                self.field_view.update_sentry_hole_scores(
                    self.referee.get_sentry_hole_scores_snapshot(),
                    trigger_update=False,
                )
            if hasattr(self.main_event_loop, "lidar_points_cm") and hasattr(
                self.field_view, "update_lidar_data"
            ):
                self.field_view.update_lidar_data(
                    lidar_points_cm,
                    lidar_point_owners,
                    trigger_update=False,
                )
            if hasattr(self.main_event_loop, "projection_points_cm") and hasattr(
                self.field_view, "update_projection_data"
            ):
                self.field_view.update_projection_data(
                    projection_points_cm,
                    trigger_update=False,
                )
            if field_need_update:
                self.field_view.update()

    def get_camera_fps(self):
        self.fps_label.setText(f"相机帧率: {self.camera.get_fps():.2f} fps")
        
    def on_reset_referee(self):
        """重置裁判系统状态"""
        if hasattr(self.referee, "reset_double_trigger_state"):
            self.referee.reset_double_trigger_state()
            self.log_text.append("已重置裁判系统状态")
            print("Request count:", self.referee.request_count)
        else:
            QMessageBox.warning(self, "错误", "裁判系统不支持重置操作")

    def test(self):
        try:
            import for_test
            info = for_test.test()
            self.log_text.append(info)
            
        except Exception as e:
            error_msg = f"Test失败: {str(e)}"
            self.log_text.append(error_msg)
            QMessageBox.critical(self, "错误", error_msg)
            self.logger.error(error_msg)

    ##########
    def open_serial_receive_window(self):
        """打开串口接收调试窗口。"""
        if self.serial_receive_window is None:
            self.serial_receive_window = SerialReceiveDebugWindow(
                self.referee,
                self,
                window_title="接收数据",
                snapshot_method_name="get_msg_queue_snapshot",
            )
        if self.serial_receive_window.isMinimized():
            self.serial_receive_window.showNormal()
        self.serial_receive_window.show()
        self.serial_receive_window.start_updates()
        self.serial_receive_window.raise_()
        self.serial_receive_window.activateWindow()

    def open_serial_send_window(self):
        """打开发送调试窗口。"""
        if self.serial_send_window is None:
            self.serial_send_window = SerialReceiveDebugWindow(
                self.referee,
                self,
                window_title="发送数据",
                snapshot_method_name="get_tx_msg_queue_snapshot",
            )
        if self.serial_send_window.isMinimized():
            self.serial_send_window.showNormal()
        self.serial_send_window.show()
        self.serial_send_window.start_updates()
        self.serial_send_window.raise_()
        self.serial_send_window.activateWindow()
    ##########

    def on_exposure_confirm(self):
        """确认设置曝光值"""
        try:
            exposure_value = self.exposure_input.value()

            # 设置相机曝光
            if hasattr(self.camera, "set_exposure"):
                success = self.camera.set_exposure(exposure_value)

                if success:
                    self.current_exposure_label.setText(f"当前: {exposure_value}μs")
                    self.current_exposure_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    self.logger.info(f"曝光值设置成功: {exposure_value}μs")

                    # 在系统日志中显示
                    self.log_text.append(
                        f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 曝光值已设置为: {exposure_value}μs"
                    )
                    self.get_current_exposure()
                else:
                    self.current_exposure_label.setStyleSheet(
                        "color: red; font-size: 12px;"
                    )
                    self.logger.error(f"曝光值设置失败: {exposure_value}μs")

                    # 显示错误信息
                    QMessageBox.warning(
                        self,
                        "设置失败",
                        f"无法设置曝光值为 {exposure_value}μs\n请检查相机连接状态",
                    )
            else:
                self.logger.warning("相机不支持曝光设置")
                QMessageBox.information(
                    self, "功能不支持", "当前相机驱动不支持曝光设置功能"
                )

        except Exception as e:
            self.logger.error(f"设置曝光时发生错误: {e}")
            QMessageBox.critical(self, "错误", f"设置曝光时发生错误:\n{str(e)}")

    def on_exposure_value_changed(self, value):
        """曝光值改变时的回调"""
        # 实时显示即将设置的值
        self.current_exposure_label.setText(f"待设置: {value}μs")
        self.current_exposure_label.setStyleSheet("color: orange; font-size: 12px;")

    def on_gain_confirm(self):
        """确认设置增益值"""
        try:
            gain_value = self.gain_input.value()

            # 设置相机增益
            if hasattr(self.camera, "set_gain"):
                success = self.camera.set_gain(gain_value)

                if success:
                    self.current_gain_label.setText(f"当前: {gain_value}dB")
                    self.current_gain_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    self.logger.info(f"增益值设置成功: {gain_value}dB")

                    # 在系统日志中显示
                    self.log_text.append(
                        f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 增益值已设置为: {gain_value}dB"
                    )
                    self.get_current_gain()
                else:
                    self.current_gain_label.setStyleSheet(
                        "color: red; font-size: 12px;"
                    )
                    self.logger.error(f"增益值设置失败: {gain_value}dB")

                    # 显示错误信息
                    QMessageBox.warning(
                        self,
                        "设置失败",
                        f"无法设置增益值为 {gain_value}dB\n请检查相机连接状态",
                    )
            else:
                self.logger.warning("相机不支持增益设置")
                QMessageBox.information(
                    self, "功能不支持", "当前相机驱动不支持增益设置功能"
                )

        except Exception as e:
            self.logger.error(f"设置增益时发生错误: {e}")
            QMessageBox.critical(self, "错误", f"设置增益时发生错误:\n{str(e)}")

    def on_gain_value_changed(self, value):
        """增益值改变时的回调"""
        # 实时显示即将设置的值
        self.current_gain_label.setText(f"待设置: {value}dB")
        self.current_gain_label.setStyleSheet("color: orange; font-size: 12px;")

    def open_keypoint_calibration(self):
        """打开关键点标定对话框"""
        try:
            # 获取当前视频帧作为背景
            current_image, _ = self.camera.get_image_latest("radar_station", timeout=1)

            if current_image is None:
                QMessageBox.warning(self, "错误", "无法获取当前视频帧，请检查相机连接")
                return

            # 确保图像是RGB格式
            background_image = self._prepare_display_image(current_image)
            if background_image is None:
                QMessageBox.warning(self, "错误", "无法处理当前视频帧")
                return

            keypoint_file = self._resolve_pnp_keypoint_file()

            # 创建并显示标定对话框
            calibration_dialog = KeypointCalibrationDialog(
                self,
                background_image,
                pnp_config=self.config,
                keypoint_file=keypoint_file,
                ########################################################################
                # 将当前已保存的 ROI 带入标定窗口，重新框选时可以在旧区域基础上修改。
                initial_outpost_roi=self.outpost_roi,
                ########################################################################
            )

            # if calibration_dialog.exec_() == QDialog.Accepted:
            # 获取标定结果
            # Wait for the calibration dialog to finish
            calibration_dialog.exec_()
            result = calibration_dialog.get_calibration_result()
            if result["success"]:
                # 在主界面显示校准结果
                self.display_calibration_result(result)
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] "
                    f"外参校准成功，残差: {result['residual']:.4f}"
                )
            else:
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 外参校准失败"
                )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开标定界面时发生错误:\n{str(e)}")
            self.logger.error(f"Error opening calibration dialog: {e}")

    def _get_initial_extrinsics(self):
        """优先返回当前已生效外参，其次返回配置中的默认外参。"""
        if self.calibrated_R is not None and self.calibrated_T is not None:
            return self.calibrated_R.copy(), self.calibrated_T.copy()

        transform_cfg = self.config.get("transform", {}) if isinstance(self.config, dict) else {}
        rotation_matrix = transform_cfg.get("R")
        translation_vector = transform_cfg.get("t")
        if rotation_matrix is None or translation_vector is None:
            return None, None

        try:
            rotation_matrix = np.array(rotation_matrix, dtype=np.float64).reshape(3, 3)
            translation_vector = np.array(translation_vector, dtype=np.float64).reshape(3, 1)
        except Exception as e:
            self.logger.warning("解析默认外参失败: %s", e)
            return None, None
        return rotation_matrix, translation_vector

    def open_manual_extrinsics_dialog(self):
        """打开手动输入外参对话框。"""
        try:
            initial_R, initial_t = self._get_initial_extrinsics()
            input_dialog = ManualExtrinsicsDialog(
                self,
                initial_rotation=initial_R,
                initial_translation=initial_t,
            )
            input_dialog.exec_()
            result = input_dialog.get_calibration_result()
            if result["success"]:
                self.display_calibration_result(result)
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] 已手动输入相机外参"
                )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开外参输入界面时发生错误:\n{str(e)}")
            self.logger.error("Error opening manual extrinsics dialog: %s", e)

    def _resolve_pnp_keypoint_file(self) -> str:
        """根据 params.yaml 的 field 配置解析 PnP 关键点文件路径。"""
        field_cfg = self.config.get("field", {}) if isinstance(self.config, dict) else {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        keypoint_file_candidates = [
            field_cfg.get("keypoints"),
            field_cfg.get("keypoint"),
        ]

        for candidate in keypoint_file_candidates:
            if not candidate:
                continue
            candidate_path = str(candidate)
            if not os.path.isabs(candidate_path):
                candidate_path = os.path.normpath(
                    os.path.join(project_root, candidate_path)
                )
            if os.path.exists(candidate_path):
                return candidate_path
            self.logger.warning("配置中的关键点文件不存在: %s", candidate_path)

        fallback_path = os.path.join(project_root, "transform", "keypoint_6.txt")
        self.logger.warning(
            "未在 field 配置中找到可用关键点文件，回退到默认文件: %s", fallback_path
        )
        return fallback_path

    def _save_calibration_extrinsics(self, result, R, t):
        """将本次 UI 外参结果保存到 recorder/extrinstics。"""
        try:
            save_dir = get_extrinsics_record_dir()
            os.makedirs(save_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"{timestamp}.txt")
            suffix = 1
            while os.path.exists(save_path):
                save_path = os.path.join(save_dir, f"{timestamp}_{suffix:02d}.txt")
                suffix += 1

            residual = result.get("residual")
            source = result.get("source", "pnp")
            record_text = format_extrinsics_record(
                R,
                t,
                residual=residual,
                source=source,
                timestamp=timestamp,
            )

            with open(save_path, "w", encoding="utf-8") as file:
                file.write(record_text)

            latest_path = os.path.join(save_dir, "latest.txt")
            with open(latest_path, "w", encoding="utf-8") as file:
                file.write(record_text)

            if hasattr(self, "log_text"):
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] "
                    f"外参已保存: {save_path}"
                )
            return save_path
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.error("保存外参失败: %s", e)
            if hasattr(self, "log_text"):
                self.log_text.append(
                    f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] "
                    f"保存外参失败: {e}"
                )
            return None

    def display_calibration_result(self, result):
        """在主界面显示校准结果"""
        if result["success"]:
            source = result.get("source", "pnp")
            is_manual_input = source == "manual_input"

            # 更新校准状态
            self.calibration_status.setText(
                "雷达标定状态: 已手动设置" if is_manual_input else "雷达标定状态: 已校准"
            )
            self.calibration_status.setStyleSheet("color: green; font-weight: bold;")

            # 显示旋转矩阵和平移向量
            R = result["rotation_matrix"]
            t = result["translation_vector"]
            self.calibrated_R = R.copy()
            self.calibrated_T = t.copy()
            self._save_calibration_extrinsics(result, R, t)
            event_loop = self._get_main_event_loop_instance()
            if event_loop is not None:
                event_loop.set_calibrated_param(self.calibrated_R, self.calibrated_T)
                event_loop.set_xy_maps_enabled(self.xy_maps_checkbox.isChecked())
            ########################################################################
            # 保存校准窗口中框选的前哨站 ROI，后续启动主循环时传入 MainEventLoop。
            # self.outpost_roi = result.get("outpost_roi")
            if result.get("outpost_roi") is not None:
                self.save_outpost_roi(
                    result.get("outpost_roi"),
                    update_result_text=False,
                    log_message=False,
                )
            ########################################################################
            self.start_tracking_btn.setEnabled(True)

            residual = result.get("residual")
            result_text = "已手动输入外参\n" if is_manual_input else "校准成功!\n"
            if residual is not None:
                result_text += f"残差: {residual:.4f}\n"
            result_text += f"旋转矩阵 R:\n"
            result_text += f"[{R[0,0]:.4f} {R[0,1]:.4f} {R[0,2]:.4f}]\n"
            result_text += f"[{R[1,0]:.4f} {R[1,1]:.4f} {R[1,2]:.4f}]\n"
            result_text += f"[{R[2,0]:.4f} {R[2,1]:.4f} {R[2,2]:.4f}]\n\n"
            result_text += f"平移向量 t:\n[{t[0,0]:.4f} {t[1,0]:.4f} {t[2,0]:.4f}]"
            ########################################################################
            # 将前哨站 ROI 四点边界和外参结果一起显示出来，便于发布/核对。
            # if self.outpost_roi is not None:
            #     x1, y1, x2, y2 = self.outpost_roi
            #     result_text += (
            #         f"\n\n前哨站 ROI:\n"
            #         f"左上: ({x1}, {y1})\n"
            #         f"右下: ({x2}, {y2})"
            #     )
            result_text += self._format_outpost_roi_result_text()
            ########################################################################

            self.calibration_result.setText(result_text)
        else:
            self.calibration_status.setText("雷达标定状态: 校准失败")
            self.calibration_status.setStyleSheet("color: red; font-weight: bold;")
            self.calibration_result.setText("校准失败，请重新标定")

    def get_current_exposure(self):
        """获取当前相机曝光值"""
        try:
            if hasattr(self.camera, "get_exposure"):
                current_exposure = self.camera.get_exposure()
                if current_exposure is not None:
                    current_exposure = int(current_exposure)
                    self.exposure_input.setValue(current_exposure)
                    self.current_exposure_label.setText(f"当前: {current_exposure}μs")
                    self.current_exposure_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    return current_exposure
        except Exception as e:
            self.logger.error(f"获取当前曝光值失败: {e}")
        return None

    def get_current_gain(self):
        """获取当前相机增益值"""
        try:
            if hasattr(self.camera, "gain"):
                current_gain = self.camera.gain
                if current_gain is not None:
                    current_gain = int(current_gain)
                    self.gain_input.setValue(current_gain)
                    self.current_gain_label.setText(f"当前: {current_gain}dB")
                    self.current_gain_label.setStyleSheet(
                        "color: green; font-size: 12px;"
                    )
                    return current_gain
        except Exception as e:
            self.logger.error(f"获取当前增益值失败: {e}")
        return None

    def _append_system_log(self, message: str):
        """向系统日志追加一条消息。"""
        if hasattr(self, "log_text") and self.log_text is not None:
            self.log_text.append(
                f"[{QDateTime.currentDateTime().toString('hh:mm:ss')}] {message}"
            )

    def _get_runtime_lidar_client(self):
        """获取运行时雷达客户端实例。"""
        from main_event_loop import MainEventLoop

        event_loop = getattr(self, "main_event_loop", None)
        if event_loop is None:
            event_loop = MainEventLoop._instance
        if event_loop is None:
            return None
        return getattr(event_loop, "lidar_client", None)

    def open_sensor_recording_dialog(self):
        """打开传感器录制配置对话框。"""
        dialog = SensorRecordingDialog(
            self,
            is_recording=self.sensor_recording_active,
            default_record_camera=self.sensor_record_camera_enabled,
            default_record_lidar=self.sensor_record_lidar_enabled,
            default_camera_fps=self.sensor_record_camera_fps,
            default_export_yolo=self.sensor_record_yolo_enabled,
            default_export_armor_yolo=self.sensor_record_armor_yolo_enabled,
            default_save_root_dir=self.sensor_record_save_root_dir,
            current_session_dir=self.sensor_record_session_dir,
        )
        dialog.exec_()
        result = dialog.get_result()
        action = result.get("action")
        if action == "start":
            self.start_sensor_recording(
                record_camera=result["record_camera"],
                record_lidar=result["record_lidar"],
                camera_fps=result["camera_fps"],
                export_yolo=result["export_yolo"],
                export_armor_yolo=result["export_armor_yolo"],
                save_root_dir=result["save_root_dir"],
            )
        elif action == "stop":
            self.stop_sensor_recording()

    def start_sensor_recording(
        self,
        record_camera: bool,
        record_lidar: bool,
        camera_fps: int,
        export_yolo: bool,
        export_armor_yolo: bool,
        save_root_dir: str,
    ):
        """开始录制传感器数据。"""
        if not record_camera and not record_lidar:
            QMessageBox.warning(self, "配置错误", "请至少选择一种要录制的传感器数据。")
            return
        if export_yolo and not record_camera:
            QMessageBox.warning(
                self,
                "配置错误",
                "导出 YOLO 标签需要同时勾选相机录制。",
            )
            return
        if export_armor_yolo and not record_camera:
            QMessageBox.warning(
                self,
                "配置错误",
                "导出装甲板 YOLO 标签需要同时勾选相机录制。",
            )
            return
        if record_camera and camera_fps <= 0:
            QMessageBox.warning(self, "配置错误", "相机录制帧率必须大于 0。")
            return

        save_root_dir = os.path.abspath(save_root_dir.strip() or "recorder")
        os.makedirs(save_root_dir, exist_ok=True)

        lidar_client = self._get_runtime_lidar_client()
        if record_lidar and lidar_client is None:
            QMessageBox.warning(
                self,
                "雷达不可用",
                "当前没有可用的雷达客户端，无法录制雷达数据。",
            )
            return

        if self.sensor_recording_active:
            self.stop_sensor_recording()

        session_name = QDateTime.currentDateTime().toString(
            "'sensor_record_'yyyyMMdd_hhmmss_zzz"
        )
        session_dir = os.path.join(save_root_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)

        self.sensor_recording_active = True
        self.sensor_record_camera_enabled = record_camera
        self.sensor_record_lidar_enabled = record_lidar
        self.sensor_record_camera_fps = int(camera_fps)
        self.sensor_record_save_root_dir = save_root_dir
        self.sensor_record_session_dir = session_dir
        self.sensor_record_camera_dir = ""
        self.sensor_record_camera_video_path = ""
        self.sensor_record_camera_timestamp_path = ""
        self.sensor_record_camera_metrics_path = ""
        self.sensor_record_camera_recorder = None
        self.sensor_record_camera_error_logged = False
        self.sensor_record_camera_frame_count = 0
        self.sensor_record_lidar_dir = ""
        self.sensor_record_yolo_enabled = bool(export_yolo)
        self.sensor_record_yolo_dataset_dir = ""
        self.sensor_record_yolo_images_dir = ""
        self.sensor_record_yolo_labels_dir = ""
        self.sensor_record_yolo_frame_count = 0
        self.sensor_record_last_yolo_stamp = None
        self.sensor_record_last_yolo_write_time = None
        self.sensor_record_yolo_wait_logged = False
        self.sensor_record_armor_yolo_enabled = bool(export_armor_yolo)
        self.sensor_record_armor_yolo_dataset_dir = ""
        self.sensor_record_armor_yolo_images_dir = ""
        self.sensor_record_armor_yolo_labels_dir = ""
        self.sensor_record_armor_yolo_frame_count = 0
        self.sensor_record_last_armor_yolo_stamp = None
        self.sensor_record_last_armor_yolo_write_time = None
        self.sensor_record_armor_yolo_wait_logged = False

        if record_camera:
            self.sensor_record_camera_dir = os.path.join(session_dir, "camera")
            os.makedirs(self.sensor_record_camera_dir, exist_ok=True)
            self.sensor_record_camera_video_path = os.path.join(
                self.sensor_record_camera_dir, "camera_record.mp4"
            )
            self.sensor_record_camera_timestamp_path = os.path.join(
                self.sensor_record_camera_dir, "camera_timestamps.csv"
            )
            self.sensor_record_camera_metrics_path = os.path.join(
                self.sensor_record_camera_dir, "recording_metrics.json"
            )
            try:
                self.sensor_record_camera_recorder = AsyncCameraRecorder(
                    camera=self.camera,
                    output_path=self.sensor_record_camera_video_path,
                    timestamp_path=self.sensor_record_camera_timestamp_path,
                    metrics_path=self.sensor_record_camera_metrics_path,
                    target_fps=float(self.sensor_record_camera_fps),
                    color_order=self.camera_color_order,
                    group_id="sensor_recorder",
                    queue_size=self.sensor_record_camera_queue_size,
                    encoder_preset=self.sensor_record_camera_encoder_preset,
                    encoder_bitrate_mbps=(
                        self.sensor_record_camera_encoder_bitrate_mbps
                    ),
                    encoder_max_bitrate_mbps=(
                        self.sensor_record_camera_encoder_max_bitrate_mbps
                    ),
                    ffmpeg_path=self.sensor_record_camera_ffmpeg_path,
                )
                self.sensor_record_camera_recorder.start()
            except Exception as e:
                if self.sensor_record_camera_recorder is not None:
                    try:
                        self.sensor_record_camera_recorder.stop()
                    except Exception:
                        pass
                self.sensor_record_camera_recorder = None
                QMessageBox.warning(
                    self,
                    "录制失败",
                    f"无法启动相机异步录像器:\n{e}",
                )
                self.sensor_recording_active = False
                self.sensor_record_camera_enabled = False
                self.sensor_record_lidar_enabled = False
                self.sensor_record_session_dir = ""
                self.sensor_record_camera_dir = ""
                self.sensor_record_camera_video_path = ""
                self.sensor_record_camera_timestamp_path = ""
                self.sensor_record_camera_metrics_path = ""
                return
            self._save_sensor_calibration_snapshot()
            self.sensor_recording_timer.start()
        else:
            self.sensor_recording_timer.stop()

        if record_lidar and lidar_client is not None:
            self.sensor_record_lidar_dir = lidar_client.start_recording(session_dir)

        if self.sensor_record_yolo_enabled:
            self.sensor_record_yolo_dataset_dir = os.path.join(
                session_dir, "yolo_car_dataset"
            )
            self.sensor_record_yolo_images_dir = os.path.join(
                self.sensor_record_yolo_dataset_dir, "images", "train"
            )
            self.sensor_record_yolo_labels_dir = os.path.join(
                self.sensor_record_yolo_dataset_dir, "labels", "train"
            )
            os.makedirs(self.sensor_record_yolo_images_dir, exist_ok=True)
            os.makedirs(self.sensor_record_yolo_labels_dir, exist_ok=True)
            self._write_sensor_record_yolo_metadata()

        if self.sensor_record_armor_yolo_enabled:
            self.sensor_record_armor_yolo_dataset_dir = os.path.join(
                session_dir, "yolo_armor_dataset"
            )
            self.sensor_record_armor_yolo_images_dir = os.path.join(
                self.sensor_record_armor_yolo_dataset_dir, "images", "train"
            )
            self.sensor_record_armor_yolo_labels_dir = os.path.join(
                self.sensor_record_armor_yolo_dataset_dir, "labels", "train"
            )
            os.makedirs(self.sensor_record_armor_yolo_images_dir, exist_ok=True)
            os.makedirs(self.sensor_record_armor_yolo_labels_dir, exist_ok=True)
            self._write_sensor_record_armor_yolo_metadata()

        self.sensor_record_btn.setText("停止录制传感器")
        self.sensor_record_btn.setStyleSheet(
            "QPushButton { background-color: #D32F2F; color: white; font-weight: bold; }"
        )
        self._append_system_log(
            f"开始录制传感器数据: 相机={'开' if record_camera else '关'} "
            f"雷达={'开' if record_lidar else '关'} "
            f"车辆YOLO导出={'开' if self.sensor_record_yolo_enabled else '关'} "
            f"装甲板YOLO导出={'开' if self.sensor_record_armor_yolo_enabled else '关'} "
            f"相机帧率={self.sensor_record_camera_fps} 目录={session_dir}"
        )

    def stop_sensor_recording(self):
        """停止录制传感器数据。"""
        if not self.sensor_recording_active:
            return

        self.sensor_recording_timer.stop()
        lidar_client = self._get_runtime_lidar_client()
        if self.sensor_record_lidar_enabled and lidar_client is not None:
            lidar_client.stop_recording()

        session_dir = self.sensor_record_session_dir
        camera_stats = None
        recorder = self.sensor_record_camera_recorder
        if recorder is not None:
            try:
                camera_stats = recorder.stop()
            except Exception as error:
                self.logger.error("停止相机异步录像器失败: %s", error)
                camera_stats = recorder.get_diagnostics()
            self.sensor_record_camera_frame_count = int(
                camera_stats.get("encoded", self.sensor_record_camera_frame_count)
            )
        camera_frame_count = self.sensor_record_camera_frame_count
        yolo_frame_count = self.sensor_record_yolo_frame_count
        armor_yolo_frame_count = self.sensor_record_armor_yolo_frame_count

        self.sensor_recording_active = False
        self.sensor_record_camera_enabled = False
        self.sensor_record_lidar_enabled = False
        self.sensor_record_session_dir = ""
        self.sensor_record_camera_dir = ""
        self.sensor_record_camera_video_path = ""
        self.sensor_record_camera_timestamp_path = ""
        self.sensor_record_camera_metrics_path = ""
        self.sensor_record_camera_recorder = None
        self.sensor_record_camera_error_logged = False
        self.sensor_record_lidar_dir = ""
        self.sensor_record_yolo_enabled = False
        self.sensor_record_yolo_dataset_dir = ""
        self.sensor_record_yolo_images_dir = ""
        self.sensor_record_yolo_labels_dir = ""
        self.sensor_record_last_yolo_stamp = None
        self.sensor_record_last_yolo_write_time = None
        self.sensor_record_yolo_wait_logged = False
        self.sensor_record_armor_yolo_enabled = False
        self.sensor_record_armor_yolo_dataset_dir = ""
        self.sensor_record_armor_yolo_images_dir = ""
        self.sensor_record_armor_yolo_labels_dir = ""
        self.sensor_record_last_armor_yolo_stamp = None
        self.sensor_record_last_armor_yolo_write_time = None
        self.sensor_record_armor_yolo_wait_logged = False

        self.sensor_record_btn.setText("录制传感器数据")
        self.sensor_record_btn.setStyleSheet("")
        camera_detail = ""
        if camera_stats is not None:
            camera_detail = (
                f" 采集观察 {int(camera_stats.get('capture_seen', 0))} 帧"
                f" 采样跳过 {int(camera_stats.get('sampled_out', 0))} 帧"
                f" 队列溢出 {int(camera_stats.get('queue_overflow_dropped', 0))} 帧"
                f" 编码平均 {float(camera_stats.get('encoder_write_avg_ms', 0.0)):.1f}ms"
            )
            fatal_error = camera_stats.get("fatal_error")
            if fatal_error:
                camera_detail += f" 编码异常={fatal_error}"
        self._append_system_log(
            f"停止录制传感器数据: 相机视频写入 {camera_frame_count} 帧 "
            f"车辆YOLO样本写入 {yolo_frame_count} 帧 "
            f"装甲板YOLO样本写入 {armor_yolo_frame_count} 帧"
            f"{camera_detail} 目录={session_dir}"
        )

    def _convert_frame_to_bgr(self, frame):
        """将当前相机帧统一转换为 OpenCV 写盘使用的 BGR 格式。"""
        if frame is None:
            return None

        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if frame.shape[2] == 4:
            if self.camera_color_order == "bgr":
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        if self.camera_color_order == "bgr":
            return np.ascontiguousarray(frame.copy())
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _build_yolo_label_line(self, class_id: int, bbox, image_width: int, image_height: int):
        """将像素框转换为 YOLO 标准标签行。"""
        if bbox is None or image_width <= 0 or image_height <= 0:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = min(max(x1, 0.0), float(image_width))
        x2 = min(max(x2, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        y2 = min(max(y2, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            return None

        x_center = ((x1 + x2) * 0.5) / float(image_width)
        y_center = ((y1 + y2) * 0.5) / float(image_height)
        width = (x2 - x1) / float(image_width)
        height = (y2 - y1) / float(image_height)
        return f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _write_yolo_dataset_metadata(self, dataset_dir: str, class_names):
        """写出 Roboflow/YOLO 可直接识别的数据集描述文件。"""
        if not dataset_dir:
            return

        normalized_class_names = [str(name) for name in class_names]
        data_yaml_path = os.path.join(dataset_dir, "data.yaml")
        classes_txt_path = os.path.join(dataset_dir, "classes.txt")
        with open(data_yaml_path, "w", encoding="utf-8") as file:
            file.write("path: .\n")
            file.write("train: images/train\n")
            file.write("val: images/train\n")
            file.write(f"nc: {len(normalized_class_names)}\n")
            file.write("names:\n")
            for class_id, class_name in enumerate(normalized_class_names):
                file.write(f"  {class_id}: {class_name}\n")
        with open(classes_txt_path, "w", encoding="utf-8") as file:
            if normalized_class_names:
                file.write("\n".join(normalized_class_names))
                file.write("\n")

    def _write_sensor_record_yolo_metadata(self):
        """写出车辆 YOLO 数据集元信息。"""
        self._write_yolo_dataset_metadata(self.sensor_record_yolo_dataset_dir, ["car"])

    def _write_sensor_record_armor_yolo_metadata(self):
        """写出装甲板 YOLO 数据集元信息。"""
        armor_class_names = self.config.get("armor_detector", {}).get("class_names", [])
        self._write_yolo_dataset_metadata(
            self.sensor_record_armor_yolo_dataset_dir,
            armor_class_names,
        )

    def _build_armor_yolo_label_line(self, detection, image_width: int, image_height: int):
        """将装甲板框转换为整帧坐标下的 YOLO 标准标签行。"""
        class_id = int(getattr(detection, "class_id", -1))
        armor_box = getattr(detection, "armor_box", None)
        car_box = getattr(detection, "car_box", None)
        armor_class_names = self.config.get("armor_detector", {}).get("class_names", [])
        if class_id < 0 or class_id >= len(armor_class_names):
            return None
        if armor_box is None or car_box is None or len(armor_box) != 4 or len(car_box) != 4:
            return None

        x1 = float(car_box[0]) + float(armor_box[0])
        y1 = float(car_box[1]) + float(armor_box[1])
        x2 = float(car_box[0]) + float(armor_box[2])
        y2 = float(car_box[1]) + float(armor_box[3])
        return self._build_yolo_label_line(class_id, [x1, y1, x2, y2], image_width, image_height)

    def _get_latest_yolo_export_payload(self):
        """读取与检测结果对齐的原始帧和时间戳。"""
        from main_event_loop import MainEventLoop

        event_loop = getattr(self, "main_event_loop", None)
        if event_loop is None:
            event_loop = MainEventLoop._instance
        if event_loop is None:
            return None, None, None

        if hasattr(event_loop, "get_latest_pipeline_output"):
            completed = event_loop.get_latest_pipeline_output()
            if completed is not None:
                return (
                    completed.frame.copy(),
                    float(completed.frame_stamp),
                    list(completed.detections),
                )

        frame = getattr(event_loop, "latest_detection_source_frame", None)
        stamp = getattr(event_loop, "latest_detection_source_stamp", None)
        detections = getattr(event_loop, "latest_detection_results", None)
        if frame is None or stamp is None or detections is None:
            return None, None, None
        return frame.copy(), float(stamp), list(detections)

    def _export_yolo_record(self, current_time: float):
        """导出整帧图片和对应的车辆 YOLO 伪标签。"""
        if (
            not self.sensor_record_yolo_enabled
            or not self.sensor_record_yolo_images_dir
            or not self.sensor_record_yolo_labels_dir
        ):
            return

        if self.sensor_record_last_yolo_write_time is not None:
            frame_interval = 1.0 / max(float(self.sensor_record_camera_fps), 1.0)
            if current_time - self.sensor_record_last_yolo_write_time < frame_interval:
                return

        export_frame, export_stamp, detections = self._get_latest_yolo_export_payload()
        if export_frame is None:
            if not self.sensor_record_yolo_wait_logged:
                self.sensor_record_yolo_wait_logged = True
                self._append_system_log("车辆YOLO导出等待雷达站检测结果，启动雷达站后会自动开始写入。")
            return

        if export_stamp == self.sensor_record_last_yolo_stamp:
            return

        frame_to_save = self._convert_frame_to_bgr(export_frame)
        if frame_to_save is None:
            return

        image_height, image_width = frame_to_save.shape[:2]
        sample_index = self.sensor_record_yolo_frame_count
        sample_name = f"frame_{sample_index:06d}_{int(round(export_stamp * 1000.0))}"
        image_path = os.path.join(self.sensor_record_yolo_images_dir, f"{sample_name}.jpg")
        label_path = os.path.join(self.sensor_record_yolo_labels_dir, f"{sample_name}.txt")

        if not cv2.imwrite(image_path, frame_to_save):
            self.logger.error("保存 YOLO 图片失败: %s", image_path)
            return

        label_lines = []
        for detection in detections:
            label_line = self._build_yolo_label_line(
                class_id=0,
                bbox=getattr(detection, "car_box", None),
                image_width=image_width,
                image_height=image_height,
            )
            if label_line is not None:
                label_lines.append(label_line)

        with open(label_path, "w", encoding="utf-8") as file:
            if label_lines:
                file.write("\n".join(label_lines))
                file.write("\n")

        self.sensor_record_yolo_frame_count += 1
        self.sensor_record_last_yolo_stamp = export_stamp
        self.sensor_record_last_yolo_write_time = current_time
        self.sensor_record_yolo_wait_logged = False

    def _export_armor_yolo_record(self, current_time: float):
        """导出整帧图片和对应的装甲板 YOLO 伪标签。"""
        if (
            not self.sensor_record_armor_yolo_enabled
            or not self.sensor_record_armor_yolo_images_dir
            or not self.sensor_record_armor_yolo_labels_dir
        ):
            return

        if self.sensor_record_last_armor_yolo_write_time is not None:
            frame_interval = 1.0 / max(float(self.sensor_record_camera_fps), 1.0)
            if current_time - self.sensor_record_last_armor_yolo_write_time < frame_interval:
                return

        export_frame, export_stamp, detections = self._get_latest_yolo_export_payload()
        if export_frame is None:
            if not self.sensor_record_armor_yolo_wait_logged:
                self.sensor_record_armor_yolo_wait_logged = True
                self._append_system_log("装甲板YOLO导出等待雷达站检测结果，启动雷达站后会自动开始写入。")
            return

        if export_stamp == self.sensor_record_last_armor_yolo_stamp:
            return

        frame_to_save = self._convert_frame_to_bgr(export_frame)
        if frame_to_save is None:
            return

        image_height, image_width = frame_to_save.shape[:2]
        sample_index = self.sensor_record_armor_yolo_frame_count
        sample_name = f"frame_{sample_index:06d}_{int(round(export_stamp * 1000.0))}"
        image_path = os.path.join(self.sensor_record_armor_yolo_images_dir, f"{sample_name}.jpg")
        label_path = os.path.join(self.sensor_record_armor_yolo_labels_dir, f"{sample_name}.txt")

        if not cv2.imwrite(image_path, frame_to_save):
            self.logger.error("保存装甲板 YOLO 图片失败: %s", image_path)
            return

        label_lines = []
        for detection in detections:
            label_line = self._build_armor_yolo_label_line(
                detection=detection,
                image_width=image_width,
                image_height=image_height,
            )
            if label_line is not None:
                label_lines.append(label_line)

        with open(label_path, "w", encoding="utf-8") as file:
            if label_lines:
                file.write("\n".join(label_lines))
                file.write("\n")

        self.sensor_record_armor_yolo_frame_count += 1
        self.sensor_record_last_armor_yolo_stamp = export_stamp
        self.sensor_record_last_armor_yolo_write_time = current_time
        self.sensor_record_armor_yolo_wait_logged = False

    def process_sensor_recording(self):
        """刷新后台录像状态，并维持可选的 YOLO 数据集导出。"""
        if (
            not self.sensor_recording_active
            or not self.sensor_record_camera_enabled
            or not self.sensor_record_camera_dir
        ):
            return

        recorder = self.sensor_record_camera_recorder
        if recorder is None:
            return
        diagnostics = recorder.get_diagnostics()
        self.sensor_record_camera_frame_count = int(diagnostics.get("encoded", 0))
        fatal_error = diagnostics.get("fatal_error")
        if fatal_error and not self.sensor_record_camera_error_logged:
            self.sensor_record_camera_error_logged = True
            self._append_system_log(f"相机异步录像异常，正在停止录制: {fatal_error}")
            self.stop_sensor_recording()
            return

        current_time = time.time()
        self._export_yolo_record(current_time)
        self._export_armor_yolo_record(current_time)

    def _save_sensor_calibration_snapshot(self):
        """把本次录像对应的相机内参和最新外参固化到 camera 目录。"""
        if not self.sensor_record_camera_dir:
            return
        snapshot_path = os.path.join(
            self.sensor_record_camera_dir, "calibration_snapshot.yaml"
        )
        extrinsics_path = os.path.join(
            self.sensor_record_camera_dir, "extrinsics.txt"
        )
        transform_cfg = self.config.get("transform", {})
        image_width = int(getattr(self.camera, "width", 0) or 0)
        image_height = int(getattr(self.camera, "height", 0) or 0)
        snapshot = {
            "version": 1,
            "recorded_wall_time": time.time(),
            "params_source": get_params_config_path(),
            "intrinsics": {
                # 宽高来自相机接口，只用于人工核验；投影运行时直接读取 img.shape。
                "image_width": image_width,
                "image_height": image_height,
                "K": transform_cfg.get("K"),
                "dist_coeffs": transform_cfg.get("dist_coeffs"),
            },
            "extrinsics": None,
        }

        rotation = getattr(self, "calibrated_R", None)
        translation = getattr(self, "calibrated_T", None)
        record_text = None
        source = "ui_runtime"
        if rotation is None or translation is None:
            latest_path = get_latest_extrinsics_path()
            try:
                rotation, translation, record_text = load_extrinsics_record(latest_path)
                source = latest_path
            except Exception as error:
                self.logger.warning("录制时没有可保存的相机外参: %s", error)

        if rotation is not None and translation is not None:
            rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
            snapshot["extrinsics"] = {
                "source": source,
                "rotation_matrix": rotation.tolist(),
                "translation_vector": translation.tolist(),
            }
            if record_text is None:
                record_text = format_extrinsics_record(
                    rotation,
                    translation,
                    source="recording_snapshot",
                )

        try:
            with open(snapshot_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(snapshot, file, allow_unicode=True, sort_keys=False)
            if record_text is not None:
                with open(extrinsics_path, "w", encoding="utf-8") as file:
                    file.write(record_text)
            self._append_system_log(f"已保存本次录像标定快照: {snapshot_path}")
        except OSError as error:
            self.logger.error("保存录像标定快照失败: %s", error)

    # def closeEvent(self, event):
    #     """窗口关闭前停止录制，避免后台线程继续写文件。"""
    #     try:
    #         self.stop_sensor_recording()
    #     except Exception as e:
    #         self.logger.error("关闭窗口时停止录制失败: %s", e)
    #     super().closeEvent(event)
    ##########
    def closeEvent(self, event):
        """窗口关闭前停止录制，并关闭串口调试子窗口。"""
        try:
            self.stop_sensor_recording()
        except Exception as e:
            self.logger.error("关闭窗口时停止录制失败: %s", e)

        try:
            if self.serial_receive_window is not None:
                self.serial_receive_window.close()
            ########################################################################
            if self.serial_send_window is not None:
                self.serial_send_window.close()
            ########################################################################
        except Exception as e:
            self.logger.error("关闭串口调试窗口失败: %s", e)

        super().closeEvent(event)
    ##########


class FieldProjectionWidget(QWidget):
    """场地投影显示组件"""

    def __init__(self):
        super().__init__()
        self.field_width_cm = 1500.0
        self.field_height_cm = 2800.0
        self.ruler_margin = 48  # 右侧和底部留给坐标卡尺
        self.map_margin = 10

        self.field_image = cv2.imread("./field/field_image.PNG")
        if self.field_image is not None:
            self.field_image = cv2.rotate(
                self.field_image, cv2.ROTATE_90_COUNTERCLOCKWISE
            )
        self._field_pixmap = None
        if self.field_image is not None:
            height, width, channel = self.field_image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                self.field_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_BGR888,
            ).copy()
            self._field_pixmap = QPixmap.fromImage(q_image)

        self.setMinimumSize(240, 180)

        self.vehicles = []  # 车辆位置数据
        self.info_wave_positions_cm = {}  # 信息波 0x0A01 对方机器人坐标
        self.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 0.0}
        self.lidar_points = []  # 雷达点位置数据
        self.lidar_point_owners = {}  # 本帧真实匹配：雷达点序号 -> class_id
        self.projection_points = []  # car 底边中心经过射线投影后的二维位置
        self.show_enemy = True
        self.show_ally = True
        self.show_guessed = False
        self.show_lidar = False
        self.show_projection = False
        self.show_info_wave = True
        self.show_tracking = True

        self.faction = None
        self._static_layer_cache_key = None
        self._static_layer_cache = None

    def paintEvent(self, event):
        """绘制场地、坐标卡尺、雷达点和车辆。"""
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            flipped = self.should_flip_field()
            field_rect = self._get_field_display_rect()
            static_layer = self._get_static_layer(field_rect, flipped)
            if static_layer is not None:
                painter.drawPixmap(0, 0, static_layer)
            else:
                self.draw_field(painter, field_rect, flipped)
                self.draw_coordinate_ruler(painter, field_rect, flipped)
                self.draw_sentry_holes(painter, field_rect, flipped)
            self.draw_sentry_hole_scores(painter, field_rect, flipped)
            self.draw_lidar_points(painter, field_rect, flipped)
            self.draw_projection_points(painter, field_rect, flipped)
            self.draw_vehicles_without_text(painter, field_rect, flipped)
            self.draw_info_wave_vehicles_without_text(painter, field_rect, flipped)
            self.draw_vehicles_text_only(painter, field_rect, flipped)
            self.draw_info_wave_vehicles_text_only(painter, field_rect, flipped)

    def _get_static_layer(self, field_rect: QRectF, flipped: bool) -> QPixmap | None:
        """缓存静态层（底图+坐标卡尺），减少高频重绘开销。"""
        widget_size = self.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            return None

        cache_key = (
            widget_size.width(),
            widget_size.height(),
            round(field_rect.x(), 2),
            round(field_rect.y(), 2),
            round(field_rect.width(), 2),
            round(field_rect.height(), 2),
            bool(flipped),
        )
        if self._static_layer_cache_key == cache_key and self._static_layer_cache is not None:
            return self._static_layer_cache

        static_layer = QPixmap(widget_size)
        static_layer.fill(Qt.transparent)
        with QPainter(static_layer) as static_painter:
            static_painter.setRenderHint(QPainter.Antialiasing)
            self.draw_field(static_painter, field_rect, flipped)
            self.draw_coordinate_ruler(static_painter, field_rect, flipped)
            self.draw_sentry_holes(static_painter, field_rect, flipped)

        self._static_layer_cache_key = cache_key
        self._static_layer_cache = static_layer
        return static_layer

    def _get_field_display_rect(self) -> QRectF:
        """计算地图实际显示区域（保持场地长宽比，不随窗口拉伸变形）。"""
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        available = QRectF(
            self.map_margin,
            self.map_margin,
            max(1.0, width - self.map_margin * 2 - self.ruler_margin),
            max(1.0, height - self.map_margin * 2 - self.ruler_margin),
        )
        field_aspect = self.field_width_cm / self.field_height_cm
        available_aspect = available.width() / available.height()

        if available_aspect > field_aspect:
            draw_height = available.height()
            draw_width = draw_height * field_aspect
            draw_x = available.left() + (available.width() - draw_width) * 0.5
            draw_y = available.top()
        else:
            draw_width = available.width()
            draw_height = draw_width / field_aspect
            draw_x = available.left()
            draw_y = available.top() + (available.height() - draw_height) * 0.5
        return QRectF(draw_x, draw_y, draw_width, draw_height)

    def _is_vehicle_visible(self, class_id: int, position=None) -> bool:
        from driver.referee.referee_comm import FACTION

        is_blue_robot = class_id < 5
        # 与追踪状态机的 class 约定保持一致：0~4 永远是蓝方，5~9 永远是红方。
        # 阵营未知时状态机默认按红方运行，因此 UI 也使用红方作为一致的回退。
        own_faction = self.faction
        if own_faction not in (FACTION.RED, FACTION.BLUE):
            own_faction = FACTION.RED
        is_ally = (
            is_blue_robot if own_faction == FACTION.BLUE else not is_blue_robot
        )
        if is_ally and not self.show_ally:
            return False
        if not is_ally and not self.show_enemy:
            return False
        if (
            position is not None
            and bool(getattr(position, "is_guessed", False))
            and not self.show_guessed
        ):
            return False
        return True

    @staticmethod
    def _source_text(position) -> str:
        """把状态机来源转换为右侧场地图中的短中文说明。"""
        source = str(getattr(position, "source", "invalid"))
        labels = {
            "camera_lidar": "视觉+雷达",
            "gray_camera_lidar": "灰甲+视觉+雷达",
            "ally_lidar": "哨兵+雷达",
            "ally_feedback": "哨兵反馈",
            "lidar_history": "历史+雷达",
            "gray_radar_guessed": "灰甲雷达猜点",
            "radar_guessed": "纯雷达猜点",
            "kf_predicted": "二维KF预测",
            "fixed_guessed": "固定点猜测",
            "invalid": "无效",
        }
        text = labels.get(source, source or "来源未知")
        if bool(getattr(position, "is_guessed", False)) and "猜" not in text:
            text = f"{text}（猜测身份）"
        return text

    def _field_to_widget(
        self, pos_x_cm: float, pos_y_cm: float, field_rect: QRectF, flipped: bool
    ) -> tuple[float, float]:
        """场地坐标映射到窗口坐标，保证与背景地图严格一致。"""
        pos_x = max(0.0, min(self.field_height_cm, float(pos_x_cm)))
        pos_y = max(0.0, min(self.field_width_cm, float(pos_y_cm)))

        if flipped:
            norm_x = pos_y / self.field_width_cm
            norm_y = pos_x / self.field_height_cm
        else:
            norm_x = (self.field_width_cm - pos_y) / self.field_width_cm
            norm_y = (self.field_height_cm - pos_x) / self.field_height_cm

        x_widget = field_rect.left() + norm_x * field_rect.width()
        y_widget = field_rect.top() + norm_y * field_rect.height()
        return x_widget, y_widget

    def _robot_style(self, class_id: int) -> tuple[QColor, QColor, str]:
        if class_id < 5:
            label = ["B1", "B2", "B3", "B4", "B7"][class_id]
            return QColor(0, 100, 255), QColor(0, 100, 255), label
        label = ["R1", "R2", "R3", "R4", "R7"][class_id - 5]
        return QColor(255, 50, 50), QColor(255, 50, 50), label

    def draw_field(self, painter, field_rect: QRectF = None, flipped: bool = False):
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        if self._field_pixmap is not None:
            painter.save()
            if flipped:
                center = field_rect.center()
                painter.translate(center)
                painter.rotate(180)
                painter.translate(-center)
            source_rect = QRectF(self._field_pixmap.rect())
            painter.drawPixmap(
                field_rect,
                self._field_pixmap,
                source_rect,
            )
            painter.restore()
        else:
            painter.fillRect(field_rect, QColor(245, 245, 245))

        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(field_rect)

    def draw_coordinate_ruler(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """绘制坐标卡尺，标注场地坐标与显示窗口映射关系。"""
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        painter.save()
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        metrics = painter.fontMetrics()

        bottom = field_rect.bottom()
        right = field_rect.right()

        # 底部 y 轴卡尺（单位 cm）
        for y_cm in range(0, int(self.field_width_cm) + 1, 100):
            x_widget, _ = self._field_to_widget(0.0, float(y_cm), field_rect, flipped)
            is_major = (y_cm % 500 == 0) or (y_cm == int(self.field_width_cm))
            tick_len = 8 if is_major else 4
            painter.drawLine(
                QPointF(x_widget, bottom), QPointF(x_widget, bottom + tick_len)
            )
            if is_major:
                text = str(y_cm)
                text_w = metrics.horizontalAdvance(text)
                baseline_y = bottom + tick_len + metrics.ascent() + 2
                painter.drawText(QPointF(x_widget - text_w * 0.5, baseline_y), text)

        # 右侧 x 轴卡尺（单位 cm）
        for x_cm in range(0, int(self.field_height_cm) + 1, 200):
            _, y_widget = self._field_to_widget(float(x_cm), 0.0, field_rect, flipped)
            is_major = (x_cm % 400 == 0) or (x_cm == int(self.field_height_cm))
            tick_len = 8 if is_major else 4
            painter.drawLine(
                QPointF(right, y_widget), QPointF(right + tick_len, y_widget)
            )
            if is_major:
                text = str(x_cm)
                painter.drawText(
                    QPointF(right + tick_len + 2, y_widget + metrics.ascent() * 0.4),
                    text,
                )

        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawText(
            QPointF(field_rect.left(), bottom + self.ruler_margin - 10), "Y(cm)"
        )
        painter.save()
        painter.translate(right + self.ruler_margin - 8, field_rect.bottom())
        painter.rotate(-90)
        painter.drawText(QPointF(0, 0), "X(cm)")
        painter.restore()

        origin_x, origin_y = self._field_to_widget(0.0, 0.0, field_rect, flipped)
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(QColor(60, 60, 60))
        painter.drawEllipse(QPointF(origin_x, origin_y), 2.5, 2.5)
        painter.drawText(QPointF(origin_x + 5, origin_y - 5), "O(0,0)")
        painter.restore()

    def _sentry_hole_centers_cm(self) -> dict[str, tuple[float, float]]:
        """返回当前阵营下两个敌方半场区域的圆心，单位厘米。"""
        from driver.referee.referee_comm import FACTION, RefereeCommManager

        centers = dict(RefereeCommManager.SENTRY_HOLES_RED_M)
        if self.faction == FACTION.BLUE:
            centers = {
                name: (
                    RefereeCommManager.FIELD_LENGTH_M - center_x,
                    RefereeCommManager.FIELD_WIDTH_M - center_y,
                )
                for name, (center_x, center_y) in centers.items()
            }
        return {
            name: (center_x * 100.0, center_y * 100.0)
            for name, (center_x, center_y) in centers.items()
        }

    def draw_sentry_holes(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """在场地图静态层绘制 long_hole 和 short_hole 圆形区域。"""
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        radius_cm = RefereeCommManager.SENTRY_HOLE_RADIUS_M * 100.0
        styles = {
            "long_hole": (QColor(255, 140, 0, 210), "long_hole"),
            "short_hole": (QColor(150, 60, 210, 210), "short_hole"),
        }
        painter.save()
        painter.setClipRect(field_rect)
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        for hole_name, (center_x_cm, center_y_cm) in (
            self._sentry_hole_centers_cm().items()
        ):
            color, label = styles[hole_name]
            center_x, center_y = self._field_to_widget(
                center_x_cm, center_y_cm, field_rect, flipped
            )
            # 场地图保持 15:28 等比例显示，因此两个方向共用同一个像素半径。
            radius_px = radius_cm / self.field_width_cm * field_rect.width()
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 28))
            painter.drawEllipse(QPointF(center_x, center_y), radius_px, radius_px)
            # 用实心点和十字准星明确标出区域圆心。
            center_radius_px = 5.0
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(center_x, center_y),
                center_radius_px,
                center_radius_px,
            )
            cross_half_length_px = 9.0
            painter.drawLine(
                QPointF(center_x - cross_half_length_px, center_y),
                QPointF(center_x + cross_half_length_px, center_y),
            )
            painter.drawLine(
                QPointF(center_x, center_y - cross_half_length_px),
                QPointF(center_x, center_y + cross_half_length_px),
            )
            painter.setPen(QPen(color, 2))
            painter.drawText(
                QPointF(center_x - radius_px + 4, center_y - radius_px + 16),
                label,
            )
        painter.restore()

    def draw_sentry_hole_scores(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """在两个区域圆心旁实时绘制当前威胁分数。"""
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        styles = {
            "long_hole": QColor(255, 140, 0, 230),
            "short_hole": QColor(150, 60, 210, 230),
        }
        painter.save()
        painter.setClipRect(field_rect)
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for hole_name, (center_x_cm, center_y_cm) in (
            self._sentry_hole_centers_cm().items()
        ):
            center_x, center_y = self._field_to_widget(
                center_x_cm, center_y_cm, field_rect, flipped
            )
            score = float(self.sentry_hole_scores.get(hole_name, 0.0))
            text = f"score={score:.2f}"
            text_width = metrics.horizontalAdvance(text)
            text_height = metrics.height()
            if center_x >= field_rect.center().x():
                text_x = center_x - text_width - 12.0
            else:
                text_x = center_x + 12.0
            text_y = center_y - 8.0
            background_rect = QRectF(
                text_x - 3.0,
                text_y - metrics.ascent() - 2.0,
                text_width + 6.0,
                text_height + 4.0,
            )
            painter.fillRect(background_rect, QColor(255, 255, 255, 205))
            painter.setPen(QPen(styles[hole_name], 2))
            painter.drawText(QPointF(text_x, text_y), text)
        painter.restore()

    def draw_vehicles_without_text(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """绘制车辆位置（不绘制文本）。"""
        if not self.show_tracking or not self.vehicles:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        robot_radius = 8
        for class_id, pos in enumerate(self.vehicles):
            if not pos.is_valid or not self._is_vehicle_visible(class_id, pos):
                continue

            marker_color, _, _ = self._robot_style(class_id)
            x_widget, y_widget = self._field_to_widget(
                pos.x, pos.y, field_rect, flipped
            )
            x_widget = int(round(x_widget))
            y_widget = int(round(y_widget))

            painter.setBrush(marker_color)
            painter.setPen(QPen(marker_color, 2))
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                x_widget - robot_radius,
                y_widget - robot_radius,
                robot_radius * 2,
                robot_radius * 2,
            )

    def draw_vehicles_text_only(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """仅绘制车辆文本，确保文字不翻转。"""
        if not self.show_tracking or not self.vehicles:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        robot_radius = 8

        for class_id, pos in enumerate(self.vehicles):
            if not pos.is_valid or not self._is_vehicle_visible(class_id, pos):
                continue

            _, text_color, label = self._robot_style(class_id)
            x_widget, y_widget = self._field_to_widget(
                pos.x, pos.y, field_rect, flipped
            )
            source_text = self._source_text(pos)
            coord_text = f"{label}\n({pos.x:.0f}, {pos.y:.0f})\n{source_text}"
            lines = coord_text.split("\n")
            max_width = max(metrics.horizontalAdvance(line) for line in lines)

            text_x = int(round(x_widget + robot_radius + 6))
            text_y = int(round(y_widget - robot_radius - line_height))
            bg_width = max_width + 6
            bg_height = line_height * len(lines) + 6
            text_x = max(0, min(self.width() - bg_width - 1, text_x))
            text_y = max(0, min(self.height() - bg_height - 1, text_y))

            background_rect = QRect(text_x, text_y, bg_width, bg_height)
            painter.fillRect(background_rect, QColor(255, 255, 255, 185))
            painter.setPen(QPen(text_color, 2))
            for i, line in enumerate(lines):
                baseline = text_y + 3 + metrics.ascent() + i * line_height
                painter.drawText(text_x + 3, baseline, line)

    def _info_wave_robot_style(self, robot_name: str) -> tuple[QColor, str]:
        """返回信息波敌方机器人的阵营颜色和编号标签。"""
        from driver.referee.referee_comm import FACTION

        role_number = {
            "hero": "1",
            "engineer": "2",
            "standard_3": "3",
            "standard_4": "4",
            "drone": "6",
            "sentry": "7",
        }.get(robot_name, "?")
        if self.faction == FACTION.BLUE:
            return QColor(255, 50, 50), f"R{role_number}"
        return QColor(0, 100, 255), f"B{role_number}"

    def _iter_valid_info_wave_positions(self):
        """过滤信息波中的空坐标、非数值坐标和场外坐标。"""
        for robot_name, coords in self.info_wave_positions_cm.items():
            try:
                pos_x = float(coords[0])
                pos_y = float(coords[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not np.isfinite(pos_x) or not np.isfinite(pos_y):
                continue
            if (pos_x == 0.0 and pos_y == 0.0) or not (
                0.0 <= pos_x <= self.field_height_cm
                and 0.0 <= pos_y <= self.field_width_cm
            ):
                continue
            yield robot_name, pos_x, pos_y

    def draw_info_wave_vehicles_without_text(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """用菱形轮廓绘制独立的信息波坐标层。"""
        if not self.show_info_wave or not self.info_wave_positions_cm:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        radius = 11
        for robot_name, pos_x, pos_y in self._iter_valid_info_wave_positions():
            team_color, _ = self._info_wave_robot_style(robot_name)
            center_x, center_y = self._field_to_widget(
                pos_x, pos_y, field_rect, flipped
            )
            points = QPolygonF(
                [
                    QPointF(center_x, center_y - radius),
                    QPointF(center_x + radius, center_y),
                    QPointF(center_x, center_y + radius),
                    QPointF(center_x - radius, center_y),
                ]
            )
            painter.setBrush(QColor(255, 255, 255, 80))
            painter.setPen(QPen(team_color, 3))
            painter.drawPolygon(points)

    def draw_info_wave_vehicles_text_only(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """绘制信息波机器人编号、坐标和明确的数据来源。"""
        if not self.show_info_wave or not self.info_wave_positions_cm:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        for robot_name, pos_x, pos_y in self._iter_valid_info_wave_positions():
            text_color, label = self._info_wave_robot_style(robot_name)
            center_x, center_y = self._field_to_widget(
                pos_x, pos_y, field_rect, flipped
            )
            lines = [label, f"({pos_x:.0f}, {pos_y:.0f})", "信息波"]
            max_width = max(metrics.horizontalAdvance(line) for line in lines)
            bg_width = max_width + 6
            bg_height = line_height * len(lines) + 6
            text_x = int(round(center_x - bg_width - 15))
            text_y = int(round(center_y + 13))
            text_x = max(0, min(self.width() - bg_width - 1, text_x))
            text_y = max(0, min(self.height() - bg_height - 1, text_y))
            painter.fillRect(
                QRect(text_x, text_y, bg_width, bg_height),
                QColor(255, 255, 255, 200),
            )
            painter.setPen(QPen(text_color, 2))
            for index, line in enumerate(lines):
                baseline = text_y + 3 + metrics.ascent() + index * line_height
                painter.drawText(text_x + 3, baseline, line)

    def draw_lidar_points(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """绘制雷达点位置。"""
        if not self.show_lidar or not self.lidar_points:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        radius = 4

        for lidar_index, (pos_x, pos_y) in enumerate(self.lidar_points):
            owner_class = self.lidar_point_owners.get(lidar_index)
            if owner_class is None:
                point_color = QColor(0, 200, 0, 180)
                pen_color = QColor(0, 120, 0, 200)
            elif owner_class < 5:
                point_color = QColor(0, 100, 255, 210)
                pen_color = QColor(0, 40, 180, 240)
            else:
                point_color = QColor(255, 50, 50, 210)
                pen_color = QColor(180, 0, 0, 240)
            painter.setBrush(point_color)
            painter.setPen(QPen(pen_color, 1))
            x_widget, y_widget = self._field_to_widget(
                pos_x, pos_y, field_rect, flipped
            )
            painter.drawEllipse(
                int(round(x_widget)) - radius,
                int(round(y_widget)) - radius,
                radius * 2,
                radius * 2,
            )

    def draw_projection_points(
        self, painter, field_rect: QRectF = None, flipped: bool = False
    ):
        """以黄色圆点和十字绘制本帧视觉 car 的二维投影位置。"""
        if not self.show_projection or not self.projection_points:
            return
        if field_rect is None:
            field_rect = self._get_field_display_rect()

        point_color = QColor(255, 220, 0, 220)
        pen_color = QColor(170, 120, 0, 240)
        radius = 5
        painter.setBrush(point_color)
        painter.setPen(QPen(pen_color, 2))
        for pos_x, pos_y in self.projection_points:
            x_widget, y_widget = self._field_to_widget(
                pos_x, pos_y, field_rect, flipped
            )
            x_center = int(round(x_widget))
            y_center = int(round(y_widget))
            painter.drawEllipse(
                x_center - radius,
                y_center - radius,
                radius * 2,
                radius * 2,
            )
            painter.drawLine(
                x_center - radius - 2, y_center, x_center + radius + 2, y_center
            )
            painter.drawLine(
                x_center, y_center - radius - 2, x_center, y_center + radius + 2
            )

    def draw_vehicles(self, painter):
        """兼容旧调用：绘制车辆点和文本。"""
        field_rect = self._get_field_display_rect()
        flipped = self.should_flip_field()
        self.draw_vehicles_without_text(painter, field_rect, flipped)
        self.draw_vehicles_text_only(painter, field_rect, flipped)

    def update_vehicle_data(self, divisions_pos, faction, trigger_update: bool = True):
        self.vehicles = divisions_pos
        self.faction = faction
        if trigger_update:
            self.update()

    def update_info_wave_data(
        self, positions_cm, trigger_update: bool = True
    ):
        """更新信息波坐标独立图层；传入空字典时清除过期显示。"""
        self.info_wave_positions_cm = dict(positions_cm or {})
        if trigger_update:
            self.update()

    def update_sentry_hole_scores(
        self, scores, trigger_update: bool = True
    ):
        """更新两个哨兵区域的实时分数。"""
        scores = scores or {}
        self.sentry_hole_scores = {
            "long_hole": float(scores.get("long_hole", 0.0)),
            "short_hole": float(scores.get("short_hole", 0.0)),
        }
        if trigger_update:
            self.update()

    def update_lidar_data(
        self, lidar_points, lidar_point_owners=None, trigger_update: bool = True
    ):
        self.lidar_points = lidar_points or []
        self.lidar_point_owners = dict(lidar_point_owners or {})
        if trigger_update:
            self.update()

    def update_projection_data(
        self, projection_points, trigger_update: bool = True
    ):
        self.projection_points = projection_points or []
        if trigger_update:
            self.update()

    def should_flip_field(self):
        from driver.referee.referee_comm import FACTION

        if self.faction is None:
            return False
        if self.faction == FACTION.BLUE:
            return True
        if self.faction == FACTION.RED:
            return False
        if self.faction == FACTION.UNKONWN:
            return False
        return False


class StatusIndicator(QWidget):
    """设备状态指示器"""

    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.connected = False
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.indicator = QLabel("●")
        self.indicator.setFixedSize(20, 20)
        self.label = QLabel(self.device_name)

        layout.addWidget(self.indicator)
        layout.addWidget(self.label)
        layout.addStretch()

        self.update_status(False)

    def update_status(self, connected):
        """更新连接状态"""
        self.connected = connected
        if connected:
            self.indicator.setStyleSheet("color: green; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 已连接")
        else:
            self.indicator.setStyleSheet("color: red; font-size: 16px;")
            self.label.setText(f"{self.device_name}: 未连接")

    def update_display_settings(self):
        """更新显示设置"""
        self.field_view.show_enemy = self.show_enemy_cb.isChecked()
        self.field_view.show_ally = self.show_ally_cb.isChecked()
        self.field_view.update()


class ClickableImageLabel(QLabel):
    """可点击的图像标签"""

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(event.x(), event.y())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动事件，显示坐标"""
        super().mouseMoveEvent(event)
        # 可以在这里添加实时坐标显示功能

    def enterEvent(self, event):
        """鼠标进入事件"""
        self.setCursor(Qt.PointCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """鼠标离开事件"""
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)


class CalibrationGuideDialog(QDialog):
    """标定示意图对话框 - 支持缩放和拖拽"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定示意图")
        self.setModal(True)
        self.resize(800, 600)

        self.setWindowFlags(
            Qt.Window  # 独立窗口
            | Qt.WindowTitleHint  # 显示标题栏
            | Qt.WindowSystemMenuHint  # 显示系统菜单
            | Qt.WindowMinMaxButtonsHint  # 显示最小化/最大化按钮
            | Qt.WindowCloseButtonHint  # 显示关闭按钮
        )

        self.init_ui()

    def init_ui(self):
        """初始化示意图界面"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("关键点标定示意图")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 15px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 图像显示和控制区域
        image_controls_layout = QVBoxLayout()

        # 图像控制按钮
        controls_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重置视图")
        self.reset_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 5px 10px; }"
        )
        self.reset_btn.clicked.connect(self.reset_image_view)

        self.fit_btn = QPushButton("适应窗口")
        self.fit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 10px; }"
        )
        self.fit_btn.clicked.connect(self.fit_image_to_window)

        # 缩放控制
        zoom_label = QLabel("缩放控制:")
        self.zoom_in_btn = QPushButton("放大 (+)")
        self.zoom_out_btn = QPushButton("缩小 (-)")

        self.zoom_in_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        self.zoom_out_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)

        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.fit_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(zoom_label)
        controls_layout.addWidget(self.zoom_out_btn)
        controls_layout.addWidget(self.zoom_in_btn)

        image_controls_layout.addLayout(controls_layout)

        # 示意图显示区域 - 使用支持缩放拖拽的控件
        self.guide_image_label = ZoomableDraggableLabel()
        self.guide_image_label.setMinimumSize(800, 500)

        # 加载并显示示意图
        self.load_guide_image()

        image_controls_layout.addWidget(self.guide_image_label, 1)  # 占据主要空间
        layout.addLayout(image_controls_layout)

        # 操作提示
        tips_label = QLabel("操作提示：鼠标滚轮缩放，左键拖拽移动图像")
        tips_label.setStyleSheet("color: #666; font-size: 12px; margin: 5px;")
        tips_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(tips_label)

        # 说明文字
        instruction_text = QTextEdit()
        instruction_text.setMaximumHeight(100)
        instruction_text.setReadOnly(True)
        instruction_text.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px;"
        )

        guide_text = """标定关键点说明：
1. 按照图中标记的顺序（1→2→3→4→5→6）依次标记关键点  2. 确保标记的点位置准确，这直接影响校准精度
3. 关键点应选择在场地边界或明显特征处  4. 避免选择在阴影或模糊区域的点  5. 标记完成后点击"开始校准"进行PnP求解"""

        instruction_text.setText(guide_text)
        layout.addWidget(instruction_text)

        # 关闭按钮
        button_layout = QHBoxLayout()

        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 10px 20px; }"
        )
        self.close_btn.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        button_layout.addStretch()

        layout.addLayout(button_layout)

    def reset_image_view(self):
        """重置图像视图"""
        self.guide_image_label.reset_view()

    def fit_image_to_window(self):
        """适应窗口大小"""
        self.guide_image_label.fit_to_window()

    def zoom_in(self):
        """放大图像"""
        current_scale = self.guide_image_label.scale_factor
        new_scale = min(self.guide_image_label.max_scale, current_scale + 0.2)
        self.guide_image_label.scale_factor = new_scale
        self.guide_image_label.update_display()

    def zoom_out(self):
        """缩小图像"""
        current_scale = self.guide_image_label.scale_factor
        new_scale = max(self.guide_image_label.min_scale, current_scale - 0.2)
        self.guide_image_label.scale_factor = new_scale
        self.guide_image_label.update_display()

    def load_guide_image(self):
        """加载标定示意图"""
        try:
            # 尝试加载外部示意图文件
            guide_image_path = "interface/calib_tutorial.jpg"
            import os

            if os.path.exists(guide_image_path):
                # 如果存在外部图片文件，加载它
                pixmap = QPixmap(guide_image_path)
                self.guide_image_label.setPixmap(pixmap)
            else:
                # 如果没有外部文件，生成一个示例示意图
                self.generate_example_guide_image()

        except Exception as e:
            # 如果加载失败，生成示例图
            self.generate_example_guide_image()

    def generate_example_guide_image(self):
        """生成示例标定示意图"""
        try:
            # 创建一个示例图像 - 增大分辨率以便缩放观察
            width, height = 1600, 1000  # 增大分辨率
            image = np.ones((height, width, 3), dtype=np.uint8) * 240  # 浅灰色背景

            # 绘制场地轮廓（矩形）
            field_margin = 100
            cv2.rectangle(
                image,
                (field_margin, field_margin),
                (width - field_margin, height - field_margin),
                (100, 100, 100),
                6,  # 增加线条粗细
            )

            # 绘制中线
            cv2.line(
                image,
                (width // 2, field_margin),
                (width // 2, height - field_margin),
                (150, 150, 150),
                4,
            )

            # 定义关键点位置（示例）
            key_points = [
                (field_margin, field_margin),  # 左上角
                (width - field_margin, field_margin),  # 右上角
                (width - field_margin, height - field_margin),  # 右下角
                (field_margin, height - field_margin),  # 左下角
                (width // 2, field_margin),  # 中上
                (width // 2, height - field_margin),  # 中下
            ]

            # 绘制关键点和标号 - 增大尺寸
            colors = [
                (0, 255, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (255, 128, 0),
                (128, 0, 255),
            ]

            for i, (x, y) in enumerate(key_points):
                color = colors[i % len(colors)]

                # 绘制关键点 - 增大半径
                cv2.circle(image, (x, y), 20, color, -1)
                cv2.circle(image, (x, y), 25, (0, 0, 0), 4)

                # 绘制标号 - 增大字体
                cv2.putText(
                    image,
                    str(i + 1),
                    (x - 15, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    4,
                )
                cv2.putText(
                    image,
                    str(i + 1),
                    (x - 15, y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    2,
                )

            # 添加标题 - 增大字体
            cv2.putText(
                image,
                "Calibration Keypoints Guide",
                (width // 2 - 360, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (50, 50, 50),
                4,
            )

            # 添加坐标轴指示 - 增大尺寸
            arrow_size = 100
            cv2.arrowedLine(
                image,
                (width - 300, height - 150),
                (width - 300 + arrow_size, height - 150),
                (0, 0, 255),
                6,
            )
            cv2.putText(
                image,
                "X",
                (width - 180, height - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            cv2.arrowedLine(
                image,
                (width - 300, height - 150),
                (width - 300, height - 150 - arrow_size),
                (0, 255, 0),
                6,
            )
            cv2.putText(
                image,
                "Y",
                (width - 330, height - 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

            # 转换为QPixmap并显示
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)

            self.guide_image_label.setPixmap(pixmap)

        except Exception as e:
            # 如果生成失败，显示文字说明
            error_text = (
                "无法生成标定示意图\n\n"
                "请按以下顺序标记关键点：\n"
                "1. 场地左上角\n"
                "2. 场地右上角\n"
                "3. 场地右下角\n"
                "4. 场地左下角\n"
                "5. 场地中线上端\n"
                "6. 场地中线下端"
            )
            self.guide_image_label.setText(error_text)
            self.guide_image_label.setStyleSheet(
                "border: 2px solid #333; background-color: #f9f9f9; "
                "font-size: 14px; color: #333; padding: 20px;"
            )

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() == Qt.Key_R:
            self.reset_image_view()
        elif event.key() == Qt.Key_F:
            self.fit_image_to_window()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        else:
            super().keyPressEvent(event)


class ZoomableDraggableImageLabel(ZoomableDraggableLabel):
    """支持缩放、拖拽和关键点标记的图像标签"""
    
    left_clicked = pyqtSignal(float, float)  # 发送原图坐标
    right_clicked = pyqtSignal()
    ########################################################################
    outpost_roi_changed = pyqtSignal(object)
    ########################################################################
    
    def __init__(self):
        super().__init__()
        self.keypoints = []  # 存储关键点在原图中的坐标
        self.click_threshold = 5  # 点击阈值（像素），小于此值认为是点击而非拖拽
        
        # 拖拽相关变量
        self.mouse_press_pos = None
        self.total_drag_distance = 0
        ########################################################################
        # 左键框选前哨站 ROI，右键拖拽移动视图。
        self.outpost_roi = None
        self.roi_selecting = False
        self.roi_start = None
        self.roi_current = None
        self.right_dragging = False
        self.last_right_pan_point = QPoint()
        ########################################################################
        
    def set_keypoints(self, keypoints):
        """设置关键点列表"""
        self.keypoints = keypoints
        self.update_display()
    
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        # self.mouse_press_pos = event.pos()
        # self.total_drag_distance = 0
        # super().mousePressEvent(event)
        ########################################################################
        self.mouse_press_pos = event.pos()
        self.total_drag_distance = 0
        if event.button() == Qt.LeftButton:
            image_pos = self.screen_to_image_coords(event.pos())
            if image_pos is not None:
                self.roi_selecting = True
                self.roi_start = image_pos
                self.roi_current = image_pos
                self.setCursor(Qt.CrossCursor)
                event.accept()
                return
        if event.button() == Qt.RightButton:
            self.right_dragging = True
            self.last_right_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)
        ########################################################################
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        # if self.mouse_press_pos is not None:
        #     # 计算拖拽距离
        #     current_distance = (event.pos() - self.mouse_press_pos).manhattanLength()
        #     self.total_drag_distance = max(self.total_drag_distance, current_distance)
        # super().mouseMoveEvent(event)
        ########################################################################
        if self.mouse_press_pos is not None:
            current_distance = (event.pos() - self.mouse_press_pos).manhattanLength()
            self.total_drag_distance = max(self.total_drag_distance, current_distance)

        if self.roi_selecting:
            image_pos = self.screen_to_image_coords(event.pos())
            if image_pos is not None:
                self.roi_current = image_pos
                self.update_display()
            event.accept()
            return

        if self.right_dragging and self.original_pixmap is not None:
            delta = event.pos() - self.last_right_pan_point
            self.image_offset += delta
            self.last_right_pan_point = event.pos()
            self.constrain_offset()
            self.update_display()
            event.accept()
            return

        super().mouseMoveEvent(event)
        ########################################################################
    
    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        # if self.mouse_press_pos is not None:
        #     # 判断是点击还是拖拽
        #     if self.total_drag_distance < self.click_threshold:
        #         # 认为是点击操作
        #         if event.button() == Qt.LeftButton:
        #             # 将屏幕坐标转换为原图坐标
        #             original_coords = self.screen_to_image_coords(event.pos())
        #             if original_coords is not None:
        #                 self.left_clicked.emit(original_coords[0], original_coords[1])
        #         elif event.button() == Qt.RightButton:
        #             self.right_clicked.emit()
        #
        # self.mouse_press_pos = None
        # self.total_drag_distance = 0
        # super().mouseReleaseEvent(event)
        ########################################################################
        if self.mouse_press_pos is not None:
            if self.total_drag_distance < self.click_threshold:
                if event.button() == Qt.LeftButton:
                    original_coords = self.screen_to_image_coords(event.pos())
                    if original_coords is not None:
                        self.left_clicked.emit(original_coords[0], original_coords[1])
                elif event.button() == Qt.RightButton:
                    self.right_clicked.emit()
            elif event.button() == Qt.LeftButton and self.roi_selecting:
                self._finish_outpost_roi_selection()

        if event.button() == Qt.LeftButton:
            self.roi_selecting = False
            self.roi_start = None
            self.roi_current = None
        elif event.button() == Qt.RightButton:
            self.right_dragging = False

        self.mouse_press_pos = None
        self.total_drag_distance = 0
        self.setCursor(Qt.ArrowCursor)
        self.update_display()
        event.accept()
        ########################################################################

    ########################################################################
    def _finish_outpost_roi_selection(self):
        """完成前哨站 ROI 框选，并输出原图坐标边界。"""
        if self.roi_start is None or self.roi_current is None:
            return

        x1 = min(self.roi_start[0], self.roi_current[0])
        y1 = min(self.roi_start[1], self.roi_current[1])
        x2 = max(self.roi_start[0], self.roi_current[0])
        y2 = max(self.roi_start[1], self.roi_current[1])
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        self.outpost_roi = (x1, y1, x2, y2)
        self.outpost_roi_changed.emit(self.outpost_roi)

    def set_outpost_roi(self, roi):
        """设置前哨站 ROI。"""
        self.outpost_roi = roi
        self.update_display()
    ########################################################################
    
    def screen_to_image_coords(self, screen_pos):
        """将屏幕坐标转换为原图坐标"""
        if self.original_pixmap is None:
            return None
        
        # 获取控件尺寸
        widget_size = self.size()
        
        # 计算缩放后的图像尺寸
        scaled_size = self.original_pixmap.size() * self.scale_factor
        
        # 计算图像在控件中的实际位置
        if scaled_size.width() < widget_size.width():
            image_left = (widget_size.width() - scaled_size.width()) // 2
        else:
            image_left = self.image_offset.x()
        
        if scaled_size.height() < widget_size.height():
            image_top = (widget_size.height() - scaled_size.height()) // 2
        else:
            image_top = self.image_offset.y()
        
        # 转换为图像内坐标
        image_x = screen_pos.x() - image_left
        image_y = screen_pos.y() - image_top
        
        # 检查是否在图像范围内
        if (image_x < 0 or image_y < 0 or 
            image_x >= scaled_size.width() or image_y >= scaled_size.height()):
            return None
        
        # 转换为原图坐标
        original_x = image_x / self.scale_factor
        original_y = image_y / self.scale_factor
        
        return (original_x, original_y)
    
    def image_to_screen_coords(self, image_x, image_y):
        """将原图坐标转换为屏幕坐标"""
        if self.original_pixmap is None:
            return None
        
        # 获取控件尺寸
        widget_size = self.size()
        
        # 计算缩放后的图像尺寸
        scaled_size = self.original_pixmap.size() * self.scale_factor
        
        # 计算缩放后的坐标
        scaled_x = image_x * self.scale_factor
        scaled_y = image_y * self.scale_factor
        
        # 计算图像在控件中的实际位置
        if scaled_size.width() < widget_size.width():
            image_left = (widget_size.width() - scaled_size.width()) // 2
        else:
            image_left = self.image_offset.x()
        
        if scaled_size.height() < widget_size.height():
            image_top = (widget_size.height() - scaled_size.height()) // 2
        else:
            image_top = self.image_offset.y()
        
        # 转换为屏幕坐标
        screen_x = scaled_x + image_left
        screen_y = scaled_y + image_top
        
        return (screen_x, screen_y)
    
    def create_display_image(self):
        """创建用于显示的图像（覆盖父类方法以添加关键点绘制）"""
        if self.scaled_pixmap is None:
            return

        # 获取控件尺寸
        widget_size = self.size()

        # 创建背景图像
        display_image = QPixmap(widget_size)
        display_image.fill(QColor(240, 240, 240))  # 浅灰色背景

        # 计算图像在控件中的位置
        image_rect = self.scaled_pixmap.rect()

        # 应用偏移
        image_rect.translate(self.image_offset)

        # 居中显示（当图像小于控件时）
        if self.scaled_pixmap.width() < widget_size.width():
            image_rect.moveLeft((widget_size.width() - self.scaled_pixmap.width()) // 2)
        if self.scaled_pixmap.height() < widget_size.height():
            image_rect.moveTop(
                (widget_size.height() - self.scaled_pixmap.height()) // 2
            )

        # 绘制图像
        painter = QPainter(display_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(image_rect, self.scaled_pixmap)
        
        # 绘制关键点
        self.draw_keypoints_on_display(painter)
        ########################################################################
        # 绘制前哨站 ROI 框选结果或当前拖拽预览。
        self.draw_outpost_roi_on_display(painter)
        ########################################################################

        # 绘制缩放信息
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setFont(QFont("Arial", 12))
        scale_text = f"缩放: {self.scale_factor:.1f}x"
        painter.drawText(10, 25, scale_text)
        
        # 绘制操作提示
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setFont(QFont("Arial", 10))
        help_text = "滚轮缩放 | 左键拖拽移动 | 单击标记关键点 | 右键撤销"
        painter.drawText(10, widget_size.height() - 10, help_text)

        painter.end()

        # 显示最终图像
        super(ZoomableDraggableLabel, self).setPixmap(display_image)

    ########################################################################
    def draw_outpost_roi_on_display(self, painter):
        """绘制前哨站 ROI。"""
        roi = self.outpost_roi
        if self.roi_selecting and self.roi_start is not None and self.roi_current is not None:
            x1 = min(self.roi_start[0], self.roi_current[0])
            y1 = min(self.roi_start[1], self.roi_current[1])
            x2 = max(self.roi_start[0], self.roi_current[0])
            y2 = max(self.roi_start[1], self.roi_current[1])
            roi = (x1, y1, x2, y2)

        if roi is None:
            return

        p1 = self.image_to_screen_coords(roi[0], roi[1])
        p2 = self.image_to_screen_coords(roi[2], roi[3])
        if p1 is None or p2 is None:
            return

        rect = QRectF(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1])).normalized()
        painter.setBrush(QColor(255, 193, 7, 45))
        painter.setPen(QPen(QColor(255, 193, 7), 3))
        painter.drawRect(rect)
    ########################################################################
    
    def draw_keypoints_on_display(self, painter):
        """在显示图像上绘制关键点"""
        if not self.keypoints:
            return
        
        # 定义关键点颜色
        colors = [
            QColor(0, 255, 0),      # 绿色
            QColor(0, 255, 255),    # 青色
            QColor(255, 0, 255),    # 洋红
            QColor(255, 255, 0),    # 黄色
            QColor(255, 128, 0),    # 橙色
            QColor(128, 0, 255),    # 紫色
        ]
        
        for i, (x, y) in enumerate(self.keypoints):
            # 转换为屏幕坐标
            screen_coords = self.image_to_screen_coords(x, y)
            if screen_coords is None:
                continue
            
            screen_x, screen_y = screen_coords
            color = colors[i % len(colors)]
            
            # 根据缩放调整关键点大小
            point_radius = max(4, int(8 * self.scale_factor))
            border_radius = max(6, int(12 * self.scale_factor))
            
            # 绘制关键点
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(int(screen_x - point_radius), int(screen_y - point_radius), 
                              point_radius * 2, point_radius * 2)
            
            # 绘制边框
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawEllipse(int(screen_x - border_radius), int(screen_y - border_radius), 
                              border_radius * 2, border_radius * 2)
            
            # 绘制点号
            font_size = max(8, int(12 * self.scale_factor))
            font = QFont("Arial", font_size, QFont.Bold)
            painter.setFont(font)
            
            # 白色背景文字
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawText(int(screen_x + border_radius + 5), int(screen_y - border_radius), str(i + 1))
            
            # 黑色前景文字
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(int(screen_x + border_radius + 5), int(screen_y - border_radius), str(i + 1))


class SensorRecordingDialog(QDialog):
    """传感器录制配置对话框。"""

    def __init__(
        self,
        parent=None,
        is_recording=False,
        default_record_camera=True,
        default_record_lidar=False,
        default_camera_fps=20,
        default_export_yolo=False,
        default_export_armor_yolo=False,
        default_save_root_dir="recorder",
        current_session_dir="",
    ):
        super().__init__(parent)
        self.setWindowTitle("录制传感器数据")
        self.setModal(True)
        self.resize(520, 320)

        self._result = {"action": None}
        self.is_recording = bool(is_recording)
        self.default_save_root_dir = default_save_root_dir
        self.current_session_dir = current_session_dir

        self.init_ui(
            default_record_camera,
            default_record_lidar,
            int(default_camera_fps),
            bool(default_export_yolo),
            bool(default_export_armor_yolo),
            default_save_root_dir,
        )

    def init_ui(
        self,
        default_record_camera: bool,
        default_record_lidar: bool,
        default_camera_fps: int,
        default_export_yolo: bool,
        default_export_armor_yolo: bool,
        default_save_root_dir: str,
    ):
        """初始化界面。"""
        layout = QVBoxLayout(self)

        title_label = QLabel("配置传感器录制")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 8px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        status_text = "当前状态：录制中" if self.is_recording else "当前状态：未录制"
        if self.current_session_dir:
            status_text += f"\n当前目录：{self.current_session_dir}"
        self.status_label = QLabel(status_text)
        self.status_label.setStyleSheet("color: #555;")
        layout.addWidget(self.status_label)

        self.record_camera_cb = QCheckBox("录制相机数据")
        self.record_camera_cb.setChecked(default_record_camera)
        self.record_camera_cb.toggled.connect(self.update_camera_fps_enabled)
        layout.addWidget(self.record_camera_cb)

        self.record_lidar_cb = QCheckBox("录制雷达数据")
        self.record_lidar_cb.setChecked(default_record_lidar)
        layout.addWidget(self.record_lidar_cb)

        self.export_yolo_cb = QCheckBox("额外导出 YOLO 车辆伪标签")
        self.export_yolo_cb.setChecked(default_export_yolo)
        layout.addWidget(self.export_yolo_cb)

        self.export_armor_yolo_cb = QCheckBox("额外导出 YOLO 装甲板伪标签")
        self.export_armor_yolo_cb.setChecked(default_export_armor_yolo)
        layout.addWidget(self.export_armor_yolo_cb)

        fps_layout = QHBoxLayout()
        fps_label = QLabel("相机录制帧率")
        self.camera_fps_input = QSpinBox()
        self.camera_fps_input.setRange(1, 240)
        self.camera_fps_input.setValue(max(1, int(default_camera_fps)))
        self.camera_fps_input.setSuffix(" fps")
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.camera_fps_input)
        fps_layout.addStretch()
        layout.addLayout(fps_layout)
        self.update_camera_fps_enabled()

        save_dir_group = QGroupBox("保存目录")
        save_dir_layout = QHBoxLayout(save_dir_group)
        self.save_root_dir_edit = QLineEdit(os.path.abspath(default_save_root_dir))
        self.browse_btn = QPushButton("选择目录")
        self.browse_btn.clicked.connect(self.choose_save_root_dir)
        save_dir_layout.addWidget(self.save_root_dir_edit)
        save_dir_layout.addWidget(self.browse_btn)
        layout.addWidget(save_dir_group)

        tips_label = QLabel(
            "相机录制默认保存为 MP4；车辆和装甲板 YOLO 导出会分别生成独立的 images/train、labels/train 和 data.yaml。"
        )
        tips_label.setStyleSheet("color: #666;")
        tips_label.setWordWrap(True)
        layout.addWidget(tips_label)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始录制")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; font-weight: bold; padding: 6px 16px; }"
        )
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.stop_btn = QPushButton("停止录制")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #C62828; color: white; font-weight: bold; padding: 6px 16px; }"
        )
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(self.is_recording)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

    def choose_save_root_dir(self):
        """选择保存目录。"""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "选择传感器数据保存目录",
            self.save_root_dir_edit.text().strip() or os.path.abspath("recorder"),
        )
        if selected_dir:
            self.save_root_dir_edit.setText(selected_dir)

    def update_camera_fps_enabled(self):
        """根据相机录制勾选状态更新帧率输入框可用性。"""
        self.camera_fps_input.setEnabled(self.record_camera_cb.isChecked())

    def on_start_clicked(self):
        """确认开始录制。"""
        save_root_dir = self.save_root_dir_edit.text().strip()
        if not save_root_dir:
            QMessageBox.warning(self, "配置错误", "请先选择保存目录。")
            return
        if not self.record_camera_cb.isChecked() and not self.record_lidar_cb.isChecked():
            QMessageBox.warning(self, "配置错误", "请至少勾选一种传感器数据。")
            return
        if self.export_yolo_cb.isChecked() and not self.record_camera_cb.isChecked():
            QMessageBox.warning(
                self,
                "配置错误",
                "导出 YOLO 标签时必须同时勾选相机录制。",
            )
            return
        if self.export_armor_yolo_cb.isChecked() and not self.record_camera_cb.isChecked():
            QMessageBox.warning(
                self,
                "配置错误",
                "导出装甲板 YOLO 标签时必须同时勾选相机录制。",
            )
            return

        self._result = {
            "action": "start",
            "record_camera": self.record_camera_cb.isChecked(),
            "record_lidar": self.record_lidar_cb.isChecked(),
            "camera_fps": int(self.camera_fps_input.value()),
            "export_yolo": self.export_yolo_cb.isChecked(),
            "export_armor_yolo": self.export_armor_yolo_cb.isChecked(),
            "save_root_dir": save_root_dir,
        }
        self.accept()

    def on_stop_clicked(self):
        """确认停止录制。"""
        self._result = {"action": "stop"}
        self.accept()

    def get_result(self):
        """返回用户配置结果。"""
        return dict(self._result)


class ManualExtrinsicsDialog(QDialog):
    """手动输入相机外参对话框。"""

    def __init__(self, parent=None, initial_rotation=None, initial_translation=None):
        super().__init__(parent)
        self.setWindowTitle("直接输入相机外参")
        self.setModal(True)
        self.resize(780, 860)

        self.initial_rotation = None if initial_rotation is None else np.array(initial_rotation, dtype=np.float64).reshape(3, 3)
        self.initial_translation = None if initial_translation is None else np.array(initial_translation, dtype=np.float64).reshape(3, 1)
        self.latest_rotation = None
        self.latest_translation = None
        self.latest_extrinsics_text = ""
        self.rotation_matrix = None
        self.translation_vector = None

        self.init_ui()
        self.fill_with_initial_extrinsics()
        _fit_window_to_screen(self, QSize(900, 1000))

    def init_ui(self):
        """初始化手动输入界面。"""
        outer_layout = QVBoxLayout(self)
        content = QWidget()
        layout = QVBoxLayout(content)
        outer_layout.addWidget(_scrollable_panel(content))

        title_label = QLabel("手动输入相机外参")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        tips_label = QLabel(
            "支持三种输入方式：\n"
            "1. 直接粘贴 3x4 或 4x4 外参矩阵；\n"
            "2. 分别输入 3x3 旋转矩阵 R 和 3x1 平移向量 t；\n"
            "3. 导入最近一次 UI 标定自动保存的外参。"
        )
        tips_label.setStyleSheet("color: #555; margin: 4px 8px;")
        tips_label.setWordWrap(True)
        layout.addWidget(tips_label)

        mode_layout = QHBoxLayout()
        self.matrix_mode_radio = QRadioButton("使用外参矩阵")
        self.rt_mode_radio = QRadioButton("使用 R+t")
        self.rt_mode_radio.setChecked(True)
        self.matrix_mode_radio.toggled.connect(self._update_input_mode_hint)
        self.rt_mode_radio.toggled.connect(self._update_input_mode_hint)
        mode_layout.addWidget(self.matrix_mode_radio)
        mode_layout.addWidget(self.rt_mode_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        matrix_group = QGroupBox("方式一：输入外参矩阵")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_input = QPlainTextEdit()
        self.matrix_input.setPlaceholderText(
            "[r11, r12, r13, tx,\n"
            " r21, r22, r23, ty,\n"
            " r31, r32, r33, tz]\n\n"
            "或输入 4x4 齐次矩阵"
        )
        self.matrix_input.setMinimumHeight(180)
        matrix_layout.addWidget(self.matrix_input)
        layout.addWidget(matrix_group)

        rt_group = QGroupBox("方式二：分别输入 R 和 t")
        rt_layout = QVBoxLayout(rt_group)

        rotation_label = QLabel("旋转矩阵 R（3x3）")
        rt_layout.addWidget(rotation_label)
        self.rotation_input = QPlainTextEdit()
        self.rotation_input.setPlaceholderText(
            "0.9799 0.0020 -0.1995\n"
            "0.0462 -0.9750 0.2175\n"
            "-0.1941 -0.2223 -0.9555"
        )
        self.rotation_input.setMinimumHeight(140)
        rt_layout.addWidget(self.rotation_input)

        translation_label = QLabel("平移向量 t（3x1）")
        rt_layout.addWidget(translation_label)
        self.translation_input = QPlainTextEdit()
        self.translation_input.setPlaceholderText("0.6379 0.3622 0.3593")
        self.translation_input.setMaximumHeight(80)
        rt_layout.addWidget(self.translation_input)

        layout.addWidget(rt_group)

        latest_group = QGroupBox("方式三：导入最近一次标定外参")
        latest_layout = QVBoxLayout(latest_group)
        self.latest_extrinsics_view = QPlainTextEdit()
        self.latest_extrinsics_view.setReadOnly(True)
        self.latest_extrinsics_view.setMaximumHeight(150)
        latest_layout.addWidget(self.latest_extrinsics_view)

        latest_button_layout = QHBoxLayout()
        self.reload_latest_btn = QPushButton("刷新最近外参")
        self.reload_latest_btn.clicked.connect(self.load_latest_extrinsics_preview)
        self.import_latest_btn = QPushButton("导入最近外参")
        self.import_latest_btn.clicked.connect(self.import_latest_extrinsics)
        latest_button_layout.addWidget(self.reload_latest_btn)
        latest_button_layout.addWidget(self.import_latest_btn)
        latest_button_layout.addStretch()
        latest_layout.addLayout(latest_button_layout)

        layout.addWidget(latest_group)
        self.load_latest_extrinsics_preview()

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
        self._update_input_mode_hint()

        button_layout = QHBoxLayout()
        self.fill_btn = QPushButton("填入当前外参")
        self.fill_btn.clicked.connect(self.fill_with_initial_extrinsics)
        self.confirm_btn = QPushButton("确认应用")
        self.confirm_btn.setStyleSheet(
            "QPushButton { background-color: #1976D2; color: white; font-weight: bold; padding: 6px 16px; }"
        )
        self.confirm_btn.clicked.connect(self.on_confirm)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(self.fill_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.confirm_btn)
        button_layout.addWidget(self.cancel_btn)
        outer_layout.addLayout(button_layout)

    def fill_with_initial_extrinsics(self):
        """用当前外参或配置默认值回填输入框。"""
        if self.initial_rotation is None or self.initial_translation is None:
            self.status_label.setText("当前没有可回填的默认外参，请直接输入。")
            return

        rotation_text = "\n".join(
            " ".join(f"{value:.6f}" for value in row)
            for row in self.initial_rotation
        )
        translation_text = " ".join(
            f"{value:.6f}" for value in self.initial_translation.flatten()
        )

        self.rotation_input.setPlainText(rotation_text)
        self.translation_input.setPlainText(translation_text)
        self.matrix_input.clear()
        self.rt_mode_radio.setChecked(True)
        self.status_label.setText("已填入当前外参，当前默认使用 R+t 模式。")

    def load_latest_extrinsics_preview(self):
        """读取并显示最近一次 UI 标定外参。"""
        latest_path = get_latest_extrinsics_path()
        try:
            rotation_matrix, translation_vector, text = load_extrinsics_record(latest_path)
            self.latest_rotation = rotation_matrix
            self.latest_translation = translation_vector
            self.latest_extrinsics_text = text
            self.latest_extrinsics_view.setPlainText(text)
            self.import_latest_btn.setEnabled(True)
        except Exception as e:
            self.latest_rotation = None
            self.latest_translation = None
            self.latest_extrinsics_text = ""
            self.latest_extrinsics_view.setPlainText(
                f"未找到可导入的最近外参。\n路径: {latest_path}\n原因: {e}"
            )
            self.import_latest_btn.setEnabled(False)

    def import_latest_extrinsics(self):
        """将最近一次外参导入到 R+t 输入区。"""
        if self.latest_rotation is None or self.latest_translation is None:
            self.load_latest_extrinsics_preview()
        if self.latest_rotation is None or self.latest_translation is None:
            QMessageBox.warning(self, "导入失败", "没有可导入的最近外参。")
            return

        rotation_text = "\n".join(
            " ".join(f"{value:.10f}" for value in row)
            for row in self.latest_rotation
        )
        translation_text = " ".join(
            f"{value:.10f}" for value in self.latest_translation.flatten()
        )
        self.rotation_input.setPlainText(rotation_text)
        self.translation_input.setPlainText(translation_text)
        self.matrix_input.clear()
        self.rt_mode_radio.setChecked(True)
        self.status_label.setText("已导入最近一次标定外参，当前使用 R+t 模式。")

    def _update_input_mode_hint(self):
        """根据当前输入模式更新提示文本。"""
        if self.matrix_mode_radio.isChecked():
            self.status_label.setText("当前模式：外参矩阵。请输入 3x4 或 4x4。")
        else:
            self.status_label.setText("当前模式：R+t。请输入 3x3 旋转矩阵和 3x1 平移向量。")

    @staticmethod
    def _extract_numbers(text):
        """从文本中提取浮点数，兼容空格、逗号、中括号等格式。"""
        return [
            float(item)
            for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
        ]

    def _parse_extrinsics(self):
        """解析用户输入的外参。"""
        if self.matrix_mode_radio.isChecked():
            matrix_text = self.matrix_input.toPlainText().strip()
            if not matrix_text:
                raise ValueError("当前选择的是外参矩阵模式，请输入 3x4 或 4x4 矩阵。")
            numbers = self._extract_numbers(matrix_text)
            if len(numbers) == 12:
                extrinsic = np.array(numbers, dtype=np.float64).reshape(3, 4)
            elif len(numbers) == 16:
                extrinsic = np.array(numbers, dtype=np.float64).reshape(4, 4)[:3, :4]
            else:
                raise ValueError("外参矩阵必须包含 12 个数（3x4）或 16 个数（4x4）。")
            rotation_matrix = extrinsic[:, :3]
            translation_vector = extrinsic[:, 3].reshape(3, 1)
            return rotation_matrix, translation_vector

        rotation_numbers = self._extract_numbers(self.rotation_input.toPlainText())
        translation_numbers = self._extract_numbers(self.translation_input.toPlainText())
        if len(rotation_numbers) != 9:
            raise ValueError("旋转矩阵 R 必须包含 9 个数。")
        if len(translation_numbers) != 3:
            raise ValueError("平移向量 t 必须包含 3 个数。")

        rotation_matrix = np.array(rotation_numbers, dtype=np.float64).reshape(3, 3)
        translation_vector = np.array(translation_numbers, dtype=np.float64).reshape(3, 1)
        return rotation_matrix, translation_vector

    @staticmethod
    def _check_rotation_matrix(rotation_matrix):
        """检查旋转矩阵是否接近正交矩阵。"""
        orthogonal_error = np.linalg.norm(
            rotation_matrix.T @ rotation_matrix - np.eye(3), ord="fro"
        )
        determinant = float(np.linalg.det(rotation_matrix))
        return orthogonal_error, determinant

    def on_confirm(self):
        """确认并应用手动输入的外参。"""
        try:
            rotation_matrix, translation_vector = self._parse_extrinsics()
            orthogonal_error, determinant = self._check_rotation_matrix(rotation_matrix)
            if orthogonal_error > 0.05 or abs(determinant - 1.0) > 0.05:
                reply = QMessageBox.question(
                    self,
                    "旋转矩阵校验提醒",
                    (
                        "输入的 R 不太像标准旋转矩阵：\n"
                        f"正交误差: {orthogonal_error:.6f}\n"
                        f"行列式: {determinant:.6f}\n\n"
                        "仍然继续使用这组外参吗？"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            self.rotation_matrix = rotation_matrix
            self.translation_vector = translation_vector
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "输入错误", f"解析外参失败：\n{str(e)}")

    def get_calibration_result(self):
        """以现有标定结果结构返回手动输入结果。"""
        if self.rotation_matrix is None or self.translation_vector is None:
            return {
                "success": False,
                "rotation_matrix": None,
                "translation_vector": None,
                "residual": None,
                "source": "manual_input",
            }
        return {
            "success": True,
            "rotation_matrix": self.rotation_matrix.copy(),
            "translation_vector": self.translation_vector.copy(),
            "residual": None,
            "source": "manual_input",
        }


class KeypointCalibrationDialog(QDialog):
    """关键点标定对话框 - 支持缩放和拖拽"""

    ########################################################################
    # def __init__(
    #     self, parent=None, background_image=None, pnp_config=None, keypoint_file=None
    # ):
    ########################################################################
    def __init__(
        self,
        parent=None,
        background_image=None,
        pnp_config=None,
        keypoint_file=None,
        initial_outpost_roi=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("关键点标定")
        self.setModal(True)
        self.resize(1800, 1200)

        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinMaxButtonsHint
            | Qt.WindowCloseButtonHint
        )

        # 背景图像和标定相关数据
        self.background_image = background_image
        self.original_image = (
            background_image.copy() if background_image is not None else None
        )
        self.pnp_config = pnp_config if isinstance(pnp_config, dict) else None
        self.keypoint_file = keypoint_file or "transform/keypoint_6.txt"
        self.keypoints = []  # 存储关键点坐标 [(x, y), ...]
        ########################################################################
        # 前哨站 ROI，格式为原图坐标 (x1, y1, x2, y2)。
        # self.outpost_roi = None
        self.outpost_roi = (
            tuple(int(round(value)) for value in initial_outpost_roi)
            if initial_outpost_roi is not None
            else None
        )
        ########################################################################

        # 标定结果
        self.calibration_success = False
        self.rotation_matrix = None
        self.translation_vector = None
        self.residual = None

        self.init_ui()
        _fit_window_to_screen(self, QSize(1800, 1200))

    def init_ui(self):
        """初始化标定界面"""
        layout = QVBoxLayout(self)

        # 标题和说明
        title_label = QLabel("相机外参标定 - 关键点标记（支持缩放拖拽）")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        instruction_label = QLabel(
            "操作说明：滚轮缩放；右键拖拽移动；左键拖拽框选前哨站区域；"
            "左键单击标点；右键单击撤销；至少标记4个关键点后校准。"
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet(
            "color: #666666; margin: 10px; padding: 10px; border: 1px solid #ddd;"
        )
        layout.addWidget(instruction_label)

        # 主要内容区域
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧：图像显示区域
        image_panel = self.create_image_panel()
        main_layout.addWidget(image_panel, 3)

        # 右侧：控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(_scrollable_panel(control_panel), 1)

        layout.addWidget(main_widget)

        # 底部按钮
        button_layout = QGridLayout()

        self.clear_btn = QPushButton("清除所有点")
        self.clear_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 8px; }"
        )
        self.clear_btn.clicked.connect(self.clear_all_points)

        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }"
        )
        self.reset_view_btn.clicked.connect(self.reset_image_view)

        self.fit_view_btn = QPushButton("适应窗口")
        self.fit_view_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.fit_view_btn.clicked.connect(self.fit_image_to_window)

        self.show_guide_btn = QPushButton("标定示意图")
        self.show_guide_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }"
        )
        self.show_guide_btn.clicked.connect(self.show_calibration_guide)

        self.calibrate_btn = QPushButton("开始校准")
        self.calibrate_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.calibrate_btn.clicked.connect(self.start_calibration)
        self.calibrate_btn.setEnabled(False)

        ########################################################################
        self.confirm_roi_btn = QPushButton("确认保存前哨站区域")
        self.confirm_roi_btn.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px; }"
        )
        self.confirm_roi_btn.clicked.connect(self.confirm_outpost_roi)
        self.confirm_roi_btn.setEnabled(self.outpost_roi is not None)
        ########################################################################

        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 8px; }"
        )
        self.close_btn.clicked.connect(self.close)

        for index, button in enumerate((
            self.clear_btn, self.reset_view_btn, self.fit_view_btn,
            self.show_guide_btn, self.calibrate_btn, self.confirm_roi_btn,
            self.close_btn,
        )):
            button_layout.addWidget(button, index // 4, index % 4)

        layout.addLayout(button_layout)

        # 设置初始图像
        if self.background_image is not None:
            self.set_background_image()

    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QGroupBox("标定图像")
        layout = QVBoxLayout(panel)

        # 创建支持缩放拖拽的图像标签
        self.image_label = ZoomableDraggableImageLabel()
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet(
            "border: 2px solid #333; background-color: #f0f0f0;"
        )

        # 连接信号
        self.image_label.left_clicked.connect(self.add_keypoint)
        self.image_label.right_clicked.connect(self.remove_last_keypoint)
        ########################################################################
        # 左键拖拽框选前哨站 ROI，右键拖拽由图像控件内部处理为移动视图。
        self.image_label.outpost_roi_changed.connect(self.set_outpost_roi)
        ########################################################################

        layout.addWidget(self.image_label)

        return panel

    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("标定控制")
        layout = QVBoxLayout(panel)

        # 关键点列表
        keypoint_label = QLabel("已标记的关键点:")
        keypoint_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(keypoint_label)

        self.keypoint_list = QListWidget()
        self.keypoint_list.setMaximumHeight(200)
        layout.addWidget(self.keypoint_list)

        # 状态信息
        status_label = QLabel("标定状态:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(status_label)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("等待标记关键点...")
        self.status_text.setReadOnly(True)
        layout.addWidget(self.status_text)

        # 校准结果显示
        result_label = QLabel("校准结果:")
        result_label.setStyleSheet("font-weight: bold; margin-top: 20px;")
        layout.addWidget(result_label)

        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(200)
        self.result_text.setPlaceholderText("校准结果将在此显示...")
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        return panel

    def set_background_image(self):
        """设置背景图像"""
        if self.background_image is not None:
            # 转换为QImage和QPixmap
            height, width, channel = self.background_image.shape
            bytes_per_line = channel * width
            q_image = QImage(
                self.background_image.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)
            
            # 设置到图像标签
            self.image_label.setPixmap(pixmap)
            ########################################################################
            # 重新打开标定窗口时显示已保存的前哨站 ROI。
            if self.outpost_roi is not None:
                self.image_label.set_outpost_roi(self.outpost_roi)
            ########################################################################

    def reset_image_view(self):
        """重置图像视图"""
        self.image_label.reset_view()

    def fit_image_to_window(self):
        """适应窗口大小"""
        self.image_label.fit_to_window()

    ########################################################################
    def set_outpost_roi(self, roi):
        """接收图像控件框选出的前哨站 ROI。"""
        self.outpost_roi = tuple(int(round(value)) for value in roi)
        self.status_text.append(f"前哨站 ROI: {self.outpost_roi}")
        ########################################################################
        if hasattr(self, "confirm_roi_btn"):
            self.confirm_roi_btn.setEnabled(True)
        ########################################################################

    def confirm_outpost_roi(self):
        """确认保存当前框选的前哨站 ROI。"""
        if self.outpost_roi is None:
            QMessageBox.warning(self, "ROI 未选择", "请先左键拖拽框选前哨站区域。")
            return

        parent = self.parent()
        if parent is not None and hasattr(parent, "save_outpost_roi"):
            if not parent.save_outpost_roi(self.outpost_roi):
                return

        self.status_text.append(f"前哨站 ROI 已保存: {self.outpost_roi}")
        QMessageBox.information(self, "ROI 已保存", "前哨站区域已保存。")
    ########################################################################

    def add_keypoint(self, x, y):
        """添加关键点"""
        # 添加到关键点列表
        self.keypoints.append((x, y))
        
        # 更新图像标签的关键点
        self.image_label.set_keypoints(self.keypoints)

        # 更新显示
        self.update_keypoint_list()
        self.update_status()

        self.status_text.append(
            f"添加关键点 {len(self.keypoints)}: ({x:.1f}, {y:.1f})"
        )

    def remove_last_keypoint(self):
        """移除最后一个关键点"""
        if self.keypoints:
            removed_point = self.keypoints.pop()
            
            # 更新图像标签的关键点
            self.image_label.set_keypoints(self.keypoints)
            
            self.update_keypoint_list()
            self.update_status()
            self.status_text.append(
                f"移除关键点: ({removed_point[0]:.1f}, {removed_point[1]:.1f})"
            )

    def clear_all_points(self):
        """清除所有关键点"""
        self.keypoints.clear()
        
        # 更新图像标签的关键点
        self.image_label.set_keypoints(self.keypoints)
        
        self.update_keypoint_list()
        self.update_status()
        self.status_text.append("清除所有关键点")

    def update_keypoint_list(self):
        """更新关键点列表显示"""
        self.keypoint_list.clear()
        for i, (x, y) in enumerate(self.keypoints):
            self.keypoint_list.addItem(f"点 {i+1}: ({x:.1f}, {y:.1f})")

    def update_status(self):
        """更新状态"""
        num_points = len(self.keypoints)
        if num_points >= 4:
            self.calibrate_btn.setEnabled(True)
            self.calibrate_btn.setText(f"开始校准 ({num_points}个点)")
        else:
            self.calibrate_btn.setEnabled(False)
            self.calibrate_btn.setText(f"需要至少4个点 (当前{num_points}个)")

    def show_calibration_guide(self):
        """显示标定示意图对话框"""
        guide_dialog = CalibrationGuideDialog(self)
        guide_dialog.exec_()

    def start_calibration(self):
        """开始PnP校准"""
        if len(self.keypoints) < 4:
            QMessageBox.warning(self, "错误", "至少需要4个关键点才能进行校准！")
            return

        try:
            # 准备图像点（2D坐标）
            image_points = np.array(self.keypoints, dtype=np.float32)

            # 加载3D物体点
            try:
                path = self.keypoint_file
                object_points = np.loadtxt(path, dtype=np.float32)
                object_points = np.atleast_2d(object_points)
            except Exception as e:
                QMessageBox.warning(
                    self, "警告", f"加载3D关键点文件失败: {path}\n错误: {e}"
                )
                return

            # 确保3D点数量足够
            if len(object_points) < len(image_points):
                QMessageBox.warning(
                    self,
                    "错误",
                    f"3D关键点数量不足！需要至少{len(image_points)}个，当前只有{len(object_points)}个",
                )
                return

            # 创建PnP求解器
            from transform.solvepnp import PnPSolver

            if self.pnp_config is not None:
                pnp_solver = PnPSolver.from_config(self.pnp_config)
            else:
                pnp_solver = PnPSolver.from_config(get_params_config_path())

            # 执行校准
            success, R, tvec, residual = pnp_solver.solve(
                object_points[: len(image_points)], image_points
            )

            if success:
                self.calibration_success = True
                self.rotation_matrix = R
                self.translation_vector = tvec
                self.residual = residual

                # 显示结果
                result_text = f"校准成功！\n\n"
                result_text += f"旋转矩阵 R:\n{R}\n\n"
                result_text += f"平移向量 t:\n{tvec.flatten()}\n\n"
                result_text += f"投影残差: {residual:.4f} 像素"

                self.result_text.setText(result_text)
                self.status_text.append(f"校准成功！残差: {residual:.4f}")

                QMessageBox.information(
                    self,
                    "校准成功",
                    f"外参校准成功完成！\n"
                    f"投影残差: {residual:.4f} 像素\n"
                    f"点击确定查看详细结果",
                )

            else:
                self.calibration_success = False
                self.result_text.setText("校准失败！\n请检查关键点标记是否正确")
                self.status_text.append("校准失败")

                QMessageBox.warning(
                    self,
                    "校准失败",
                    "外参校准失败！\n"
                    "请检查关键点标记是否正确，\n"
                    "确保点的顺序与3D模型匹配",
                )

        except Exception as e:
            self.calibration_success = False
            error_msg = f"校准过程中发生错误: {str(e)}"
            self.result_text.setText(error_msg)
            self.status_text.append(error_msg)
            QMessageBox.critical(self, "错误", f"校准过程中发生错误:\n{str(e)}")

    def get_calibration_result(self):
        """获取校准结果"""
        if self.rotation_matrix is None or self.translation_vector is None:
            return {
                "success": False,
                "rotation_matrix": None,
                "translation_vector": None,
                "residual": None,
                "keypoints": [],
                ########################################################################
                "outpost_roi": self.outpost_roi,
                ########################################################################
            }
        return {
            "success": self.calibration_success,
            "rotation_matrix": self.rotation_matrix.copy(),
            "translation_vector": self.translation_vector.copy(),
            "residual": self.residual,
            "keypoints": self.keypoints,
            ########################################################################
            "outpost_roi": self.outpost_roi,
            ########################################################################
        }

    def keyPressEvent(self, event):
        """键盘快捷键"""
        if event.key() == Qt.Key_R:
            self.reset_image_view()
        elif event.key() == Qt.Key_F:
            self.fit_image_to_window()
        elif event.key() == Qt.Key_C:
            self.clear_all_points()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

def launch():
    """启动雷达站主界面"""
    qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    QCoreApplication.setLibraryPaths([qt_plugin_path])
    os.environ["QT_PLUGIN_PATH"] = qt_plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        qt_plugin_path, "platforms"
    )
    os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 设置应用样式
    window = RadarStationMainWindow()
    window.show()
    # 让 Ctrl+C 触发 Qt 事件循环退出
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start()
    app.setProperty("_sigint_timer", sigint_timer)
    return app.exec_()


if __name__ == "__main__":
    launch()
