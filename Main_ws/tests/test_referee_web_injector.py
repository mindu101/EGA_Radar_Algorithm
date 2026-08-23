import socket
import struct
import time

import pytest

from driver.referee.referee_comm import FACTION, RefereeCommManager
from driver.referee.serial_comm import RefereeSerialManager
from driver.referee.serial_protocol import RefereeGenericMessage
from driver.referee.tx_policy import RefereeTxPolicy
from driver.referee.udp_protocol import InfoMsgBag
from tools.referee_web_injector import (
    AttachedInfoWaveTransport,
    AttachedLateOverrideTransport,
    DryRunTransport,
    InfoWaveSnapshotFactory,
    InjectionController,
    MESSAGE_KEYS,
    PAGE_CONFIRMATION,
    PacketFactory,
    default_config,
    normalize_config,
)


def test_late_attach_overrides_formal_manager_sources_without_owning_serial(
    monkeypatch, tmp_path
):
    import tools.referee_web_injector as injector_module

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        late_port = probe.getsockname()[1]
    session = "pytest-late-override-session"
    marker = tmp_path / "full-chain-test.env"
    marker.write_text(
        "mode=full_chain_test\nrelay_port=5010\nisolation_port=15010\n"
        f"session={session}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(injector_module, "FULL_CHAIN_TEST_MARKER", marker)
    monkeypatch.setattr(
        AttachedLateOverrideTransport,
        "_formal_main_running",
        lambda self: True,
    )

    manager = object.__new__(RefereeCommManager)
    manager.web_late_test_session = session
    manager.web_late_test_port = late_port
    manager.web_late_override_lock = __import__("threading").Lock()
    manager.web_late_override_config = None
    manager.web_late_override_received_monotonic = None
    manager.web_late_rx_socket = None
    manager.web_late_rx_stop_event = None
    manager.web_late_rx_thread = None
    manager.game_progress = manager.ACTIVE_GAME_PROGRESS
    manager.robot_id = 9
    manager.faction = FACTION.RED
    manager.last_radar2client_tx_time = None
    manager.radar2client_tx_fps = 0.0
    manager.enable_info_wave_robot_multicast = True
    manager.info_wave_robot_multicast_receiver_credits = [0, 0, 0]
    manager.get_logger = lambda: type(
        "_Logger", (), {"warning": lambda *_: None, "debug": lambda *_: None}
    )()
    sent_frames = []
    manager.tx = lambda frame: sent_frames.append(bytes(frame)) or True

    config = normalize_config(default_config())
    for key in MESSAGE_KEYS:
        config["messages"][key]["enabled"] = True
    config["messages"]["radar_map"]["coordinates"]["enemy_hero_x"] = 1234
    config["messages"]["decision"].update(
        radar_cmd=2, key_command=2, key_text="ABCDEF"
    )
    config["messages"]["engineer"].update(
        aerial_ammo=0x1234, remaining_coins=0x5678
    )
    config["messages"]["robot_multicast"]["hero"].update(
        hp_percent=75,
        ammo=0x1111,
        defense=10,
        negative_defense=20,
        targetable=True,
    )

    transport = AttachedLateOverrideTransport(port=late_port)
    try:
        manager._start_web_late_override_receiver()
        transport.start()
        assert transport.send_config(config) is True
        deadline = time.monotonic() + 1.0
        while (
            manager._get_web_late_override("radar_map") is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        manager._send_radar2client()
        manager._send_radar_decision()
        manager._send_radar2dart()
        manager._send_info_wave()
        manager._send_radar2sentry()
        manager._send_info_wave_robot_multicast_one()

        assert len(sent_frames) == 6
        assert int.from_bytes(sent_frames[0][7:9], "little") == 1234
        sub_ids = [int.from_bytes(frame[7:9], "little") for frame in sent_frames[1:]]
        assert sub_ids == [0x0121, 0x0234, 0x02AA, 0x0233, 0x02AB]
        assert sent_frames[1][13:21] == b"\x02\x02ABCDEF"
        assert sent_frames[3][13:17] == bytes.fromhex("34 12 78 56")
        assert sent_frames[5][13:19] == bytes.fromhex("4B 11 11 0A 14 01")
    finally:
        transport.stop()
        manager._stop_web_late_override_receiver()


def test_attached_info_wave_requires_test_start_marker(monkeypatch, tmp_path):
    import tools.referee_web_injector as injector_module

    monkeypatch.setattr(
        injector_module,
        "FULL_CHAIN_TEST_MARKER",
        tmp_path / "missing-test-marker.env",
    )
    transport = AttachedInfoWaveTransport(relay_port=5010)
    with pytest.raises(RuntimeError, match="--test-info-wave-injection"):
        transport.start()


def test_attached_late_override_requires_test_start_marker(monkeypatch, tmp_path):
    import tools.referee_web_injector as injector_module

    monkeypatch.setattr(
        injector_module,
        "FULL_CHAIN_TEST_MARKER",
        tmp_path / "missing-test-marker.env",
    )
    transport = AttachedLateOverrideTransport(port=5011)
    with pytest.raises(RuntimeError, match="--test-info-wave-injection"):
        transport.start()


def test_attached_info_wave_only_sends_to_authorized_formal_relay(
    monkeypatch, tmp_path
):
    import tools.referee_web_injector as injector_module

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    relay_port = receiver.getsockname()[1]
    marker = tmp_path / "full-chain-test.env"
    marker.write_text(
        f"mode=full_chain_test\nrelay_port={relay_port}\n"
        "isolation_port=15010\nsession=test-session\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(injector_module, "FULL_CHAIN_TEST_MARKER", marker)
    monkeypatch.setattr(
        AttachedInfoWaveTransport,
        "_formal_relay_running",
        lambda self: True,
    )

    transport = AttachedInfoWaveTransport(relay_port=relay_port)
    try:
        transport.start()
        assert transport.send(b"IF\x03test") is True
        payload, _address = receiver.recvfrom(1024)
        assert payload == b"IF\x03test"
        assert transport.status()["attached_to_start"] is True
    finally:
        transport.stop()
        receiver.close()


def _enabled_config(side="red"):
    config = default_config()
    config["side"] = side
    for key in MESSAGE_KEYS:
        config["messages"][key]["enabled"] = True
    return normalize_config(config)


def test_web_injector_defaults_are_safe_and_disabled():
    config = normalize_config(default_config())

    assert all(not config["messages"][key]["enabled"] for key in MESSAGE_KEYS)
    sentry = config["messages"]["sentry"]
    assert sentry["hero_targetable"] is True
    assert sentry["engineer_targetable"] is False
    assert sentry["prefer_long_hole"] is False
    assert config["messages"]["health"]["enabled"] is False


def test_web_mode_can_switch_between_early_and_late_only_while_stopped():
    created = []

    def factory(requested_mode):
        transport = DryRunTransport()
        created.append((requested_mode, transport))
        return transport, requested_mode

    controller = InjectionController(
        transport=DryRunTransport(),
        max_active_sec=1.0,
        mode="serial",
        transport_factory=factory,
    )
    assert controller.set_mode("info-wave-attach") == "info-wave-attach"
    assert controller.info_wave_mode is True
    assert controller.message_keys == ("radar_map", "health", "engineer", "sentry")

    controller.start(PAGE_CONFIRMATION)
    with pytest.raises(ValueError, match="请先停止注入"):
        controller.set_mode("late-attach")
    controller.stop()

    assert controller.set_mode("late-attach") == "late-attach"
    assert controller.info_wave_mode is False
    assert len(created) == 2


def test_info_wave_early_injection_builds_five_independent_command_updates():
    config = normalize_config(default_config())
    coordinates = config["messages"]["radar_map"]["coordinates"]
    for index, (robot_key, _label) in enumerate(
        (
            ("enemy_hero", ""),
            ("enemy_engineer", ""),
            ("enemy_standard_3", ""),
            ("enemy_standard_4", ""),
            ("enemy_drone", ""),
            ("enemy_sentry", ""),
        ),
        start=1,
    ):
        coordinates[f"{robot_key}_x"] = index * 10 + 1
        coordinates[f"{robot_key}_y"] = index * 10 + 2
    config["messages"]["health"].update(
        hero=101,
        engineer=102,
        standard_3=103,
        standard_4=104,
        reserved=105,
        sentry=106,
    )
    config["messages"]["engineer"].update(
        aerial_ammo=0x1234,
        remaining_coins=0x5678,
    )
    robot_summary = config["messages"]["robot_multicast"]
    robot_summary["hero"].update(ammo=0x1111, defense=10, negative_defense=20)
    robot_summary["standard_3"].update(ammo=0x2222, defense=30, negative_defense=40)
    robot_summary["standard_4"].update(ammo=0x3333, defense=50, negative_defense=60)
    robot_summary["sentry"].update(ammo=0x4444, defense=70, negative_defense=80)
    robot_summary["hero"]["targetable"] = True
    robot_summary["engineer"]["targetable"] = False
    robot_summary["standard_3"]["targetable"] = False
    robot_summary["standard_4"]["targetable"] = True
    robot_summary["sentry"]["targetable"] = False

    factory = InfoWaveSnapshotFactory()
    frames = []
    for message_key in ("radar_map", "health", "engineer", "sentry"):
        frames.extend(factory.build(message_key, config))

    assert [label for label, _frame in frames] == [
        "0x0A01",
        "0x0A02",
        "0x0A03",
        "0x0A04",
        "0x0A05",
    ]
    messages = [InfoMsgBag.from_bytes(frame) for _label, frame in frames]
    assert [message.seq for message in messages] == [1, 2, 3, 4, 5]
    assert [message.updated_cmd_id for message in messages] == list(InfoMsgBag.CMD_IDS)
    assert [message.valid_mask for message in messages] == [1, 3, 7, 15, 31]
    assert [message.fresh_mask for message in messages] == [1, 3, 7, 15, 31]
    assert all(len(message.pack()) == 102 for message in messages)

    assert struct.unpack("<12H", messages[-1].get_cmd_payload(0x0A01)) == (
        11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62
    )
    assert struct.unpack("<6H", messages[-1].get_cmd_payload(0x0A02)) == (
        101, 102, 103, 104, 105, 106
    )
    assert struct.unpack("<5H", messages[-1].get_cmd_payload(0x0A03)) == (
        0x1111,
        0x2222,
        0x3333,
        0x1234,
        0x4444,
    )
    assert struct.unpack("<HHI", messages[-1].get_cmd_payload(0x0A04))[0] == 0x5678
    payload_0a05 = messages[-1].get_cmd_payload(0x0A05)
    assert payload_0a05[3:5] == bytes([10, 20])
    assert payload_0a05[17:19] == bytes([30, 40])
    assert payload_0a05[24:26] == bytes([50, 60])
    assert payload_0a05[31:33] == bytes([70, 80])
    assert payload_0a05[35:41] == bytes([0, 0, 1, 1, 0, 1])


def test_info_wave_early_injection_controller_emits_info_packets_not_serial_frames():
    transport = DryRunTransport()
    controller = InjectionController(
        transport=transport,
        max_active_sec=1.0,
        mode="info-wave-dry-run",
    )
    config = default_config()
    config["messages"]["engineer"].update(enabled=True, frequency_hz=20)
    controller.update_config(config)
    controller.start(PAGE_CONFIRMATION)
    time.sleep(0.08)
    controller.stop()

    assert len(transport.frames) >= 2
    messages = [InfoMsgBag.from_bytes(frame) for frame in transport.frames]
    assert {message.updated_cmd_id for message in messages} == {0x0A03, 0x0A04}
    assert all(frame[:3] == b"IF\x03" for frame in transport.frames)


def test_early_injection_packets_drive_formal_engineer_and_sentry_builders():
    """早注只提供解析结果，最终0x02AA/0x0233必须由正式管理器生成。"""
    config = normalize_config(default_config())
    config["messages"]["engineer"].update(
        aerial_ammo=0x1234,
        remaining_coins=0x5678,
    )
    robot_summary = config["messages"]["robot_multicast"]
    robot_summary["hero"]["targetable"] = True
    robot_summary["engineer"]["targetable"] = False
    robot_summary["standard_3"]["targetable"] = True
    robot_summary["standard_4"]["targetable"] = False
    robot_summary["sentry"]["targetable"] = True

    manager = object.__new__(RefereeCommManager)
    manager.game_progress = manager.ACTIVE_GAME_PROGRESS
    manager.robot_id = 9
    manager.faction = type("_Faction", (), {"name": "RED"})()
    manager.info_msg = None
    manager.info_msg_received_monotonic = None
    manager.info_msg_valid_0a05_latest_time = None
    manager.info_wave_payload_cache = manager._make_empty_info_wave_payload_cache()
    manager.info_0a01_payload = None
    manager.info_enemy_positions_cm = None
    manager.info_0a01_last_update_time = None
    manager.info_0a01_last_snapshot_seq = None
    manager.info_0a01_update_count = 0
    manager.last_info_0a01_log_time = 0.0
    manager.sent_frames = []
    manager.tx = lambda frame: manager.sent_frames.append(frame) or True

    factory = InfoWaveSnapshotFactory()
    for message_key in ("engineer", "sentry"):
        for _label, frame in factory.build(message_key, config):
            manager._handle_udp_rx_message(InfoMsgBag.from_bytes(frame))

    manager._send_info_wave()
    manager._send_radar2sentry()

    engineer_frame, sentry_frame = manager.sent_frames
    assert engineer_frame[13:17] == bytes.fromhex("34 12 78 56")
    assert sentry_frame[13] == 0xD5


def test_web_injector_uses_official_map_layout():
    config = _enabled_config()
    coordinates = config["messages"]["radar_map"]["coordinates"]
    for index, field_name in enumerate(coordinates, start=1):
        coordinates[field_name] = index

    frame = PacketFactory.build("radar_map", config)

    assert frame[5:7] == bytes.fromhex("05 03")
    assert int.from_bytes(frame[1:3], "little") == 48
    assert struct.unpack("<24H", frame[7:55]) == tuple(range(1, 25))
    assert len(frame) == 57


def test_web_injector_builds_human_fields_into_interactive_packets():
    config = _enabled_config()
    config["messages"]["decision"].update(
        radar_cmd=7,
        key_command=2,
        key_text="A1b2C3",
    )
    config["messages"]["dart"].update(
        trigger_state=3,
        remaining_count=2,
        autoaim_stage=6,
    )
    config["messages"]["engineer"].update(
        aerial_ammo=0x1234,
        remaining_coins=0x5678,
    )
    config["messages"]["sentry"].update(
        hero_targetable=True,
        engineer_targetable=False,
        standard_3_targetable=True,
        standard_4_targetable=False,
        sentry_targetable=True,
        prefer_long_hole=True,
    )
    robot_summary = config["messages"]["robot_multicast"]
    robot_summary["hero"].update(
        hp_percent=75,
        ammo=0x1111,
        defense=10,
        negative_defense=20,
        targetable=True,
    )
    robot_summary["engineer"].update(
        hp_percent=80,
        ammo=0xFFFF,
        defense=30,
        negative_defense=40,
        targetable=False,
    )

    decision = PacketFactory.build("decision", config)
    dart = PacketFactory.build("dart", config)
    engineer = PacketFactory.build("engineer", config)
    sentry = PacketFactory.build("sentry", config)
    robot_multicast = PacketFactory.build("robot_multicast", config)

    assert decision[5:7] == bytes.fromhex("01 03")
    assert decision[7:13] == bytes.fromhex("21 01 09 00 80 80")
    assert decision[13:21] == b"\x07\x02A1b2C3"
    assert dart[7:13] == bytes.fromhex("34 02 09 00 06 00")
    assert dart[13:16] == bytes([3, 2, 6])
    assert engineer[7:13] == bytes.fromhex("AA 02 09 00 02 00")
    assert engineer[13:17] == bytes.fromhex("34 12 78 56")
    assert sentry[7:13] == bytes.fromhex("33 02 09 00 07 00")
    assert sentry[13] == 0xF5
    assert robot_multicast[7:13] == bytes.fromhex("AB 02 09 00 01 00")
    assert robot_multicast[13:25] == bytes.fromhex(
        "4B 11 11 0A 14 01 50 00 00 1E 28 00"
    )
    assert int.from_bytes(robot_multicast[1:3], "little") == 36
    assert len(robot_multicast) == 45


def test_web_injector_blue_side_uses_fixed_blue_ids():
    config = _enabled_config(side="blue")

    decision = PacketFactory.build("decision", config)
    dart = PacketFactory.build("dart", config)
    engineer = PacketFactory.build("engineer", config)
    sentry = PacketFactory.build("sentry", config)
    robot_multicast = PacketFactory.build("robot_multicast", config)

    assert decision[9:13] == bytes.fromhex("6D 00 80 80")
    assert dart[9:13] == bytes.fromhex("6D 00 6A 00")
    assert engineer[9:13] == bytes.fromhex("6D 00 66 00")
    assert sentry[9:13] == bytes.fromhex("6D 00 6B 00")
    assert robot_multicast[9:13] == bytes.fromhex("6D 00 65 00")


def test_frame_preview_does_not_consume_real_serial_sequence():
    config = _enabled_config()
    RefereeGenericMessage._seq_counter = 41

    PacketFactory.preview(config)

    assert RefereeGenericMessage._seq_counter == 41
    frame = PacketFactory.build("sentry", config)
    assert frame[3] == 41
    assert RefereeGenericMessage._seq_counter == 42


def test_controller_requires_page_confirmation_and_auto_stops():
    controller = InjectionController(
        transport=DryRunTransport(),
        max_active_sec=0.05,
        mode="dry-run",
    )

    with pytest.raises(ValueError, match=PAGE_CONFIRMATION):
        controller.start("错误确认")

    controller.start(PAGE_CONFIRMATION)
    assert controller.snapshot()["active"] is True
    time.sleep(0.12)
    assert controller.snapshot()["active"] is False
    assert "自动停止" in controller.snapshot()["last_error"]


def test_running_controller_uses_updated_fields_on_next_send_cycle():
    transport = DryRunTransport()
    controller = InjectionController(
        transport=transport,
        max_active_sec=1.0,
        mode="dry-run",
    )
    first = default_config()
    first["messages"]["engineer"].update(
        enabled=True,
        frequency_hz=20,
        aerial_ammo=123,
        remaining_coins=456,
    )
    controller.update_config(first)
    controller.start(PAGE_CONFIRMATION)
    time.sleep(0.08)

    second = default_config()
    second["messages"]["engineer"].update(
        enabled=True,
        frequency_hz=20,
        aerial_ammo=321,
        remaining_coins=654,
    )
    controller.update_config(second)
    time.sleep(0.12)
    controller.stop()

    user_fields = [frame[13:17] for frame in transport.frames]
    assert struct.pack("<HH", 123, 456) in user_fields
    assert struct.pack("<HH", 321, 654) in user_fields


def test_running_controller_applies_frequency_change_without_restart():
    transport = DryRunTransport()
    controller = InjectionController(
        transport=transport,
        max_active_sec=2.0,
        mode="dry-run",
    )
    config = default_config()
    config["messages"]["engineer"].update(enabled=True, frequency_hz=10)
    controller.update_config(config)
    controller.start(PAGE_CONFIRMATION)
    time.sleep(0.24)
    count_before_slowdown = controller.snapshot()["send_counts"]["engineer"]
    assert count_before_slowdown >= 2

    config["messages"]["engineer"]["frequency_hz"] = 1
    controller.update_config(config)
    time.sleep(0.3)
    count_during_slow_period = controller.snapshot()["send_counts"]["engineer"]
    assert count_during_slow_period == count_before_slowdown

    config["messages"]["engineer"]["frequency_hz"] = 20
    controller.update_config(config)
    time.sleep(0.13)
    count_after_speedup = controller.snapshot()["send_counts"]["engineer"]
    controller.stop()

    assert count_after_speedup >= count_during_slow_period + 2


def test_default_seven_hz_interactive_scheduler_does_not_starve_dart_packet():
    controller = InjectionController(
        transport=DryRunTransport(),
        max_active_sec=1.0,
        mode="dry-run",
    )
    config = default_config()
    config["messages"]["decision"]["enabled"] = True
    config["messages"]["dart"]["enabled"] = True
    controller.update_config(config)
    controller.start(PAGE_CONFIRMATION)
    time.sleep(0.46)
    counts = controller.snapshot()["send_counts"]
    controller.stop()

    assert counts["decision"] >= 2
    assert counts["dart"] >= 2


def test_main_and_web_injector_cannot_hold_the_same_serial_lock():
    first = object.__new__(RefereeSerialManager)
    second = object.__new__(RefereeSerialManager)
    for manager in (first, second):
        manager.serial_lock_file = None
        manager.serial_lock_path = None

    port = "/tmp/rm2026-referee-web-lock-test"
    assert first._acquire_serial_lock(port) is True
    try:
        assert second._acquire_serial_lock(port) is False
    finally:
        first._release_serial_lock()

    assert second._acquire_serial_lock(port) is True
    second._release_serial_lock()


def test_web_injector_rejects_non_semantic_or_invalid_values():
    config = default_config()
    config["messages"]["decision"]["key_text"] = "ABC!"
    with pytest.raises(ValueError, match="6个ASCII"):
        normalize_config(config)

    config = default_config()
    config["messages"]["decision"]["key_command"] = 1
    with pytest.raises(ValueError, match="只能选择"):
        normalize_config(config)

    config = default_config()
    config["messages"]["radar_map"]["coordinates"]["enemy_hero_x"] = 70000
    with pytest.raises(ValueError, match="0～65535"):
        normalize_config(config)

    config = default_config()
    config["messages"]["engineer"]["frequency_hz"] = 20.1
    with pytest.raises(ValueError, match="0.1～20 Hz"):
        normalize_config(config)


def test_web_injector_and_main_share_the_same_frequency_policy():
    config = default_config()
    assert config["messages"]["radar_map"]["frequency_hz"] == 5.0
    assert config["messages"]["decision"]["frequency_hz"] == 5.0
    assert config["messages"]["dart"]["frequency_hz"] == 5.0
    assert config["messages"]["engineer"]["frequency_hz"] == 5.0
    assert config["messages"]["sentry"]["frequency_hz"] == 5.0
    assert config["messages"]["robot_multicast"]["frequency_hz"] == 8.0
    assert RefereeCommManager.RADAR2CLIENT_TX_HZ == RefereeTxPolicy.RADAR2CLIENT_TX_HZ
    assert RefereeCommManager.INTERACTIVE_ROUND_HZ == RefereeTxPolicy.INTERACTIVE_ROUND_HZ
    assert (
        RefereeCommManager.INTERACTIVE_ROUND_GAP_SEC
        == RefereeTxPolicy.INTERACTIVE_ROUND_GAP_SEC
    )
    assert (
        RefereeCommManager.INFO_WAVE_ENGINEER_TX_HZ
        == RefereeTxPolicy.INFO_WAVE_ENGINEER_TX_HZ
    )
    assert (
        RefereeCommManager.INFO_WAVE_SENTRY_TX_HZ
        == RefereeTxPolicy.INFO_WAVE_SENTRY_TX_HZ
    )
