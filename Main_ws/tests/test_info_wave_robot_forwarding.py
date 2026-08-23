import struct
import threading
import time

import pytest

from driver.referee.referee_comm import FACTION, RefereeCommManager


class _InfoSnapshot:
    def __init__(self, valid_cmd_ids=(), fresh_mask=0):
        self.valid_cmd_ids = set(valid_cmd_ids)
        self.fresh_mask = int(fresh_mask)

    def is_cmd_valid(self, cmd_id: int) -> bool:
        return cmd_id in self.valid_cmd_ids


def _make_manager(valid_cmd_ids=(), fresh_mask=0) -> RefereeCommManager:
    """构造不连接串口和 ROS2 的最小信息波转发管理器。"""
    manager = object.__new__(RefereeCommManager)
    manager.game_progress = manager.ACTIVE_GAME_PROGRESS
    manager.robot_id = 9
    manager.faction = FACTION.RED
    manager.info_msg = _InfoSnapshot(valid_cmd_ids, fresh_mask)
    manager.info_msg_received_monotonic = time.monotonic()
    manager.info_wave_payload_cache = manager._make_empty_info_wave_payload_cache()
    manager.info_enemy_positions_cm = None
    manager.info_0a01_last_update_time = None
    manager.info_wave_position_timeout_sec = 0.25
    manager.enable_info_wave_robot_multicast = False
    manager.info_wave_robot_multicast_receiver_credits = [0, 0, 0]
    manager.info_wave_robot_max_health = [0, 0, 0, 0, 0]
    manager.sentry_prefer_long_hole = False
    manager.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 0.0}
    manager.sent_frames = []
    manager.tx = lambda frame: manager.sent_frames.append(frame) or True
    return manager


def test_engineer_packet_contains_only_aerial_ammo_and_remaining_coins():
    """0x02AA user_data 只保留两个 uint16 小端字段，不携带 fresh。"""
    manager = _make_manager(valid_cmd_ids=(0x0A03, 0x0A04))
    projectile_payload = bytearray(10)
    economy_payload = bytearray(8)
    projectile_payload[6:8] = bytes.fromhex("34 12")
    economy_payload[0:2] = bytes.fromhex("78 56")
    manager.info_wave_payload_cache[0x0A03] = bytes(projectile_payload)
    manager.info_wave_payload_cache[0x0A04] = bytes(economy_payload)

    manager._send_info_wave()

    frame = manager.sent_frames[-1]
    assert int.from_bytes(frame[1:3], byteorder="little") == 10
    assert frame[7:13] == bytes.fromhex("AA 02 09 00 02 00")
    assert frame[13:17] == bytes.fromhex("34 12 78 56")
    assert len(frame) == 19


def test_engineer_packet_sends_when_only_0a03_is_valid():
    """只有 0x0A03 有效时也发送，尚未有效的 0x0A04 字段保持为零。"""
    manager = _make_manager(valid_cmd_ids=(0x0A03,))
    projectile_payload = bytearray(10)
    projectile_payload[6:8] = bytes.fromhex("34 12")
    manager.info_wave_payload_cache[0x0A03] = bytes(projectile_payload)

    manager._send_info_wave()

    assert manager.sent_frames[-1][13:17] == bytes.fromhex("34 12 00 00")


def test_engineer_packet_sends_when_only_0a04_is_valid():
    """只有 0x0A04 有效时也发送，尚未有效的 0x0A03 字段保持为零。"""
    manager = _make_manager(valid_cmd_ids=(0x0A04,))
    economy_payload = bytearray(8)
    economy_payload[0:2] = bytes.fromhex("78 56")
    manager.info_wave_payload_cache[0x0A04] = bytes(economy_payload)

    manager._send_info_wave()

    assert manager.sent_frames[-1][13:17] == bytes.fromhex("00 00 78 56")


def test_engineer_packet_waits_when_neither_source_is_valid():
    """两个来源都未有效时不发送，避免把全零占位误认为有效数据。"""
    manager = _make_manager(valid_cmd_ids=())

    manager._send_info_wave()

    assert manager.sent_frames == []


def test_engineer_packet_is_not_sent_before_game_starts():
    manager = _make_manager(valid_cmd_ids=(0x0A03, 0x0A04))
    manager.game_progress = 3

    manager._send_info_wave()

    assert manager.sent_frames == []


def _fill_robot_multicast_payloads(manager):
    health = bytearray(12)
    projectile = bytearray(10)
    buff = bytearray(41)

    for offset, value in (
        (0, 0x1234),
        (2, 0x1A2B),
        (4, 0x2345),
        (6, 0x3456),
        (10, 0x4567),
    ):
        health[offset : offset + 2] = struct.pack("<H", value)
    for offset, value in ((0, 0x1111), (2, 0x2222), (4, 0x3333), (8, 0x4444)):
        projectile[offset : offset + 2] = struct.pack("<H", value)

    buff[3], buff[4], buff[36] = 10, 20, 0
    buff[10], buff[11], buff[37] = 15, 25, 0
    buff[17], buff[18], buff[38] = 30, 40, 1
    buff[24], buff[25], buff[39] = 50, 60, 0
    buff[31], buff[32], buff[40] = 70, 80, 0
    buff[35] = 5

    manager.info_wave_payload_cache[0x0A02] = bytes(health)
    manager.info_wave_payload_cache[0x0A03] = bytes(projectile)
    manager.info_wave_payload_cache[0x0A05] = bytes(buff)


def test_robot_multicast_user_data_contains_four_fixed_records():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    _fill_robot_multicast_payloads(manager)

    user_data = manager._build_info_wave_robot_multicast_user_data()

    assert len(user_data) == 30
    assert user_data == bytes.fromhex(
        "64 11 11 0A 14 01 "
        "64 00 00 0F 19 01 "
        "64 22 22 1E 28 00 "
        "64 33 33 32 3C 01 "
        "64 44 44 46 50 00"
    )


def test_robot_multicast_health_percentage_uses_per_robot_maximum_cache():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    _fill_robot_multicast_payloads(manager)

    maximum_health = bytearray(12)
    current_health = bytearray(12)
    for offset, maximum, current in (
        (0, 1000, 500),
        (2, 1500, 750),
        (4, 2000, 1500),
        (6, 3000, 3000),
        (10, 4000, 0),
    ):
        maximum_health[offset : offset + 2] = struct.pack("<H", maximum)
        current_health[offset : offset + 2] = struct.pack("<H", current)

    manager._update_info_wave_robot_max_health(bytes(maximum_health))
    manager.info_wave_payload_cache[0x0A02] = bytes(current_health)

    user_data = manager._build_info_wave_robot_multicast_user_data()

    assert [user_data[offset] for offset in (0, 6, 12, 18, 24)] == [
        50,
        50,
        75,
        100,
        0,
    ]
    assert manager.info_wave_robot_max_health == [1000, 1500, 2000, 3000, 4000]


def test_robot_multicast_health_maximum_only_increases_and_zero_is_safe():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    zero_health = bytes(12)

    manager._update_info_wave_robot_max_health(zero_health)
    manager.info_wave_payload_cache[0x0A02] = zero_health
    manager.info_wave_payload_cache[0x0A03] = bytes(10)
    manager.info_wave_payload_cache[0x0A05] = bytes(41)

    assert manager._build_info_wave_robot_multicast_user_data()[0::6] == bytes(5)
    assert manager.info_wave_robot_max_health == [0, 0, 0, 0, 0]


def test_robot_multicast_switch_off_does_not_send():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    _fill_robot_multicast_payloads(manager)

    assert manager._send_info_wave_robot_multicast_one() is False
    assert manager.sent_frames == []


def test_robot_multicast_waits_until_all_three_sources_are_valid():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03))
    manager.enable_info_wave_robot_multicast = True
    _fill_robot_multicast_payloads(manager)

    assert manager._send_info_wave_robot_multicast_one() is False
    assert manager.sent_frames == []


def test_robot_multicast_weights_red_hero_lower_than_two_infantries():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    manager.enable_info_wave_robot_multicast = True
    _fill_robot_multicast_payloads(manager)

    for _ in range(5):
        assert manager._send_info_wave_robot_multicast_one() is True

    assert [int.from_bytes(frame[7:9], "little") for frame in manager.sent_frames] == [
        0x02AB,
        0x02AB,
        0x02AB,
        0x02AB,
        0x02AB,
    ]
    assert [int.from_bytes(frame[9:11], "little") for frame in manager.sent_frames] == [
        9,
        9,
        9,
        9,
        9,
    ]
    assert [int.from_bytes(frame[11:13], "little") for frame in manager.sent_frames] == [
        3,
        4,
        1,
        3,
        4,
    ]
    assert all(int.from_bytes(frame[1:3], "little") == 36 for frame in manager.sent_frames)
    assert all(len(frame) == 45 for frame in manager.sent_frames)
    assert manager.sent_frames[0][13:43] == manager.sent_frames[1][13:43]


def test_robot_multicast_weights_blue_receivers_and_sender_id():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    manager.enable_info_wave_robot_multicast = True
    manager.robot_id = 109
    manager.faction = FACTION.BLUE
    _fill_robot_multicast_payloads(manager)

    for _ in range(5):
        manager._send_info_wave_robot_multicast_one()

    assert [int.from_bytes(frame[9:11], "little") for frame in manager.sent_frames] == [
        109,
        109,
        109,
        109,
        109,
    ]
    assert [int.from_bytes(frame[11:13], "little") for frame in manager.sent_frames] == [
        103,
        104,
        101,
        103,
        104,
    ]


def test_radar_decision_has_priority_over_optional_robot_multicast():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    manager.enable_info_wave_robot_multicast = True
    manager._send_radar_decision = lambda: True
    manager._send_info_wave_robot_multicast_one = lambda: (_ for _ in ()).throw(
        AssertionError("决策槽被占用时不应发送可选摘要")
    )

    assert manager._send_radar_decision_or_info_wave_robot_multicast() is True


def test_idle_radar_decision_slot_is_given_to_robot_multicast():
    manager = _make_manager(valid_cmd_ids=(0x0A02, 0x0A03, 0x0A05))
    calls = []
    manager._send_radar_decision = lambda: False
    manager._send_info_wave_robot_multicast_one = lambda: calls.append(
        "multicast"
    ) or True

    assert manager._send_radar_decision_or_info_wave_robot_multicast() is True
    assert calls == ["multicast"]


def test_sentry_flags_follow_main_status_pose_and_fresh_rules():
    """新鲜 0x0A05 按五个主状态和哨兵姿态生成可瞄准位。"""
    manager = _make_manager(
        valid_cmd_ids=(0x0A05,),
        fresh_mask=1 << manager_cmd_index(0x0A05),
    )
    payload = bytearray(41)
    payload[36] = 1
    payload[37] = 2
    payload[38] = 0
    payload[39] = 3
    payload[40] = 0
    payload[35] = 5
    manager.info_wave_payload_cache[0x0A05] = bytes(payload)

    manager._send_radar2sentry()

    frame = manager.sent_frames[-1]
    # 英雄/工程/4号/哨兵不可瞄准，3号可瞄准；无坐标时 bit5 默认 short。
    assert frame[13] == 0xC4
    assert int.from_bytes(frame[1:3], byteorder="little") == 7
    assert len(frame) == 16


def test_sentry_fresh_alive_engineer_is_targetable():
    """工程仅在新鲜状态明确等于 0 时置 1。"""
    manager = _make_manager(
        valid_cmd_ids=(0x0A05,),
        fresh_mask=1 << manager_cmd_index(0x0A05),
    )
    manager.info_wave_payload_cache[0x0A05] = bytes(41)

    manager._send_radar2sentry()

    assert manager.sent_frames[-1][13] == 0xDF


def test_sentry_stale_cache_uses_safe_defaults_and_still_sends():
    """超过 fresh 时忽略旧状态，工程为 0，其余目标为 1。"""
    manager = _make_manager(valid_cmd_ids=(0x0A05,), fresh_mask=0)
    manager.info_wave_payload_cache[0x0A05] = bytes([5] * 41)

    manager._send_radar2sentry()

    assert manager.sent_frames[-1][13] == 0xDD


def test_python_udp_freshness_guard_uses_three_second_window():
    """主程序保留 3 秒内的 fresh_mask，超过 3 秒强制清零。"""
    manager = _make_manager(valid_cmd_ids=(0x0A05,), fresh_mask=0x1F)
    now = time.monotonic()

    manager.info_msg_received_monotonic = now - 2.9
    assert manager._current_info_fresh_mask(now) == 0x1F

    manager.info_msg_received_monotonic = now - 3.1
    assert manager._current_info_fresh_mask(now) == 0


def test_sentry_packet_is_sent_during_game_even_without_valid_0a05():
    manager = _make_manager(
        valid_cmd_ids=(),
        fresh_mask=1 << manager_cmd_index(0x0A05),
    )

    manager._send_radar2sentry()

    assert manager.sent_frames[-1][13] == 0xDD


def test_sentry_hole_centers_are_mirrored_for_blue_faction():
    """蓝方区域圆心必须绕 28m×15m 场地中心作 180 度对称。"""
    manager = _make_manager()
    assert manager.get_sentry_hole_centers_m() == {
        "long_hole": (15.5, 1.56),
        "short_hole": (19.0, 12.0),
    }

    manager.robot_id = 109
    manager.faction = FACTION.BLUE
    assert manager.get_sentry_hole_centers_m() == {
        "long_hole": (12.5, 13.44),
        "short_hole": (9.0, 3.0),
    }


def test_sentry_bit5_prefers_region_with_lower_info_wave_score():
    """short_hole 内有敌人时 long 分数更低，bit5 应置 1。"""
    manager = _make_manager()
    manager.info_enemy_positions_cm = {
        "hero": (1900, 1200),
        "engineer": (1550, 156),
        "standard_3": (0, 0),
        "standard_4": (0, 0),
        "sentry": (0, 0),
    }
    manager.info_0a01_last_update_time = time.monotonic()

    manager._send_radar2sentry()

    assert manager.sentry_hole_scores == {
        "long_hole": 0.0,
        "short_hole": 2.0,
    }
    # 工程位于 long_hole 圆心但不参与评分；bit5=1 建议前往 long_hole。
    assert manager.sent_frames[-1][13] & 0x20


def test_sentry_score_uses_same_linear_function_for_both_holes():
    """两个区域均使用 2-0.4×距离的线性函数，不按半径归一化。"""
    manager = _make_manager()
    scores = manager._calculate_sentry_hole_scores(
        {
            "hero": (1550, 156),
            "standard_3": (1650, 156),
            "standard_4": (1900, 1350),
        }
    )

    assert scores["long_hole"] == pytest.approx(3.6)
    assert scores["short_hole"] == pytest.approx(1.4)


def test_sentry_long_hole_radius_is_larger_than_short_hole_radius():
    """距圆心 4.1m 属于长洞范围，距短洞圆心 3.1m 则不计分。"""
    manager = _make_manager()
    scores = manager._calculate_sentry_hole_scores(
        {
            "hero": (1960, 156),
            "standard_3": (2210, 1200),
        }
    )

    assert scores["long_hole"] == pytest.approx(0.36)
    assert scores["short_hole"] == 0.0


def test_sentry_score_snapshot_is_independent_for_ui():
    """UI 获取分数副本后不能反向修改通信管理器内部状态。"""
    manager = _make_manager()
    manager.sentry_hole_scores = {"long_hole": 1.25, "short_hole": 2.5}

    snapshot = manager.get_sentry_hole_scores_snapshot()
    snapshot["long_hole"] = 99.0

    assert manager.sentry_hole_scores == {
        "long_hole": 1.25,
        "short_hole": 2.5,
    }


def test_sentry_bit5_uses_mirrored_blue_regions():
    """蓝方 short_hole 圆心为 (9,3)，该处有敌人时应建议 long_hole。"""
    manager = _make_manager()
    manager.robot_id = 109
    manager.faction = FACTION.BLUE
    manager.info_enemy_positions_cm = {"sentry": (900, 300)}
    manager.info_0a01_last_update_time = time.monotonic()

    manager._send_radar2sentry()

    assert manager.sentry_hole_scores["long_hole"] == 0.0
    assert manager.sentry_hole_scores["short_hole"] == 2.0
    assert manager.sent_frames[-1][13] & 0x20


def test_sentry_stale_position_defaults_to_short_hole():
    """信息波坐标过期后清空旧分数，并默认建议 short_hole。"""
    manager = _make_manager()
    manager.sentry_prefer_long_hole = True
    manager.sentry_hole_scores = {"long_hole": 0.25, "short_hole": 1.5}
    manager.info_enemy_positions_cm = {"hero": (1550, 156)}
    manager.info_0a01_last_update_time = time.monotonic() - 1.0

    manager._send_radar2sentry()

    assert manager.sentry_hole_scores == {"long_hole": 0.0, "short_hole": 0.0}
    assert manager.sentry_prefer_long_hole is False
    assert not (manager.sent_frames[-1][13] & 0x20)


def test_sentry_equal_scores_use_short_hole():
    """两侧均无机器人或分数相等时选择 short_hole。"""
    manager = _make_manager()
    manager.info_enemy_positions_cm = {}
    manager.info_0a01_last_update_time = time.monotonic()

    manager._send_radar2sentry()

    assert manager.sentry_prefer_long_hole is False
    assert not (manager.sent_frames[-1][13] & 0x20)


def test_sentry_new_round_resets_hole_choice_to_short():
    """新小局清空信息波缓存时，区域建议恢复为 short_hole。"""
    manager = _make_manager()
    manager.sentry_prefer_long_hole = True
    manager.sentry_hole_scores = {"long_hole": 0.0, "short_hole": 2.0}

    manager._reset_info_wave_enemy_positions()

    assert manager.sentry_prefer_long_hole is False
    assert manager.sentry_hole_scores == {"long_hole": 0.0, "short_hole": 0.0}


def test_sentry_packet_is_not_sent_before_game_starts():
    manager = _make_manager(valid_cmd_ids=(), fresh_mask=0)
    manager.game_progress = 3

    manager._send_radar2sentry()

    assert manager.sent_frames == []


def test_forwarding_frequencies_match_robot_requirements():
    assert RefereeCommManager.INTERACTIVE_ROUND_HZ == 7.0
    assert RefereeCommManager.INFO_WAVE_ENGINEER_TX_HZ == 7.0
    assert RefereeCommManager.INFO_WAVE_SENTRY_TX_HZ == 7.0
    assert RefereeCommManager.INTERACTIVE_ROUND_GAP_SEC > 0.033
    assert RefereeCommManager.OPTIONAL_MULTICAST_SLOT_WEIGHTS == (5, 5, 5, 5, 8)
    assert sum(RefereeCommManager.OPTIONAL_MULTICAST_SLOT_WEIGHTS) == 28
    assert RefereeCommManager.OPTIONAL_MULTICAST_SLOT_GAP_SEC > 0.035


def test_interactive_packets_use_one_ordered_seven_hz_scheduler():
    """四种 0x0301 由同一循环轮询，相邻槽位严格大于 33 ms。"""
    manager = object.__new__(RefereeCommManager)
    manager.interactive_tx_stop_event = threading.Event()
    events = []

    def record(name):
        events.append((name, time.monotonic()))
        if len(events) >= 8:
            manager.interactive_tx_stop_event.set()
        return True

    manager._send_radar_decision = lambda: record("decision")
    manager._send_radar2dart = lambda: record("dart")
    manager._send_info_wave = lambda: record("engineer")
    manager._send_radar2sentry = lambda: record("sentry")

    manager._interactive_tx_loop()

    assert [name for name, _ in events] == [
        "decision",
        "dart",
        "engineer",
        "sentry",
        "decision",
        "dart",
        "engineer",
        "sentry",
    ]


def test_optional_multicast_scheduler_uses_five_five_five_five_eight_slots():
    """开关开启后每28槽严格按5/5/5/5/8分配，总频率为28 Hz。"""
    manager = object.__new__(RefereeCommManager)
    manager.enable_info_wave_robot_multicast = True
    manager.interactive_tx_stop_event = threading.Event()
    events = []

    def record(name):
        events.append((name, time.monotonic()))
        if len(events) >= 28:
            manager.interactive_tx_stop_event.set()
        return True

    manager._send_radar_decision_or_info_wave_robot_multicast = lambda: record(
        "decision"
    )
    manager._send_radar2dart = lambda: record("dart")
    manager._send_info_wave = lambda: record("engineer")
    manager._send_radar2sentry = lambda: record("sentry")
    manager._send_info_wave_robot_multicast_one = lambda: record("multicast")

    manager._interactive_tx_loop()

    names = [name for name, _ in events]
    assert names.count("decision") == 5
    assert names.count("dart") == 5
    assert names.count("engineer") == 5
    assert names.count("sentry") == 5
    assert names.count("multicast") == 8
    intervals = [
        events[index + 1][1] - events[index][1]
        for index in range(len(events) - 1)
    ]
    assert min(intervals) >= 0.033
    adjacent_gaps = [
        events[index][1] - events[index - 1][1]
        for index in range(1, len(events))
    ]
    assert min(adjacent_gaps) > 0.033
    message_periods = [
        events[index + 4][1] - events[index][1]
        for index in range(4)
    ]
    assert all(abs(period - 1.0 / 7.0) < 0.02 for period in message_periods)


def manager_cmd_index(cmd_id: int) -> int:
    return RefereeCommManager.INFO_WAVE_CMD_IDS.index(cmd_id)
