# -*- coding: utf-8 -*-
"""统一解析正式运行与同步回放运行时使用的配置路径。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


PARAMS_CONFIG_ENV = "RM_PARAMS_CONFIG"
TEST_CONFIG_ENV = "RM_REPLAY_TEST_CONFIG"
EXTRINSICS_IMPORT_ENV = "RM_EXTRINSICS_IMPORT_PATH"


def get_params_config_path(default: str = "config/params.yaml") -> str:
    """返回当前进程唯一的 params.yaml 路径。"""
    return os.path.abspath(os.environ.get(PARAMS_CONFIG_ENV, default))


def get_test_config_path() -> str | None:
    value = os.environ.get(TEST_CONFIG_ENV, "").strip()
    return os.path.abspath(value) if value else None


def load_params_config(default: str = "config/params.yaml") -> dict:
    with open(get_params_config_path(default), "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_test_config() -> dict:
    path = get_test_config_path()
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_extrinsics_import_path(default_dir: str) -> str:
    """测试时优先导入随录像保存的外参，否则使用全局 latest.txt。"""
    override = os.environ.get(EXTRINSICS_IMPORT_ENV, "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    return os.path.join(default_dir, "latest.txt")
