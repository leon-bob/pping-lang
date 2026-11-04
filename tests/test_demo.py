"""pping_lang.demo(pping-demo 入口)冒烟:零时长跑一轮,不依赖 GPU/vLLM。

只验证入口能起能收(插件初始化 → 注入路径 → drain/stop/close),规则触发
语义由 tests/test_e2e_marquee.py 覆盖。
"""
from __future__ import annotations

import os

from pping_lang.demo import DEMO_DB, main


def test_demo_main_smoke(monkeypatch):
    monkeypatch.setenv("DEMO_DURATION_S", "0")
    _snap = dict(os.environ)  # main() 会真改 env(DB 路径/间隔等),快照还原防泄漏
    try:
        assert main() == 0
        assert DEMO_DB.exists()  # 走到了建库/落盘那一步
    finally:
        os.environ.clear()
        os.environ.update(_snap)
