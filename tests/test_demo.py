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


def test_demo_skips_bench_seed_without_duckdb(monkeypatch, capsys):
    """HAVE_DUCKDB=False → 种子跳过、提示出现,主流程照跑(duckdb 属 [bench] extra)。"""
    from pping_lang import demo as demo_mod

    monkeypatch.setattr(demo_mod, "HAVE_DUCKDB", False)
    monkeypatch.setenv("DEMO_DURATION_S", "0")
    _snap = dict(os.environ)
    try:
        assert main() == 0
        out = capsys.readouterr().out
        assert "pping-lang[bench]" in out
    finally:
        os.environ.clear()
        os.environ.update(_snap)


def test_demo_holds_on_tty(monkeypatch, capsys):
    """交互式终端(isatty True)→ 结束横幅后挂住等 Enter;EOF/输入异常不炸。"""
    import builtins

    class _FakeStdin:
        @staticmethod
        def isatty() -> bool:
            return True

    def _fake_input(prompt: str = "") -> str:
        print(prompt)  # 真 input 会把 prompt 打到 stdout,替身保持同形
        raise EOFError

    monkeypatch.setattr("sys.stdin", _FakeStdin())
    monkeypatch.setattr(builtins, "input", _fake_input)
    monkeypatch.setenv("DEMO_DURATION_S", "0")
    monkeypatch.setenv("PPING_LANG_API_PORT", "18766")  # 避开前一个 demo 测试的 18765
    _snap = dict(os.environ)
    try:
        assert main() == 0
        assert "Dashboard live at" in capsys.readouterr().out
    finally:
        os.environ.clear()
        os.environ.update(_snap)


def test_demo_fires_double_low(monkeypatch, capsys):
    """3 秒真实跑:合成指标喂满规则 A(双低)的三条 check,断言 fire_count ≥ 1。

    锁住 demo 剧本与现役 DiagnosisEngine 的契约 —— 引擎/规则再演进时,demo
    fired 0 会在测试里炸而不是在客户终端上。"""
    import re

    monkeypatch.setenv("DEMO_DURATION_S", "3")
    _snap = dict(os.environ)
    try:
        assert main() == 0
        out = capsys.readouterr().out
        m = re.search(r"fired (\d+) times", out)
        assert m, f"输出里没有 fire 统计:\n{out[-400:]}"
        assert int(m.group(1)) >= 1, f"规则 A 未触发:\n{out[-400:]}"
    finally:
        os.environ.clear()
        os.environ.update(_snap)
