"""builder.py(.so 现编)+ cli.py(pping-vllm 包装)单测。

实际 g++ 编译要 GPU/CUPTI 环境(由 runw 端到端覆盖);这里测不依赖 GPU 的逻辑：
缓存路径、显式 .so 路径、源缺失报错、cli 设 env + exec、cli 降级。
"""
from __future__ import annotations

import os

import pytest

from pping_lang import cli
from pping_lang.native import builder
from pping_lang.native.builder import BuildError, ensure_so


def test_cache_so_respects_home(monkeypatch, tmp_path):
    monkeypatch.setenv("PPING_LANG_HOME", str(tmp_path))
    p = builder._cache_so()
    assert p == tmp_path / "lib" / "libppingcupti.so"


def test_ensure_so_explicit_path_skips_build(monkeypatch, tmp_path):
    """PPING_LANG_PCS_SO 指向已存在文件 → 直接返回,不触发 _detect/编译。"""
    so = tmp_path / "libppingcupti.so"
    so.write_bytes(b"\x7fELF stub")
    monkeypatch.setenv("PPING_LANG_PCS_SO", str(so))
    # _detect 失败也无所谓(显式路径已命中);确保不抛
    monkeypatch.setattr(builder, "_detect", lambda: (_ for _ in ()).throw(BuildError("no cupti")))
    path, libdir = ensure_so()
    assert path == str(so)
    assert libdir == ""  # _detect 失败 → libdir 空,但仍返回


def test_ensure_so_raises_when_source_missing(monkeypatch, tmp_path):
    """无显式 .so 且打包源码缺失 → BuildError(不依赖 GPU)。"""
    monkeypatch.delenv("PPING_LANG_PCS_SO", raising=False)
    monkeypatch.setattr(builder, "_CPP", tmp_path / "nonexistent.cpp")
    with pytest.raises(BuildError, match="源码缺失"):
        ensure_so()


def test_glob1_returns_first_or_none():
    assert builder._glob1(["/definitely/nope/*.xyz"]) is None
    # 当前文件目录里一定有 .py
    here = os.path.dirname(builder.__file__)
    hit = builder._glob1([os.path.join(here, "*.py")])
    assert hit and hit.endswith(".py")


def test_detect_covers_conda_venv_layout(monkeypatch, tmp_path):
    """conda/venv 布局:nvidia/triton pip 包在当前解释器的 site-packages 下(而非 vllm
    官方镜像的 /usr/local/lib/python3*/dist-packages)。伪造该布局,_detect 必须能找到
    cupti 库/头与 triton 的 cuda.h;本机无驱动时,只允许最后一步(找 libcuda)失败。"""
    cupti = tmp_path / "nvidia" / "cuda_cupti"
    (cupti / "lib" / "libcupti.so.1").parent.mkdir(parents=True)
    (cupti / "lib" / "libcupti.so.1").write_bytes(b"")
    (cupti / "include" / "cupti_pcsampling.h").parent.mkdir(parents=True)
    (cupti / "include" / "cupti_pcsampling.h").write_bytes(b"")
    tri = tmp_path / "triton" / "backends" / "nvidia" / "include"
    (tri / "crt").mkdir(parents=True)
    (tri / "cuda.h").write_bytes(b"")
    (tri / "crt" / "host_defines.h").write_bytes(b"")
    monkeypatch.setattr(builder.sysconfig, "get_paths", lambda *a, **k: {"purelib": str(tmp_path)})
    try:
        det = builder._detect()
        assert det["cupti_libdir"] == str(cupti / "lib")   # 命中的是 conda 布局,不是镜像路径
    except BuildError as e:
        # 无 GPU 机器:探测应一路走到 libcuda 才断;更早失败 = conda 布局没被覆盖(regression)
        assert "libcuda" in str(e), f"failed before libcuda step: {e}"


def test_cli_sets_env_and_execs(monkeypatch, tmp_path):
    captured: dict = {}
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text("#!/bin/sh\n")
    monkeypatch.setattr("pping_lang.native.builder.ensure_so",
                        lambda *a, **k: ("/fake/libppingcupti.so", "/fake/cupti/lib"))
    monkeypatch.setattr(cli.shutil, "which", lambda _name: str(fake_vllm))
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: captured.update(path=path, argv=argv))
    monkeypatch.setattr(cli.sys, "argv", ["pping-vllm", "serve", "my-model", "--port", "8000"])
    # cli 用 os.environ.setdefault 真改 env;快照 + finally 还原,绝不泄漏给后续 test
    _keys = {"CUDA_INJECTION64_PATH", "PPING_LANG_PCS_SO", "PPING_LANG_ENABLE_PCS", "LD_LIBRARY_PATH"}
    _snap = dict(os.environ)
    for k in _keys:
        os.environ.pop(k, None)
    try:
        cli.pping_vllm_main()
        assert os.environ["CUDA_INJECTION64_PATH"] == "/fake/libppingcupti.so"
        assert os.environ["PPING_LANG_PCS_SO"] == "/fake/libppingcupti.so"
        assert os.environ["PPING_LANG_ENABLE_PCS"] == "1"
        assert "/fake/cupti/lib" in os.environ["LD_LIBRARY_PATH"]
        assert captured["path"] == str(fake_vllm)
        assert captured["argv"] == [str(fake_vllm), "serve", "my-model", "--port", "8000"]
    finally:
        os.environ.clear()
        os.environ.update(_snap)


def test_extract_opt():
    """--host VALUE / --host=VALUE / 缺失 / 末次优先。"""
    assert cli._extract_opt(["serve", "m", "--host", "0.0.0.0", "--port", "8000"], "--host") == "0.0.0.0"
    assert cli._extract_opt(["serve", "m", "--host=1.2.3.4"], "--host") == "1.2.3.4"
    assert cli._extract_opt(["serve", "m"], "--host") is None
    assert cli._extract_opt(["--host", "a", "--host", "b"], "--host") == "b"


def test_cli_dispatch_version(capsys):
    assert cli.main(["version"]) == 0
    assert "pping-lang" in capsys.readouterr().out


def test_cli_dispatch_help_and_unknown(capsys):
    assert cli.main([]) == 0
    assert "demo" in capsys.readouterr().out
    assert cli.main(["nope"]) == 2  # 未知子命令 → usage 进 stderr,退出码 2


def test_cli_dispatch_demo_routes(monkeypatch):
    called = {}
    monkeypatch.setattr("pping_lang.demo.main", lambda: called.update(ok=True) or 0)
    assert cli.main(["demo"]) == 0
    assert called.get("ok")


def test_cli_dispatch_serve_routes(monkeypatch):
    got: dict = {}
    monkeypatch.setattr(cli, "pping_vllm_main", lambda a: got.update(args=a))
    assert cli.main(["serve", "m", "--port", "8000"]) == 0
    assert got["args"] == ["m", "--port", "8000"]


def test_cli_dashboard_host_follows_vllm_host(monkeypatch, tmp_path):
    """`pping-vllm serve --host 0.0.0.0` → dashboard 也绑 0.0.0.0(容器里 docker 转发可达);
    无 --host 时不动(用插件默认 127.0.0.1);显式 PPING_LANG_API_HOST 可覆盖(setdefault)。"""
    captured: dict = {}
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text("#!/bin/sh\n")
    monkeypatch.setattr("pping_lang.native.builder.ensure_so",
                        lambda *a, **k: ("/fake/so", "/fake/lib"))
    monkeypatch.setattr(cli.shutil, "which", lambda _name: str(fake_vllm))
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: captured.update(argv=argv))
    monkeypatch.setattr(cli.sys, "argv",
                        ["pping-vllm", "serve", "m", "--host", "0.0.0.0", "--port", "8000"])
    _snap = dict(os.environ)
    os.environ.pop("PPING_LANG_API_HOST", None)
    try:
        cli.pping_vllm_main()
        assert os.environ["PPING_LANG_API_HOST"] == "0.0.0.0"
    finally:
        os.environ.clear()
        os.environ.update(_snap)


def test_cli_degrades_when_build_fails(monkeypatch, tmp_path):
    """ensure_so 抛错 → 不崩,仍 exec vllm,且不设注入 env(降级)。"""
    captured: dict = {}
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text("#!/bin/sh\n")

    def _boom(*_a, **_k):
        raise BuildError("g++ 没装")

    monkeypatch.setattr("pping_lang.native.builder.ensure_so", _boom)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: str(fake_vllm))
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: captured.update(path=path, argv=argv))
    monkeypatch.setattr(cli.sys, "argv", ["pping-vllm", "serve", "m"])
    _snap = dict(os.environ)
    os.environ.pop("CUDA_INJECTION64_PATH", None)
    try:
        cli.pping_vllm_main()
        assert captured["argv"] == [str(fake_vllm), "serve", "m"]
        assert "CUDA_INJECTION64_PATH" not in os.environ  # 降级：没设注入
    finally:
        os.environ.clear()
        os.environ.update(_snap)


def test_cli_errors_without_vllm(monkeypatch, tmp_path):
    monkeypatch.setattr("pping_lang.native.builder.ensure_so",
                        lambda *a, **k: ("/fake/so", "/fake/lib"))
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    # which 返 None 时回退到 dirname(sys.executable)/vllm;指到无 vllm 的临时目录
    monkeypatch.setattr(cli.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(cli.sys, "argv", ["pping-vllm", "serve", "m"])
    _snap = dict(os.environ)  # pping_vllm_main 在 raise 前已 setdefault 改 env,还原防泄漏
    try:
        with pytest.raises(SystemExit, match="找不到 vllm"):
            cli.pping_vllm_main()
    finally:
        os.environ.clear()
        os.environ.update(_snap)
