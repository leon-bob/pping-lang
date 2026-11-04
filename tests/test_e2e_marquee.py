"""端到端：合成 SchedulerStats 经完整 plugin 数据流 → 派生指标落 JSONL。

链路：合成 SchedulerStats → VllmStatsCollector(算派生比值)→ Sink → metrics.jsonl

不依赖 NVML / 真 vLLM / DuckDB。验证的是 plugin.record 全链路把 cudagraph/perf 的
**派生指标**(padding_ratio / mfu_ratio / mem_bw_util_ratio)算出来并持久化。

诊断触发的端到端(DiagnosisEngine 纯内存环 → sink.recent_diagnoses)见
test_diagnosis_runtime.py;诊断求值核逻辑见 test_diagnosis_engine.py。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from pping_lang.hardware import GPUPeak
from pping_lang.metrics_catalog import M
from pping_lang.plugin import PpingLangStatLogger
from pping_lang.sink.metric_log import JsonlStore


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    monkeypatch.setenv("PPING_LANG_DB_PATH", str(tmp_path / "e2e.duckdb"))
    monkeypatch.setenv("PPING_LANG_INSTANCE_ID", "e2e")
    monkeypatch.setenv("PPING_LANG_FLUSH_INTERVAL_S", "0.2")
    monkeypatch.setenv("PPING_LANG_DISABLE_NVML", "1")
    monkeypatch.setenv("PPING_LANG_DISABLE_API", "1")  # avoid uvicorn port collisions
    monkeypatch.setenv("PPING_LANG_DISABLE_RULES", "1")  # plumbing test — no diag thread
    monkeypatch.setenv("PPING_LANG_DIAGNOSIS_PRINT", "0")

    p = PpingLangStatLogger(vllm_config=None, engine_index=0)
    p.log_engine_initialized()
    # Inject demo GPU peak (no NVML, so plugin couldn't auto-detect)
    if p._collector is not None:
        p._collector._gpu_peak = GPUPeak(bf16_tflops=989.0, mem_bw_gbs=3350.0)
    yield p, tmp_path
    if p._diag_engine is not None:
        p._diag_engine.stop()
    if p._sink is not None and not p._sink._closed:
        p._sink.close()


def _bad_scheduler() -> SimpleNamespace:
    """高 padding (70%) + 低 MFU (~5%) 的合成 SchedulerStats。"""
    return SimpleNamespace(
        num_running_reqs=8,
        num_waiting_reqs=2,
        kv_cache_usage=0.45,
        prefix_cache_stats=SimpleNamespace(queries=100, hits=15),
        cudagraph_stats=SimpleNamespace(
            num_unpadded_tokens=300,
            num_padded_tokens=1000,  # 70% padding
            num_paddings=10,
            runtime_mode="PIECEWISE",
        ),
        perf_stats=SimpleNamespace(
            num_flops_per_gpu=int(4.9e12),         # ~5% MFU @ dt=0.1s, H100
            num_read_bytes_per_gpu=int(1675e8),    # 50% bw util
            num_write_bytes_per_gpu=0,
            debug_stats=None,
        ),
    )


def test_record_never_raises_into_vllm(plugin):
    """vLLM 版本漂移 → collect 抛 AttributeError,record 必须吞掉(vLLM 主循环不能被插件炸)。"""
    p, _ = plugin
    assert p._collector is not None

    def _boom(*_a, **_k):
        raise AttributeError("vLLM 0.x 改了 stats 字段名")

    p._collector.collect = _boom
    p.record(_bad_scheduler(), None)  # 不抛即通过


def test_derived_ratio_metrics_reach_jsonl(plugin):
    """派生指标 padding_ratio / mfu_ratio / mem_bw_util_ratio 都进了 metrics.jsonl。"""
    p, store_dir = plugin

    for _ in range(20):
        p.record(_bad_scheduler(), None)
        time.sleep(0.05)
    p._sink._drain()
    p._sink.close()

    store = JsonlStore(store_dir, "e2e")
    for name in (
        M.VLLM_CUDAGRAPH_PADDING_RATIO,
        M.VLLM_PERF_MFU_RATIO,
        M.VLLM_PERF_MEM_BW_UTIL_RATIO,
    ):
        pts = store.recent_metric_points(name, since_ns=0, limit=100)
        assert pts, f"{name} 应当落到 JSONL"


def test_padding_ratio_value_is_correct(plugin):
    """派生 padding_ratio 应 ≈ 0.70(num_padded=1000, unpadded=300 → 700/1000)。"""
    p, store_dir = plugin

    for _ in range(10):
        p.record(_bad_scheduler(), None)
        time.sleep(0.05)
    p._sink._drain()
    p._sink.close()

    store = JsonlStore(store_dir, "e2e")
    val = store.aggregate_metric(M.VLLM_CUDAGRAPH_PADDING_RATIO, 0, "max")
    assert val is not None
    assert abs(val - 0.70) < 0.05
