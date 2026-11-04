"""端到端 demo — 不需要真 vLLM/GPU，喂合成 stats 触发现役诊断引擎的规则。

跑法（wheel 自带，pip 装上即用）：
    pping-demo
    # 或: python -m pping_lang.demo
    # 时长: DEMO_DURATION_S=600 pping-demo

剧本：低 MFU（~5%）+ HBM 空闲（mem_util 8%）+ 持续在途请求 → 触发规则 A
「双低（算力、带宽均有余量）」。dashboard 起在 demo 专用端口 18765（不与生产
dashboard 的 8765 冲突）。cudagraph/perf 派生指标照常喂 dashboard 各 tab。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

DEMO_DB = Path(tempfile.gettempdir()) / "pping-lang-demo.duckdb"

try:  # bench 历史(duckdb)属 [bench] extra;缺了 demo 其余照跑,只是压测 tab 无种子
    import duckdb as _duckdb  # noqa: F401
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False


def make_unhealthy_scheduler_stats(step: int) -> SimpleNamespace:
    """合成 SchedulerStats: 高 padding (~70%) + 低 MFU (~5%)。"""
    # cudagraph: 1000 padded tokens, 300 unpadded → 70% padding ratio
    cudagraph = SimpleNamespace(
        num_unpadded_tokens=300,
        num_padded_tokens=1000,
        num_paddings=10,
        runtime_mode="PIECEWISE",
    )
    # perf: ~5% MFU 假设 step 间隔 ~100ms
    #   target: flops / (peak_flops * dt) ≈ 0.05
    #   peak_flops = 989e12, dt = 0.1s → flops_per_step = 0.05 * 989e12 * 0.1 ≈ 4.9e12
    perf = SimpleNamespace(
        num_flops_per_gpu=int(4.9e12),
        num_read_bytes_per_gpu=int(0.5 * 3350e9 * 0.1),  # 50% bw util
        num_write_bytes_per_gpu=0,
        debug_stats=None,
    )
    return SimpleNamespace(
        num_running_reqs=8,
        num_waiting_reqs=2,
        num_skipped_waiting_reqs=0,
        current_wave=step // 10,
        kv_cache_usage=0.45,
        prefix_cache_stats=SimpleNamespace(queries=100, hits=15),
        kv_cache_eviction_events=[],
        spec_decoding_stats=None,
        cudagraph_stats=cudagraph,
        perf_stats=perf,
        waiting_lora_adapters={},
        running_lora_adapters={},
    )


def make_iteration_stats(step: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_generation_tokens=50 + step % 10,
        prompt_token_stats=SimpleNamespace(
            total=200, local_cache_hit=30, external_kv_transfer=0,
        ),
        num_preempted_reqs=0,
        num_corrupted_reqs=0,
        time_to_first_tokens_iter=[0.15, 0.18],
        inter_token_latencies_iter=[0.025, 0.026],
        finished_requests=[],
    )


def _seed_demo_bench_runs(db_path: Path) -> None:
    """Insert 3 synthetic bench_runs so the dashboard 压测 tab isn't empty.

    Demo has no real vLLM to hit; this just shows what a populated history
    looks like. Real bench writes go through the 压测 tab → POST /api/bench/start.
    """
    import time as _t

    import duckdb as _ddb

    from pping_lang.bench import store as _bs
    from pping_lang.bench.measurement import LatencyStats, RunSummary
    from pping_lang.bench.scenarios.schema import SLO, StaticScenario

    _baseline_slo = SLO.from_spec("ttft:p99<500ms;tpot:p99<50ms")

    conn = _ddb.connect(str(db_path))
    try:
        _bs.init_bench_table(conn)
        now_ns = _t.monotonic_ns()
        seeds = [
            dict(
                offset_min=45, name="baseline-128-100",
                concurrency=16, duration=60,
                summary=RunSummary(
                    total=247, ok=247, errors=0, duration_s=60.2,
                    ttft_ms=LatencyStats(n=247, p50=142, p90=305, p95=350, p99=387, mean=178),
                    tpot_ms=LatencyStats(n=247, p50=28, p99=41, mean=30),
                    e2e_ms=LatencyStats(n=247, p50=1850, p99=4220, mean=2100),
                    output_tokens_total=24700, input_tokens_total=128000,
                    output_throughput_tps=410.4, input_throughput_tps=2127.0,
                ),
                slo=_baseline_slo,
                status="pass",
            ),
            dict(
                offset_min=22, name="stress-concurrency-64",
                concurrency=64, duration=60,
                summary=RunSummary(
                    total=1024, ok=1019, errors=5, duration_s=60.5,
                    ttft_ms=LatencyStats(n=1019, p50=384, p90=820, p95=940, p99=1180, mean=435),
                    tpot_ms=LatencyStats(n=1019, p50=42, p99=83, mean=49),
                    e2e_ms=LatencyStats(n=1019, p50=4500, p99=8700, mean=5100),
                    output_tokens_total=101900, input_tokens_total=512000,
                    output_throughput_tps=1684.3, input_throughput_tps=8460.8,
                    error_samples=["http_503: kv cache exhausted"],
                ),
                slo=_baseline_slo,
                status="fail",
            ),
            dict(
                offset_min=5, name="quick-smoke-8",
                concurrency=8, duration=30,
                summary=RunSummary(
                    total=124, ok=124, errors=0, duration_s=30.1,
                    ttft_ms=LatencyStats(n=124, p50=98, p90=180, p95=210, p99=265, mean=115),
                    tpot_ms=LatencyStats(n=124, p99=32, p50=22, mean=24),
                    e2e_ms=LatencyStats(n=124, p50=1480, p99=3200, mean=1620),
                    output_tokens_total=12400, input_tokens_total=64000,
                    output_throughput_tps=411.9, input_throughput_tps=2126.2,
                ),
                slo=None,
                status="n/a",
            ),
        ]

        for i, s in enumerate(seeds, start=1):
            scenario = StaticScenario(
                name=s["name"],
                endpoint="http://localhost:8000",
                model="Qwen/Qwen2.5-32B-Instruct",
                prompt_tokens=500, output_tokens=100,
                concurrency=s["concurrency"],
                duration_s=s["duration"], num_requests=None,
                warmup_s=5, timeout_s=30.0,
                slo=s["slo"],
            )
            started_at = now_ns - int(s["offset_min"] * 60 * 1e9)
            finished_at = started_at + int(s["summary"].duration_s * 1e9)
            run_id = f"static-demo-{i:03d}"
            try:
                _bs.insert_running(conn, run_id, scenario, "static", started_at)
                _bs.mark_done(conn, run_id, finished_at, s["summary"], slo_status=s["status"])
            except _ddb.ConstraintException:
                # Already seeded on a previous demo run (DB persists)
                pass
    finally:
        conn.close()


def main() -> int:
    # Demo 配置：缩短 flush / eval 间隔便于即时看到效果。
    # 放 main() 里(env 副作用不外溢),且必须在 PpingLangStatLogger 初始化前设好。
    if DEMO_DB.exists():
        DEMO_DB.unlink()
    os.environ["PPING_LANG_DB_PATH"] = str(DEMO_DB)
    os.environ["PPING_LANG_INSTANCE_ID"] = "demo"
    os.environ["PPING_LANG_FLUSH_INTERVAL_S"] = "1.0"      # default 5s → 1s
    os.environ["PPING_LANG_RULE_EVAL_INTERVAL_S"] = "1.0"
    os.environ["PPING_LANG_DISABLE_NVML"] = "1"            # demo 跑在无 GPU 机器上
    # demo 专用端口:生产 dashboard 常占 8765,同机跑 demo 别抢端口
    os.environ.setdefault("PPING_LANG_API_PORT", "18765")

    # Demo: populate /api/system so the dashboard hero card has something to show
    os.environ.setdefault("PPING_LANG_INFO_VLLM_VERSION", "0.20.2")
    os.environ.setdefault("PPING_LANG_INFO_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    os.environ.setdefault("PPING_LANG_INFO_GPU", "NVIDIA H100 80GB HBM3")
    os.environ.setdefault("PPING_LANG_INFO_GPU_COUNT", "8")
    os.environ.setdefault("PPING_LANG_INFO_BF16_TFLOPS", "989")
    os.environ.setdefault("PPING_LANG_INFO_MEM_BW_GBS", "3350")

    from pping_lang.hardware import GPUPeak
    from pping_lang.plugin import PpingLangStatLogger

    duration_s = int(os.environ.get("DEMO_DURATION_S", "12"))

    # H100 SXM peak — 用于 MFU 派生
    demo_gpu_peak = GPUPeak(bf16_tflops=989.0, mem_bw_gbs=3350.0)

    print("=" * 70)
    print("pping-lang demo — synthetic vLLM stats: low MFU + idle HBM + live requests")
    print("=" * 70)
    print(f"Data dir: {DEMO_DB.parent}")
    if HAVE_DUCKDB:
        print(f"  metrics.jsonl(指标流) / diagnoses.jsonl(诊断) / {DEMO_DB.name}(bench 历史)")
    else:
        print("  metrics.jsonl(指标流) / diagnoses.jsonl(诊断) "
              "(bench 历史需 'pping-lang[bench]',未装,已跳过)")
    print(f"Duration: {duration_s}s, step every 100ms")
    print("Expecting rule A 「双低」 to fire within a few seconds...\n")

    plugin = PpingLangStatLogger(vllm_config=None, engine_index=0)
    plugin.log_engine_initialized()

    # Inject demo GPU peak for MFU derivation (since NVML disabled)
    if plugin._collector is not None:
        plugin._collector._gpu_peak = demo_gpu_peak

    # Pre-populate a sample bench run so the 压测 tab has something to show
    # (real bench needs an actual vLLM endpoint to hit; demo just shows the UI).
    if HAVE_DUCKDB:
        _seed_demo_bench_runs(DEMO_DB)
    else:
        print("(跳过 bench 历史种子:未装 duckdb —— pip install 'pping-lang[bench]' 可启用)\n")

    start = time.monotonic()
    step = 0
    try:
        while time.monotonic() - start < duration_s:
            plugin.record(
                make_unhealthy_scheduler_stats(step),
                make_iteration_stats(step),
            )
            # 合成 NVML 侧信号:demo 禁了真 NVML,而 A(双低)的 roofline 手段第三条
            # check 要 gpu.mem_util_pct(HBM 控制器空闲%)—— 没这个点 demo 永远 fired 0。
            # 8% = 带宽大量余量;ts 用墙钟(recent() 按墙钟窗口过滤)。
            if plugin._sink is not None:
                from pping_lang.metrics_catalog import M
                from pping_lang.types import MetricPoint

                plugin._sink.push_metric(MetricPoint(
                    ts_ns=time.time_ns(), name=M.GPU_MEM_UTIL_PCT, value=8.0))
            time.sleep(0.1)
            step += 1
    finally:
        # Force-flush any pending data, then stop bg threads
        if plugin._sink is not None:
            plugin._sink._drain()
        if plugin._diag_engine is not None:
            plugin._diag_engine.stop()
        if plugin._sink is not None:
            plugin._sink.close()

    print(f"\n{'=' * 70}")
    print(f"Demo done. {step} synthetic steps pushed in {duration_s}s.")
    if plugin._diag_engine is not None:
        print(f"Rule eval ran {plugin._diag_engine.eval_count} times, "
              f"fired {plugin._diag_engine.fire_count} times.")
    print(f"Diagnoses: {DEMO_DB.parent / 'diagnoses.jsonl'}")
    if HAVE_DUCKDB:
        print(f"Bench:     duckdb {DEMO_DB} -c 'SELECT name, status FROM bench_runs;'")
    print("=" * 70)
    # 交互式终端:挂住进程,浏览器慢慢看(dashboard 是进程内服务,退出即关);
    # CI/非 TTY(管道、pytest)自动跳过,不阻塞。想加长数据窗口仍用 DEMO_DURATION_S。
    if sys.stdin is not None and sys.stdin.isatty():
        host = os.environ.get("PPING_LANG_API_HOST", "127.0.0.1")
        port = os.environ.get("PPING_LANG_API_PORT", "18765")
        try:
            input(f"\nDashboard live at http://{host}:{port} — 按 Enter 退出...")
        except EOFError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
