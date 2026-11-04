"""pping-lang bench — 两类子模块：

1. **microbench**：pping-lang 自身热路径基准（push_metric / collect / record overhead），
   验证不超 RFC §3 / design §8.3 预算。`python -m pping_lang.bench.microbench` 跑。

2. **load bench (静态 / 动态)**：对 vLLM endpoint 打流量、客户端测 TTFT/TPOT/e2e，
   跑完联动诊断引擎自动给建议。完整设计见 docs/bench-design-v0.1.md。
   v0.1 实现 StaticScenario，动态场景留 Week 2。
"""
from pping_lang.bench.measurement import (
    LatencyStats,
    RequestSample,
    RunSummary,
    aggregate,
    latency_stats,
)
from pping_lang.bench.microbench import (
    bench_collect_scheduler,
    bench_push_metric,
    bench_record_overhead,
    main,
)
from pping_lang.bench.prompts import (
    BUILTIN_DATASETS,
    available_builtins,
    load_prompts,
)
from pping_lang.bench.runner import SweepPoint, SweepResult, run_static, run_sweep
from pping_lang.bench.scenarios.schema import SLO, ApiKind, StaticScenario, Threshold

# OpenAIStreamClient / synthesize_prompt 走惰性导出:client.py 顶层 import httpx,
# 而 httpx 只在 extras([bench]/[dev])里 —— 急切导入会让纯 `pip install pping-lang`
# 的用户连 demo 都跑不起来(plugin → api.routes → bench.store 会被 __init__ 连坐)。
_LAZY = {"OpenAIStreamClient", "synthesize_prompt"}


def __getattr__(name: str):
    if name in _LAZY:
        from pping_lang.bench import client  # noqa: PLC0415

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # microbench (existing)
    "bench_push_metric",
    "bench_collect_scheduler",
    "bench_record_overhead",
    "main",
    # load bench — public surface
    "StaticScenario",
    "SLO",
    "Threshold",
    "ApiKind",
    "OpenAIStreamClient",
    "RequestSample",
    "LatencyStats",
    "RunSummary",
    "aggregate",
    "latency_stats",
    "synthesize_prompt",
    "load_prompts",
    "available_builtins",
    "BUILTIN_DATASETS",
    "run_static",
    "run_sweep",
    "SweepResult",
    "SweepPoint",
]
