"""FastAPI app + 核心 API 端点 (Day 6/8)。

依赖通过闭包注入（避免 FastAPI Depends 的样板）。

Day 6: GET 端点（health, metrics, diagnoses, rules, instances）+ /  dashboard
Day 8: POST/PUT/DELETE/test 端点 — 规则 CRUD via RuleStore
Day 9: 规则热加载（store 改动让 DiagnosisEngine 下一轮即看到）

读路径：实时短窗走 Sink 内存环;长窗/历史走 JsonlStore(扫 metrics.jsonl);
bench_runs(可变行)仍用 DuckDB。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pping_lang.api.queries import open_conn  # bench_runs only (mutable rows → DuckDB)
from pping_lang.api.schemas import BenchStartIn, RuleIn, RuleTestRequest
from pping_lang.bench import store as bench_store

# 注意:不要在模块顶层 import pping_lang.bench.client —— 它顶层要 httpx(仅 extras
# 提供),会把无 extras 安装的 dashboard/demo 一起拖死。用点内惰性导入。
from pping_lang.bench.runner import run_static
from pping_lang.bench.scenarios.schema import SLO, StaticScenario
from pping_lang.clock import wall_ns
from pping_lang.hardware import GPUPeak
from pping_lang.metrics_catalog import ALLOWED_METRICS, M
from pping_lang.rules.diagnosis_config import (
    WORKLOAD_FORMS,
)
from pping_lang.rules.diagnosis_config import (
    from_dict as diag_config_from_dict,
)
from pping_lang.rules.diagnosis_config import (
    to_dict as diag_config_to_dict,
)
from pping_lang.rules.diagnosis_rules import DIAGNOSIS_RULES
from pping_lang.rules.engine import _OP_TO_FN
from pping_lang.rules.schema import Condition, Rule
from pping_lang.sink.metric_log import JsonlStore

_UI_DIR = Path(__file__).parent.parent / "ui"
_UI_INDEX = _UI_DIR / "index.html"


def _read_ui(name: str, fallback: str = "") -> str:
    try:
        return (_UI_DIR / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("[pping-lang] UI asset not found: %s", name)
        return fallback


def _percentile(values: list[float], q: float) -> float | None:
    """Sorted-list linear-interpolation percentile. q in [0, 1]. None if empty."""
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return values[0]
    s = sorted(values)
    rank = q * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def _extract_arch(vllm_config: Any) -> dict[str, Any] | None:
    """Pull transformer arch knobs out of vllm_config.model_config.hf_config.

    Returns the fields we need to estimate parameter count + per-step
    FLOPS/bytes when vllm doesn't emit perf_stats (i.e. <0.20). None if any
    required field is missing — caller falls back to "no data" UI.
    """
    if vllm_config is None:
        return None
    mc = getattr(vllm_config, "model_config", None)
    if mc is None:
        return None
    hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
    if hf is None:
        return None
    needed = ("hidden_size", "num_hidden_layers", "intermediate_size",
              "num_attention_heads", "vocab_size")
    out: dict[str, Any] = {}
    for k in needed:
        v = getattr(hf, k, None)
        if v is None:
            return None
        out[k] = int(v)
    out["num_key_value_heads"] = int(getattr(hf, "num_key_value_heads", None)
                                     or out["num_attention_heads"])
    out["tie_word_embeddings"] = bool(getattr(hf, "tie_word_embeddings", False))
    out["torch_dtype"] = str(getattr(hf, "torch_dtype", "bfloat16"))
    return out


def _estimate_params(arch: dict[str, Any]) -> int:
    """Sum of weights for a standard decoder-only transformer (Llama/Qwen family).

    Formula per layer:
      attention = 2·h² + 2·h·Hkv·head_dim   (Q, O + K, V under GQA)
      mlp       = 3·h·i                      (gate + up + down, SwiGLU)
      norms     = 2·h
    Plus embedding (tied = counted once) and final norm. Off by a few %
    against the model card's "0.5B / 7B / 70B" round numbers — enough for
    roofline scale.
    """
    h   = arch["hidden_size"]
    L   = arch["num_hidden_layers"]
    i   = arch["intermediate_size"]
    H   = arch["num_attention_heads"]
    Hkv = arch["num_key_value_heads"]
    V   = arch["vocab_size"]
    head_dim = h // H
    per_layer = 2 * h * h + 2 * h * Hkv * head_dim + 3 * h * i + 2 * h
    total = L * per_layer + V * h + h
    if not arch["tie_word_embeddings"]:
        total += V * h
    return total


# fp16/bf16 = 2 bytes; fp32 = 4. vLLM serves fp16/bf16 by default.
_DTYPE_BYTES: dict[str, int] = {
    "float16": 2, "bfloat16": 2, "torch.float16": 2, "torch.bfloat16": 2,
    "float32": 4, "torch.float32": 4,
}


def _summary(values: list[float]) -> dict[str, float | int | None]:
    """{p50, p95, p99, avg, n} — one pass for the dashboard latency cards.

    Single-number summaries lie about skewed distributions. Reporting p50/p95/p99
    together with avg and n lets the reader judge typical experience, tail,
    central tendency, and sample-size confidence at once.
    """
    n = len(values)
    if n == 0:
        return {"p50": None, "p95": None, "p99": None, "avg": None, "n": 0}
    s = sorted(values)
    return {
        "p50": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "avg": sum(s) / n,
        "n":   n,
    }

def _kernel_findings(
    *,
    class_shares: list[dict[str, Any]],
    top_kernels: list[dict[str, Any]],
    sync_share: float | None,
    memcpy_share: float | None,
    in_graph: float | None,
    launch_rate: float | None,
    overhead_cb_ms: float | None,
    window_s: float | None,
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """把 kernel 指标翻译成人话结论 —— pping-lang 的命根子：给结论不给数字。

    全部从阶段 1a 已采集的量派生(无需 PC Sampling)。每条 {level, claim, title, detail}。
    lang 选 "zh"/"en";默认 zh 保持既有行为不变。
    """
    en = lang == "en"
    findings: list[dict[str, Any]] = []
    _label = (
        {"gemm": "GEMM", "attention": "Attention", "comm": "Comm (NCCL)",
         "norm": "Norm", "activation": "Activation", "rotary": "Rotary", "other": "Other"}
        if en else
        {"gemm": "GEMM(矩阵乘)", "attention": "Attention", "comm": "通信(NCCL)",
         "norm": "Norm", "activation": "Activation", "rotary": "Rotary", "other": "其它"}
    )

    # 1. 哪类 kernel 主导 → X-bound
    if class_shares and class_shares[0]["pct"] >= 60:
        c = class_shares[0]
        lbl = _label.get(c["cls"], c["cls"])
        findings.append({
            "level": "info", "claim": "derived", "title": f"{lbl}-bound",
            "detail": (
                f"{c['pct']:.0f}% of GPU compute time is concentrated in {lbl} — "
                f"a classic compute-bound shape. To speed up: grow the batch to amortize, "
                f"check tiling, or quantize."
                if en else
                f"GPU 计算时间 {c['pct']:.0f}% 集中在 {lbl}，"
                f"典型 compute-bound 形态。提速方向：增大 batch 摊薄、检查 tiling、或量化。"
            ),
        })
    # 2. 单个 kernel 主导 → 优化它收益最大
    if top_kernels and top_kernels[0]["pct"] >= 40:
        k = top_kernels[0]
        findings.append({
            "level": "info", "claim": "derived",
            "title": "Single kernel dominates" if en else "单 kernel 主导",
            "detail": (
                f"A single kernel takes {k['pct']:.0f}% of GPU time ({k['name'][:46]}); "
                f"optimizing or replacing it yields the biggest win."
                if en else
                f"单个 kernel 占 GPU 时间 {k['pct']:.0f}%（{k['name'][:46]}），"
                f"优化或替换它收益最大。"
            ),
        })
    # 3. 同步等待高 → launch-bound
    if sync_share is not None and sync_share >= 15:
        findings.append({
            "level": "warning", "claim": "hypothesis",
            "title": "Launch-bound (high sync wait)" if en else "Launch-bound（同步等待高）",
            "detail": (
                f"Sync wait is {sync_share:.0f}% of wall-clock — the GPU is waiting on the CPU "
                f"to dispatch kernels. A larger batch / higher CUDA Graph coverage can ease this."
                if en else
                f"同步等待占墙钟 {sync_share:.0f}%，GPU 在等 CPU 派发 kernel。"
                f"增大 batch / 提高 CUDA graph 覆盖可缓解。"
            ),
        })
    # 4. GEMM 碎片化(kernel zoo)—— 对标 zymtrace 那张图的观察
    gemm_variants = [k for k in top_kernels if k.get("cls") == "gemm"]
    if len(gemm_variants) >= 8:
        findings.append({
            "level": "info", "claim": "hypothesis",
            "title": "GEMM kernel fragmentation" if en else "GEMM kernel 碎片化",
            "detail": (
                f"{len(gemm_variants)} distinct GEMM variants appear (a kernel zoo); "
                f"tiling may be suboptimal — the same matmul is split into many shapes."
                if en else
                f"出现 {len(gemm_variants)} 种不同 GEMM 变体(kernel zoo)，"
                f"tiling 可能次优 —— 同一矩阵乘被拆成多种形状。"
            ),
        })
    # 5. 内存拷贝偏多
    if memcpy_share is not None and memcpy_share >= 10:
        findings.append({
            "level": "warning", "claim": "derived",
            "title": "memcpy overhead" if en else "内存拷贝开销",
            "detail": (
                f"memcpy is {memcpy_share:.0f}% of wall-clock — a lot of data movement (H2D/D2H)."
                if en else
                f"memcpy 占墙钟 {memcpy_share:.0f}%，数据搬运偏多(H2D/D2H)。"
            ),
        })
    # 6. CUDA Graph 覆盖低 + launch 频繁 → launch 开销
    if in_graph is not None and in_graph < 40 and launch_rate and launch_rate > 5000:
        findings.append({
            "level": "info", "claim": "hypothesis",
            "title": "Low CUDA Graph coverage" if en else "CUDA Graph 覆盖低",
            "detail": (
                f"Only {in_graph:.0f}% of kernels run inside a graph, at {launch_rate:.0f} launches/s; "
                f"launch overhead may be high."
                if en else
                f"仅 {in_graph:.0f}% kernel 在 graph 内、每秒 {launch_rate:.0f} 次 launch，"
                f"launch 开销可能偏高。"
            ),
        })
    # 7. 吃自己狗粮：采集器自身开销
    if overhead_cb_ms is not None and window_s and window_s > 0:
        ov_pct = 100.0 * (overhead_cb_ms / 1000.0) / window_s
        if ov_pct >= 3:
            findings.append({
                "level": "warning", "claim": "measurement",
                "title": "Collector overhead high" if en else "采集开销偏高",
                "detail": (
                    f"This collector costs about {ov_pct:.1f}% (in-process Python, gets pricier at "
                    f"high kernel rates). Consider downsampling; phase 1b injection can eliminate it."
                    if en else
                    f"本采集器开销约占 {ov_pct:.1f}%（进程内 Python，高 kernel 率下变贵)。"
                    f"建议降采样;阶段 1b 注入式可消除。"
                ),
            })
    return findings


_STALL_LABEL: dict[str, str] = {
    "memory_dependency": "访存依赖", "shared_dependency": "shared/MIO 依赖",
    "memory_throttle": "访存子系统压力", "math_pipe": "计算管线",
    "exec_dependency": "执行依赖", "sync": "同步", "fetch_control": "取指/控制流",
    "dispatch": "调度分发", "scheduler_slack": "调度余量", "other": "其它",
}
_STALL_ADVICE: dict[str, str] = {
    "memory_dependency": "kernel 大量时间在等全局/本地内存返回。结合 batch/shape 判断是延迟还是带宽"
                         " —— 增大 batch 摊薄、改 tiling、提高 L2 命中。",
    "math_pipe": "数学/Tensor 管线接近饱和(compute-bound)。可量化/降精度;Tensor Core kernel 别一概归为算力不足。",
    "shared_dependency": "shared memory 依赖 —— 查 tiling、bank conflict、MMA pipeline。",
    "memory_throttle": "内存子系统资源入口被塞(MIO/LG/常量缓存),区别于单纯依赖等待。",
    "exec_dependency": "指令级执行依赖链长 —— 看能否打散依赖、提高 ILP。",
    "sync": "同步/屏障开销偏高 —— 查 tile/block 同步粒度。",
    "fetch_control": "取指/控制流开销 —— fused mega-kernel 或动态分支较多。",
}
_STALL_LABEL_EN: dict[str, str] = {
    "memory_dependency": "memory dependency", "shared_dependency": "shared/MIO dependency",
    "memory_throttle": "memory-subsystem pressure", "math_pipe": "math pipe",
    "exec_dependency": "execution dependency", "sync": "sync", "fetch_control": "fetch/control-flow",
    "dispatch": "dispatch", "scheduler_slack": "scheduler slack", "other": "other",
}
_STALL_ADVICE_EN: dict[str, str] = {
    "memory_dependency": "The kernel spends much time waiting on global/local memory returns. Use batch/shape"
                         " to tell latency from bandwidth — grow the batch to amortize, change tiling, raise L2 hits.",
    "math_pipe": "The math/Tensor pipe is near saturation (compute-bound). Quantize / lower precision; don't lump Tensor-Core kernels in as compute-starved.",
    "shared_dependency": "Shared-memory dependency — check tiling, bank conflicts, MMA pipeline.",
    "memory_throttle": "Memory-subsystem resource intake is throttled (MIO/LG/constant cache), distinct from plain dependency waits.",
    "exec_dependency": "Long instruction-level execution dependency chains — break dependencies and raise ILP.",
    "sync": "Sync/barrier overhead is high — check tile/block sync granularity.",
    "fetch_control": "Fetch/control-flow overhead — fused mega-kernels or many dynamic branches.",
}


def _stall_findings(result: dict[str, Any] | None, lang: str = "zh") -> list[dict[str, Any]]:
    """把 PC Sampling stall 分解翻成人话结论(Deep Evidence 的"给结论"层)。

    口径遵循设计文档 §11(Codex 评审)：不过度归因(访存依赖不单独断言带宽 vs 延迟);
    scheduler_slack 高是好事(非 latency-starved);真瓶颈排除 slack/other。
    lang 选 "zh"/"en";默认 zh 保持既有行为不变。
    """
    en = lang == "en"
    LBL = _STALL_LABEL_EN if en else _STALL_LABEL
    ADV = _STALL_ADVICE_EN if en else _STALL_ADVICE
    findings: list[dict[str, Any]] = []
    if not result or not result.get("available"):
        return findings
    shares: list[dict[str, Any]] = result.get("stall_shares") or []
    # 1. 真瓶颈主导类(排除调度余量 / 其它)
    real = [s for s in shares if s["cls"] not in ("scheduler_slack", "other")]
    if real and real[0]["pct"] >= 35:
        c = real[0]
        lbl = LBL.get(c["cls"], c["cls"])
        findings.append({
            "level": "info", "claim": "derived",
            "title": f"{lbl}-dominated stall" if en else f"{lbl}主导 stall",
            "detail": (f"{c['pct']:.0f}% of stall samples are in {lbl}. " + ADV.get(c["cls"], ""))
                      if en else
                      (f"stall 样本 {c['pct']:.0f}% 集中在{lbl}。" + ADV.get(c["cls"], "")),
        })
    # 2. scheduler_slack 高 → 非 latency-starved(好事,提示瓶颈在别处)
    slack = next((s for s in shares if s["cls"] == "scheduler_slack"), None)
    if slack and slack["pct"] >= 40:
        findings.append({
            "level": "info", "claim": "hypothesis",
            "title": "Plenty of scheduler slack (not latency-starved)" if en else "调度余量充足(非 latency-starved)",
            "detail": (f"{slack['pct']:.0f}% of stalls are not-selected (warps ready but the scheduler picked "
                       f"others) — eligible warps are plentiful, so latency isn't the current bottleneck; "
                       f"look at the dominant stall class for the real one.")
                      if en else
                      (f"{slack['pct']:.0f}% 的 stall 是 not-selected(warp 就绪但调度器选了别的)——"
                       f" eligible warp 充足,当前不是延迟瓶颈,真正瓶颈看主导 stall 类。"),
        })
    # 3. 主导 kernel 的 stall
    table: list[dict[str, Any]] = result.get("kernel_table") or []
    if table and table[0].get("dominant_stall"):
        k = table[0]
        dom = LBL.get(k["dominant_stall"], k["dominant_stall"])
        findings.append({
            "level": "info", "claim": "derived",
            "title": "Top kernel's stall" if en else "主导 kernel 的 stall",
            "detail": (f"The most-sampled kernel ({k['kernel'][:46]}) is mainly stalled on {dom} "
                       f"({k['dominant_pct']:.0f}%).")
                      if en else
                      (f"采样最多的 kernel（{k['kernel'][:46]}）主要卡在 {dom}"
                       f"（{k['dominant_pct']:.0f}%）。"),
        })
    # 4. 诚实：取证采样有丢样就标注
    ov: dict[str, Any] = result.get("overhead") or {}
    if (ov.get("hwfull") or 0) > 0 or (ov.get("dropped") or 0) > 0:
        findings.append({
            "level": "warning", "claim": "measurement",
            "title": "Sampling had drops" if en else "取证采样有丢样",
            "detail": (f"HW buffer full {ov.get('hwfull', 0)}×, dropped {ov.get('dropped', 0)} — the sampling "
                       f"period can be lengthened; this distribution is still indicative but not exhaustive.")
                      if en else
                      (f"HW 缓冲满 {ov.get('hwfull', 0)} 次、丢样 {ov.get('dropped', 0)} ——"
                       f" 采样周期可调长;本次分布仍可参考但非全量。"),
        })
    return findings


_BUILTIN_DESCRIPTIONS: dict[str, str] = {
    "mixed-short": "短问答 + 闲聊 + 简单指令（每条 50–180 tokens）",
    "mixed-long":  "长上下文：文档摘要 / 长指令 / 转录稿（每条 1000–3000 tokens）",
    "code":        "代码相关：写函数 / 解释代码 / 修 bug / 重构",
}
_BUILTIN_DESCRIPTIONS_EN: dict[str, str] = {
    "mixed-short": "Short Q&A + chit-chat + simple instructions (50–180 tokens each)",
    "mixed-long":  "Long context: doc summaries / long instructions / transcripts (1000–3000 tokens each)",
    "code":        "Code-related: write functions / explain code / fix bugs / refactor",
}

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pping_lang.collector.cupti import CuptiKernelCollector
    from pping_lang.collector.nvml import NvmlSampler
    from pping_lang.rules.store import RuleStore
    from pping_lang.sink.base import Sink


def build_app(
    *,
    db_path: str,
    instance_id: str,
    engine_index: int,
    sink: Sink,
    rule_store: RuleStore,
    diag_engine: Any = None,
    nvml: NvmlSampler | None = None,
    cupti: CuptiKernelCollector | None = None,
    version: str = "0.0.1.dev0",
    vllm_config: Any = None,
    gpu_peak: GPUPeak | None = None,
    gpu_name: str | None = None,
    cmdline: list[str] | None = None,
    env_snapshot: dict[str, str] | None = None,
) -> FastAPI:
    """Construct the FastAPI app with deps wired via closure."""
    app = FastAPI(
        title="pping-lang",
        version=version,
        description="vLLM 性能诊断插件 — HTTP API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 拆出 vendor 资源：CSS/JS 单独文件,index.html 只剩 markup(读进闭包,启动时一次)
    ui_html = _read_ui("index.html", "<h1>pping-lang UI missing</h1>")
    ui_css = _read_ui("dashboard.css")
    ui_js = _read_ui("dashboard.js")
    # vendor 资源本地化：Alpine/Chart 不再走 CDN,离线/air-gapped GPU 机也能渲染
    ui_vendor_alpine = _read_ui("vendor/alpine.min.js")
    ui_vendor_chart = _read_ui("vendor/chart.umd.min.js")

    # 冷/长窗读端：扫 LocalSink 落的 JSONL(与 sink 共享同目录文件)。实时短窗走内存环。
    metric_store = JsonlStore(Path(db_path).parent, instance_id)

    # === GET / — dashboard ===
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # index.html 同样 no-cache：无缓存头时浏览器走启发式缓存,普通 F5 可能拿旧版
        # (实测踩过：布局改了用户刷新看不到)。js/css 的同款处理见下。
        return HTMLResponse(ui_html, headers={"Cache-Control": "no-cache, must-revalidate"})

    # 自研 JS/CSS 每次部署都会变 —— no-cache 让浏览器每次校验(没变 304,很便宜),
    # 避免普通 F5 复用旧 dashboard.js 导致新功能(如行级归因)静默不渲染。
    _NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}

    @app.get("/dashboard.css")
    def dashboard_css() -> Response:
        return Response(ui_css, media_type="text/css; charset=utf-8", headers=_NOCACHE)

    @app.get("/dashboard.js")
    def dashboard_js() -> Response:
        return Response(ui_js, media_type="application/javascript; charset=utf-8", headers=_NOCACHE)

    @app.get("/vendor/alpine.min.js")
    def vendor_alpine() -> Response:
        return Response(ui_vendor_alpine, media_type="application/javascript; charset=utf-8")

    @app.get("/vendor/chart.umd.min.js")
    def vendor_chart() -> Response:
        return Response(ui_vendor_chart, media_type="application/javascript; charset=utf-8")

    # === GET /api/health ===
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": version,
            "instance_id": instance_id,
            "engine_index": engine_index,
            "sink": {
                "dropped_metrics": sink.dropped_metrics,
                "downsampled_metrics": getattr(sink, "downsampled_metrics", 0),
                "dropped_diags": sink.dropped_diags,
                "flush_errors": sink.flush_errors,
                "queue_depth": sink.queue_depth,
            },
            "nvml": {
                "enabled": nvml.enabled if nvml else False,
                "num_gpus": nvml.num_gpus if nvml else 0,
            },
            "rules": {
                "num": len(rule_store.list()),
                "eval_count": getattr(diag_engine, "eval_count", 0) if diag_engine else 0,
                "fire_count": getattr(diag_engine, "fire_count", 0) if diag_engine else 0,
            },
        }

    # === GET /api/system — environment / model / GPU info for dashboard hero ===
    @app.get("/api/system")
    def system() -> dict[str, Any]:
        # vLLM version: real install first, then env override (demo)
        vllm_ver: str | None = None
        try:
            import vllm as _v
            vllm_ver = getattr(_v, "__version__", None)
        except Exception:
            vllm_ver = None
        if not vllm_ver:
            vllm_ver = os.environ.get("PPING_LANG_INFO_VLLM_VERSION") or None

        # Two distinct identities from vllm_config.model_config:
        #   `model`               → what `vllm serve` was pointed at (often a disk path
        #                            from a local cache like ~/.cache/modelscope/...)
        #   `served_model_name`   → the name clients use in OpenAI-protocol calls
        #                            (--served-model-name). May be a list of aliases.
        # Dashboard prefers served_model_name for display + bench form prefill —
        # it's the user-facing identity that matches what the OpenAI request body
        # says. The disk path is still surfaced via cmdline in the startup modal.
        model: str | None = None
        served_model_name: str | None = None
        if vllm_config is not None:
            mc = getattr(vllm_config, "model_config", None)
            if mc is not None:
                model = getattr(mc, "model", None)
                smn = getattr(mc, "served_model_name", None)
                if isinstance(smn, (list, tuple)) and smn:
                    served_model_name = str(smn[0])
                elif isinstance(smn, str):
                    served_model_name = smn
        if not model:
            model = os.environ.get("PPING_LANG_INFO_MODEL") or None
        if not served_model_name:
            served_model_name = os.environ.get("PPING_LANG_INFO_SERVED_MODEL_NAME") or model

        # GPU: NVML-detected first, then env override
        name = gpu_name or os.environ.get("PPING_LANG_INFO_GPU") or None
        count = nvml.num_gpus if (nvml and nvml.num_gpus) else None
        if count is None:
            env_count = os.environ.get("PPING_LANG_INFO_GPU_COUNT")
            if env_count and env_count.isdigit():
                count = int(env_count)

        peak: dict[str, float] | None = None
        if gpu_peak is not None:
            peak = {
                "bf16_tflops": gpu_peak.bf16_tflops,
                "mem_bw_gbs": gpu_peak.mem_bw_gbs,
            }
        else:
            env_tflops = os.environ.get("PPING_LANG_INFO_BF16_TFLOPS")
            env_bw = os.environ.get("PPING_LANG_INFO_MEM_BW_GBS")
            if env_tflops and env_bw:
                try:
                    peak = {"bf16_tflops": float(env_tflops), "mem_bw_gbs": float(env_bw)}
                except ValueError:
                    peak = None

        return {
            "version": version,          # pping-lang 包版本(与 /api/health 一致)
            "vllm_version": vllm_ver,
            "model": model,
            "served_model_name": served_model_name,
            # vLLM OpenAI 端点(从启动 cmdline 解析真实 --host/--port,0.0.0.0→127.0.0.1)。
            # 压测在服务端跑,前端用这个预填 endpoint,不靠 :8000 的猜测(端口被改过就会猜错)。
            "vllm_endpoint": _vllm_base_url(),
            "gpu_name": name,
            "gpu_count": count,
            "gpu_peak": peak,
            "instance_id": instance_id,
            "engine_index": engine_index,
            # Startup-info modal payload — captured at plugin init time
            "cmdline": cmdline or [],
            "env": env_snapshot or {},
            "resolved_config": _resolved_vllm_config(vllm_config),
        }

    def _resolved_vllm_config(vc: Any) -> dict[str, Any] | None:
        """Dump vllm_config's basic-typed fields per sub-config, for the
        startup-info modal. Best-effort: skips anything that isn't a plain
        scalar so we never spill non-serializable engine internals."""
        if vc is None:
            return None
        out: dict[str, dict[str, Any]] = {}
        for sub_name in ("model_config", "scheduler_config", "cache_config", "parallel_config"):
            sub = getattr(vc, sub_name, None)
            if sub is None:
                continue
            sub_data: dict[str, Any] = {}
            for attr in dir(sub):
                if attr.startswith("_") or callable(getattr(sub, attr, None)):
                    continue
                v = getattr(sub, attr, None)
                if isinstance(v, (str, int, float, bool)) or v is None:
                    sub_data[attr] = v
            if sub_data:
                out[sub_name] = sub_data
        return out or None

    # === GET /api/metrics/available ===
    @app.get("/api/metrics/available")
    def metrics_available() -> dict[str, list[str]]:
        return {"metrics": sorted(ALLOWED_METRICS)}

    # === GET /api/metrics/recent ===
    @app.get("/api/metrics/recent")
    def metrics_recent(
        name: str = Query(..., description="Metric name (must be in catalog)"),
        seconds: int = Query(60, ge=1, le=86400),
        limit: int = Query(1000, ge=1, le=10000),
    ) -> dict[str, Any]:
        if name not in ALLOWED_METRICS:
            raise HTTPException(422, f"unknown metric {name!r}")
        # Short-window reads (the dashboard's poll path) go straight to the
        # in-memory ring buffer — no roundtrip, no flush wait. Long-window reads
        # scan the JSONL persistence (cold path) since the ring isn't sized for it.
        if seconds <= 60:
            ring = sink.recent(name, seconds)
            points = [
                {"ts_ns": ts, "value": v, "engine_idx": 0, "gpu_idx": -1}
                for v, ts in ring[-limit:]
            ]
            return {"name": name, "seconds": seconds, "points": points}
        since_ns = wall_ns() - int(seconds * 1e9)
        try:
            points = metric_store.recent_metric_points(name, since_ns, limit)
        except Exception:
            points = []
        return {"name": name, "seconds": seconds, "points": points}

    # === GET /api/metrics/snapshot ===
    # "Latest value per metric within window" — live read path, in-memory.
    @app.get("/api/metrics/snapshot")
    def metrics_snapshot(
        seconds: int = Query(30, ge=1, le=3600),
    ) -> dict[str, Any]:
        cutoff_ns = wall_ns() - int(seconds * 1e9)
        latest: dict[str, dict[str, Any]] = {}
        for name in ALLOWED_METRICS:
            row = sink.latest(name)
            if row is None:
                continue
            value, ts_ns = row
            if ts_ns >= cutoff_ns:
                latest[name] = {
                    "value": value, "ts_ns": ts_ns,
                    "engine_idx": 0, "gpu_idx": -1,
                }
        return {"window_seconds": seconds, "metrics": latest}

    # === GET /api/kpis — curated KPI bundle for dashboard (one round-trip) ===
    # Live read path: ALL data comes from Sink's in-memory ring buffers, no
    # DuckDB roundtrip, no checkpoint dance. Latency = push→memory (<1μs),
    # so the dashboard sees a metric within one poll interval of arrival.
    @app.get("/api/kpis")
    def kpis(
        window: int = Query(60, ge=5, le=3600, description="Aggregation window (s)"),
    ) -> dict[str, Any]:
        def latest_val(name: str) -> float | None:
            row = sink.latest(name)
            return row[0] if row else None

        def values_in_window(name: str) -> list[float]:
            return [v for v, _ts in sink.recent(name, window)]

        # Latency cards: report a full summary, not a single number — see _summary.
        ttft = _summary(values_in_window(M.VLLM_REQ_TTFT_MS))

        # TPOT preferred (per-finished-request), fall back to ITL (per-iter)
        # when the model/version never emits TPOT. We label which source won so
        # the dashboard can flag "this is ITL, not TPOT" honestly.
        tpot_vals = values_in_window(M.VLLM_REQ_TPOT_MS)
        tpot_source = "tpot"
        if not tpot_vals:
            tpot_vals = values_in_window(M.VLLM_REQ_ITL_MS)
            tpot_source = "itl"
        tpot = _summary(tpot_vals)

        # 端到端请求延迟——首页 SLO 三件套(TTFT/TPOT/E2E)的第三项。
        e2e = _summary(values_in_window(M.VLLM_REQ_E2E_LATENCY_MS))

        # Output throughput: sum gen tokens / window seconds (system aggregate)
        gen_vals = values_in_window(M.VLLM_ITER_GEN_TOKENS)
        output_tps = (sum(gen_vals) / window) if gen_vals else None

        # Per-request decode rate: derived directly from TPOT. With TPOT in ms,
        # 1000/TPOT_p50 gives tokens/sec a single user feels typing on screen.
        per_req_decode_tps = (1000.0 / tpot["p50"]) if tpot["p50"] else None

        preempt_vals = values_in_window(M.VLLM_ITER_PREEMPTED_REQS)
        preempt_per_min = (sum(preempt_vals) * 60.0 / window) if preempt_vals else None

        return {
            "window_seconds": window,
            "kpis": {
                # Latency: full summaries instead of one percentile each.
                "ttft": ttft,                       # {p50, p95, p99, avg, n}
                "tpot": tpot,                       # ditto
                "tpot_source": tpot_source,         # "tpot" | "itl"
                "e2e": e2e,                          # ditto — 端到端(vLLM e2e_latency 直方图)
                # Throughput pair: system-aggregate + per-request feel
                "output_tps": output_tps,
                "per_req_decode_tps": per_req_decode_tps,
                # Backwards-compatible flat keys (older UI builds, bench cards)
                "ttft_p50_ms": ttft["p50"],
                "ttft_p99_ms": ttft["p99"],
                "tpot_p50_ms": tpot["p50"],
                "tpot_p99_ms": tpot["p99"],
                "e2e_p50_ms": e2e["p50"],
                "e2e_p99_ms": e2e["p99"],
                # Scheduler & efficiency latches
                "kv_cache": latest_val(M.VLLM_SCHEDULER_KV_CACHE_USAGE_RATIO),
                "running_reqs": latest_val(M.VLLM_SCHEDULER_RUNNING_REQS),
                "waiting_reqs": latest_val(M.VLLM_SCHEDULER_WAITING_REQS),
                "mfu": latest_val(M.VLLM_PERF_MFU_RATIO),
                "padding_ratio": latest_val(M.VLLM_CUDAGRAPH_PADDING_RATIO),
                "prefix_cache_hit": latest_val(M.VLLM_SCHEDULER_PREFIX_CACHE_HIT_RATIO),
                "preempt_per_min": preempt_per_min,
                "gpu_util": latest_val(M.GPU_UTIL_PCT),
                "gpu_mem_used_pct": latest_val(M.GPU_MEM_USED_PCT),
                "gpu_mem_bw_pct": latest_val(M.GPU_MEM_UTIL_PCT),
            },
        }

    # === GET /api/kernels — CUPTI kernel 级时间分解 (阶段 1a) ===
    # 一个 round-trip 喂 dashboard 的 kernel 面板。全部走 Sink 内存实时环,
    # 无 DuckDB。enabled=False 表示 CUPTI 未启用 / 无 GPU / 还没数据。
    @app.get("/api/kernels")
    def kernels(
        window: int = Query(60, ge=5, le=3600, description="latest 取值窗口 (s)"),
        lang: str = Query("zh", description="findings 语言 zh/en"),
    ) -> dict[str, Any]:
        cutoff_ns = wall_ns() - int(window * 1e9)

        def fresh_val(name: str) -> float | None:
            row = sink.latest(name)
            if row is None:
                return None
            value, ts_ns = row
            return value if ts_ns >= cutoff_ns else None

        # kernel 类占比(占总 kernel 计算时间);按占比降序便于 UI 直接画条形
        class_metrics = {
            "attention": M.KERNEL_SHARE_ATTENTION_PCT,
            "gemm": M.KERNEL_SHARE_GEMM_PCT,
            "norm": M.KERNEL_SHARE_NORM_PCT,
            "rotary": M.KERNEL_SHARE_ROTARY_PCT,
            "activation": M.KERNEL_SHARE_ACTIVATION_PCT,
            "comm": M.KERNEL_SHARE_COMM_PCT,
            "other": M.KERNEL_SHARE_OTHER_PCT,
        }
        _shares: list[tuple[str, float]] = []
        for cls, metric in class_metrics.items():
            pct = fresh_val(metric)
            if pct is not None:
                _shares.append((cls, pct))
        _shares.sort(key=lambda t: t[1], reverse=True)
        class_shares = [{"cls": cls, "pct": pct} for cls, pct in _shares]

        gpu_busy = fresh_val(M.KERNEL_GPU_BUSY_PCT)
        launch_rate = fresh_val(M.KERNEL_LAUNCH_COUNT_PER_S)
        # per-kernel 原始明细(未聚合 mangled 名)直接读 collector,不走 metric 管道
        top_kernels = cupti.top_kernels() if cupti is not None else []
        # 数据时刻：这批 kernel 数据是哪一窗的、采集于多久前。让前端能说清
        # "实时 / X 秒前 / 无流量已过期",而不是把陈旧数据当现状。
        snapshot_age_s: float | None = None
        rollup_window_s: float | None = None
        if cupti is not None and cupti.last_snapshot_ts is not None:
            snapshot_age_s = max(0.0, (wall_ns() - cupti.last_snapshot_ts) / 1e9)
            if cupti.last_window_ns > 0:
                rollup_window_s = cupti.last_window_ns / 1e9
        # enabled：任一 kernel 指标在窗口内有值,或有原始明细
        enabled = (
            bool(class_shares) or gpu_busy is not None
            or launch_rate is not None or bool(top_kernels)
        )

        memcpy_share = fresh_val(M.KERNEL_MEMCPY_SHARE_PCT)
        sync_share = fresh_val(M.KERNEL_SYNC_SHARE_PCT)
        in_graph = fresh_val(M.KERNEL_IN_GRAPH_PCT)
        overhead_cb_ms = fresh_val(M.PPING_LANG_CUPTI_CB_MS)
        # 结论层：把数字翻译成人话(诊断不止观测)
        findings = _kernel_findings(
            class_shares=class_shares, top_kernels=top_kernels,
            sync_share=sync_share, memcpy_share=memcpy_share, in_graph=in_graph,
            launch_rate=launch_rate, overhead_cb_ms=overhead_cb_ms,
            window_s=rollup_window_s, lang=lang,
        )

        return {
            "window_seconds": window,
            "enabled": enabled,
            "snapshot_age_s": snapshot_age_s,        # 这批数据采集于多少秒前(None=无 collector)
            "rollup_window_s": rollup_window_s,      # 这批数据聚合的窗宽(秒)
            "findings": findings,                    # [{level, title, detail}] 诊断结论
            "class_shares": class_shares,            # [{cls, pct}], 降序, 占 kernel 计算时间
            "top_kernels": top_kernels,              # [{name, cls, count, total_ms, mean_us, pct, in_graph_pct}]
            "memcpy_share_pct": memcpy_share,                           # 占墙钟
            "sync_share_pct": sync_share,                               # 占墙钟
            "gpu_busy_pct": gpu_busy,                                   # 占墙钟
            "launch_count_per_s": launch_rate,
            "mean_dur_us": fresh_val(M.KERNEL_MEAN_DUR_US),
            "in_graph_pct": in_graph,
            # 自我观测：回调开销 + 丢弃数(5% 预算守护的可见性)
            "overhead_cb_ms": overhead_cb_ms,
            "dropped_total": fresh_val(M.PPING_LANG_CUPTI_DROPPED_TOTAL),
        }

    # === GET /api/kernels/flamegraph — Python 调用栈 → kernel 火焰图 (on-demand) ===
    # 仅 capture_stacks 模式有数据。对标 zymtrace 的 Python→kernel 火焰图。
    @app.get("/api/kernels/flamegraph")
    def kernels_flamegraph() -> dict[str, Any]:
        tree = cupti.flamegraph() if cupti is not None else None
        return {
            "available": tree is not None,
            "tree": tree,   # {name, kind, value(ns), children:[...]} 或 None
        }

    # === GET /api/kernels/timeline — Nsight-style 执行时间线 (最近 N 条 kernel) ===
    @app.get("/api/kernels/timeline")
    def kernels_timeline(
        max_events: int = Query(800, ge=50, le=4000),
    ) -> dict[str, Any]:
        tl = cupti.timeline(max_events) if cupti is not None else None
        return {"available": tl is not None, "timeline": tl}

    # === GET /api/kernels/trends — kernel 聚合指标的实时时序(给趋势图,读内存环)===
    @app.get("/api/kernels/trends")
    def kernels_trends(
        seconds: int = Query(180, ge=10, le=3600),
    ) -> dict[str, Any]:
        cmap = {
            "gpu_busy": M.KERNEL_GPU_BUSY_PCT, "sync": M.KERNEL_SYNC_SHARE_PCT,
            "memcpy": M.KERNEL_MEMCPY_SHARE_PCT,
            "attention": M.KERNEL_SHARE_ATTENTION_PCT, "gemm": M.KERNEL_SHARE_GEMM_PCT,
            "comm": M.KERNEL_SHARE_COMM_PCT, "norm": M.KERNEL_SHARE_NORM_PCT,
            "activation": M.KERNEL_SHARE_ACTIVATION_PCT, "rotary": M.KERNEL_SHARE_ROTARY_PCT,
            "other": M.KERNEL_SHARE_OTHER_PCT,
        }
        series = {
            key: [{"t": ts, "v": v} for v, ts in sink.recent(m, seconds)]
            for key, m in cmap.items()
        }
        return {
            "seconds": seconds, "now_ns": wall_ns(),
            "available": any(series[k] for k in series), "series": series,
        }

    # === GET /api/kernels/trace — Chrome Trace Event JSON(Perfetto/chrome://tracing)===
    @app.get("/api/kernels/trace")
    def kernels_trace() -> dict[str, Any]:
        tr = cupti.chrome_trace() if cupti is not None else None
        return {"available": tr is not None, "trace": tr}

    # === Deep Evidence(阶段 2 PC Sampling 按需取证)===
    # POST 触发一个短窗取证(阻塞 ~window 秒,FastAPI 在 threadpool 跑 sync def,
    # 不卡事件循环)。不可用时 fail-closed 返回 available=False + error,不抛。
    @app.post("/api/kernels/deep_evidence")
    def deep_evidence(
        window: float = Query(5.0, ge=0.0, le=30.0, description="取证窗时长(秒)"),
        period_log2: int = Query(16, ge=5, le=31, description="采样周期 2^N 周期(过小会打满 HW 缓冲楔死采样)"),
        lang: str = Query("zh", description="findings 语言 zh/en"),
    ) -> dict[str, Any]:
        if cupti is None:
            return {"available": False, "error": "CUPTI collector not configured", "findings": []}
        result = cupti.run_deep_evidence(window_s=window, period_log2=period_log2)
        result["findings"] = _stall_findings(result, lang)
        return result

    # GET 读最近一次取证结果 + 当前是否可用(给 UI 渲染面板,不触发新采集)。
    @app.get("/api/kernels/deep_evidence")
    def deep_evidence_last(
        lang: str = Query("zh", description="findings 语言 zh/en"),
    ) -> dict[str, Any]:
        last = cupti.last_stall_result() if cupti is not None else None
        available = cupti.pc_sampling_available() if cupti is not None else False
        return {
            "available_now": available,
            "last": last,
            "findings": _stall_findings(last, lang) if last else [],
        }

    # === Deep Profile(#1/#7):常驻 launch 配置 → 理论占用率/受限资源/wave 量化 ===
    # 与深窗无关:数据来自 launch 回调常驻采集(随 PC Sampling 窗回流),不暂停服务。
    # #2/#3 由 roofline_est 纯软件估算(family 级,arch × 形状反推 × PCS time_pct);
    # #4 实测 L2/DRAM 与 row 级实测占用率/Tensor 仍是深窗指标 → 字段为 null,UI 显示 "—"。
    @app.post("/api/kernels/deep_profile")
    def deep_profile() -> dict[str, Any]:
        from pping_lang.collector.deep_profile import build_deep_profile  # noqa: PLC0415
        from pping_lang.hardware import (  # noqa: PLC0415
            read_gpu_peak,
            read_sm_clock_hz,
            read_sm_limits,
        )
        if cupti is None:
            return {"available": False, "error": "CUPTI collector not configured", "kernels": []}
        return build_deep_profile(
            cupti.last_stall_result(), read_sm_limits(),
            arch=_arch, peak=read_gpu_peak(), sm_clock_hz=read_sm_clock_hz(),
            calibration=_kernel_calib,
        )

    # === GET /api/latency_trends — TTFT / TPOT / E2E bucketed p50+p99 over time ===
    # Same dual-path strategy as /api/kpis: ≤200s windows served from the live
    # ring buffer (per-metric ring holds ~2000 points; at typical req rates
    # that covers 200s+). Longer windows still go to DuckDB. Lets the trend
    # charts paint within one poll cycle of a metric arriving instead of
    # waiting for WAL→checkpoint.
    def _bucketed_from_memory(
        name: str, since_ns: int, until_ns: int, buckets: int,
    ) -> list[dict[str, Any]]:
        pts = sink.recent(name, max(1, (until_ns - since_ns) / 1e9))
        if not pts:
            return []
        bucket_width_ns = max(1, (until_ns - since_ns) // buckets)
        groups: dict[int, list[float]] = {}
        for v, ts in pts:
            if ts < since_ns or ts >= until_ns:
                continue
            b = int((ts - since_ns) // bucket_width_ns)
            if 0 <= b < buckets:
                groups.setdefault(b, []).append(v)
        out: list[dict[str, Any]] = []
        for b in sorted(groups):
            vals = groups[b]
            out.append({
                "t":   (b * bucket_width_ns) / 1e9,
                "avg": sum(vals) / len(vals),
                "p50": _percentile(vals, 0.50),
                "p99": _percentile(vals, 0.99),
                "n":   len(vals),
            })
        return out

    @app.get("/api/latency_trends")
    def latency_trends(
        seconds: int = Query(300, ge=30, le=86400),
        buckets: int = Query(30, ge=5, le=240),
    ) -> dict[str, Any]:
        now_ns = wall_ns()
        since_ns = now_ns - int(seconds * 1e9)
        # ≤900s: serve from memory (ring may not cover the whole window under
        # extreme push rates, but it'll always have the *most recent* data,
        # which is what users actually look at on the dashboard). >900s falls
        # back to DuckDB for genuine historical replays.
        if seconds <= 900:
            ttft = _bucketed_from_memory(M.VLLM_REQ_TTFT_MS, since_ns, now_ns, buckets)
            tpot = _bucketed_from_memory(M.VLLM_REQ_TPOT_MS, since_ns, now_ns, buckets)
            tpot_source = "tpot"
            if not tpot:
                tpot = _bucketed_from_memory(M.VLLM_REQ_ITL_MS, since_ns, now_ns, buckets)
                tpot_source = "itl"
            e2e = _bucketed_from_memory(M.VLLM_REQ_E2E_LATENCY_MS, since_ns, now_ns, buckets)
        else:
            # >900s: genuine historical replay — scan the JSONL persistence (cold).
            ttft = metric_store.bucketed_quantiles(M.VLLM_REQ_TTFT_MS, since_ns, now_ns, buckets)
            tpot = metric_store.bucketed_quantiles(M.VLLM_REQ_TPOT_MS, since_ns, now_ns, buckets)
            tpot_source = "tpot"
            if not tpot:
                tpot = metric_store.bucketed_quantiles(M.VLLM_REQ_ITL_MS, since_ns, now_ns, buckets)
                tpot_source = "itl"
            e2e = metric_store.bucketed_quantiles(M.VLLM_REQ_E2E_LATENCY_MS, since_ns, now_ns, buckets)
        return {
            "seconds": seconds,
            "buckets": buckets,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "tpot_source": tpot_source,
            "e2e_ms": e2e,
        }

    # Precompute arch/params once at app-build time — vllm_config is immutable
    # for the lifetime of the process.
    _arch = _extract_arch(vllm_config)
    _params = _estimate_params(_arch) if _arch else None
    _dtype_b = _DTYPE_BYTES.get(_arch["torch_dtype"], 2) if _arch else 2
    # #4:ncu 离线标定表(decode M=1 口径的 L2/DRAM 实测)。没有文件 → None,面板显示 —
    from pping_lang.collector.kernel_calibration import load_calibration  # noqa: PLC0415
    _kernel_calib = load_calibration()

    # === GET /api/roofline — live Roofline scatter (points + peak roofs) ===
    # Two data sources, chosen at request time:
    #   "measured"   — direct from vllm perf_stats (flops/read_bytes/write_bytes),
    #                   only available on vllm >=0.20. Each point is what the
    #                   engine actually saw.
    #   "analytical" — estimated from iter token counts + model parameter count:
    #                   FLOPS ≈ 2 · params · tokens   (mul+add per param per
    #                                                  token — Kaplan form)
    #                   Bytes ≈ params · dtype_bytes  (weights read once per
    #                                                  fwd pass; KV ignored)
    #                   AI    ≈ 2 · tokens / dtype_bytes
    #                   For bf16 (dtype_bytes=2): AI ≈ tokens. So decode
    #                   (tokens=1) at AI≈1 (memory-bound); prefill (tokens=N)
    #                   at AI≈N (compute-bound for large N).
    #                   Always computable, even on old vllm. Reported as
    #                   "estimate" in the response so the UI can flag it.
    @app.get("/api/roofline")
    def roofline(
        seconds: int = Query(60, ge=5, le=3600),
    ) -> dict[str, Any]:
        points: list[dict[str, Any]] = []
        data_source = "measured"
        formula: str | None = None

        # Measured path: join flops + read_b + write_b from live ring buffers.
        # Collector pushes all three at the same monotonic ts so they line up.
        flops_pts = sink.recent(M.VLLM_PERF_FLOPS_PER_GPU, seconds)
        read_pts = {ts: v for v, ts in sink.recent(M.VLLM_PERF_READ_BYTES_PER_GPU, seconds)}
        write_pts = {ts: v for v, ts in sink.recent(M.VLLM_PERF_WRITE_BYTES_PER_GPU, seconds)}
        last_ts: int | None = None
        for flops_v, ts in flops_pts:
            read_b = read_pts.get(ts)
            if read_b is None:
                continue
            total_b = read_b + write_pts.get(ts, 0.0)
            if last_ts is None:
                last_ts = ts
                continue
            dt_s = (ts - last_ts) / 1e9
            last_ts = ts
            if dt_s <= 0 or total_b <= 0 or flops_v <= 0:
                continue
            points.append({
                "ai": flops_v / total_b,
                "throughput_tflops": flops_v / dt_s / 1e12,
                "ts_ns": ts,
            })

        # Fallback: vllm <0.20 doesn't emit perf_stats. Estimate from token
        # counts + model architecture.
        if not points and _params:
            data_source = "analytical"
            formula = (
                f"FLOPS≈2·params·tokens, Bytes≈params·{_dtype_b} "
                f"(params={_params/1e9:.2f}B, dtype={_arch['torch_dtype']})"
            )
            gen_pts = sink.recent(M.VLLM_ITER_GEN_TOKENS, seconds)
            prompt_pts = {ts: v for v, ts in sink.recent(M.VLLM_ITER_PROMPT_TOKENS, seconds)}
            last_ts = None
            bytes_per_step = _params * _dtype_b  # weights read once per fwd pass
            for gen_v, ts in gen_pts:
                tokens = gen_v + prompt_pts.get(ts, 0.0)
                if tokens <= 0:
                    continue
                if last_ts is None:
                    last_ts = ts
                    continue
                dt_s = (ts - last_ts) / 1e9
                last_ts = ts
                if dt_s <= 0:
                    continue
                flops = 2.0 * _params * tokens
                points.append({
                    "ai": flops / bytes_per_step,
                    "throughput_tflops": flops / dt_s / 1e12,
                    "ts_ns": ts,
                })

        # Peak from gpu_peak param, or fall back to env (demo)
        peak: dict[str, float] | None = None
        if gpu_peak is not None:
            peak = {
                "compute_tflops": gpu_peak.bf16_tflops,
                "mem_bw_tbs": gpu_peak.mem_bw_gbs / 1000.0,
            }
        else:
            env_tflops = os.environ.get("PPING_LANG_INFO_BF16_TFLOPS")
            env_bw = os.environ.get("PPING_LANG_INFO_MEM_BW_GBS")
            if env_tflops and env_bw:
                try:
                    peak = {
                        "compute_tflops": float(env_tflops),
                        "mem_bw_tbs": float(env_bw) / 1000.0,
                    }
                except ValueError:
                    peak = None

        return {
            "seconds": seconds,
            "points": points,
            "peak": peak,
            "data_source": data_source,        # "measured" | "analytical"
            "formula": formula,                # analytical only — explanation string
            "params_billion": (_params / 1e9) if _params else None,
            # P0-C：最近一次实测 scaling sweep(没跑过为 None)→ 图上叠实测扩展曲线
            "scaling": _scaling["result"],
        }

    # 同一规则每个评估周期(默认 1s)都会复触发,内存环里就堆成 D3a×N。
    # UI 要的是「当前命中了哪些事实」——每条规则一张卡、取最近一次状态。
    # recent_diagnoses 是 newest-first,故按 rule_id 取首次出现 = 最新一次。
    def _dedup_latest_by_rule(diags: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[Any] = set()
        out: list[dict[str, Any]] = []
        for d in diags:
            rid = d.get("rule_id")
            if rid in seen:
                continue
            seen.add(rid)
            out.append(d)
        return out

    # === GET /api/diagnoses ===
    @app.get("/api/diagnoses")
    def diagnoses(
        seconds: int = Query(300, ge=1, le=86400),
        limit: int = Query(200, ge=1, le=2000),
    ) -> dict[str, Any]:
        # 内存诊断环：命中即可见,无 DuckDB 读、无刷盘滞后
        since_ns = wall_ns() - int(seconds * 1e9)
        try:
            diags = sink.recent_diagnoses(since_ns, limit)
        except Exception:
            diags = []
        return {"window_seconds": seconds, "diagnoses": _dedup_latest_by_rule(diags)}

    # === GET /api/diagnoses/history ===
    @app.get("/api/diagnoses/history")
    def diagnoses_history(
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict[str, Any]:
        try:
            diags = sink.recent_diagnoses(since_ns=0, limit=limit)
        except Exception:
            diags = []
        return {"diagnoses": _dedup_latest_by_rule(diags)}

    # === GET /api/diagnosis_rules — 现役事实规则 + 中心配置(阈值已按配置解析)===
    # 这是诊断引擎真正在跑的规则(代码+配置驱动,非旧 RuleStore 的自由 CRUD)。
    def _active_config():
        # 引擎在跑就读它的(热配置);否则回退磁盘配置,UI 仍可浏览。
        if diag_engine is not None:
            return diag_engine.config
        from pping_lang.rules.diagnosis_config import load_config  # noqa: PLC0415
        return load_config()

    def _serialize_check(cfg, c) -> dict[str, Any]:
        resolved = getattr(cfg, c.threshold_ref) if c.threshold_ref else c.threshold
        return {
            "metric": c.metric, "op": c.op,
            "threshold": resolved, "threshold_ref": c.threshold_ref,
            "window_seconds": c.window_seconds, "aggregation": c.aggregation,
        }

    def _serialize_rules(cfg) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in DIAGNOSIS_RULES:
            detectors = [
                {
                    "key": det.key, "name": det.name, "layer": det.layer,
                    "checks": [_serialize_check(cfg, c) for c in det.checks],
                }
                for det in r.detectors
            ]
            out.append({
                "id": r.id, "name": r.name, "kind": r.kind, "severity": r.severity,
                "detectors": detectors,
                "hypothesis": r.hypothesis, "suggestion": r.suggestion,
            })
        return out

    @app.get("/api/diagnosis_rules")
    def diagnosis_rules() -> dict[str, Any]:
        cfg = _active_config()
        return {
            "active": diag_engine is not None,
            "rules": _serialize_rules(cfg),               # 4 瓶颈诊断规则(只读)
            "config": diag_config_to_dict(cfg),
            "workload_forms": list(WORKLOAD_FORMS),
        }

    # === PUT /api/diagnosis_config — 改中心 SLA/阈值,热生效 ===
    @app.put("/api/diagnosis_config")
    def update_diagnosis_config(body: dict = Body(...)) -> dict[str, Any]:
        try:
            cfg = diag_config_from_dict(body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid diagnosis config: {e}")
        applied = diag_engine is not None
        if applied:
            diag_engine.set_config(cfg)
        return {
            "applied": applied,        # False = 引擎没在跑(只校验,未热加载)
            "config": diag_config_to_dict(cfg),
            "rules": _serialize_rules(cfg),
        }

    # === GET /api/rules ===
    @app.get("/api/rules")
    def rules_list() -> dict[str, Any]:
        return {
            "rules": [_rule_to_dict(r, rule_store.is_default(r.id))
                      for r in rule_store.list()]
        }

    # === GET /api/rules/{rule_id} ===
    @app.get("/api/rules/{rule_id}")
    def rule_get(rule_id: str) -> dict[str, Any]:
        r = rule_store.get(rule_id)
        if r is None:
            raise HTTPException(404, f"rule {rule_id!r} not found")
        return _rule_to_dict(r, rule_store.is_default(rule_id))

    # === POST /api/rules — create new rule ===
    @app.post("/api/rules", status_code=201)
    def rule_create(rule_in: RuleIn) -> dict[str, Any]:
        if rule_store.get(rule_in.id) is not None:
            raise HTTPException(409, f"rule {rule_in.id!r} exists; use PUT to update")
        rule = _rule_in_to_rule(rule_in)
        try:
            rule_store.upsert(rule)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return _rule_to_dict(rule, rule_store.is_default(rule.id))

    # === PUT /api/rules/{rule_id} — update existing rule ===
    @app.put("/api/rules/{rule_id}")
    def rule_update(rule_id: str, rule_in: RuleIn) -> dict[str, Any]:
        if rule_in.id != rule_id:
            raise HTTPException(
                422,
                f"rule_id mismatch: path={rule_id!r}, body.id={rule_in.id!r}",
            )
        rule = _rule_in_to_rule(rule_in)
        try:
            rule_store.upsert(rule)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return _rule_to_dict(rule, rule_store.is_default(rule.id))

    # === DELETE /api/rules/{rule_id} ===
    @app.delete("/api/rules/{rule_id}", status_code=204)
    def rule_delete(rule_id: str) -> Response:
        try:
            rule_store.delete(rule_id)
        except KeyError:
            raise HTTPException(404, f"rule {rule_id!r} not found")
        return Response(status_code=204)

    # === POST /api/rules/{rule_id}/test — preview without firing ===
    @app.post("/api/rules/{rule_id}/test")
    def rule_test(
        rule_id: str,
        body: RuleTestRequest | None = None,
    ) -> dict[str, Any]:
        # Use override if provided, else look up by id
        if body and body.override is not None:
            rule = _rule_in_to_rule(body.override)
        else:
            rule = rule_store.get(rule_id)
            if rule is None:
                raise HTTPException(404, f"rule {rule_id!r} not found")
        try:
            cond = rule.condition
            cutoff = wall_ns() - int(cond.window_seconds * 1e9)
            value = metric_store.aggregate_metric(cond.metric, cutoff, cond.aggregation)
            fired = (
                _OP_TO_FN[cond.op](value, cond.threshold)
                if value is not None and cond.op in _OP_TO_FN
                else None
            )
        except Exception as e:
            logger.exception("test eval failed for %s", rule_id)
            raise HTTPException(500, f"eval failed: {e}")
        return {
            "rule_id": rule.id,
            "would_fire": fired,
            "value": value,
            "threshold": rule.condition.threshold,
            "data_available": value is not None,
            "metric": rule.condition.metric,
            "window_seconds": rule.condition.window_seconds,
            "aggregation": rule.condition.aggregation,
        }

    # === GET /api/instances ===
    @app.get("/api/instances")
    def instances() -> dict[str, list[str]]:
        try:
            ids = metric_store.list_instances()
        except Exception:
            ids = []
        return {"instances": ids}

    # ─────────────────────────────────────────────────────────────────────
    #  BENCH API — see docs/bench-design-v0.1.md §10
    # ─────────────────────────────────────────────────────────────────────

    # In-memory registry of currently-running bench runs (asyncio Task references).
    # DB holds the canonical state; this dict is just for "is X live right now".
    _bench_runs: dict[str, dict[str, Any]] = {}

    # Initialize bench_runs table once. Re-init at every endpoint is also defensive
    # (in case of race with first vLLM step on a fresh DB).
    try:
        _init_conn = open_conn(db_path)
        try:
            bench_store.init_bench_table(_init_conn)
        finally:
            _init_conn.close()
    except Exception as e:
        logger.warning("[pping-lang] bench_runs table init deferred: %s", e)

    async def _execute_bench(
        run_id: str, scenario: StaticScenario,
    ) -> None:
        """Run a bench in-process; finalize DB on completion or failure."""
        # 惰性导入:client.py 顶层要 httpx(仅 extras 提供),dashboard/API 本身不能连坐
        from pping_lang.bench.client import OpenAIStreamClient  # noqa: PLC0415

        try:
            async with OpenAIStreamClient(
                scenario.endpoint, timeout_s=scenario.timeout_s,
            ) as client:
                summary = await run_static(scenario, client)
            conn = open_conn(db_path)
            try:
                bench_store.init_bench_table(conn)
                slo = bench_store.evaluate_slo(summary, scenario)
                bench_store.mark_done(
                    conn, run_id, time.monotonic_ns(), summary, slo_status=slo,
                )
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001
            logger.exception("[bench] run %s failed", run_id)
            try:
                conn = open_conn(db_path)
                try:
                    bench_store.mark_failed(
                        conn, run_id, time.monotonic_ns(),
                        f"{type(e).__name__}: {e}",
                    )
                finally:
                    conn.close()
            except Exception:
                logger.exception("[bench] failed to record failure for %s", run_id)
        finally:
            _bench_runs.pop(run_id, None)

    # === P0-C:roofline 实测 scaling 闭环 ===
    # 理论 batch-scaling envelope 是线性带宽外推;这里串行压测 B∈{1,4,16,64} 把**实测**
    # 扩展曲线叠上去 —— 缺口从哪个 B 张开 = 真实瓶颈位置(调度/KV/launch),外推变实测。
    _scaling: dict[str, Any] = {"running": False, "progress": None, "result": None, "error": None}

    def _vllm_base_url() -> str:
        """vLLM OpenAI 端点：插件与 vllm 同机,从启动 cmdline 解析 --host/--port。"""
        host, port = "127.0.0.1", "8000"
        args = cmdline or []
        for i, a in enumerate(args):
            if a == "--port" and i + 1 < len(args):
                port = args[i + 1]
            elif a.startswith("--port="):
                port = a.split("=", 1)[1]
            elif a == "--host" and i + 1 < len(args):
                host = args[i + 1]
            elif a.startswith("--host="):
                host = a.split("=", 1)[1]
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}/v1"

    def _served_model() -> str | None:
        mc = getattr(vllm_config, "model_config", None)
        if mc is None:
            return None
        smn = getattr(mc, "served_model_name", None)
        if isinstance(smn, (list, tuple)) and smn:
            return str(smn[0])
        if isinstance(smn, str) and smn:
            return smn
        m = getattr(mc, "model", None)
        return str(m) if m else None

    def _scaling_verdict(pts: list[dict[str, Any]]) -> dict[str, Any] | None:
        """实测点 vs 理论 envelope 的缺口分析 → 可行动结论(P0-C 的'给结论'层)。

        decode AI≈并发 B → envelope(B) = min(mem_bw_tbs×B, peak_tflops)。
        缺口 <30% 视为"跟随";首个 ≥30% 的 B = 瓶颈转移点。
        """
        if gpu_peak is None or not pts:
            return None
        peak_c = gpu_peak.bf16_tflops
        bw_tbs = gpu_peak.mem_bw_gbs / 1000.0
        rows: list[dict[str, Any]] = []
        for p in pts:
            if not p.get("tflops"):
                continue
            env = min(bw_tbs * p["b"], peak_c)
            gap = max(0.0, 1.0 - p["tflops"] / env) * 100.0
            rows.append({**p, "envelope_tflops": round(env, 2), "gap_pct": round(gap, 1)})
        if not rows:
            return None
        # 扩展效率(关键口径):B=1 的缺口是**每步固定开销**(调度/采样/launch,不随并发变),
        # 不能当扩展问题报。真正要看的是相对基线的加速比 ÷ 理想加速比：
        #   eff(B) = (tps_B / tps_b0) / (B / b0)
        # eff ≈100% = 线性扩展(固定开销被摊薄,贴 envelope 斜率);掉头点 = 收益递减点。
        base = rows[0]
        for r in rows:
            ideal = r["b"] / base["b"]
            actual = (r["tps"] / base["tps"]) if base["tps"] else 0.0
            r["eff_pct"] = round(100.0 * actual / ideal, 1) if ideal > 0 else None
        eff_t = 70.0
        linear = [r for r in rows[1:] if (r["eff_pct"] or 0) >= eff_t]
        diverge = [r for r in rows[1:] if (r["eff_pct"] or 0) < eff_t]
        base_note = (f"基线(并发 {base['b']})距 envelope 有 {base['gap_pct']:.0f}% 固定开销"
                     f"(调度/采样/launch,摊薄即可,不是扩展问题)。")
        if not rows[1:]:
            text = base_note
        elif not diverge:
            last = rows[-1]
            text = (base_note + f"扩展到并发 {last['b']} 仍保持 {last['eff_pct']:.0f}% 线性效率"
                    f"—— 余量真实可兑现,继续加并发仍有收益。")
        else:
            d0 = diverge[0]
            head = (f"并发 ≤{linear[-1]['b']} 近线性扩展(效率 ≥{eff_t:.0f}%);"
                    if linear else "")
            text = (base_note + head +
                    f"并发 {d0['b']} 扩展效率掉到 {d0['eff_pct']:.0f}% —— 收益递减,瓶颈转向"
                    f"调度/KV cache/带宽饱和;检查 max_num_seqs、gpu_memory_utilization(KV 容量),"
                    f"或就此并发档做容量规划。")
        return {"rows": rows, "text": text}

    async def _run_scaling_sweep(levels: list[int], per_level_s: int,
                                 endpoint: str, model: str) -> None:
        try:
            pts: list[dict[str, Any]] = []
            for i, b in enumerate(levels):
                _scaling["progress"] = f"并发 {b} 压测中({i + 1}/{len(levels)})"
                scenario = StaticScenario(
                    name=f"scaling-b{b}", endpoint=endpoint, model=model,
                    prompt_tokens=64, output_tokens=200, concurrency=b,
                    duration_s=per_level_s, warmup_s=4, timeout_s=90.0,
                    api="completions", prompt_source="synthetic",
                )
                scenario.validate()
                from pping_lang.bench.client import OpenAIStreamClient  # noqa: PLC0415

                async with OpenAIStreamClient(endpoint, timeout_s=scenario.timeout_s) as client:
                    summary = await run_static(scenario, client)
                tps = summary.output_throughput_tps
                pts.append({
                    "b": b, "tps": round(tps, 1),
                    # decode 每 token FLOPs ≈ 2·params(与 roofline analytical 同口径)
                    "tflops": round(2.0 * _params * tps / 1e12, 3) if _params else None,
                    "ok": summary.ok, "errors": summary.errors,
                    "tpot_p50_ms": summary.tpot_ms.p50,
                })
            _scaling["result"] = {
                "points": pts,
                "verdict": _scaling_verdict(pts),
                "per_level_s": per_level_s,
                "finished_at": time.time(),
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("[pping-lang] scaling sweep failed")
            _scaling["error"] = f"{type(e).__name__}: {e}"
        finally:
            _scaling["running"] = False
            _scaling["progress"] = None

    @app.post("/api/roofline/scaling_sweep", status_code=202)
    async def roofline_scaling_sweep(
        levels: str = Query("1,4,16,64", description="并发档,逗号分隔"),
        per_level_s: int = Query(25, ge=5, le=120, description="每档压测秒数"),
    ) -> dict[str, Any]:
        if _scaling["running"]:
            raise HTTPException(409, "scaling sweep already running")
        if not _params:
            raise HTTPException(400, "Missing model architecture info; cannot convert tok/s to TFLOPs")
        model = _served_model()
        if not model:
            raise HTTPException(400, "Cannot resolve model name (vllm_config unavailable)")
        try:
            lv = sorted({int(x) for x in levels.split(",") if x.strip()})
        except ValueError:
            raise HTTPException(422, f"Failed to parse levels: {levels!r}")
        if not lv:
            raise HTTPException(422, "levels is empty")
        _scaling.update(running=True, error=None, progress="启动中")
        asyncio.create_task(_run_scaling_sweep(lv, per_level_s, _vllm_base_url(), model))
        return {"status": "running", "levels": lv, "per_level_s": per_level_s,
                "eta_s": len(lv) * (per_level_s + 6)}

    @app.get("/api/roofline/scaling")
    def roofline_scaling() -> dict[str, Any]:
        return {"running": _scaling["running"], "progress": _scaling["progress"],
                "error": _scaling["error"], "result": _scaling["result"]}

    # === GET /api/bench/prompt-sources — UI dropdown discovery ===
    @app.get("/api/bench/prompt-sources")
    def bench_prompt_sources(
        lang: str = Query("zh", description="标签语言 zh/en"),
    ) -> dict[str, Any]:
        from pping_lang.bench.prompts import available_builtins, load_prompts
        en = lang == "en"
        items: list[dict[str, Any]] = [
            {
                "value": "synthetic",
                "label": "Synthetic fill (synthetic)" if en else "合成填充 (synthetic)",
                "description": ("Repeats the 'the quick brown fox' template up to prompt_tokens length"
                                if en else "按 prompt_tokens 长度循环 the quick brown fox 句模板"),
                "uses_prompt_tokens": True,
            },
        ]
        descs = _BUILTIN_DESCRIPTIONS_EN if en else _BUILTIN_DESCRIPTIONS
        for name in available_builtins():
            try:
                size = len(load_prompts(f"builtin:{name}"))
            except Exception:
                size = 0
            items.append({
                "value": f"builtin:{name}",
                "label": (f"Built-in {name} ({size})" if en else f"内置 {name} ({size} 条)"),
                "description": descs.get(name, ""),
                "uses_prompt_tokens": False,
            })
        return {"sources": items}

    # === GET /api/bench/runs — list past + currently running ===
    @app.get("/api/bench/runs")
    def bench_list(
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        conn = open_conn(db_path)
        try:
            try:
                bench_store.init_bench_table(conn)
                runs = bench_store.list_runs(conn, limit=limit)
            except Exception:
                runs = []
        finally:
            conn.close()
        # Mark which runs are alive in this process right now
        for r in runs:
            r["live"] = r["run_id"] in _bench_runs
        # now_ns lets the UI compute "X seconds ago" against the server's
        # monotonic clock (started_at_ns uses time.monotonic_ns, not wall clock)
        return {"runs": runs, "now_ns": time.monotonic_ns()}

    # === GET /api/bench/runs/{run_id} — single run detail ===
    @app.get("/api/bench/runs/{run_id}")
    def bench_detail(run_id: str) -> dict[str, Any]:
        conn = open_conn(db_path)
        try:
            run = bench_store.get_run(conn, run_id)
        finally:
            conn.close()
        if run is None:
            raise HTTPException(404, f"bench run {run_id!r} not found")
        run["live"] = run_id in _bench_runs
        return run

    # === GET /api/bench/status — currently running snapshot ===
    @app.get("/api/bench/status")
    def bench_status() -> dict[str, Any]:
        return {
            "running": [
                {
                    "run_id": rid,
                    "scenario_name": meta["scenario_name"],
                    "started_at_ns": meta["started_at_ns"],
                }
                for rid, meta in _bench_runs.items()
            ],
        }

    # === POST /api/bench/start — kick off a new run, returns 202 ===
    @app.post("/api/bench/start", status_code=202)
    async def bench_start(body: BenchStartIn) -> dict[str, Any]:
        # v0.1 API does single static runs only — sweep stays CLI-only.
        if body.sweep:
            raise HTTPException(
                501,
                "sweep mode not yet supported via API in v0.1; use `python -m pping_lang.bench static --sweep ...`",
            )

        # num_requests wins over duration_s if both set
        duration_s = body.duration_s
        num_requests = body.num_requests
        if num_requests is not None:
            duration_s = None

        name = body.name or f"adhoc-{int(time.time())}"
        try:
            slo_obj = SLO.from_spec(body.slo) if body.slo else None
            scenario = StaticScenario(
                name=name,
                endpoint=body.endpoint,
                model=body.model,
                prompt_tokens=body.prompt_tokens,
                output_tokens=body.output_tokens,
                concurrency=body.concurrency,
                duration_s=duration_s,
                num_requests=num_requests,
                warmup_s=body.warmup_s,
                timeout_s=body.timeout_s,
                api=body.api,
                slo=slo_obj,
                prompt_source=body.prompt_source,
            )
            scenario.validate()
        except (ValueError, TypeError) as e:
            raise HTTPException(422, f"invalid scenario: {e}")

        # Persist initial 'running' row
        conn = open_conn(db_path)
        try:
            bench_store.init_bench_table(conn)
            run_id = bench_store.generate_run_id(conn, "static")
            started_at_ns = time.monotonic_ns()
            bench_store.insert_running(
                conn, run_id, scenario, "static", started_at_ns,
            )
        finally:
            conn.close()

        # Fire-and-forget asyncio task; finalization is in _execute_bench
        task = asyncio.create_task(_execute_bench(run_id, scenario))
        _bench_runs[run_id] = {
            "task": task,
            "scenario_name": scenario.name,
            "started_at_ns": started_at_ns,
        }

        return {
            "run_id": run_id,
            "status": "running",
            "started_at_ns": started_at_ns,
            "scenario_name": scenario.name,
        }

    # === Autopilot(M0：默认 SimSandbox + StubAgent 跑通闭环;真 GPU 沙盒为下一增量)===
    try:
        from pathlib import Path as _Path  # noqa: PLC0415

        from pping_lang.autopilot.api import register_autopilot_routes  # noqa: PLC0415
        _ap_model = (os.environ.get("PPING_LANG_INFO_SERVED_MODEL_NAME")
                     or os.environ.get("PPING_LANG_INFO_MODEL")
                     or "Qwen/Qwen2.5-0.5B-Instruct")
        # session_dir 可被 env 覆盖,指向共享挂载卷 → host CLI 跑的真 session 直接在 dashboard 显示
        _ap_dir = (os.environ.get("PPING_LANG_AUTOPILOT_DIR")
                   or str(_Path(db_path).parent / "autopilot"))
        _ap_bridge = os.environ.get("PPING_LANG_AUTOPILOT_BRIDGE_URL")
        register_autopilot_routes(app, model=_ap_model, sim=True, session_dir=_ap_dir,
                                  bridge_url=_ap_bridge)
    except Exception:  # noqa: BLE001
        logger.exception("[pping-lang] autopilot routes registration failed")

    return app


def _rule_to_dict(r: Rule, is_default: bool = False) -> dict[str, Any]:
    return {
        "id": r.id,
        "name": r.name,
        "severity": r.severity,
        "category": r.category,
        "claim": r.claim,
        "enabled": r.enabled,
        "is_default": is_default,
        "condition": {
            "metric": r.condition.metric,
            "op": r.condition.op,
            "threshold": r.condition.threshold,
            "window_seconds": r.condition.window_seconds,
            "aggregation": r.condition.aggregation,
        },
        "message": r.message,
        "suggestion": r.suggestion,
    }


def _rule_in_to_rule(rin: RuleIn) -> Rule:
    return Rule(
        id=rin.id,
        name=rin.name,
        severity=rin.severity,
        category=rin.category,
        condition=Condition(
            metric=rin.condition.metric,
            op=rin.condition.op,
            threshold=rin.condition.threshold,
            window_seconds=rin.condition.window_seconds,
            aggregation=rin.condition.aggregation,
        ),
        message=rin.message,
        suggestion=rin.suggestion,
        claim=rin.claim,
        enabled=rin.enabled,
    )
