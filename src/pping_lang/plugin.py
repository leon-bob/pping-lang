"""vLLM stat_logger_plugins entry point — see design §6.2 / pre-impl-rfc §4.

Day 3 status：完整数据流通了。
- log_engine_initialized: 创建 LocalSink + NvmlSampler + VllmStatsCollector
- record(): 提取真实 stats → push 到 sink；同时写 self-overhead 心跳
- NVML 后台线程独立采样 GPU 物理层
- GPU peak 表查找成功则计算 MFU / mem_bw_util_ratio

环境变量：
- PPING_LANG_DB_PATH              落盘目录基准(取其父目录)：指标/诊断写 JSONL,
                                  bench_runs 仍用同目录 .duckdb。默认 ~/.pping-lang/local.duckdb
- PPING_LANG_INSTANCE_ID          默认 local-{engine_index}
- PPING_LANG_NVML_INTERVAL_S      默认 0.1（NVML 采样间隔，秒）
- PPING_LANG_DISABLE_NVML         设为 1 关闭 NVML 采样
- PPING_LANG_ENABLE_PCS           设为 1 开启 PC sampling（Deep Evidence,采样实测;长期主干）。
                                  需 .so 注入(CUDA_INJECTION64_PATH)+ 单进程
                                  (VLLM_ENABLE_V1_MULTIPROCESSING=0)。与 CUPTI 互斥,优先。
- PPING_LANG_PCS_PERIOD_LOG2      PC sampling 采样周期 log2(默认 12 = 每 4096 cycle)
- PPING_LANG_ENABLE_CUPTI         设为 1 开启 CUPTI Activity kernel 级采集（默认关；仅 Linux x86_64/aarch64;PCS 未开时才用）
- PPING_LANG_CUPTI_ROLLUP_S       默认 1.0（CUPTI kernel 指标 roll-up 间隔，秒）
- PPING_LANG_CUPTI_TOP_N          默认 100（每窗保留的 per-kernel 明细行数上限）
- PPING_LANG_FLUSH_INTERVAL_S     默认 5.0（Sink → JSONL 批量追加间隔）
- PPING_LANG_RETENTION_SECONDS    默认 7200（JSONL 保存时间窗口;时间为主、磁盘兜底,
                                  洪流下窗口缩水但磁盘有界,见 sink/metric_log.AppendLog）
- PPING_LANG_RULE_EVAL_INTERVAL_S 默认 1.0
- PPING_LANG_DISABLE_RULES        设为 1 关闭规则引擎
- PPING_LANG_RULES_PATH           可选 JSON 文件覆盖默认规则
- PPING_LANG_DIAGNOSIS_PRINT      默认 1，设 0 关闭终端打印
- PPING_LANG_API_HOST             默认 127.0.0.1（容器场景设 0.0.0.0）
- PPING_LANG_API_PORT             默认 8765
- PPING_LANG_DISABLE_API          设为 1 关闭 HTTP API
- OTEL_EXPORTER_OTLP_ENDPOINT     标准 OTel env，设了即开 OTel 出口（与 LocalSink 并行）
- PPING_LANG_DISABLE_OTEL         即使设了 OTLP endpoint 也强制关
"""
from __future__ import annotations

import atexit
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from pping_lang.api.routes import build_app
from pping_lang.api.server import ApiServer
from pping_lang.clock import wall_ns
from pping_lang.collector.cupti import CuptiKernelCollector
from pping_lang.collector.nvml import NvmlSampler, detect_first_gpu_name
from pping_lang.collector.vllm_stats import VllmStatsCollector
from pping_lang.hardware import GPUPeak, lookup_peak
from pping_lang.metrics_catalog import M
from pping_lang.rules.store import RuleStore
from pping_lang.sink.base import Sink
from pping_lang.sink.local import LocalSink
from pping_lang.sink.metric_log import DEFAULT_RETENTION_S
from pping_lang.sink.tee import TeeSink
from pping_lang.types import MetricPoint

logger = logging.getLogger(__name__)

# 允许 vllm 缺失时也能 import（dev / test 环境无需安装巨型 vllm）。
# 真实 `vllm serve` 时一定能 import 成功，此分支永不走。
try:
    from vllm.v1.metrics.loggers import StatLoggerBase

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

    class StatLoggerBase:  # type: ignore[no-redef]
        """Fallback used only when vllm is not installed (dev/test only)."""

        def __init__(self, vllm_config=None, engine_index: int = 0):  # noqa: D401
            pass


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import (
        IterationStats,
        MultiModalCacheStats,
        SchedulerStats,
    )


DEFAULT_DB_PATH = Path.home() / ".pping-lang" / "local.duckdb"


# Env-var prefixes exposed in /api/system. Anything else is filtered out to
# avoid leaking shell history / unrelated process env / accidental secrets.
# 注意：不收 PPING_LANG_* —— 那是插件自己的配置参数(DB 路径/端口/采样间隔/feature flag),
# 不是 vLLM workload 的上下文,对看面板的人是噪声(启动信息只展示 vLLM/模型/GPU/CUDA 环境)。
_ENV_PREFIXES_INCLUDED = (
    "VLLM_",
    "HF_",
    "CUDA_",
    "TORCH_",
    "NCCL_",
    "TRITON_",
    "MODELSCOPE_",
)
# Single keys (not prefixes) also included
_ENV_KEYS_INCLUDED = frozenset({
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
})
# Names containing any of these substrings get their VALUE masked (key kept)
_ENV_SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "API_KEY")


def _snapshot_relevant_env() -> dict[str, str]:
    """Return a dict of env vars relevant to vLLM/pping-lang/CUDA, with
    obvious secrets masked. Captured once at plugin init for the dashboard."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        included = (
            k in _ENV_KEYS_INCLUDED
            or any(k.startswith(p) for p in _ENV_PREFIXES_INCLUDED)
        )
        if not included:
            continue
        if any(marker in k for marker in _ENV_SECRET_MARKERS):
            out[k] = "***"
        else:
            out[k] = v
    return out


class PpingLangStatLogger(StatLoggerBase):
    """v0.1 主入口。每个 vLLM EngineCore 进程实例化一次。

    生命周期：
      __init__               — 构造（不做重 I/O）
      log_engine_initialized — 创建 LocalSink + NvmlSampler + VllmStatsCollector
      record(...)            — 热路径：collector.collect + 心跳 metric
      log()                  — v1 几乎不调用，逻辑放 record()
      record_sleep_state(..) — v0.1 no-op
    """

    def __init__(
        self,
        vllm_config: VllmConfig | None = None,
        engine_index: int = 0,
    ) -> None:
        super().__init__(vllm_config, engine_index)
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._sink: Sink | None = None
        self._nvml: NvmlSampler | None = None
        self._cupti: CuptiKernelCollector | None = None
        self._collector: VllmStatsCollector | None = None
        self._diag_engine = None  # DiagnosisEngine(4 瓶颈诊断引擎,见 rules/diagnosis_runtime)
        self._api: ApiServer | None = None
        self._rule_store: RuleStore | None = None
        self._collect_fail_seen: set[str] = set()  # 采集异常按类型只告警一次(10Hz 热路径防刷屏)
        # 重 I/O 延迟到 log_engine_initialized（pre-impl-rfc §4.2）
        logger.info(
            "[pping-lang] plugin instantiated for engine_index=%d (vllm_available=%s)",
            engine_index,
            _VLLM_AVAILABLE,
        )

    def log_engine_initialized(self) -> None:
        # 1) Sink
        db_path_str = os.environ.get("PPING_LANG_DB_PATH")
        db_path = Path(db_path_str) if db_path_str else DEFAULT_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        instance_id = os.environ.get(
            "PPING_LANG_INSTANCE_ID", f"local-{self.engine_index}"
        )
        flush_interval = float(
            os.environ.get("PPING_LANG_FLUSH_INTERVAL_S", "5.0")
        )
        queue_size = int(
            os.environ.get("PPING_LANG_SINK_QUEUE_SIZE", "65536")
        )
        retention_s = float(
            os.environ.get("PPING_LANG_RETENTION_SECONDS", str(DEFAULT_RETENTION_S))
        )
        local_sink = LocalSink(
            db_path=db_path,
            instance_id=instance_id,
            retention_s=retention_s,
            flush_interval_s=flush_interval,
            queue_size=queue_size,
        )
        # Optional OTel sink (additive, alongside LocalSink) — see Day 11
        sinks: list[Sink] = [local_sink]
        otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint and os.environ.get("PPING_LANG_DISABLE_OTEL") != "1":
            try:
                from pping_lang.otel.sink import OTelSink
                otel_sink = OTelSink(
                    endpoint=otel_endpoint,
                    instance_id=instance_id,
                    flush_interval_s=flush_interval,
                )
                sinks.append(otel_sink)
                logger.info("[pping-lang] OTel sink enabled → %s", otel_endpoint)
            except Exception as e:
                logger.warning("[pping-lang] OTel setup failed, continuing without: %s", e)

        # If single sink, use directly; if multiple, wrap in TeeSink for fan-out
        self._sink = sinks[0] if len(sinks) == 1 else TeeSink(*sinks)
        atexit.register(self._sink.close)

        # 2) GPU peak + name (best-effort) — skip if NVML disabled (no GPU expected)
        gpu_name: str | None = None
        gpu_peak = None
        if os.environ.get("PPING_LANG_DISABLE_NVML") != "1":
            gpu_name = detect_first_gpu_name()
            if gpu_name is not None:
                gpu_peak = lookup_peak(gpu_name)
                if gpu_peak is None:
                    logger.warning(
                        "[pping-lang] unknown GPU %r — MFU / Roofline disabled. "
                        "Add to pping_lang.hardware._GPU_PEAK_TABLE if needed.",
                        gpu_name,
                    )

        # 3) vLLM stats collector
        self._collector = VllmStatsCollector(
            self._sink, engine_index=self.engine_index, gpu_peak=gpu_peak,
        )

        # 4) NVML sampler (optional — disabled by env or by absence of GPU)
        if os.environ.get("PPING_LANG_DISABLE_NVML") != "1":
            interval_s = float(
                os.environ.get("PPING_LANG_NVML_INTERVAL_S", "0.1")
            )
            self._nvml = NvmlSampler(
                self._sink,
                engine_index=self.engine_index,
                interval_s=interval_s,
            )
            self._nvml.start()
            atexit.register(self._nvml.stop)

        # 4b) Kernel 级采集（opt-in，默认关）—— 两条**硬件互斥**的路(Activity 与 PC
        #     sampling 抢同一套性能计数器,不能同时开):
        #       PPING_LANG_ENABLE_PCS=1   → PC sampling(Deep Evidence,采样实测;长期主干)
        #       PPING_LANG_ENABLE_CUPTI=1 → CUPTI Activity(精确 μs;可选,PCS 未开时才用)
        #     PC sampling 需 .so 注入(CUDA_INJECTION64_PATH,驱动 cuInit 时加载)+ 采集须与
        #     引擎**同进程**：多进程 serve 下插件在前端进程、碰不到 EngineCore 的 kernel,
        #     故单进程(VLLM_ENABLE_V1_MULTIPROCESSING=0)才生效。详见设计 §11/§12。
        if os.environ.get("PPING_LANG_ENABLE_PCS") == "1":
            from pping_lang.collector.cupti import FakeActivitySource  # noqa: PLC0415
            from pping_lang.engine_pcs import FilePcSampling  # noqa: PLC0415
            # 多进程 serve：本进程(前端 stat_logger)没有 CUDA context、不能本地采样。
            # 真采集由 EngineCore 进程的 general_plugin(engine_pcs.init_engine_pcs)做,
            # 把每窗结果写共享文件;这里只读文件喂 Deep Evidence。单进程时两者同进程也成立。
            pcs = FilePcSampling()
            self._cupti = CuptiKernelCollector(
                self._sink,
                engine_index=self.engine_index,
                source=FakeActivitySource(available=False),
                pc_sampling=pcs,
            )
            logger.info(
                "[pping-lang] PC sampling: 前端读共享结果文件(EngineCore general_plugin 采集)",
            )
        elif os.environ.get("PPING_LANG_ENABLE_CUPTI") == "1":
            rollup_s = float(os.environ.get("PPING_LANG_CUPTI_ROLLUP_S", "1.0"))
            top_n = int(os.environ.get("PPING_LANG_CUPTI_TOP_N", "100"))
            self._cupti = CuptiKernelCollector(
                self._sink,
                engine_index=self.engine_index,
                rollup_interval_s=rollup_s,
                top_n=top_n,
            )
            self._cupti.start()  # source 不可用时内部优雅禁用 + warning
            atexit.register(self._cupti.stop)

        # 5) Rule store (defaults + optional user JSON overrides)
        rules_path = os.environ.get("PPING_LANG_RULES_PATH")
        self._rule_store = RuleStore(
            override_path=Path(rules_path) if rules_path else None
        )

        # 6) 规则引擎
        eval_interval = float(
            os.environ.get("PPING_LANG_RULE_EVAL_INTERVAL_S", "1.0")
        )
        # 旧扁平 RuleEngine(周期查 DuckDB)已随「插件去 DuckDB」退役;现役诊断器是下面
        # 的 DiagnosisEngine(纯内存环评估)。/api/rules CRUD 直接走 RuleStore。
        # DiagnosisEngine —— 事实规则决策引擎,现役诊断器(默认开)。
        if os.environ.get("PPING_LANG_DISABLE_RULES") != "1":
            from pping_lang.api.routes import (  # noqa: PLC0415
                _DTYPE_BYTES,
                _estimate_params,
                _extract_arch,
            )
            from pping_lang.rules.diagnosis_config import load_config  # noqa: PLC0415
            from pping_lang.rules.diagnosis_runtime import DiagnosisEngine  # noqa: PLC0415
            _arch = _extract_arch(self.vllm_config)
            _params = _estimate_params(_arch) if _arch else None
            _dtb = _DTYPE_BYTES.get(_arch["torch_dtype"], 2) if _arch else 2
            self._diag_engine = DiagnosisEngine(
                sink=self._sink, config=load_config(),
                params=_params, dtype_bytes=_dtb,
                peak_compute_tflops=(gpu_peak.bf16_tflops if gpu_peak else None),
                peak_mem_bw_tbs=(gpu_peak.mem_bw_gbs / 1000.0 if gpu_peak else None),
                engine_index=self.engine_index, eval_interval_s=eval_interval,
            )
            self._diag_engine.start()
            atexit.register(self._diag_engine.stop)

        # 7) HTTP API + dashboard (optional)
        if os.environ.get("PPING_LANG_DISABLE_API") != "1":
            api_host = os.environ.get("PPING_LANG_API_HOST", "127.0.0.1")
            api_port = int(os.environ.get("PPING_LANG_API_PORT", "8765"))
            # Snapshot vLLM startup context for the dashboard 启动信息 modal —
            # captured once at plugin init so subsequent env mutations don't
            # alter what users see (and so the dashboard can show env values
            # even if vLLM scrubs them later).
            import sys as _sys

            from pping_lang import __version__ as _version
            cmdline_snapshot = list(_sys.argv)
            env_snapshot = _snapshot_relevant_env()

            app = build_app(
                db_path=str(db_path),
                instance_id=instance_id,
                engine_index=self.engine_index,
                sink=self._sink,
                rule_store=self._rule_store,
                diag_engine=self._diag_engine,
                nvml=self._nvml,
                cupti=self._cupti,
                version=_version,
                vllm_config=self.vllm_config,
                gpu_peak=gpu_peak,
                gpu_name=gpu_name,
                cmdline=cmdline_snapshot,
                env_snapshot=env_snapshot,
            )
            self._api = ApiServer(app, host=api_host, port=api_port)
            self._api.start()
            atexit.register(self._api.stop)

        logger.info(
            "[pping-lang] ready: db=%s instance=%s engine=%d nvml=%s cupti=%s gpu_peak=%s "
            "rules=%d api=%s",
            db_path, instance_id, self.engine_index,
            self._nvml.enabled if self._nvml else False,
            self._cupti.enabled if self._cupti else False,
            gpu_peak,
            len(self._rule_store.list()) if self._rule_store else 0,
            self._api.url if self._api else "disabled",
        )

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        if self._sink is None:
            return
        t0 = time.monotonic_ns()  # 测耗时差值,用 monotonic
        if self._collector is not None:
            try:
                self._collector.collect(scheduler_stats, iteration_stats)
            except Exception as e:  # noqa: BLE001 — 兜底:vLLM 内部 stats 字段随版本漂移,
                self._note_collect_failure(e)  # 采集炸不能炸 serve(承诺:任何失败不影响 vLLM)
        elapsed_us = (time.monotonic_ns() - t0) / 1000.0
        self._sink.push_metric(MetricPoint(
            ts_ns=wall_ns(),  # 落库 ts,跨进程/重启可比
            name=M.PPING_LANG_RECORD_OVERHEAD_US,
            value=elapsed_us,
            engine_idx=self.engine_index,
        ))

    def log(self) -> None:
        # v1 中几乎不被调用 (vllm-project/vllm#20175)。逻辑放 record()。
        pass

    def _note_collect_failure(self, e: Exception) -> None:
        """采集链路异常:按类型只告警一次,永不外抛(版本漂移 → 降级,不崩 serve)。"""
        kind = type(e).__name__
        if kind in self._collect_fail_seen:
            return
        self._collect_fail_seen.add(kind)
        logger.warning(
            "[pping-lang] 指标采集失败(%s: %s),后续同类异常静默忽略 —— 不影响 vLLM 服务。"
            "多为 vLLM 版本间 stats 字段变动,欢迎带上 vLLM 版本号反馈",
            kind, e,
        )

    def record_sleep_state(self, is_awake: int, level: int) -> None:
        # v0.1 no-op
        pass

    # === internals ===

    def _detect_gpu_peak(self) -> GPUPeak | None:
        name = detect_first_gpu_name()
        if name is None:
            return None
        peak = lookup_peak(name)
        if peak is None:
            logger.warning(
                "[pping-lang] unknown GPU %r — MFU / Roofline disabled. "
                "Add to pping_lang.hardware._GPU_PEAK_TABLE if needed.", name,
            )
        return peak
