"""pping-lang — vLLM 性能诊断插件。

主入口：PpingLangStatLogger（通过 vllm.stat_logger_plugins entry point 自动加载）。
"""
from pping_lang.plugin import PpingLangStatLogger
from pping_lang.sink import LocalSink, Sink
from pping_lang.types import Diagnosis, MetricPoint, Severity

try:  # 版本单一来源:pyproject(经安装元数据);源码树未安装时兜底
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("pping-lang")
except PackageNotFoundError:  # pragma: no cover — 只在未安装的源码树出现
    __version__ = "0.0.0.dev0"
__all__ = [
    "PpingLangStatLogger",
    "MetricPoint",
    "Diagnosis",
    "Severity",
    "Sink",
    "LocalSink",
    "__version__",
]
