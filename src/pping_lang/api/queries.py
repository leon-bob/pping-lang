"""DuckDB 连接辅助 —— 现仅供 bench_runs 用。

历史：这里曾有一整套指标/诊断的 DuckDB 查询(latest_per_metric / bucketed_quantiles /
recent_diagnoses / list_instances …)。随着「插件去 DuckDB、指标落 JSONL」,这些读路径
已迁到 `sink.metric_log.JsonlStore`(长窗扫描)与 Sink 内存环(实时短窗)。只剩 bench 的
`bench_runs`(可变行,UPDATE 语义)仍用 DuckDB,故保留 `open_conn`。

duckdb 自 0.1.7 起是可选依赖(`[bench]` extra)—— 惰性导入,缺它时调用方优雅降级。
"""
from __future__ import annotations

from typing import Any


def open_conn(db_path: str) -> Any:
    """Open a DuckDB connection. Caller is responsible for close().

    Raises ImportError when duckdb isn't installed (see callers for graceful paths).
    """
    import duckdb  # noqa: PLC0415 — 可选依赖,惰性

    return duckdb.connect(db_path)
