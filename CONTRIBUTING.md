# Contributing to pping-lang

谢谢关注！项目在 v0.1 阶段快速演进，欢迎 issue / PR。

## 开发环境

```bash
git clone https://github.com/leon/pping-lang.git
cd pping-lang
python3 -m venv .venv && source .venv/bin/activate   # Debian/Ubuntu 系统 Python 受 PEP 668 保护
pip install -e ".[dev]"
pytest
```

vLLM 不是 dev 必须依赖（plugin.py 有 fallback），但要测真实集成需要：

```bash
pip install -e ".[dev,vllm]"
```

## 项目结构

```
pping_lang/
├── plugin.py            # vLLM stat_logger entry point + 生命周期
├── types.py             # MetricPoint / Diagnosis (frozen + slots)
├── metrics_catalog.py   # 单一 metric 名常量源 + ALLOWED_METRICS
├── hardware.py          # GPU peak FLOPs / 带宽表
├── collector/           # NVML + vLLM stats → MetricPoint
├── sink/                # base ABC + Local (DuckDB) / Tee / (v0.2 Remote)
├── rules/               # schema / defaults / loader / store / engine
├── api/                 # FastAPI app + routes + queries
├── otel/                # OTLP exporter sink
├── report/              # HTML 报告生成 (Jinja2 + Plotly)
├── ui/                  # 单文件 dashboard HTML
└── bench/               # 微基准
```

## 提交规范

- commit message 中文/英文都行；用第一行简洁概括（< 72 字符）
- 重要改动加 body 解释为什么（不只是做了什么）
- 测试通过再提：`pytest -q`
- 改 hot path 的提交必须跑：`pytest tests/test_perf.py -v`

## 写测试

新代码必须配套测试。看现有测试找风格：
- 单元测试：`tests/test_<module>.py`
- 集成测试：用真 LocalSink + 临时 DuckDB（`tmp_path` fixture）
- e2e：`tests/test_e2e_marquee.py` 跑完整数据流
- 性能 regression：`tests/test_perf.py` 守预算

UI 改动**必须** headless 浏览器实地验：

```bash
DEMO_DURATION_S=600 python examples/embedded/demo.py &
# 然后用 browse / playwright / 手动浏览器看 http://localhost:8765
```

类似 Chart.js × Alpine.js proxy 冲突这种 bug 单元测试覆盖不到，必须真渲染。

## 加新 metric

1. 加常量到 `pping_lang/metrics_catalog.py:M`
2. 命名守卫测试（`test_metrics_catalog.py`）会自动覆盖
3. 在 collector 里 emit
4. 如有派生计算，单元测试覆盖

## 加新规则

写在 `pping_lang/rules/defaults.py:DEFAULT_RULES`，跑：

```bash
pytest tests/test_rules_defaults.py -v
```

模板渲染、metric ∈ catalog、字段完整性都自动验。

## 设计决策记录

大改动看：
- [docs/pping-lang-design-v0.2.md](docs/pping-lang-design-v0.2.md) — 整体设计
- [docs/pping-lang-pre-impl-rfc.md](docs/pping-lang-pre-impl-rfc.md) — 5 项关键决策的来龙去脉

新增"为什么这么写"的设计决策应记录到 RFC 或新 ADR 文档。
