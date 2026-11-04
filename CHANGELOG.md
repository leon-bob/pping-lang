# Changelog

按 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，遵循 SemVer。

## [Unreleased]

## [0.1.3] — 2026-11-04

### Fixed

- `pping-demo` fired 0：demo 剧本还是旧 RuleEngine 的（padding/MFU 警告规则已随引擎重写下线），且现役规则 A（双低）的第三条 check 需要 NVML `gpu.mem_util_pct` 而 demo 禁了 NVML —— 现补合成该指标，数秒内稳定触发「双低」；新增 3 秒触发契约测试，引擎再演进时 demo 失配在测试里炸而不是在客户终端上
- demo dashboard 默认端口改 18765：与生产 dashboard 的 8765 同机并跑不再 bind 失败

### Changed

- README 方式二 docker run 补 `--cap-add SYS_ADMIN` 与 `--ipc host`：宿主机驱动默认 `RmProfilingAdminOnly=1`，缺 cap 时 Kernel 级采集被拒并静默降级（runw 实测定位）；镜像 tag 同步 0.21-0.1.2

## [0.1.2] — 2026-11-04

### Added

- `pping-demo` 命令：离线 demo 打进 wheel（`pping_lang/demo.py` + console script），pip 用户无需 clone 仓库即可跑；README/CONTRIBUTING 同步，旧 `examples/embedded/demo.py` 保留为兼容 shim

### Fixed

- demo 收尾引用引擎重写前的旧属性 `_rule_engine`（现为 `_diag_engine`）→ 跑完循环即 AttributeError；此前无人端到端跑过，随 `pping-demo` 冒烟测试一并锁住

## [0.1.1] — 2026-11-03

### Added

- Docker 快速开始镜像 `docker/Dockerfile`：基于已验证配对 `vllm/vllm-openai:v0.21.0`，内置 g++/CUPTI 编译链并构建期预编译 `.so`，入口 `pping-vllm serve`（参数透传）
- README 快速上手改双路径（中英同步）：方式一 conda/venv 独立环境 `pip install "pping-lang[vllm]"` 一条命令装齐（extra 内置已验证版本配对）；方式二 Docker 完全可复现

### Fixed

- 安装指引补充 PEP 668（Debian 12+ / Ubuntu 24.04+ 系统 Python）场景：先建 venv 或用 uv，插件须与 vLLM 同环境（README 中英同步、CONTRIBUTING、autopilot-quickstart）
- builder 探测只认 vllm 官方镜像的 dist-packages 路径布局 → conda/venv 环境（nvidia/triton 包在当前解释器 site-packages 下）探测必失败、`pping-vllm` 静默降级丢 Kernel 级采集；现优先按当前解释器探测（新增单测覆盖 conda 布局）
- autopilot-quickstart 的 `pip install pping-lang[bench]` 加引号——zsh 下裸 `[bench]` 会被当 glob 报 `no matches found`

## [0.1.0] — 2026-07-27

正式版：跳出 alpha。诊断引擎重写成规则决策树、Autopilot 从 0 到 M0+M1（诊断驱动自主调优的旗舰能力）、
全站中英双语、GitHub Pages 展示站、PC Sampling 稳定性修复。

### Added

**诊断引擎重写**
- FactRule 决策树 schema + 10 条内置规则（四大瓶颈：算力 / 带宽 / 容量 / 空载守卫）
- 纯规则求值核心（precondition guard、any/all、ratio、regime gate）
- 无 `perf_stats` 时的 analytical operating point（regime / MFU / AI 推算，不依赖 vLLM 0.20+）
- `DiagnosisEngine` 运行时接入插件（默认启用，legacy DuckDB RuleEngine 下线）
- 自定义规则复用同一 DiagnosisEngine；规则配置改成弹窗表单

**Sink 与持久化**
- JSONL 持久化取代 DuckDB 存储；规则求值直接读内存 ring，DuckDB 移出诊断热路径
- Sink 背压下自适应降采样，替代原先的尾部无序丢弃

**Autopilot —— 旗舰能力，从无到有（M0 + M1）**
- 沙盒化诊断驱动自动调优 Agent：观察诊断 → 假设一个参数 → 压测验证 → 留下 / 回滚
- 支持 OpenAI 兼容 / Anthropic / Kimi Coding 三种 Agent 供应商；真流式（SSE）调用，替代阻塞式整段等待
- 6 种业务负载形态（chat / RAG / agent / reasoning / code / custom）一键给出匹配的压测负载与 SLA 默认值
- E2E p99 SLA 从硬性关卡改为仅监控（TTFT / TPOT 仍卡关，避免因通用默认值不合模型实际而误判"无改善"）
- session 结束给结构化摘要：因果链叙述 + 逐轮账本（每个参数的 from→to、判定、Δ）
- 全中英双语（Agent 应答语言可配置、UI 全量 i18n），支持随时中断
- 一大批真机 dogfood 修复：waiting 队列作为 load-binding 判据、agent HTTP 调用重试退避、
  Kimi max_tokens 避免 thinking 饿死输出、vLLM 硬拒绝的 knob 不再提议、p99 延迟按 median-of-N 去噪、
  session 边界正确性、baseline 失败自愈

**UI / i18n**
- 全站 i18n 框架（中 / 英切换），KPI / Kernel / Bench / Rules / Autopilot 全量翻译
- Linear 风格主题（冷灰 + 靛蓝）+ sticky tabs
- GitHub Pages 双语展示站（`docs/index.html`），含实测 runw 数据与 Autopilot 真机轨迹

### Changed
- "规则决策树"改名为"规则"（去掉不必要的术语包装）
- Autopilot 启动按钮改名为"执行"（更准确描述动作）

### Fixed
- PC Sampling：禁用会在 Blackwell + CUDA 13 上与在飞推理死锁的 reprime 自愈机制
- 指标 / 诊断时间戳改存墙钟而非单调钟（避免跨进程 / 重启后时间线错位）
- `/api/diagnoses` 按 `rule_id` 去重
- Autopilot 大量正确性修复（详见上）及 dashboard 面板的一系列展示 / 状态 bug

### Tests
- 测试数从 209 增至 612

## [0.1.0a1] — 2026-06-14

首个 alpha。第 1-3 周采集/诊断/Sink 完成后，继续做完 CUPTI kernel 级剖析（阶段 1a/2/3）、
压测闭环、dashboard 诊断面板与 pip 一键打包。

### Added

**采集与派生**（Day 1-3）
- vLLM `stat_logger_plugins` entry point + `PpingLangStatLogger`
- NVML 周期采样（util / mem / power / temp / sm/mem clock，按 GPU index 区分）
- vLLM SchedulerStats / IterationStats / FinishedRequestStats 完整字段映射
- cudagraph_stats / perf_stats 派生指标：**`padding_ratio`** / **`mfu_ratio`** / **`mem_bw_util_ratio`**
- 16 款常见 GPU 的 BF16 peak FLOPs + 内存带宽常量表（hardware.py）

**诊断**（Day 4 / Day 9）
- 微型 DSL（10 字段 Rule + Condition），加载时校验 metric ∈ catalog
- 10 条内置规则覆盖 6 个 category，含 marquee `high-cudagraph-padding` + `low-mfu`
- 周期 SQL 评估（DuckDB QUANTILE_CONT 支持 p50/p95/p99）
- 触发即终端打印 + 推 Diagnosis 到 sink
- 抑制窗口避免重复告警
- **规则热加载**：CRUD 后下次评估即生效，无需重启 vLLM

**Sink 与存储**（Day 2 / Day 11）
- Fire-and-forget Sink ABC，push <5μs（实测 156ns）
- LocalSink → DuckDB（schema 含 instance_id + gpu_idx）
- TeeSink fan-out 多 sink 并联
- OTelSink → OTLP gRPC（metric 名 `pping_lang.<name>`，含 engine_idx + gpu_idx attrs）

**HTTP API + UI**（Day 6-8）
- 14 个端点：health / metrics (available/recent/snapshot) / diagnoses / rules CRUD / rule test / report / instances
- 单文件 HTML dashboard（Alpine.js + Chart.js via CDN，~12KB）
- 三 tab：实时（KPI + chart + 诊断卡片）、规则（CRUD 表单 + inline 测试）、报告

**报告**（Day 10）
- 单 HTML 文件含 Executive Summary / 关键问题 / 趋势 / 配置审计 / Roofline / 规则汇总
- Plotly via CDN 默认（~30KB），inline 选项（~3.5MB 邮件离线）
- Roofline 散点 + 理论上界折线（compute roof + memory bandwidth roof）
- 配置审计含 4 条启发式建议（batch / prefix cache / KV / padding）

**性能**（Day 12）
- 微基准：push_metric 156ns mean，collect 22μs，record 12μs
- CI regression test 守 hot path 预算
- 自我观测指标：`pping_lang.overhead.record_us` / `rule_eval_ms` / `sink.dropped_total`

**CUPTI kernel 级剖析**（阶段 1a）
- 注入式 `.so`（`CUDA_INJECTION64_PATH`）抢占 CUPTI subscriber，避开 PyTorch Kineto 冲突
- Activity API per-kernel GPU 时间分解 + 算子分类（gemm / attention / comm / norm / activation / …）
- Kernel tab：占比、主导 stall、CUDA Graph 覆盖率、launch 频率、自身开销（cb_ms / dropped）
- 服务端 findings：compute-bound / launch-bound / memcpy 开销 / CUDA Graph 覆盖低 等结论

**PC Sampling**（阶段 2）
- SASS 指令 + stall reason 采样，库内预聚合（不全量回流，守 5% 开销）
- 多进程 `vllm serve` 工作：EngineCore 进程驱动 PC sampling + 跨进程共享 JSON 回流前端
- Deep Evidence 面板：warp 三态、全局 stall 分解、PerfWorks 原始 reason 下钻、测量方法卡
- 按需触发 5–30s 采样的深度分析端点

**行级归因**（阶段 3，对标 zymtrace 阶段 2）
- PC → 源码行 / SASS 热点双轨；源码行显示该行 Python 原文
- kernel → Python launch 栈（向外归因 MVP），派发器噪音清洗后露出真算子名（addmm/mm/linear）
- decode 覆盖 cuBLAS GEMV / FlashInfer 采样；区分 `_ZN` mangled vs Triton

**Roofline 调优地图**（P0-C 实测闭环）
- 一键压测扫并发，实测 scaling 曲线叠加理论 envelope（外推 → 实测）
- scaling verdict 用扩展效率口径（相对基线加速比 / 理想加速比，eff<70% = 收益递减点）
- 散点聚合成簇心 + operating point / batch scaling envelope / ridge point 语义标签

**压测**（bench）
- OpenAI 协议静态压测（chat / completions）+ SLO 校验，结果落 DuckDB 可回看
- 并发 sweep（CLI）
- 任选两条 run 逐指标 A/B/Δ% 对比 + 图表化对比面板（每卡 A/B 双横条 + Δ% 优劣徽标）

**打包与硬件**
- `pping-vllm serve <model>` 一条命令，`.so` 现编（cu12/cu13 自适应，编不出自动降级）
- roofline 峰值改从 CUDA 设备属性现读现算（`read_gpu_peak`），回退型号表 / env
- GPU 名现读 `torch.cuda.get_device_name` + 可选部署标签（`PPING_DEPLOY_TAG`）

**Dashboard 演进**
- Kernel tab 扩成完整诊断面板；HTTP API 扩到 36 端点（含 kernels / deep_evidence / roofline / scaling_sweep / bench）
- KPI 卡 + TTFT/TPOT 分位分布条 + 用户延迟趋势（平均为主、p99 退为浅线参考）

### Changed
- 项目命名从 `llmprof` 改为 `pping-lang`（PyPI 名带连字符，Python 模块带下划线）
- roofline 从「分类器」升级为「调优地图」：operating point + batch scaling envelope，文案改工程口吻
- 用户延迟趋势 / TTFT·TPOT 改以平均为主显，p50/p95/p99 退为分布/参考

### Fixed
- cudagraph 采样：HW 缓冲溢出会永久楔死 CUPTI 会话 → 默认周期 2^12→2^16；capture 打断早 prime 采样 → 检测空窗自愈 reprime
- correlate 拼 `dirName`+`fileName` 成全路径（否则只得 basename，读不到源码原文）
- UI no-cache（`dashboard.js` / `css` / `index.html`，防 F5 复用旧版致新功能不渲染）
- Deep Evidence findings 卡片黏连（`x-show` 与内联 `display:flex` 同元素冲突，拆两层）

### Tests
- 初版 209 测试（采集 / 诊断 / Sink / API CRUD / e2e marquee / 性能 regression）
- 后续新增覆盖：bench（runner / schema / SLO / sweep / store / prompts）、CUPTI collector、scaling、API server 生命周期

---

模板（未来 release）：

## [0.x.0] — YYYY-MM-DD

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
