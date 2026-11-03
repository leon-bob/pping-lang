# Autopilot 真机调优 Quickstart

在**你自己的 GPU 机器**上跑一次真实的 Autopilot 调优 session：Agent 在一次性容器沙盒里迭代
「诊断 → 改一个参数 → 压测 → 留下/回滚」，最后给出实测验证过的 `vllm serve` 推荐命令。

> **先读这三行**
> 1. **你的 serve 会被暂停**：单卡机器上候选沙盒要独占 GPU，session 期间指定的主 serve 容器会被 `docker stop`，结束后自动重启（`--restore-cmd` 可自定义恢复动作）。
> 2. **不碰生产语义**：产出只是「推荐配置 + 全程推理轨迹」，应用到生产永远是你的手动决定。
> 3. **压测是唯一真相**：Agent 无权声明压测没证实的收益；变差 / 破 SLA 一律自动回滚。

## 前置条件

- 单卡 NVIDIA GPU 机器（实测环境：RTX 5060 Ti 16GB / Blackwell；其他架构理论可跑、未实测）
- Docker + NVIDIA Container Toolkit（`docker run --gpus all` 可用）
- Python ≥ 3.10，`pip install 'pping-lang[bench]'`（或源码 `pip install -e '.[bench]'`；Debian/Ubuntu 系统 Python 受 PEP 668 保护，先进 venv，见 README 快速上手）
- 一个 LLM API key 作为 Agent（OpenAI 兼容端点 / Anthropic / Kimi Coding 均可）
- vLLM 官方镜像：`docker pull vllm/vllm-openai:v0.21.0`（动作空间按 0.21 校准）

## 一条命令跑一个 session

```bash
export AGENT_KEY=sk-...          # 你的 LLM API key（只进内存与请求，不落盘）

python -m pping_lang.autopilot.run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --image vllm/vllm-openai:v0.21.0 --serve-cmd "" \
  --session-dir ./autopilot-sessions \
  --agent-base-url https://api.deepseek.com/v1 \
  --agent-model deepseek-chat \
  --rounds 6 --minutes 25 \
  --target throughput --ttft 8000 --tpot 50 \
  --bench-concurrency 48 --bench-repeats 3
```

跑完终端会打印逐轮判定和推荐命令，完整轨迹落在 `./autopilot-sessions/ap-*.jsonl`。

关键参数说明：

| 参数 | 说明 |
|---|---|
| `--serve-cmd ""` | 官方镜像的 entrypoint 已是 `vllm serve`，留空即可；用自带 pping 插件的镜像时按镜像形态给 |
| `--serve-container <名字>` | 机器上有在跑的 serve 容器时给：session 期间停它腾卡、结束自动重启 |
| `--restore-cmd '<shell>'` | serve 靠 `docker exec` 起（容器主进程是 `sleep infinity`）的部署形态必须给完整恢复命令 |
| `--bench-concurrency` | **必须明显高于**基线 `max_num_seqs`（不能只是等于！等于时 waiting 永远测不出 >0，Agent 会误判"准入闸没绑定"而不去试提并发——留出 1.5× 左右余量） |
| `--bench-repeats 3` | median-of-3 去噪：p99 延迟在 SLA 边界不再因单次离群翻转判定 |
| `--baseline-max-num-seqs / --baseline-gpu-util` | 基线起点；默认 32 / 0.70（故意朴素的常见配置） |
| `--workload <形态>` | 业务负载形态：一次性给 bench 负载参数（prompt/output tokens、并发）+ SLA 默认值，让 bench、SLA、诊断说同一种负载语言，见下方形态表；不给则退回各参数各自的旧默认（等价 `custom`） |
| `--quality-gate` | 放开 T2 质量类参数（kv-cache fp8 / 量化 / 投机解码），带输出等价检查 |
| `--dash-port 8013` | 候选容器带 pping 插件时发布其 dashboard：Agent 直接读**真诊断**而非启发式 |
| `--agent-lang zh\|en` | Agent 自由文本应答语言（rationale/reason/思考过程）；默认 `zh`。不指定时 LLM 会跟着上下文里的英文技术标识符默认答英文，即使 system prompt 本身是中文 |
| `--search-mode agent\|grid\|bo` | 参数搜索策略；默认 `agent`（LLM 诊断驱动）。`grid`/`bo` 用 `--search-width` 控制每个参数展开的候选档数 |

### `--workload` 负载形态

显式 flag（如 `--bench-concurrency`）> 形态默认 > 各参数原有兜底；`custom` 不施加任何默认，行为等价不给这个 flag。

| 形态 | 说明 | bench 默认（prompt/output tokens · 并发） | SLA 默认（TTFT/TPOT/E2E p99） |
|---|---|---|---|
| `chat` | 对话（短进短出） | 500 / 128 · 64 | 1000ms / 50ms / 3000ms |
| `rag` | RAG 问答（长 prompt） | 4000 / 256 · 16 | 3000ms / 50ms / 8000ms |
| `agent` | Agent 工具循环 | 2000 / 512 · 32 | 1000ms / 50ms / 15000ms |
| `reasoning` | 长推理（长输出） | 1000 / 4096 · 16 | 1000ms / 30ms / 90000ms |
| `code` | 代码补全（硬延迟闸） | 300 / 128 · 16 | 100ms / 20ms / 2000ms |
| `custom` | 全手动，不注入任何默认 | — | — |

Agent 供应商任选其一：

```bash
# OpenAI 兼容(DeepSeek/OpenRouter/本地 vLLM…)
--agent-base-url https://openrouter.ai/api/v1 --agent-model anthropic/claude-sonnet-4
# Anthropic 官方
--agent-provider anthropic --agent-model claude-sonnet-4
# 不给 agent 参数 → 确定性启发式(StubAgent),可用于无 key 冒烟
```

## 结果怎么读

- 每轮记录 = 诊断快照（regime + 证据）→ Agent 假设（挂在证据上）→ 实测 scorecard → `kept / reverted / tie`；
- **`reverted` 是特性不是故障**：×5 的吞吐破了 TPOT SLA 一样会被回滚——这正是可信的来源；
- Agent 判「已近最优」会给出署名理由（如"准入闸没绑定，瓶颈在提供的负载"），不编造收益；
- 机器上若跑着带 pping 插件的 dashboard，session 结束后 Autopilot tab 自动显示这条轨迹。

## 已知边界（诚实声明）

- 动作空间按 **vLLM 0.21 (V1)** 校准（默认值、`unsupported` 参数标记均针对该版本）；
- 单卡单 session；多卡 TP/PP 调优在路线图 M3；
- 收益与起点强相关：朴素基线常见 ×2~6，已调过的基线只应期待诚实的边际收益；
- 每轮成本约 5-7 分钟（容器冷启动 + 压测），6 轮 session ≈ 30-45 分钟；
- 浏览器 UI 一键真调优依赖 host 侧 bridge 编排（含调优期间面板端口接管），当前作为参考实现随部署脚本提供，通用化打包在路线图上。
