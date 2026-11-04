"""pping-lang CLI：统一入口 `pping-lang <子命令>`(demo/serve/version)。

    pping-lang demo            离线 demo(无需 GPU/vLLM)
    pping-lang serve <args>    等价 pping-vllm:设 Kernel 级采集 env 后 exec vllm serve

pping-vllm：一条命令把 pping-lang 接到 vllm(纯 pip 装上即用)。

    pping-vllm serve <model> [vllm 的参数...]

等价于：现编/取缓存 libppingcupti.so → 设好注入 + PC sampling 的环境变量 → `vllm serve ...`。
之后 CUDA 驱动在 cuInit 注入 .so、EngineCore 的 general_plugin 采样、前端 dashboard 读结果
(见 engine_pcs.py)。.so 编不出来时自动降级：仍有 KPI / roofline / NVML / 诊断,只是没 Kernel 级采集。
"""
from __future__ import annotations

import os
import shutil
import sys


def _extract_opt(args: list[str], name: str) -> str | None:
    """从 vllm 参数里取 `--name VALUE` 或 `--name=VALUE` 的值(末次出现优先)。"""
    val: str | None = None
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith(name + "="):
            val = a.split("=", 1)[1]
    return val


_USAGE = """用法: pping-lang <子命令>
  pping-lang demo            离线 demo(无需 GPU/vLLM,合成指标 → 「双低」诊断)
  pping-lang serve <args>    设 Kernel 级采集后 exec `vllm serve <args>`(等价 pping-vllm)
  pping-lang version         版本
(兼容别名: pping-demo / pping-vllm)"""


def main(argv: list[str] | None = None) -> int:
    """`pping-lang <子命令>` 分发。console script / python -m pping_lang 共用。"""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "demo":
        from pping_lang.demo import main as demo_main  # noqa: PLC0415 — 延迟导入便于测试替身
        return demo_main()
    if cmd == "serve":
        pping_vllm_main(rest)          # exec 成功不返回;失败自己 raise
        return 0
    if cmd == "version":
        from pping_lang import __version__  # noqa: PLC0415
        print(f"pping-lang {__version__}")
        return 0
    print(f"未知子命令 {cmd!r}\n{_USAGE}", file=sys.stderr)
    return 2


def pping_vllm_main(args: list[str] | None = None) -> None:
    args = sys.argv[1:] if args is None else args

    # dashboard host 默认跟随 vllm 的 --host:`serve` 本就是对外服务,容器/远程里 vllm
    # 监听 0.0.0.0 时 dashboard 也该如此,否则 docker 端口转发够不到 127.0.0.1 的监听
    # (插件默认 127.0.0.1 是安全保守值;仅在用户显式给了 vllm --host 时跟随,且 setdefault
    # 让显式的 PPING_LANG_API_HOST 仍可覆盖)。
    vllm_host = _extract_opt(args, "--host")
    if vllm_host:
        os.environ.setdefault("PPING_LANG_API_HOST", vllm_host)

    # 1) 确保 .so(现编或取缓存),设注入 + PCS 环境(均 setdefault,用户可覆盖)
    try:
        from pping_lang.native.builder import ensure_so
        so, cupti_libdir = ensure_so()
        os.environ.setdefault("CUDA_INJECTION64_PATH", so)  # 驱动 cuInit 时加载 → EngineCore 抢 subscriber
        os.environ.setdefault("PPING_LANG_PCS_SO", so)
        if cupti_libdir:
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = cupti_libdir + (f":{ld}" if ld else "")
        os.environ.setdefault("PPING_LANG_ENABLE_PCS", "1")
        # P3 行级归因：开 per-PC 直方图(源码行/SASS 热点)。每样本多一次 hash,默认开,
        # 想省这点开销可显式设 0。
        os.environ.setdefault("PPING_LANG_PCS_PC_HIST", "1")
        # P3 launch 栈：kernel ← 启动它的 host 代码(向外归因,闭源 GEMM ← nn.Linear)。
        # 首见某 kernel 抓一次栈,开销极小;默认开。
        os.environ.setdefault("PPING_LANG_PCS_LAUNCH_STACK", "1")
        print(f"[pping-vllm] PC sampling 就绪：.so={so}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[pping-vllm] .so 准备失败,降级(无 Kernel 级采集,其余照常):{e}", file=sys.stderr)

    # 2) exec vllm,透传全部参数(env 已设好,vllm 及其 EngineCore 子进程继承)
    vllm = shutil.which("vllm") or os.path.join(os.path.dirname(sys.executable), "vllm")
    if not os.path.exists(vllm):
        raise SystemExit("[pping-vllm] 找不到 vllm,请先 pip install vllm")
    os.execv(vllm, [vllm, *args])
