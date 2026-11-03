"""运行时把打包的 ppingcupti.cpp 编成 libppingcupti.so —— 实现"纯 pip 装上即用"。

在目标(vLLM)环境里现编：自动探测 cu12/cu13 的 libcupti + cupti 头 + cuda.h + libcuda
stub(逻辑同 deploy/runw/build_so.sh),g++ 编译,缓存到 ~/.pping-lang/lib/。

入口：`ensure_so()` → 返回 (so_path, cupti_libdir);失败抛 BuildError(调用方降级)。
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import sysconfig
from pathlib import Path

logger = logging.getLogger("pping_lang.native.builder")

_NATIVE_DIR = Path(__file__).resolve().parent
_CPP = _NATIVE_DIR / "ppingcupti.cpp"


class BuildError(RuntimeError):
    pass


def _cache_so() -> Path:
    root = Path(os.environ.get("PPING_LANG_HOME", str(Path.home() / ".pping-lang")))
    return root / "lib" / "libppingcupti.so"


def _glob1(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _nvidia_roots() -> list[str]:
    """nvidia pip 包根目录候选(顺序即优先级):当前解释器在前(conda/venv 布局,
    torch 的 nvidia-cuda-cupti 装在 $CONDA_PREFIX 的 site-packages 下),vllm 官方
    镜像的系统 dist-packages 在后。只认后者时 conda 环境探测必失败 → 静默降级。"""
    return [
        os.path.join(sysconfig.get_paths()["purelib"], "nvidia"),
        "/usr/local/lib/python3*/dist-packages/nvidia",
    ]


def _detect() -> dict[str, str]:
    """探测编译所需路径。任一缺失抛 BuildError。"""
    mysite, site = _nvidia_roots()
    cupti_lib = _glob1([
        f"{mysite}/*/lib/libcupti.so.1*",
        f"{site}/*/lib/libcupti.so.1*",
        "/usr/local/cuda*/lib64/libcupti.so.1*",
        "/usr/local/cuda*/targets/*/lib/libcupti.so.1*",
    ])
    if not cupti_lib:
        raise BuildError("找不到 libcupti.so.1*(需 nvidia-cuda-cupti-cuXX 或 CUDA toolkit)")
    cupti_h = _glob1([
        f"{mysite}/*/include/cupti_pcsampling.h",
        f"{site}/*/include/cupti_pcsampling.h",
        "/usr/local/cuda*/include/cupti_pcsampling.h",
        "/usr/local/cuda*/extras/CUPTI/include/cupti_pcsampling.h",
    ])
    if not cupti_h:
        raise BuildError("找不到 cupti_pcsampling.h")
    # cuda.h：必须挑**同时有 crt/host_defines.h** 的 include 目录 —— cupti 头会层层
    # include 到 crt/host_defines.h;triton 自带整套 CUDA 头(含 crt/),而 nvidia/cuXX/
    # include 只有 cupti+部分头、缺 crt/ → 选错就 "fatal error: crt/host_defines.h"。
    cuda_inc: str | None = None
    purelib = sysconfig.get_paths()["purelib"]
    for ch in sorted(set(
        glob.glob(os.path.join(purelib, "triton/backends/nvidia/include/cuda.h"))
        + glob.glob("/usr/local/lib/python3*/dist-packages/triton/backends/nvidia/include/cuda.h")
        + glob.glob("/usr/local/cuda*/include/cuda.h")
        + glob.glob(f"{site}/*/include/cuda.h")
    )):
        d = os.path.dirname(ch)
        if os.path.exists(os.path.join(d, "crt", "host_defines.h")):
            cuda_inc = d
            break
    if cuda_inc is None:
        raise BuildError("找不到含 crt/host_defines.h 的 cuda.h include 目录(通常 triton 自带)")
    # libcuda 链接：优先 stub,否则真 .so.1
    stub = _glob1([
        "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
        "/usr/local/cuda*/lib64/stubs/libcuda.so",
        "/usr/lib/**/stubs/libcuda.so",
    ])
    if stub:
        link_cuda = ["-L" + os.path.dirname(stub), "-lcuda"]
    else:
        real = _glob1(["/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib/**/libcuda.so.1"])
        if not real:
            raise BuildError("找不到 libcuda(stub 或 .so.1)")
        link_cuda = ["-L" + os.path.dirname(real), "-l:libcuda.so.1"]
    return {
        "cupti_libdir": os.path.dirname(cupti_lib),
        "cupti_soname": os.path.basename(cupti_lib),
        "cupti_inc": os.path.dirname(cupti_h),
        "cuda_inc": cuda_inc,
        "link_cuda": link_cuda,  # type: ignore[dict-item]
    }


def ensure_so(force: bool = False) -> tuple[str, str]:
    """确保 libppingcupti.so 存在(必要时现编),返回 (so 路径, cupti 库目录)。

    缓存命中(.so 比 .cpp 新)直接返回。`PPING_LANG_PCS_SO` 显式指定则用它。
    """
    explicit = os.environ.get("PPING_LANG_PCS_SO")
    det_libdir = ""
    if explicit and os.path.exists(explicit) and not force:
        try:
            det_libdir = _detect()["cupti_libdir"]
        except BuildError:
            pass
        return explicit, det_libdir

    if not _CPP.exists():
        raise BuildError(f"打包的源码缺失：{_CPP}(wheel 未含 native/*.cpp?)")

    det = _detect()
    out = _cache_so()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 缓存命中：.so 比 .cpp 和 .h 都新
    if out.exists() and not force:
        hdr = _NATIVE_DIR / "ppingcupti.h"
        newest_src = max(_CPP.stat().st_mtime, hdr.stat().st_mtime if hdr.exists() else 0)
        if out.stat().st_mtime >= newest_src:
            return str(out), det["cupti_libdir"]

    cmd = [
        "g++", "-O2", "-fPIC", "-std=c++17",
        f"-I{det['cupti_inc']}", f"-I{det['cuda_inc']}", f"-I{_NATIVE_DIR}",
        "-shared", f"-L{det['cupti_libdir']}", f"-Wl,-rpath,{det['cupti_libdir']}",
        "-o", str(out), str(_CPP),
        f"-l:{det['cupti_soname']}", *det["link_cuda"], "-pthread", "-ldl",
    ]
    logger.info("[pping-lang] 编译 libppingcupti.so:cupti=%s @ %s",
                det["cupti_soname"], det["cupti_libdir"])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"g++ 编译失败 rc={proc.returncode}:\n{proc.stderr[-2000:]}")
    logger.info("[pping-lang] libppingcupti.so 编好 → %s", out)
    return str(out), det["cupti_libdir"]
