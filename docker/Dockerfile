# pping-lang × vLLM 快速开始镜像(README「接入 vLLM · 方式二」)。
# 配对固定:vLLM v0.21.0(真机验证版本)× pping-lang —— 升 vLLM 时连 base tag 一起换并重验。
# 构建(仓库根为上下文):docker build -t pping-vllm:0.21-0.1.2 -f docker/Dockerfile .
FROM vllm/vllm-openai:v0.21.0

ARG PPING_LANG_VERSION=0.1.6

# 编译链盘点(2026-11 在 v0.21.0 镜像上实测):
#   libcupti.so.13 + cupti_pcsampling.h —— 已有(torch 依赖链的 nvidia/cu13 带入)
#   triton 的 cuda.h(含 crt/)          —— 已有(vllm → triton)
#   libcuda stub(cuda-cudart-dev-13-0) —— 已有
#   g++                                —— 缺,本层补上(现编 libppingcupti.so 的编译器)
#   换 cu12 base 时按同法盘点,cudart-dev cu12 需 apt 补:cuda-cudart-dev-12-x
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/* \
 && uv pip install --system "pping-lang==${PPING_LANG_VERSION}"

# 构建期预编译 .so:缓存(~/.pping-lang/lib)烙进镜像层,容器启动零编译;
# 探测/编译失败直接 fail build —— 早暴露,不带病出厂(运行时另有自动降级兜底)
# (上游最终镜像里确定存在的是 uv + python3.12,裸 pip/python3 不保证)
RUN python3.12 -c "from pping_lang.native.builder import ensure_so; print(ensure_so())"

# 容器内绑 0.0.0.0:曝光与否交给 `docker run -p`(边界在端口映射,不在进程);
# 插件侧是 setdefault 语义,显式给 vllm --host / PPING_LANG_API_HOST 仍可覆盖
ENV PPING_LANG_API_HOST=0.0.0.0

EXPOSE 8000 8765

# 与上游 entrypoint(`vllm serve`)同形,套上 pping-vllm 薄包装,参数原样透传:
#   docker run --gpus all -p 8000:8000 -p 8765:8765 pping-vllm:0.21-0.1.2 <model>
ENTRYPOINT ["pping-vllm", "serve"]
