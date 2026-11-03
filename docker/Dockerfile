# pping-lang × vLLM 快速开始镜像(README「接入 vLLM · 方式二」)。
# 配对固定:vLLM v0.21.0(真机验证版本)× pping-lang —— 升 vLLM 时连 base tag 一起换并重验。
# 构建(仓库根为上下文):docker build -t pping-vllm:0.21-0.1.0 -f docker/Dockerfile .
FROM vllm/vllm-openai:v0.21.0

# 换 cu12 base 时同步改这两个(co12: cudart-dev-12-x / nvidia-cuda-cupti-cu12)
ARG CUDA_APT=13-0
ARG CUPTI_PIP=cu13
ARG PPING_LANG_VERSION=0.1.0

# Kernel 级 PC Sampling 的编译链,补齐 base 镜像缺的三样:
#   g++               现编 libppingcupti.so 的编译器
#   cuda-cudart-dev   libcuda stub —— 仅链接期用;真 libcuda 由 nvidia 运行时注入,build 时不存在
#   nvidia-cuda-cupti libcupti + cupti_pcsampling.h(torch 未必全带,显式装保证探测必中)
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends g++ cuda-cudart-dev-${CUDA_APT} \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir nvidia-cuda-cupti-${CUPTI_PIP} "pping-lang==${PPING_LANG_VERSION}"

# 构建期预编译 .so:缓存(~/.pping-lang/lib)烙进镜像层,容器启动零编译;
# 探测/编译失败直接 fail build —— 早暴露,不带病出厂(运行时另有自动降级兜底)
RUN python3 -c "from pping_lang.native.builder import ensure_so; print(ensure_so())"

# 容器内绑 0.0.0.0:曝光与否交给 `docker run -p`(边界在端口映射,不在进程);
# 插件侧是 setdefault 语义,显式给 vllm --host / PPING_LANG_API_HOST 仍可覆盖
ENV PPING_LANG_API_HOST=0.0.0.0

EXPOSE 8000 8765

# 与上游 entrypoint(`vllm serve`)同形,套上 pping-vllm 薄包装,参数原样透传:
#   docker run --gpus all -p 8000:8000 -p 8765:8765 pping-vllm:0.21-0.1.0 <model>
ENTRYPOINT ["pping-vllm", "serve"]
