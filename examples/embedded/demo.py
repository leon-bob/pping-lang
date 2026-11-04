"""兼容 shim —— 实现已移入包内(`pping_lang/demo.py`),随 wheel 分发。

新跑法(无需仓库 clone):
    pping-demo                     # 或 python -m pping_lang.demo
旧跑法(仓库内,仍可用):
    python examples/embedded/demo.py
"""
from pping_lang.demo import main

if __name__ == "__main__":
    raise SystemExit(main())
