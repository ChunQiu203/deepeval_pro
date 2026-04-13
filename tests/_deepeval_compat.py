# =============================================================================
# 测试辅助：兼容 DeepEval 将 GEval.model 从 str 包装为 GPTModel 等对象
# =============================================================================

from __future__ import annotations

from typing import Any


def metric_judge_model_name(metric: Any) -> str:
    """从指标对象上取出「裁判模型名」字符串，便于单元断言。"""
    raw = getattr(metric, "model", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    fn = getattr(raw, "get_model_name", None)
    if callable(fn):
        return str(fn())
    return str(raw)
