# ==================================================
# 该模块由 PM&算法逻辑 开发，负责自定义UX指标的实现
# 预留接口，当前为占位实现，用于本地测试，正式实现完成后替换此文件即可
# (注：现已替换为对接DeepEval的真实GEval指标工厂函数)
# ==================================================

from .geval_metrics import (
    get_trust_metric,
    get_satisfaction_metric,
    get_understanding_metric,
    get_control_metric,
    get_efficiency_metric,
    get_cognitive_load_metric,
    get_safety_metric,
    get_dependency_metric,
    get_anthropomorphism_metric,
    get_empathy_metric,
    create_metric
)

__all__ = [
    "get_trust_metric",
    "get_satisfaction_metric",
    "get_understanding_metric",
    "get_control_metric",
    "get_efficiency_metric",
    "get_cognitive_load_metric",
    "get_safety_metric",
    "get_dependency_metric",
    "get_anthropomorphism_metric",
    "get_empathy_metric",
    "create_metric"
]