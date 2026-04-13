# =============================================================================
# 测试：resolve_metrics（不调用大模型 API）
# =============================================================================
# 作用：验证「声明 / 工厂 / 已有指标」能否正确解析为 DeepEval 指标实例，以及 default_judge 是否生效。
# 运行：在项目根目录执行  pytest tests/test_resolve_metrics.py -q
# =============================================================================

from __future__ import annotations

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from tests._deepeval_compat import metric_judge_model_name
from ux_evaluator.judge.resolve import resolve_metrics
from ux_evaluator.judge.specs import CustomGEvalSpec


def test_custom_geval_spec_requires_criteria_or_steps() -> None:
    """缺少 criteria 与 evaluation_steps 时应立即报错（与 DeepEval GEval 规则一致）。"""
    with pytest.raises(ValueError, match="criteria"):
        CustomGEvalSpec(name="bad", criteria=None, evaluation_steps=None)


def test_resolve_custom_spec_to_geval() -> None:
    """CustomGEvalSpec 应被解析为 GEval 实例。"""
    spec = CustomGEvalSpec(
        name="Politeness",
        criteria="Score how polite the reply is.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    out = resolve_metrics([spec])
    assert len(out) == 1
    assert isinstance(out[0], GEval)
    assert out[0].name == "Politeness"
    assert out[0].threshold == 0.7


def test_default_judge_applied_to_spec_without_model() -> None:
    """spec 未写 model 时，应使用 resolve_metrics 的 default_judge。"""
    spec = CustomGEvalSpec(
        name="Trust",
        criteria="Score perceived trustworthiness.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    out = resolve_metrics([spec], default_judge="gpt-4.1-mini")
    assert metric_judge_model_name(out[0]) == "gpt-4.1-mini"


def test_spec_model_overrides_default_judge() -> None:
    """spec 自带 model 时，不应被 default_judge 覆盖。"""
    spec = CustomGEvalSpec(
        name="X",
        criteria="...",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model="override-model",
    )
    out = resolve_metrics([spec], default_judge="gpt-4.1-mini")
    assert metric_judge_model_name(out[0]) == "override-model"


def test_factory_returns_metric() -> None:
    """支持传入无参工厂：延迟构造 GEval。"""
    def _factory() -> GEval:
        return GEval(
            name="Factory",
            criteria="Always return a score.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        )

    out = resolve_metrics([_factory], default_judge="judge-from-default")
    assert len(out) == 1
    assert isinstance(out[0], GEval)
    assert metric_judge_model_name(out[0]) == "judge-from-default"
