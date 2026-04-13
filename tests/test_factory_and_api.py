# =============================================================================
# 测试：build_judge、evaluate_ux、create_ux_evaluator（不调用真实大模型）
# =============================================================================

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ux_evaluator.judge.api import create_ux_evaluator, evaluate_ux
from ux_evaluator.judge.backends.openai_judge_llm import OpenAICompatibleJudgeLLM
from ux_evaluator.judge.backends.resilience import MinIntervalGate
from ux_evaluator.judge.config import JudgeConfig, JudgeMode
from ux_evaluator.judge.factory import build_judge


def test_build_judge_deepeval_string_returns_model_name() -> None:
    cfg = JudgeConfig(mode=JudgeMode.DEEPEVAL_STRING, model="gpt-4.1-mini")
    assert build_judge(cfg) == "gpt-4.1-mini"


def test_build_judge_openai_compatible_returns_wrapper() -> None:
    cfg = JudgeConfig(
        mode=JudgeMode.OPENAI_COMPATIBLE,
        model="gpt-4.1-mini",
        api_key="sk-test",
        api_base=None,
        max_retries=2,
        min_interval_sec=0.0,
        timeout_sec=30.0,
    )
    judge = build_judge(cfg)
    assert isinstance(judge, OpenAICompatibleJudgeLLM)
    assert judge.get_model_name() == "gpt-4.1-mini"


def test_build_judge_litellm_requires_optional_dependency() -> None:
    pytest.importorskip("litellm", reason="未安装 litellm 时跳过 LITELLM 构造用例")
    cfg = JudgeConfig(mode=JudgeMode.LITELLM, model="openai/gpt-4o-mini")
    judge = build_judge(cfg)
    from ux_evaluator.judge.backends.litellm_judge_llm import LiteLLMJudgeLLM

    assert isinstance(judge, LiteLLMJudgeLLM)


def test_evaluate_ux_with_judge_config_mock_evaluate() -> None:
    mock_evaluate = MagicMock(return_value={"ok": True})
    with patch("ux_evaluator.judge.evaluator.get_evaluate", return_value=mock_evaluate):
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        from ux_evaluator.judge.specs import CustomGEvalSpec

        tc = LLMTestCase(input="hi", actual_output="hello")
        spec = CustomGEvalSpec(
            name="X",
            criteria="check",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        )
        cfg = JudgeConfig(mode=JudgeMode.DEEPEVAL_STRING, model="gpt-4.1-mini")
        out = evaluate_ux([tc], [spec], judge=cfg)
    assert out == {"ok": True}
    mock_evaluate.assert_called_once()


def test_create_ux_evaluator_binds_judge() -> None:
    ev = create_ux_evaluator(judge=JudgeConfig(mode=JudgeMode.DEEPEVAL_STRING, model="m"))
    assert ev._default_judge == "m"


def test_min_interval_gate_delays_second_call() -> None:
    import time

    gate = MinIntervalGate(0.08)
    t0 = time.monotonic()
    gate.acquire()
    gate.acquire()
    assert time.monotonic() - t0 >= 0.07
