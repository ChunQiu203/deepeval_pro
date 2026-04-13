# =============================================================================
# 测试：UXEvaluator.run（Mock evaluate，不调用大模型）
# =============================================================================
# 作用：验证最终会调用 deepeval.evaluate，且传入的 metrics 已是解析后的具体指标对象。
# 运行：pytest tests/test_evaluator_run.py -q
# =============================================================================

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from tests._deepeval_compat import metric_judge_model_name
from ux_evaluator.judge.evaluator import UXEvaluator
from ux_evaluator.judge.specs import CustomGEvalSpec


def test_ux_evaluator_invokes_deepeval_evaluate_with_concrete_metrics() -> None:
    mock_evaluate = MagicMock(return_value={"mocked": True})

    with patch("ux_evaluator.judge.evaluator.get_evaluate", return_value=mock_evaluate):
        ev = UXEvaluator(default_judge="gpt-4.1-mini")
        # LLMTestCase 仅需 input + actual_output 即可参与多数指标
        from deepeval.test_case import LLMTestCase

        tc = LLMTestCase(input="你好", actual_output="您好，有什么可以帮您？")

        spec = CustomGEvalSpec(
            name="Politeness",
            criteria="评估礼貌程度。",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        )

        result = ev.run([tc], [spec])

    assert result == {"mocked": True}
    mock_evaluate.assert_called_once()
    kwargs = mock_evaluate.call_args.kwargs
    assert kwargs["test_cases"] == [tc]
    assert len(kwargs["metrics"]) == 1
    assert isinstance(kwargs["metrics"][0], GEval)
    assert metric_judge_model_name(kwargs["metrics"][0]) == "gpt-4.1-mini"
