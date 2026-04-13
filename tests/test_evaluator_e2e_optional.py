# =============================================================================
# 测试：端到端（真实调用裁判模型，可选）
# =============================================================================
# 作用：验证 resolve + evaluate 在真实 DeepEval 流程下能跑通（会消耗 API 额度）。
#
# 如何运行：
#   1）配置环境变量 OPENAI_API_KEY（DeepEval 默认常用 OpenAI 兼容接口）
#   2）在项目根目录执行：
#        pytest tests/test_evaluator_e2e_optional.py -q -m integration
#
# 若未设置密钥，测试会自动跳过，避免 CI 或本地误跑产生费用。
# =============================================================================

from __future__ import annotations

import os

import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ux_evaluator.judge.evaluator import UXEvaluator
from ux_evaluator.judge.specs import CustomGEvalSpec


pytestmark = pytest.mark.integration


def _real_openai_key_configured() -> bool:
    """避免 conftest 注入的占位密钥触发真实请求并产生 401。"""
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return False
    if "placeholder" in key.lower():
        return False
    return True


@pytest.mark.skipif(
    not _real_openai_key_configured(),
    reason="需要真实 OPENAI_API_KEY（非 tests/conftest 占位值）才能跑 LLM-as-a-Judge",
)
def test_real_evaluate_smoke() -> None:
    """小样本冒烟：一条用例 + 一条 GEval；模型名可用环境变量覆盖。"""
    judge = os.getenv("UX_EVAL_JUDGE_MODEL", "gpt-4.1-mini")
    ev = UXEvaluator(default_judge=judge)

    tc = LLMTestCase(
        input="1+1等于几？",
        actual_output="等于2。",
    )
    spec = CustomGEvalSpec(
        name="CorrectnessSmoke",
        criteria="判断回答是否正确且简洁。",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.3,
    )

    # 不 assert 具体分数结构（deepeval 版本间返回类型可能不同），只要求不抛异常且能拿到结果对象
    result = ev.run([tc], [spec])
    assert result is not None
