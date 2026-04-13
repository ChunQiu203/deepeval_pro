import os
from typing import List, Dict, Any

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

from ux_evaluator.dataset.loader import TestCase


class MyCustomModel(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key, base_url):
        # 使用 LangChain 的 ChatOpenAI 来处理所有 OpenAI 格式的接口
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=base_url  # 这里的 base_url 会直接指向你的模型供应商
        )

    def load_model(self):
        return self.model

    # deepeval 在评估指标时会调用这个同步方法
    def generate(self, prompt: str, *args, **kwargs) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    # deepeval 在异步评估时会调用这个异步方法
    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom LLM via LangChain"


class UXJudge:
    def __init__(self, model: str, retry: int = 3, base_url: str = None):
        """兼容外层 batch_evaluate.py 的类初始化接口"""
        self.model_name = model
        self.retry = retry

        # --- 实例化并使用 ---
        # 从环境变量中获取 API Key
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 API Key，请确保已设置环境变量")

        # 假设你想用 DeepSeek 或者其他兼容接口
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        self.custom_model = MyCustomModel(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url
        )

    # 入口函数
    def batch_evaluate(self, test_cases: List[TestCase], metrics: List[Any]) -> List[Dict[str, Any]]:
        """
        运行情感支持大模型回答的多维度评估。
        直接使用传入的 metrics 列表进行批量测算（对接全局定义的评估指标）。
        """
        print(f"🚀 开始评估回复的多维度表现...")

        # 1. 构建测试用例 (Test Case)，将自定义的 TestCase 转换为 DeepEval 的 LLMTestCase
        deepeval_cases = []
        for tc in test_cases:
            deepeval_cases.append(LLMTestCase(
                input=tc.input,
                actual_output=tc.actual_output,
                retrieval_context=tc.retrieval_context if tc.retrieval_context else None
            ))

        # 3. 运行评估
        results = evaluate(
            test_cases=deepeval_cases,
            metrics=metrics,
        )

        # 4. 提取并打印每个具体指标的得分摘要
        print("\n📊 评估得分摘要:")

        formatted_results = []
        # 确保 results 存在且包含测试结果
        if results and hasattr(results, 'test_results'):
            for i, test_case_result in enumerate(results.test_results):
                metric_dict = {}
                overall_passed = True

                # 遍历该用例下所有指标的计算数据
                for metric_data in test_case_result.metrics_data:
                    # 这里的 name 会自动匹配指标类定义的名称（如 Trustworthiness）
                    name = metric_data.name
                    score = metric_data.score
                    print(f"- {name:<20}: {score}")

                    # 兼容不同版本的 DeepEval，优先取 success
                    passed = getattr(metric_data, 'success', False)

                    metric_dict[name] = {
                        "score": score,
                        "passed": passed
                    }
                    if not passed:
                        overall_passed = False

                formatted_results.append({
                    "test_case": test_cases[i].to_dict(),
                    "metrics": metric_dict,
                    "overall_passed": overall_passed
                })
        else:
            print("❌ 未获取到有效的评估结果数据。")

        return formatted_results


if __name__ == "__main__":
    from ux_evaluator.metrics.geval_metrics import get_trust_metric, get_understanding_metric

    sample_input = "我最近真的觉得快崩溃了，工作永远做不完，家里还有一堆烂摊子，感觉自己好失败。"

    sample_output = "听起来你现在背负了非常大的压力，工作和家庭的双重重担让你感到喘不过气，这种感觉一定很辛苦吧。在这么困难的情况下你还在努力支撑，已经非常不容易了。你现在最想先从哪一件小事开始理清头绪呢？如果你愿意，我们可以一起慢慢梳理。"

    # 模拟本地测试调用
    tc = TestCase(input=sample_input, actual_output=sample_output)

    judge = UXJudge(model="qwen3.5-flash")

    # 动态组装指标
    test_metrics = [
        get_trust_metric(judge.custom_model),
        get_understanding_metric(judge.custom_model)
    ]

    # 现在调用时只需传入测试用例和指标列表即可
    judge.batch_evaluate(
        test_cases=[tc],
        metrics=test_metrics
    )