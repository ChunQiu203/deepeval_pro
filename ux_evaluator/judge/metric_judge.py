#python -m ux_evaluator.judge.metric_judge

import os
from typing import List, Dict, Any
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

from ux_evaluator.dataset.loader import TestCase

import json
from collections import defaultdict
from ux_evaluator.metrics.geval_metrics import get_trust_metric, get_understanding_metric,create_metric

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_project_env() -> None:
    """Load .env from project root for IDE runs that don't inject env vars."""
    if load_dotenv is None:
        return

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


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

        _load_project_env()

        # --- 实例化并使用 ---
        # 从环境变量中获取 API Key
        self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未找到 API Key。请在环境变量或项目根目录 .env 中设置 QWEN_API_KEY（或 DASHSCOPE_API_KEY）。"
            )

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
    
    def evaluate_from_json(
        self,
        json_path: str,
        metrics: List[Any]
    ) -> Dict[str, Any]:
        """
        从 JSON 文件读取测试数据并进行评测（类内入口函数）
        + 返回平均分统计结果
        """

        

        print(f"📂 正在读取测试文件: {json_path}")

        # 1. 读取 JSON 文件
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2. 转换为 TestCase 列表
        test_cases = []
        for item in data:
            tc = TestCase(
                input=item.get("input", ""),
                actual_output=item.get("actual_output", ""),
                retrieval_context=item.get("retrieval_context", None)
            )
            test_cases.append(tc)

        print(f"✅ 读取完成，共 {len(test_cases)} 条测试用例")

        # 3. 执行评测
        results = self.batch_evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print("📊 正在计算平均分...")

        # 4. 统计平均分
        metric_sum = defaultdict(float)
        metric_count = defaultdict(int)

        overall_scores = []

        for case in results:
            case_total = 0
            case_metric_num = 0

            for metric_name, metric_info in case["metrics"].items():
                score = metric_info["score"]

                if score is not None:
                    metric_sum[metric_name] += score
                    metric_count[metric_name] += 1

                    case_total += score
                    case_metric_num += 1

            # 每个 case 的平均分（可选）
            if case_metric_num > 0:
                overall_scores.append(case_total / case_metric_num)

        # 各指标平均
        metric_avg = {
            name: metric_sum[name] / metric_count[name]
            for name in metric_sum
        }

        # 总体平均（所有 case 的平均）
        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        print("\n📈 平均得分结果:")
        for name, avg in metric_avg.items():
            print(f"- {name:<20}: {avg:.4f}")

        print(f"\n🌟 Overall Score: {overall_avg:.4f}")

        return {
            "detail_results": results,   # 每条样本详细结果
            "metric_average": metric_avg,  # 每个指标平均
            "overall_average": overall_avg  # 总体平均
        }
    def evaluate_from_files(
        self,
        data_path: str,
        metrics_path: str
    ) -> Dict[str, Any]:
        """
        从数据 JSON + 指标 JSON 进行完整评测（配置驱动入口）

        参数：
        - data_path: 测试数据 JSON 路径
        - metrics_path: 指标配置 JSON 路径
        """

        print(f"📂 加载数据文件: {data_path}")
        print(f"🧩 加载指标文件: {metrics_path}")

        # =========================
        # 1. 加载测试数据
        # =========================
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = []
        for i, item in enumerate(data):
            try:
                # 基本校验（增强鲁棒性）
                if "input" not in item or "actual_output" not in item:
                    print(f"⚠️ 第{i}条数据缺失字段，跳过")
                    continue

                tc = TestCase(
                    input=item.get("input", ""),
                    actual_output=item.get("actual_output", ""),
                    retrieval_context=item.get("retrieval_context", None)
                )
                test_cases.append(tc)

            except Exception as e:
                print(f"❌ 第{i}条数据解析失败: {e}")

        print(f"✅ 成功加载 {len(test_cases)} 条测试数据")

        # =========================
        # 2. 加载指标配置
        # =========================
        with open(metrics_path, "r", encoding="utf-8") as f:
            metric_configs = json.load(f)

        metrics = []
        for cfg in metric_configs:
            try:
                metric = create_metric(
                    name=cfg.get("name"),
                    model=self.custom_model,
                    criteria=cfg.get("criteria", ""),
                    threshold=cfg.get("threshold", 0.5),
                    evaluation_params=cfg.get("evaluation_params", None),
                    strict_mode=cfg.get("strict_mode", False)
                )
                metrics.append(metric)

            except Exception as e:
                print(f"❌ 指标加载失败: {cfg.get('name')} - {e}")

        print(f"✅ 成功加载 {len(metrics)} 个评测指标")

        # =========================
        # 3. 执行评测
        # =========================
        results = self.batch_evaluate(
            test_cases=test_cases,
            metrics=metrics
        )

        print("📊 正在计算平均分...")

        # =========================
        # 4. 统计平均分
        # =========================
        metric_sum = defaultdict(float)
        metric_count = defaultdict(int)
        overall_scores = []

        for case in results:
            case_total = 0
            case_metric_num = 0

            for metric_name, metric_info in case["metrics"].items():
                score = metric_info["score"]

                if score is not None:
                    metric_sum[metric_name] += score
                    metric_count[metric_name] += 1

                    case_total += score
                    case_metric_num += 1

            if case_metric_num > 0:
                overall_scores.append(case_total / case_metric_num)

        metric_avg = {
            name: metric_sum[name] / metric_count[name]
            for name in metric_sum
        }

        overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        # =========================
        # 5. 输出结果
        # =========================
        print("\n📈 平均得分结果:")
        for name, avg in metric_avg.items():
            print(f"- {name:<20}: {avg:.4f}")

        print(f"\n🌟 Overall Score: {overall_avg:.4f}")

        return {
            "detail_results": results,
            "metric_average": metric_avg,
            "overall_average": overall_avg
        }


if __name__ == "__main__":
    

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
    # judge.batch_evaluate(
    #     test_cases=[tc],
    #     metrics=test_metrics
    # )
    # results = judge.evaluate_from_json(
    #     json_path="ux_evaluator/dataset/testcases.json",
    #     metrics=test_metrics
    # )
    results = judge.evaluate_from_files(
        data_path="ux_evaluator/dataset/testcases.json",
        metrics_path="ux_evaluator/dataset/metrics.json"
    )