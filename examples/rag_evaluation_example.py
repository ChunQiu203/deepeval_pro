# examples/rag_evaluation_example.py
"""
RAG应用用户体验评估示例
你可以直接运行此脚本，测试整个流程是否正常
"""
from ux_evaluator.dataset import DatasetLoader
from ux_evaluator.judge import UXJudge
from ux_evaluator.metrics import get_trust_metric, get_satisfaction_metric


def main():
    print("=== RAG应用用户体验评估示例 ===")

    # 1. 加载测试数据集
    print("\n1. 加载测试数据集...")
    loader = DatasetLoader(clean_data=True)
    test_cases = loader.load_from_file("sample_data.json")
    print(f"   成功加载 {len(test_cases)} 个测试用例")

    # 2. 初始化评判器 (调整顺序：优先初始化大模型实例)
    print("\n2. 初始化LLM评判器...")
    judge = UXJudge(model="qwen-turbo")

    # 3. 初始化评估指标
    print("\n3. 初始化评估指标...")
    metrics = [
        get_trust_metric(custom_model=judge.custom_model, threshold=0.5),
        get_satisfaction_metric(custom_model=judge.custom_model, threshold=0.5)
    ]
    print(f"   已初始化 {len(metrics)} 个UX指标: 信任度、满意度")

    # 4. 执行评估
    print("\n4. 执行评估...")
    results = judge.batch_evaluate(test_cases, metrics)

    # 5. 打印结果
    print("\n5. 评估结果:")
    for i, res in enumerate(results):
        print(f"\n--- 测试用例 {i + 1} ---")
        print(f"用户问题: {res['test_case']['input']}")
        print(f"LLM回答: {res['test_case']['actual_output']}")
        print(f"评估结果:")
        for metric_name, metric_res in res['metrics'].items():
            print(f"  - {metric_name}: 得分={metric_res['score']}, 通过={metric_res['passed']}")
        print(f"整体通过: {res['overall_passed']}")

    print("\n=== 示例运行完成 ===")


if __name__ == "__main__":
    main()