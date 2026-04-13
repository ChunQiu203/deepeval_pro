# scripts/batch_evaluate.py
"""
批量自动化评估脚本，负责加载数据集、执行评估、导出结果
预留了与其他模块的对接接口
"""
import argparse
import json
import os
from typing import List, Dict, Any

import yaml

# 导入核心模块
from ux_evaluator.dataset import DatasetLoader
from ux_evaluator.judge import UXJudge
from ux_evaluator.metrics import (
    get_trust_metric,
    get_satisfaction_metric
)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_metrics(metrics_config: List[Dict[str, Any]], custom_model) -> List[Any]:
    """根据配置初始化指标"""
    metrics = []
    for m in metrics_config:
        name = m["name"]
        threshold = m.get("threshold", 0.5)
        if name == "trust":
            metrics.append(get_trust_metric(custom_model=custom_model, threshold=threshold))
        elif name == "satisfaction":
            metrics.append(get_satisfaction_metric(custom_model=custom_model, threshold=threshold))
        else:
            raise ValueError(f"不支持的指标: {name}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="UX Evaluator 批量自动化评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径(相对于项目根目录)"
    )
    args = parser.parse_args()

    # 获取当前脚本所在的上一级目录（即项目根目录）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 将传入的相对路径转为绝对路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    # 1. 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    # 同样转换数据文件路径
    data_path = config["dataset"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)

    # 2. 加载数据集
    print(f"加载数据集: {data_path}")
    loader = DatasetLoader(clean_data=config["dataset"]["clean_data"])
    test_cases = loader.load_from_file(data_path)
    print(f"成功加载 {len(test_cases)} 个测试用例")

    # 3. 初始化Judge和指标
    print("初始化LLM评判器...")
    judge = UXJudge(
        model=config["judge"]["model"],
        retry=config["judge"]["retry_times"]
    )

    print("初始化评估指标...")
    metrics = init_metrics(config["metrics"], judge.custom_model)

    # 4. 执行批量评估
    print("开始批量评估...")
    results = judge.batch_evaluate(test_cases, metrics)
    print(f"评估完成，共处理 {len(results)} 个测试用例")

    # 5. 导出结果 (同样转换输出路径)
    output_path = config["batch"]["output_path"]
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到: {output_path}")

    # 6. 打印汇总统计
    passed_count = sum(1 for r in results if r["overall_passed"])
    print(f"\n评估汇总:")
    print(f"总测试用例: {len(results)}")
    print(f"通过用例: {passed_count}")
    print(f"通过率: {passed_count / len(results) * 100:.2f}%")


if __name__ == "__main__":
    main()