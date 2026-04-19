"""Batch evaluation entrypoint for the UX evaluator."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ux_evaluator.dataset import DatasetLoader
from ux_evaluator.judge import UXJudge
from ux_evaluator.metrics import create_metric, get_metric_by_key, list_metric_keys


METRIC_NAME_ALIASES = {
    "信任感": "trust",
    "理解度": "understanding",
    "掌控感": "control",
    "效率": "efficiency",
    "认知负荷": "cognitive_load",
    "满意度": "satisfaction",
    "安全性": "safety",
    "非依赖": "dependency",
    "拟人化": "anthropomorphism",
    "共情性": "empathy",
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(project_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _normalize_metric_key(value: str) -> str:
    key = str(value).strip()
    if not key:
        return key
    return METRIC_NAME_ALIASES.get(key, METRIC_NAME_ALIASES.get(key.lower(), key))


def _build_metric(metric_config: Dict[str, Any], custom_model) -> Any:
    metric_key = metric_config.get("metric_key") or metric_config.get("key")
    metric_name = metric_config.get("name")
    threshold = metric_config.get("threshold", 0.5)
    criteria = metric_config.get("criteria")
    evaluation_params = metric_config.get("evaluation_params")
    strict_mode = metric_config.get("strict_mode", False)

    candidate = metric_key or metric_name
    if candidate:
        normalized_key = _normalize_metric_key(candidate)
        if normalized_key in list_metric_keys():
            return get_metric_by_key(custom_model, normalized_key, threshold=threshold)
        if criteria:
            return create_metric(
                name=metric_name or normalized_key,
                model=custom_model,
                criteria=criteria,
                threshold=threshold,
                evaluation_params=evaluation_params,
                strict_mode=strict_mode,
            )
        raise ValueError(f"不支持的指标: {candidate}")

    raise ValueError("每个指标配置都需要 name 或 metric_key")


def init_metrics(metrics_config: List[Dict[str, Any]], custom_model) -> List[Any]:
    """Build DeepEval metrics from config entries."""
    if not metrics_config:
        raise ValueError("metrics_config 不能为空")
    return [_build_metric(metric_cfg, custom_model) for metric_cfg in metrics_config]


def _load_metrics_config(config: Dict[str, Any], project_root: str) -> List[Dict[str, Any]]:
    metrics_path = config.get("metrics_path") or config.get("batch", {}).get("metrics_path")
    if metrics_path:
        metrics_path = _resolve_path(project_root, metrics_path)
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return [loaded]
        if isinstance(loaded, list):
            return loaded
        raise ValueError("metrics_path must point to a JSON object or array")

    metrics = config.get("metrics", [])
    if not metrics:
        raise ValueError("配置中没有 metrics")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="UX Evaluator batch evaluation script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路径(相对于项目根目录)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    data_path = _resolve_path(project_root, config["dataset"]["data_path"])
    print(f"加载数据集: {data_path}")
    loader = DatasetLoader(clean_data=config["dataset"].get("clean_data", True))
    test_cases = loader.load_from_file(data_path)
    print(f"成功加载 {len(test_cases)} 个测试用例")

    print("初始化 LLM Judge...")
    judge = UXJudge(
        model=config["judge"]["model"],
        retry=config["judge"].get("retry_times", 3),
        base_url=config["judge"].get("base_url"),
    )

    print("初始化评估指标...")
    metrics_config = _load_metrics_config(config, project_root)
    metrics = init_metrics(metrics_config, judge.custom_model)
    print(f"成功加载 {len(metrics)} 个评估指标")

    print("开始批量评估...")
    results = judge.batch_evaluate(test_cases, metrics)
    summary = judge._aggregate_results(results)

    output_path = _resolve_path(project_root, config["batch"]["output_path"])
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到: {output_path}")
    print("\n评估汇总:")
    print(f"总测试用例: {len(results)}")

    metric_average = summary["metric_average"]
    if metric_average:
        for name, avg in metric_average.items():
            print(f"- {name:<20}: {avg:.4f}")

    overall_average = summary["overall_average"]
    print(f"Overall Score: {overall_average:.4f}")

    passed_count = sum(1 for result in results if result["overall_passed"])
    print(f"通过用例: {passed_count}")
    if results:
        print(f"通过率: {passed_count / len(results) * 100:.2f}%")


if __name__ == "__main__":
    main()
