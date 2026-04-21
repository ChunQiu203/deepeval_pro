"""将批量评测结果可视化为简洁仪表盘。"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _normalize_metric_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned.endswith("[GEval]"):
        cleaned = cleaned.replace("[GEval]", "").strip()
    return cleaned


def load_results(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError("评测结果必须是 JSON 数组")
    return results


def summarize_results(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    metric_sum = defaultdict(float)
    metric_count = defaultdict(int)
    metric_pass = defaultdict(int)
    overall_passed = 0
    all_scores = []

    rows = []
    for idx, case in enumerate(results, start=1):
        passed = bool(case.get("overall_passed"))
        overall_passed += int(passed)

        row = {"case_index": idx, "overall_passed": passed}
        for metric_name, metric_info in case.get("metrics", {}).items():
            normalized_name = _normalize_metric_name(metric_name)
            score = metric_info.get("score")
            passed_metric = bool(metric_info.get("passed"))

            row[normalized_name] = score
            if score is not None:
                metric_sum[normalized_name] += float(score)
                metric_count[normalized_name] += 1
                all_scores.append(float(score))
            if passed_metric:
                metric_pass[normalized_name] += 1

        rows.append(row)

    df = pd.DataFrame(rows)
    total_cases = len(results)
    metric_avg = {
        metric: metric_sum[metric] / metric_count[metric]
        for metric in metric_sum
        if metric_count[metric] > 0
    }
    metric_pass_rate = {
        metric: metric_pass[metric] / total_cases * 100 if total_cases else 0.0
        for metric in metric_count
    }

    summary = {
        "total_cases": total_cases,
        "passed_cases": overall_passed,
        "failed_cases": total_cases - overall_passed,
        "pass_rate": (overall_passed / total_cases * 100) if total_cases else 0.0,
        "metric_average": metric_avg,
        "metric_pass_rate": metric_pass_rate,
        "all_scores": all_scores,
    }
    return summary, df


def plot_dashboard(summary: Dict[str, Any], output_path: str) -> None:
    metric_avg = summary["metric_average"]
    metric_pass_rate = summary["metric_pass_rate"]
    all_scores = summary["all_scores"]

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("评测结果仪表盘", fontsize=18, fontweight="bold")

    # 1. 总体通过 / 未通过
    ax = axes[0, 0]
    passed = summary["passed_cases"]
    failed = summary["failed_cases"]
    ax.pie(
        [passed, failed],
        labels=["通过", "未通过"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )
    ax.set_title(f"总体通过率：{summary['pass_rate']:.2f}%")

    # 2. 指标平均分
    ax = axes[0, 1]
    if metric_avg:
        metric_names = list(metric_avg.keys())
        metric_values = [metric_avg[name] for name in metric_names]
        ax.barh(metric_names, metric_values, color="#3498db")
        ax.set_xlim(0, 1)
        ax.set_xlabel("平均分")
        ax.set_title("指标平均分")
        for i, value in enumerate(metric_values):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center")
    else:
        ax.text(0.5, 0.5, "未找到指标分数", ha="center", va="center")
        ax.set_axis_off()

    # 3. 指标通过率
    ax = axes[1, 0]
    if metric_pass_rate:
        metric_names = list(metric_pass_rate.keys())
        metric_values = [metric_pass_rate[name] for name in metric_names]
        ax.barh(metric_names, metric_values, color="#9b59b6")
        ax.set_xlim(0, 100)
        ax.set_xlabel("通过率 (%)")
        ax.set_title("指标通过率")
        for i, value in enumerate(metric_values):
            ax.text(value + 1, i, f"{value:.1f}%", va="center")
    else:
        ax.text(0.5, 0.5, "未找到指标通过数据", ha="center", va="center")
        ax.set_axis_off()

    # 4. 分数分布
    ax = axes[1, 1]
    if all_scores:
        ax.hist(all_scores, bins=min(10, len(all_scores)), color="#f39c12", edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel("分数")
        ax.set_ylabel("数量")
        ax.set_title("分数分布")
    else:
        ax.text(0.5, 0.5, "未找到分数数据", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_text(summary: Dict[str, Any], output_path: str) -> None:
    lines = [
        "=== 评测结果摘要 ===",
        f"样本总数: {summary['total_cases']}",
        f"通过样本数: {summary['passed_cases']}",
        f"通过率: {summary['pass_rate']:.2f}%",
        "",
        "指标平均分:",
    ]

    metric_avg = summary["metric_average"]
    if not metric_avg:
        lines.append("  (未找到指标分数)")
    else:
        for name in sorted(metric_avg.keys()):
            lines.append(f"  - {name}: {metric_avg[name]:.4f}")

    lines.append("")
    lines.append("指标通过率:")
    metric_pass_rate = summary["metric_pass_rate"]
    if not metric_pass_rate:
        lines.append("  (未找到指标通过数据)")
    else:
        for name in sorted(metric_pass_rate.keys()):
            lines.append(f"  - {name}: {metric_pass_rate[name]:.2f}%")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="将评测结果可视化为图表。")
    parser.add_argument("--input", required=True, help="评测结果 JSON 文件路径")
    parser.add_argument(
        "--output-dir",
        default="results/visualizations",
        help="图表和摘要文本的输出目录",
    )
    parser.add_argument(
        "--prefix",
        default="evaluation_dashboard",
        help="生成文件的名称前缀",
    )
    args = parser.parse_args()

    results = load_results(args.input)
    summary, _ = summarize_results(results)

    os.makedirs(args.output_dir, exist_ok=True)
    chart_path = os.path.join(args.output_dir, f"{args.prefix}.png")
    summary_path = os.path.join(args.output_dir, f"{args.prefix}.txt")

    plot_dashboard(summary, chart_path)
    save_summary_text(summary, summary_path)

    print(f"图表已保存: {chart_path}")
    print(f"摘要已保存: {summary_path}")
    print(f"通过率: {summary['pass_rate']:.2f}%")


if __name__ == "__main__":
    main()
