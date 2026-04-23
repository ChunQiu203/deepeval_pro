import json
import argparse
import os
from collections import defaultdict


def _safe_preview(text, limit=60):
    if text is None:
        return ""
    value = str(text).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _collect_failed_cases(results, top_failures=3):
    failed_cases = []
    for idx, case in enumerate(results, start=1):
        if case.get("overall_passed"):
            continue

        metric_details = case.get("metrics", {}) or {}
        failed_metrics = []
        for metric_name, metric_info in metric_details.items():
            passed = bool(metric_info.get("passed", False))
            if not passed:
                score = metric_info.get("score")
                failed_metrics.append((metric_name, score))

        test_case = case.get("test_case", {}) or {}
        failed_cases.append(
            {
                "index": idx,
                "input_preview": _safe_preview(test_case.get("input", "")),
                "failed_metrics": failed_metrics,
            }
        )

    return failed_cases[: max(top_failures, 0)]


def _collect_metric_pass_rates(results):
    metric_passed = defaultdict(int)
    metric_total = defaultdict(int)

    for case in results:
        for metric_name, metric_info in (case.get("metrics", {}) or {}).items():
            metric_total[metric_name] += 1
            if bool(metric_info.get("passed", False)):
                metric_passed[metric_name] += 1

    metric_rates = []
    for name in sorted(metric_total.keys()):
        total = metric_total[name]
        passed = metric_passed[name]
        rate = (passed / total * 100) if total else 0.0
        metric_rates.append((name, rate, passed, total))
    return metric_rates


def generate_txt_report(total, passed, pass_rate, metric_sum, metric_count, metric_rates, failed_cases):
    """生成原生纯文本格式报告"""
    lines = [
        "=== QA Report ===",
        f"Total cases: {total}",
        f"Passed cases: {passed}",
        f"Pass rate: {pass_rate:.2f}%",
        "",
        "Average metric scores:",
    ]
    if not metric_sum:
        lines.append("  (no metric scores found)")
    else:
        for name in sorted(metric_sum.keys()):
            avg = metric_sum[name] / metric_count[name]
            lines.append(f"  - {name}: {avg:.4f}")

    lines.extend(["", "Metric pass rates:"])
    if not metric_rates:
        lines.append("  (no metric pass-rate data found)")
    else:
        for name, rate, passed_num, total_num in metric_rates:
            lines.append(f"  - {name}: {rate:.2f}% ({passed_num}/{total_num})")

    lines.extend(["", "Failed case summary:"])
    if not failed_cases:
        lines.append("  (no failed cases)")
    else:
        for item in failed_cases:
            failed_metrics_text = ", ".join(
                f"{metric_name}={score:.4f}" if isinstance(score, (int, float)) else f"{metric_name}=n/a"
                for metric_name, score in item["failed_metrics"]
            )
            preview = item["input_preview"] or "(empty input)"
            lines.append(f"  - Case #{item['index']}: {preview}")
            lines.append(f"    failed metrics: {failed_metrics_text or 'no metric-level fail details'}")
    return "\n".join(lines)


def generate_markdown_report(total, passed, pass_rate, metric_sum, metric_count, metric_rates, failed_cases):
    """生成 Markdown 格式报告，包含表格和清晰的层级"""
    lines = [
        "# UX Evaluator 质量评估报告",
        "",
        "## 总体概况",
        f"- **总测试用例数**: `{total}`",
        f"- **通过的用例数**: `{passed}`",
        f"- **总体通过率**: **`{pass_rate:.2f}%`**",
        "",
        "## 📈 指标平均分得分",
    ]

    if not metric_sum:
        lines.append("> *暂无指标得分数据*")
    else:
        lines.extend(["| 指标名称 | 平均得分 |", "| :--- | :--- |"])
        for name in sorted(metric_sum.keys()):
            avg = metric_sum[name] / metric_count[name]
            lines.append(f"| {name} | {avg:.4f} |")

    lines.extend(["", "##  各指标通过率"])
    if not metric_rates:
        lines.append("> *暂无指标通过率数据*")
    else:
        lines.extend(["| 指标名称 | 通过率 | 通过数/总数 |", "| :--- | :--- | :--- |"])
        for name, rate, passed_num, total_num in metric_rates:
            lines.append(f"| {name} | **{rate:.2f}%** | {passed_num}/{total_num} |")

    lines.extend(["", "## ❌ 失败用例摘要 (Top Failures)"])
    if not failed_cases:
        lines.append("**所有抽查用例均已通过**")
    else:
        for item in failed_cases:
            failed_metrics_text = ", ".join(
                f"`{metric_name}={score:.4f}`" if isinstance(score, (int, float)) else f"`{metric_name}=n/a`"
                for metric_name, score in item["failed_metrics"]
            )
            preview = item["input_preview"] or "(空输入)"
            lines.extend([
                f"### Case #{item['index']}",
                f"- **输入预览**: > {preview}",
                f"- **未通过指标**: {failed_metrics_text or '*无具体失败指标详情*'}",
                ""
            ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate QA summary report from evaluation results.")
    parser.add_argument("--input", required=True, help="Path to evaluation_results.json")
    parser.add_argument("--output", help="Optional path to save report text")
    parser.add_argument("--format", choices=["txt", "md"], default="txt", help="Report output format: txt or md")
    parser.add_argument("--top-failures", type=int, default=3, help="How many failed cases to include (default: 3)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    passed = sum(1 for r in results if r.get("overall_passed"))
    pass_rate = (passed / total * 100) if total else 0.0

    metric_sum = defaultdict(float)
    metric_count = defaultdict(int)

    for case in results:
        for metric_name, metric_info in case.get("metrics", {}).items():
            score = metric_info.get("score")
            if score is not None:
                metric_sum[metric_name] += score
                metric_count[metric_name] += 1

    metric_rates = _collect_metric_pass_rates(results)
    failed_cases = _collect_failed_cases(results, top_failures=args.top_failures)

    # 根据 format 参数选择生成逻辑
    if args.format == "md":
        report_text = generate_markdown_report(total, passed, pass_rate, metric_sum, metric_count, metric_rates,
                                               failed_cases)
    else:
        report_text = generate_txt_report(total, passed, pass_rate, metric_sum, metric_count, metric_rates,
                                          failed_cases)

    print(report_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")


if __name__ == "__main__":
    main()