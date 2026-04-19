import json
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate a simple QA summary report from evaluation results.")
    parser.add_argument("--input", required=True, help="Path to evaluation_results.json")
    parser.add_argument("--output", help="Optional path to save report text")
    args = parser.parse_args()

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

    report_text = "\n".join(lines)
    print(report_text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")


if __name__ == "__main__":
    main()