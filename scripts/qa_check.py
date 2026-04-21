"""Unified QA gate script for tests, evaluation, and report generation."""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _run_step(command: list[str], title: str) -> None:
    print(f"\n[QA] {title}")
    print("[QA] CMD:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {title}")


def _validate_artifact(path: str, required: bool = True) -> None:
    if not required:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required artifact not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Artifact is empty: {path}")


def _load_results(results_path: str) -> list[dict[str, Any]]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Evaluation results JSON must be a list")
    return data


def _calculate_pass_rate(results: Iterable[dict[str, Any]]) -> float:
    rows = list(results)
    total = len(rows)
    if total == 0:
        return 0.0
    passed = sum(1 for row in rows if row.get("overall_passed"))
    return passed / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QA gate checks for deepeval_pro")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Config path for batch evaluation")
    parser.add_argument("--results-path", default="results/evaluation_results.json", help="Evaluation results JSON path")
    parser.add_argument("--report-path", default="tests/reports/qa_report.txt", help="Generated report output path")
    parser.add_argument("--pytest-target", default="tests", help="Pytest target path")
    parser.add_argument("--top-failures", type=int, default=3, help="Failed cases shown in report")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Optional minimum pass rate threshold in [0,1], e.g. 0.7",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest execution")
    parser.add_argument("--skip-eval", action="store_true", help="Skip batch evaluation execution")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation execution")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    results_path = _resolve_path(args.results_path)
    report_path = _resolve_path(args.report_path)
    pytest_target = _resolve_path(args.pytest_target)

    try:
        if not args.skip_tests:
            _run_step([sys.executable, "-m", "pytest", pytest_target, "-q"], "Run tests")

        if not args.skip_eval:
            _run_step(
                [sys.executable, "scripts/batch_evaluate.py", "--config", config_path],
                "Run batch evaluation",
            )

        if not args.skip_report:
            report_dir = os.path.dirname(report_path)
            if report_dir:
                os.makedirs(report_dir, exist_ok=True)
            _run_step(
                [
                    sys.executable,
                    "scripts/generate_report.py",
                    "--input",
                    results_path,
                    "--output",
                    report_path,
                    "--top-failures",
                    str(args.top_failures),
                ],
                "Generate QA report",
            )

        _validate_artifact(results_path, required=True)
        _validate_artifact(report_path, required=not args.skip_report)

        results = _load_results(results_path)
        pass_rate = _calculate_pass_rate(results)
        print(f"\n[QA] Pass rate: {pass_rate:.2%} ({len(results)} cases)")

        if args.min_pass_rate is not None:
            if not 0 <= args.min_pass_rate <= 1:
                raise ValueError("--min-pass-rate must be within [0, 1]")
            if pass_rate < args.min_pass_rate:
                raise RuntimeError(
                    f"Pass rate check failed: {pass_rate:.2%} < {args.min_pass_rate:.2%}"
                )
            print(f"[QA] Threshold check passed: >= {args.min_pass_rate:.2%}")

        print("[QA] Gate passed")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[QA] Gate failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

