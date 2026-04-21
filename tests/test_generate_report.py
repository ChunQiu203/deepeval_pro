import json
from pathlib import Path

from scripts.generate_report import _collect_failed_cases, main


def _write_results(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_collect_failed_cases_handles_missing_fields():
    results = [
        {
            "overall_passed": False,
            "metrics": {
                "Trustworthiness": {"score": 0.2, "passed": False},
                "Understanding": {"score": 0.8, "passed": True},
            },
            "test_case": {},
        }
    ]

    failed = _collect_failed_cases(results, top_failures=3)

    assert len(failed) == 1
    assert failed[0]["index"] == 1
    assert failed[0]["input_preview"] == ""
    assert failed[0]["failed_metrics"] == [("Trustworthiness", 0.2)]


def test_generate_report_includes_failed_summary(tmp_path, monkeypatch):
    input_file = tmp_path / "evaluation_results.json"
    output_file = tmp_path / "qa_report.txt"
    data = [
        {
            "overall_passed": True,
            "metrics": {
                "Trustworthiness": {"score": 0.9, "passed": True}
            },
            "test_case": {"input": "good case"},
        },
        {
            "overall_passed": False,
            "metrics": {
                "Trustworthiness": {"score": 0.3, "passed": False},
                "Safety": {"score": 0.2, "passed": False},
            },
            "test_case": {"input": "bad case"},
        },
    ]
    _write_results(input_file, data)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_report.py",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--top-failures",
            "1",
        ],
    )

    main()

    report = output_file.read_text(encoding="utf-8")
    assert "=== QA Report ===" in report
    assert "Pass rate: 50.00%" in report
    assert "Metric pass rates:" in report
    assert "Trustworthiness: 50.00% (1/2)" in report
    assert "Safety: 0.00% (0/1)" in report
    assert "Failed case summary:" in report
    assert "Case #2: bad case" in report
    assert "Trustworthiness=0.3000" in report
    assert "Safety=0.2000" in report


def test_generate_report_handles_no_failed_cases(tmp_path, monkeypatch):
    input_file = tmp_path / "evaluation_results.json"
    output_file = tmp_path / "qa_report.txt"
    data = [
        {
            "overall_passed": True,
            "metrics": {
                "Trustworthiness": {"score": 0.9, "passed": True}
            },
            "test_case": {"input": "all good"},
        }
    ]
    _write_results(input_file, data)

    monkeypatch.setattr(
        "sys.argv",
        ["generate_report.py", "--input", str(input_file), "--output", str(output_file)],
    )

    main()

    report = output_file.read_text(encoding="utf-8")
    assert "Metric pass rates:" in report
    assert "Trustworthiness: 100.00% (1/1)" in report
    assert "Failed case summary:" in report
    assert "(no failed cases)" in report

