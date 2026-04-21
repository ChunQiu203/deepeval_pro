import json

import pytest

from scripts.qa_check import _calculate_pass_rate, _load_results, _validate_artifact


def test_calculate_pass_rate_mixed_results():
    results = [
        {"overall_passed": True},
        {"overall_passed": False},
        {"overall_passed": True},
    ]
    assert _calculate_pass_rate(results) == pytest.approx(2 / 3)


def test_calculate_pass_rate_empty_results():
    assert _calculate_pass_rate([]) == 0.0


def test_load_results_requires_list(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        _load_results(str(bad_file))


def test_validate_artifact_checks_presence_and_non_empty(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        _validate_artifact(str(missing), required=True)

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _validate_artifact(str(empty), required=True)

    ok = tmp_path / "ok.txt"
    ok.write_text("ok", encoding="utf-8")
    _validate_artifact(str(ok), required=True)

