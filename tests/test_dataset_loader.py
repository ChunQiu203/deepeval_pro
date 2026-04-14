import json
import pytest
from ux_evaluator.dataset.loader import DatasetLoader


def test_load_from_json_filters_empty_rows(tmp_path):
    data = [
        {"input": "问题1", "actual_output": "回答1"},
        {"input": "", "actual_output": "回答2"},
        {"input": "问题3", "actual_output": ""},
    ]

    file_path = tmp_path / "sample.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = DatasetLoader(clean_data=True)
    cases = loader.load_from_file(str(file_path))

    assert len(cases) == 1
    assert cases[0].input == "问题1"
    assert cases[0].actual_output == "回答1"


def test_load_from_file_raises_when_path_not_exists():
    loader = DatasetLoader(clean_data=True)

    with pytest.raises(FileNotFoundError):
        loader.load_from_file("not_exists.json")


def test_load_from_json_supports_alias_fields(tmp_path):
    data = [
        {"question": "别名问题", "answer": "别名回答"}
    ]

    file_path = tmp_path / "alias.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = DatasetLoader(clean_data=True)
    cases = loader.load_from_file(str(file_path))

    assert len(cases) == 1
    assert cases[0].input == "别名问题"
    assert cases[0].actual_output == "别名回答"

def test_load_from_json_keeps_empty_rows_when_clean_data_false(tmp_path):
    data = [
        {"input": "", "actual_output": "回答A"},
        {"input": "问题B", "actual_output": ""},
    ]

    file_path = tmp_path / "keep_empty.json"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loader = DatasetLoader(clean_data=False)
    cases = loader.load_from_file(str(file_path))

    assert len(cases) == 2
    assert cases[0].actual_output == "回答A"
    assert cases[1].input == "问题B"