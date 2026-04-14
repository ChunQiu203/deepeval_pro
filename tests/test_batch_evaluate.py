import pytest
from scripts.batch_evaluate import init_metrics
from scripts.batch_evaluate import load_config

def test_init_metrics_raises_for_unsupported_metric():
    with pytest.raises(ValueError, match="不支持的指标"):
        init_metrics([{"name": "not_supported"}], custom_model=object())

def test_load_config_reads_yaml(tmp_path):
    config_text = """
judge:
  model: qwen-turbo
  retry_times: 3
metrics:
  - name: trust
    threshold: 0.5
dataset:
  data_path: examples/sample_data.json
  clean_data: true
batch:
  batch_size: 10
  output_path: results/evaluation_results.json
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    config = load_config(str(config_path))

    assert config["judge"]["model"] == "qwen-turbo"
    assert config["metrics"][0]["name"] == "trust"
    assert config["dataset"]["clean_data"] is True
