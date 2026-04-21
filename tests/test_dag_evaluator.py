import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepeval.models.base_model import DeepEvalBaseLLM

from ux_evaluator.metrics.dag import (
    ConversationCase,
    ConversationDAGEvaluator,
    ConversationTurn,
    load_dag_config,
    load_turn_metric_configs,
)


class FakeModel(DeepEvalBaseLLM):
    def load_model(self):
        return self

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return "ok"

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return "ok"

    def get_model_name(self):
        return "fake-model"


class FakeJudge:
    def __init__(self):
        self.custom_model = FakeModel()

    def _build_metrics_from_keys(self, metric_keys, threshold=0.5):
        return [{"metric_key": metric_key, "threshold": threshold} for metric_key in metric_keys]

    def batch_evaluate(self, test_cases, metrics):
        score_map = {
            "Trustworthiness": 0.8,
            "Understanding": 0.6,
            "Conversation_Consistency": 0.9,
            "Goal_Completion": 0.7,
        }
        results = []
        for test_case in test_cases:
            metric_dict = {}
            for metric in metrics:
                if isinstance(metric, dict):
                    if metric["metric_key"] == "trust":
                        metric_dict["Trustworthiness"] = {"score": score_map["Trustworthiness"], "passed": True}
                    if metric["metric_key"] == "understanding":
                        metric_dict["Understanding"] = {"score": score_map["Understanding"], "passed": True}
                else:
                    metric_dict[metric.name] = {"score": score_map[metric.name], "passed": True}

            results.append(
                {
                    "test_case": test_case.to_dict(),
                    "metrics": metric_dict,
                    "overall_passed": True,
                }
            )
        return results


def _write_json_in_workspace(payload, filename: str) -> str:
    workspace = Path(__file__).resolve().parents[1]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{filename}",
        dir=workspace,
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(json.dumps(payload, ensure_ascii=False))
        return temp_file.name


def test_conversation_dag_evaluator_groups_results_by_case():
    case = ConversationCase(
        case_id="conv_001",
        scene="refund",
        turns=[
            ConversationTurn(turn_id=1, user="u1", assistant="a1"),
            ConversationTurn(turn_id=2, user="u2", assistant="a2"),
        ],
        final_expectation={"should_follow_policy": True},
        tags=["refund"],
    )

    evaluator = ConversationDAGEvaluator(judge=FakeJudge())
    result = evaluator.evaluate_cases(
        cases=[case],
        metric_keys=["trust", "understanding"],
    )

    assert result["turn_result_count"] == 2
    assert result["summary"]["case_count"] == 1
    assert result["summary"]["passed_case_count"] == 1
    assert result["summary"]["metric_average"]["Trustworthiness"] == 0.8
    assert result["summary"]["metric_average"]["Understanding"] == 0.6
    assert result["summary"]["dag_node_average"]["turn_level"] == 0.7
    assert result["summary"]["dag_node_average"]["conversation_consistency"] == 0.9
    assert result["summary"]["dag_node_average"]["goal_completion"] == 0.7

    case_result = result["case_results"][0]
    assert case_result["case_id"] == "conv_001"
    assert case_result["evaluated_turn_count"] == 2
    assert case_result["overall_passed"] is True
    assert case_result["metric_average"]["Trustworthiness"] == 0.8
    assert case_result["metric_average"]["Understanding"] == 0.6
    assert case_result["dag"]["node_results"]["turn_level"]["score"] == 0.7
    assert case_result["dag"]["node_results"]["conversation_consistency"]["score"] == 0.9
    assert case_result["dag"]["node_results"]["goal_completion"]["score"] == 0.7


def test_load_dag_config_sorts_dependencies():
    payload = {
        "nodes": [
            {
                "key": "goal_completion",
                "type": "conversation_metric",
                "name": "Goal_Completion",
                "dependencies": ["turn_level", "conversation_consistency"],
                "criteria": "goal completion criteria",
            },
            {
                "key": "turn_level",
                "type": "turn_metrics",
                "name": "Turn_Level_Quality",
                "dependencies": [],
            },
            {
                "key": "conversation_consistency",
                "type": "conversation_metric",
                "name": "Conversation_Consistency",
                "dependencies": ["turn_level"],
                "criteria": "consistency criteria",
            },
        ]
    }
    path = _write_json_in_workspace(payload, "test_dag_config.json")

    try:
        node_specs = load_dag_config(path)
        assert [node.key for node in node_specs] == [
            "turn_level",
            "conversation_consistency",
            "goal_completion",
        ]
    finally:
        os.remove(path)


def test_conversation_dag_evaluator_uses_custom_config():
    node_specs = load_dag_config()
    custom_node_specs = [node for node in node_specs if node.key != "goal_completion"]

    case = ConversationCase(
        case_id="conv_002",
        scene="support",
        turns=[ConversationTurn(turn_id=1, user="u1", assistant="a1")],
    )

    evaluator = ConversationDAGEvaluator(judge=FakeJudge(), node_specs=custom_node_specs)
    result = evaluator.evaluate_cases(cases=[case], metric_keys=["trust"])

    dag_nodes = result["dag_nodes"]
    assert [node["key"] for node in dag_nodes] == ["turn_level", "conversation_consistency"]
    assert "goal_completion" not in result["case_results"][0]["dag"]["node_results"]


def test_load_turn_metric_configs_reads_json():
    payload = [
        {"key": "trust", "threshold": 0.5},
        {"name": "CustomMetric", "criteria": "custom criteria", "threshold": 0.7},
    ]
    path = _write_json_in_workspace(payload, "test_metrics_config.json")

    try:
        metric_configs = load_turn_metric_configs(path)
        assert len(metric_configs) == 2
        assert metric_configs[0]["key"] == "trust"
        assert metric_configs[1]["name"] == "CustomMetric"
    finally:
        os.remove(path)


def test_conversation_dag_evaluator_builds_turn_metrics_from_configs():
    evaluator = ConversationDAGEvaluator(judge=FakeJudge())
    metric_configs = [
        {"key": "trust", "threshold": 0.5},
        {"name": "CustomMetric", "criteria": "custom criteria", "threshold": 0.7},
    ]

    metrics = evaluator.build_turn_metrics_from_configs(metric_configs)

    assert len(metrics) == 2
    assert isinstance(metrics[0], dict)
    assert metrics[0]["metric_key"] == "trust"
    assert metrics[1].name == "CustomMetric"
