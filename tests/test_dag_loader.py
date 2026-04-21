import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ux_evaluator.metrics.dag import load_conversation_cases


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


def test_load_conversation_cases_from_wrapped_json():
    payload = {
        "version": "1.0",
        "cases": [
            {
                "case_id": "conv_001",
                "scene": "refund",
                "user_profile": {"persona": "anxious"},
                "global_context": {"business_rules": ["7-day refund"]},
                "turns": [
                    {
                        "turn_id": 1,
                        "user": "Can I get a refund?",
                        "assistant": "Please tell me whether it was activated.",
                        "retrieval_context": ["Refund depends on activation state"],
                    },
                    {
                        "turn_id": 2,
                        "user": "It was activated.",
                        "assistant": "Activated orders usually cannot use no-reason refunds.",
                    },
                ],
                "final_expectation": {"should_follow_policy": True},
                "tags": ["refund", "multi_turn"],
            }
        ],
    }

    path = _write_json_in_workspace(payload, "test_dag_conversation.json")

    try:
        cases = load_conversation_cases(path)

        assert len(cases) == 1
        assert cases[0].case_id == "conv_001"
        assert len(cases[0].turns) == 2
        assert cases[0].global_context["business_rules"] == ["7-day refund"]
    finally:
        os.remove(path)


def test_conversation_case_to_turn_test_cases_keeps_history():
    payload = {
        "cases": [
            {
                "case_id": "conv_002",
                "scene": "support",
                "turns": [
                    {
                        "user": "My device is broken.",
                        "assistant": "Can you describe the issue?",
                    },
                    {
                        "user": "It won't boot.",
                        "assistant": "Please try charging it for 30 minutes first.",
                        "retrieval_context": ["Boot issues may be caused by low battery"],
                    },
                ],
            }
        ]
    }

    path = _write_json_in_workspace(payload, "test_dag_turn_cases.json")

    try:
        cases = load_conversation_cases(path)
        test_cases = cases[0].to_turn_test_cases()

        assert len(test_cases) == 2
        assert test_cases[0].context == ["Scene: support"]
        assert test_cases[1].context == [
            "Scene: support",
            "User: My device is broken.",
            "Assistant: Can you describe the issue?",
        ]
        assert test_cases[1].retrieval_context == ["Boot issues may be caused by low battery"]
        assert test_cases[1].metadata["case_id"] == "conv_002"
        assert test_cases[1].metadata["turn_id"] == 2
    finally:
        os.remove(path)


def test_load_conversation_cases_requires_turns():
    payload = {"cases": [{"case_id": "conv_003"}]}

    path = _write_json_in_workspace(payload, "test_dag_invalid.json")

    try:
        with pytest.raises(ValueError, match="case.turns"):
            load_conversation_cases(path)
    finally:
        os.remove(path)
