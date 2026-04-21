from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from deepeval.test_case import LLMTestCaseParams

from ux_evaluator.dataset.loader import TestCase
from ux_evaluator.metrics.geval_metrics import create_metric


@dataclass
class ConversationTurn:
    """One turn in a multi-turn conversation."""

    turn_id: int
    user: str
    assistant: str
    retrieval_context: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    expected: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_turn_id: int) -> "ConversationTurn":
        user = str(data.get("user", "")).strip()
        assistant = str(data.get("assistant", "")).strip()
        if not user:
            raise ValueError("turn.user must not be empty")
        if not assistant:
            raise ValueError("turn.assistant must not be empty")

        turn_id = data.get("turn_id", default_turn_id)
        if not isinstance(turn_id, int):
            raise ValueError("turn.turn_id must be an integer")

        retrieval_context = data.get("retrieval_context") or []
        if not isinstance(retrieval_context, list):
            raise ValueError("turn.retrieval_context must be a list")

        tool_calls = data.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ValueError("turn.tool_calls must be a list")

        expected = data.get("expected") or {}
        if not isinstance(expected, dict):
            raise ValueError("turn.expected must be an object")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("turn.metadata must be an object")

        return cls(
            turn_id=turn_id,
            user=user,
            assistant=assistant,
            retrieval_context=[str(item) for item in retrieval_context],
            tool_calls=tool_calls,
            expected=expected,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user": self.user,
            "assistant": self.assistant,
            "retrieval_context": self.retrieval_context,
            "tool_calls": self.tool_calls,
            "expected": self.expected,
            "metadata": self.metadata,
        }


@dataclass
class ConversationCase:
    """A full conversation case for chain-level evaluation."""

    case_id: str
    turns: List[ConversationTurn]
    scene: str = ""
    user_profile: Dict[str, Any] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    final_expectation: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_case_id: str) -> "ConversationCase":
        case_id = str(data.get("case_id", default_case_id)).strip()
        if not case_id:
            raise ValueError("case.case_id must not be empty")

        turns_data = data.get("turns")
        if not isinstance(turns_data, list) or not turns_data:
            raise ValueError("case.turns must be a non-empty list")

        turns = [
            ConversationTurn.from_dict(turn_data, default_turn_id=index + 1)
            for index, turn_data in enumerate(turns_data)
        ]

        user_profile = data.get("user_profile") or {}
        global_context = data.get("global_context") or {}
        final_expectation = data.get("final_expectation") or {}
        metadata = data.get("metadata") or {}
        tags = data.get("tags") or []

        for value, field_name in (
            (user_profile, "case.user_profile"),
            (global_context, "case.global_context"),
            (final_expectation, "case.final_expectation"),
            (metadata, "case.metadata"),
        ):
            if not isinstance(value, dict):
                raise ValueError(f"{field_name} must be an object")
        if not isinstance(tags, list):
            raise ValueError("case.tags must be a list")

        return cls(
            case_id=case_id,
            turns=turns,
            scene=str(data.get("scene", "")),
            user_profile=user_profile,
            global_context=global_context,
            final_expectation=final_expectation,
            tags=[str(tag) for tag in tags],
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "scene": self.scene,
            "user_profile": self.user_profile,
            "global_context": self.global_context,
            "turns": [turn.to_dict() for turn in self.turns],
            "final_expectation": self.final_expectation,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def build_history(self, turn_index: int) -> List[str]:
        """Return the conversation history before the current turn."""
        history: List[str] = []
        for turn in self.turns[:turn_index]:
            history.append(f"User: {turn.user}")
            history.append(f"Assistant: {turn.assistant}")
        return history

    def build_global_context_items(self) -> List[str]:
        """Normalize global context into a flat list of strings."""
        items: List[str] = []
        if self.scene:
            items.append(f"Scene: {self.scene}")

        for key, value in self.global_context.items():
            if isinstance(value, list):
                items.extend(f"{key}: {item}" for item in value)
            else:
                items.append(f"{key}: {value}")

        for key, value in self.user_profile.items():
            items.append(f"user_profile.{key}: {value}")

        return items

    def to_turn_test_cases(self, include_global_context: bool = True) -> List[TestCase]:
        """
        Expand a conversation case into turn-level test cases for evaluation.

        Each test case keeps prior dialogue in `context` and current retrieval
        context in `retrieval_context`, so chain-level metrics can reason over
        both the local turn and the full session state.
        """
        global_items = self.build_global_context_items() if include_global_context else []
        test_cases: List[TestCase] = []

        for index, turn in enumerate(self.turns):
            context = global_items + self.build_history(index)
            metadata = {
                "case_id": self.case_id,
                "turn_id": turn.turn_id,
                "scene": self.scene,
                "tags": self.tags,
                "expected": turn.expected,
                "final_expectation": self.final_expectation,
                "tool_calls": turn.tool_calls,
            }
            metadata.update(self.metadata)
            metadata.update(turn.metadata)

            test_cases.append(
                TestCase(
                    input=turn.user,
                    actual_output=turn.assistant,
                    context=context,
                    retrieval_context=turn.retrieval_context,
                    metadata=metadata,
                )
            )

        return test_cases

    def build_transcript(self) -> str:
        """Build a normalized transcript for conversation-level evaluation."""
        transcript_lines: List[str] = []
        for turn in self.turns:
            transcript_lines.append(f"User: {turn.user}")
            transcript_lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(transcript_lines)

    def to_conversation_test_case(self) -> TestCase:
        """
        Collapse the full conversation into one synthetic test case.

        This is used by DAG nodes that reason about the whole conversation,
        such as cross-turn consistency and end-to-end goal completion.
        """
        expectation_lines = [f"{key}: {value}" for key, value in self.final_expectation.items()]
        input_lines = [
            f"case_id: {self.case_id}",
            f"scene: {self.scene}" if self.scene else "scene: ",
            "conversation_goal:",
        ]
        if self.user_profile:
            input_lines.extend(f"user_profile.{key}: {value}" for key, value in self.user_profile.items())
        if expectation_lines:
            input_lines.append("final_expectation:")
            input_lines.extend(expectation_lines)

        retrieval_context = self.build_global_context_items()
        for turn in self.turns:
            retrieval_context.extend(turn.retrieval_context)
            for tool_call in turn.tool_calls:
                retrieval_context.append(f"tool_call: {json.dumps(tool_call, ensure_ascii=False)}")

        return TestCase(
            input="\n".join(input_lines),
            actual_output=self.build_transcript(),
            retrieval_context=retrieval_context,
            metadata={
                "case_id": self.case_id,
                "scene": self.scene,
                "tags": self.tags,
                "final_expectation": self.final_expectation,
                "turn_count": len(self.turns),
            },
        )


def _load_json_payload(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            items = []
            for line in file:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            return items


def load_conversation_cases(file_path: str) -> List[ConversationCase]:
    """
    Load multi-turn conversation cases from JSON or JSONL.

    Supported top-level formats:
    - {"version": "1.0", "cases": [...]}
    - [...]
    - JSONL with one case per line
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"conversation data file not found: {file_path}")

    payload = _load_json_payload(file_path)
    if isinstance(payload, dict):
        cases_data = payload.get("cases")
        if cases_data is None:
            raise ValueError("top-level JSON object must contain a 'cases' field")
    elif isinstance(payload, list):
        cases_data = payload
    else:
        raise ValueError("conversation payload must be an object or a list")

    if not isinstance(cases_data, list) or not cases_data:
        raise ValueError("conversation cases must be a non-empty list")

    cases: List[ConversationCase] = []
    for index, case_data in enumerate(cases_data):
        if not isinstance(case_data, dict):
            raise ValueError("each conversation case must be an object")
        cases.append(ConversationCase.from_dict(case_data, default_case_id=f"case_{index + 1:04d}"))
    return cases


def summarize_conversation_cases(cases: List[ConversationCase]) -> Dict[str, Any]:
    """Build a compact summary for quick validation from the CLI."""
    total_turns = sum(len(case.turns) for case in cases)
    summary: Dict[str, Any] = {
        "case_count": len(cases),
        "turn_count": total_turns,
    }

    if cases:
        first_case = cases[0]
        first_test_case = first_case.to_turn_test_cases()[0]
        summary["first_case"] = {
            "case_id": first_case.case_id,
            "scene": first_case.scene,
            "turn_count": len(first_case.turns),
            "first_turn_input": first_test_case.input,
            "first_turn_context": first_test_case.context,
            "first_turn_retrieval_context": first_test_case.retrieval_context,
            "first_turn_metadata": first_test_case.metadata,
        }

    return summary


def load_dag_config(config_path: Optional[str] = None) -> List[DAGNodeSpec]:
    """Load DAG node definitions from JSON, or fall back to the built-in default."""
    if config_path is None:
        payload = DEFAULT_DAG_CONFIG
    else:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"dag config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

    node_payloads = payload.get("nodes")
    if not isinstance(node_payloads, list) or not node_payloads:
        raise ValueError("dag config must contain a non-empty 'nodes' list")

    node_specs: List[DAGNodeSpec] = []
    for index, node_payload in enumerate(node_payloads):
        if not isinstance(node_payload, dict):
            raise ValueError("each dag node config must be an object")

        key = str(node_payload.get("key", "")).strip()
        node_type = str(node_payload.get("type", "")).strip()
        display_name = str(node_payload.get("name", "")).strip()
        dependencies = node_payload.get("dependencies") or []
        criteria = node_payload.get("criteria")

        if not key:
            raise ValueError(f"dag node at index {index} is missing key")
        if node_type not in {"turn_metrics", "conversation_metric"}:
            raise ValueError(f"unsupported dag node type for '{key}': {node_type}")
        if not display_name:
            raise ValueError(f"dag node '{key}' is missing name")
        if not isinstance(dependencies, list):
            raise ValueError(f"dag node '{key}' dependencies must be a list")
        if node_type == "conversation_metric" and (not criteria or not str(criteria).strip()):
            raise ValueError(f"conversation_metric node '{key}' must define criteria")

        node_specs.append(
            DAGNodeSpec(
                key=key,
                node_type=node_type,
                display_name=display_name,
                dependencies=[str(dep) for dep in dependencies],
                criteria=str(criteria).strip() if criteria is not None else None,
            )
        )

    return validate_and_sort_dag_nodes(node_specs)


def validate_and_sort_dag_nodes(node_specs: List[DAGNodeSpec]) -> List[DAGNodeSpec]:
    """Validate DAG node references and return a topologically sorted list."""
    node_by_key: Dict[str, DAGNodeSpec] = {}
    for node_spec in node_specs:
        if node_spec.key in node_by_key:
            raise ValueError(f"duplicate dag node key: {node_spec.key}")
        node_by_key[node_spec.key] = node_spec

    for node_spec in node_specs:
        for dependency in node_spec.dependencies:
            if dependency not in node_by_key:
                raise ValueError(f"dag node '{node_spec.key}' depends on unknown node '{dependency}'")
        if node_spec.node_type == "turn_metrics" and node_spec.dependencies:
            raise ValueError("turn_metrics nodes cannot depend on other nodes")

    ordered: List[DAGNodeSpec] = []
    visited: Set[str] = set()
    visiting: Set[str] = set()

    def visit(node_key: str) -> None:
        if node_key in visited:
            return
        if node_key in visiting:
            raise ValueError(f"dag config contains a cycle involving '{node_key}'")

        visiting.add(node_key)
        node_spec = node_by_key[node_key]
        for dependency in node_spec.dependencies:
            visit(dependency)
        visiting.remove(node_key)
        visited.add(node_key)
        ordered.append(node_spec)

    for node_spec in node_specs:
        visit(node_spec.key)

    return ordered


def load_turn_metric_configs(metrics_config_path: str) -> List[Dict[str, Any]]:
    """Load turn-level metric definitions from a JSON file."""
    if not os.path.exists(metrics_config_path):
        raise FileNotFoundError(f"metrics config file not found: {metrics_config_path}")

    with open(metrics_config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("metrics config must be a non-empty list")

    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"metric config at index {index} must be an object")

    return payload


@dataclass(frozen=True)
class DAGNodeSpec:
    """Definition of one evaluation node in the conversation DAG."""

    key: str
    node_type: str
    display_name: str
    dependencies: List[str]
    criteria: Optional[str] = None


TURN_LEVEL_NODE = DAGNodeSpec(
    key="turn_level",
    node_type="turn_metrics",
    display_name="Turn_Level_Quality",
    dependencies=[],
)

CONVERSATION_CONSISTENCY_NODE = DAGNodeSpec(
    key="conversation_consistency",
    node_type="conversation_metric",
    display_name="Conversation_Consistency",
    dependencies=["turn_level"],
    criteria="""
Evaluate the full multi-turn conversation as an end-to-end interaction.
The `actual_output` field contains the entire conversation transcript, including
all user and assistant turns in chronological order. The `retrieval_context`
field contains the global business rules, user profile, turn-level retrieval
context, and tool call traces.

Score from 1 to 10 according to:
1. Cross-turn consistency: does the assistant remain logically consistent across turns?
2. Context carry-over: does the assistant correctly use and remember earlier turns?
3. Policy stability: does the assistant keep following the same rules and boundaries throughout the conversation?
4. Persona stability: does the assistant maintain a stable style and strategy across turns?

10 means the conversation is coherent and stable end to end.
1 means the assistant contradicts itself or loses track of the conversation.
""",
)

GOAL_COMPLETION_NODE = DAGNodeSpec(
    key="goal_completion",
    node_type="conversation_metric",
    display_name="Goal_Completion",
    dependencies=["turn_level", "conversation_consistency"],
    criteria="""
Evaluate whether the assistant successfully moved the conversation toward the
user's goal and the final expectation. The `input` field contains the case
metadata, user profile, and final expectation. The `actual_output` field
contains the full conversation transcript. The `retrieval_context` field
contains the available background rules and evidence.

Score from 1 to 10 according to:
1. Goal progress: did the assistant make meaningful progress toward the user's objective?
2. Actionability: did the assistant provide clear next steps or a usable resolution?
3. Expectation alignment: did the conversation outcome match the stated final expectation?
4. End-to-end usefulness: after reading the full conversation, would a user feel the task was handled well?

10 means the conversation handled the goal effectively from start to finish.
1 means the assistant failed to help, derailed, or left the goal unresolved.
""",
)


DEFAULT_DAG_CONFIG: Dict[str, Any] = {
    "nodes": [
        {
            "key": TURN_LEVEL_NODE.key,
            "type": TURN_LEVEL_NODE.node_type,
            "name": TURN_LEVEL_NODE.display_name,
            "dependencies": TURN_LEVEL_NODE.dependencies,
        },
        {
            "key": CONVERSATION_CONSISTENCY_NODE.key,
            "type": CONVERSATION_CONSISTENCY_NODE.node_type,
            "name": CONVERSATION_CONSISTENCY_NODE.display_name,
            "dependencies": CONVERSATION_CONSISTENCY_NODE.dependencies,
            "criteria": CONVERSATION_CONSISTENCY_NODE.criteria,
        },
        {
            "key": GOAL_COMPLETION_NODE.key,
            "type": GOAL_COMPLETION_NODE.node_type,
            "name": GOAL_COMPLETION_NODE.display_name,
            "dependencies": GOAL_COMPLETION_NODE.dependencies,
            "criteria": GOAL_COMPLETION_NODE.criteria,
        },
    ]
}


class ConversationDAGEvaluator:
    """Evaluate multi-turn conversation cases with the existing judge pipeline."""

    def __init__(self, judge: Any, node_specs: Optional[List[DAGNodeSpec]] = None):
        self.judge = judge
        self.node_specs = validate_and_sort_dag_nodes(node_specs or load_dag_config())

    def evaluate_cases(
        self,
        cases: List[ConversationCase],
        metric_keys: Optional[List[str]] = None,
        metrics: Optional[List[Any]] = None,
        threshold: float = 0.5,
        include_global_context: bool = True,
    ) -> Dict[str, Any]:
        expanded_cases: List[TestCase] = []
        for case in cases:
            expanded_cases.extend(case.to_turn_test_cases(include_global_context=include_global_context))

        built_metrics = metrics or self._build_turn_metrics(metric_keys or [], threshold=threshold)
        turn_results = self.judge.batch_evaluate(expanded_cases, built_metrics)
        conversation_node_results = self._evaluate_conversation_nodes(cases, threshold=threshold)
        case_results = self._group_turn_results_by_case(
            cases=cases,
            turn_results=turn_results,
            conversation_node_results=conversation_node_results,
            node_specs=self.node_specs,
        )
        summary = self._build_summary(case_results, node_specs=self.node_specs)

        return {
            "case_results": case_results,
            "summary": summary,
            "turn_result_count": len(turn_results),
            "metric_keys": metric_keys or [],
            "dag_nodes": [
                {
                    "key": spec.key,
                    "type": spec.node_type,
                    "display_name": spec.display_name,
                    "dependencies": spec.dependencies,
                }
                for spec in self.node_specs
            ],
        }

    def evaluate_file(
        self,
        json_path: str,
        metric_keys: Optional[List[str]] = None,
        metrics: Optional[List[Any]] = None,
        threshold: float = 0.5,
        include_global_context: bool = True,
    ) -> Dict[str, Any]:
        cases = load_conversation_cases(json_path)
        return self.evaluate_cases(
            cases=cases,
            metric_keys=metric_keys,
            metrics=metrics,
            threshold=threshold,
            include_global_context=include_global_context,
        )

    def _build_turn_metrics(self, metric_keys: List[str], threshold: float) -> List[Any]:
        if not metric_keys:
            return []
        return self.judge._build_metrics_from_keys(metric_keys, threshold=threshold)

    def build_turn_metrics_from_configs(self, metric_configs: List[Dict[str, Any]]) -> List[Any]:
        metrics: List[Any] = []
        for config in metric_configs:
            metric_key = config.get("metric_key") or config.get("key")
            if metric_key:
                built_metrics = self.judge._build_metrics_from_keys(
                    [str(metric_key)],
                    threshold=float(config.get("threshold", 0.5)),
                )
                if not built_metrics:
                    raise ValueError(f"unsupported metric key in metrics config: {metric_key}")
                metrics.append(built_metrics[0])
                continue

            metrics.append(
                create_metric(
                    name=str(config.get("name", "")).strip(),
                    model=self.judge.custom_model,
                    criteria=str(config.get("criteria", "")),
                    threshold=float(config.get("threshold", 0.5)),
                    evaluation_params=config.get("evaluation_params"),
                    strict_mode=bool(config.get("strict_mode", False)),
                )
            )
        return metrics

    def _build_conversation_metric(self, node_spec: DAGNodeSpec, threshold: float) -> Any:
        return create_metric(
            name=node_spec.display_name,
            model=self.judge.custom_model,
            criteria=node_spec.criteria or "",
            threshold=threshold,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            strict_mode=False,
        )

    def _evaluate_conversation_nodes(
        self,
        cases: List[ConversationCase],
        threshold: float,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        test_cases = [case.to_conversation_test_case() for case in cases]
        results_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {
            case.case_id: {} for case in cases
        }

        for node_spec in self.node_specs:
            if node_spec.node_type != "conversation_metric":
                continue

            metric = self._build_conversation_metric(node_spec, threshold=threshold)
            evaluation_results = self.judge.batch_evaluate(test_cases, [metric])

            for case, evaluation_result in zip(cases, evaluation_results):
                metric_payload = evaluation_result["metrics"].get(node_spec.display_name, {})
                results_by_case[case.case_id][node_spec.key] = {
                    "display_name": node_spec.display_name,
                    "dependencies": node_spec.dependencies,
                    "score": metric_payload.get("score"),
                    "passed": metric_payload.get("passed"),
                }

        return results_by_case

    def _group_turn_results_by_case(
        self,
        cases: List[ConversationCase],
        turn_results: List[Dict[str, Any]],
        conversation_node_results: Dict[str, Dict[str, Dict[str, Any]]],
        node_specs: List[DAGNodeSpec],
    ) -> List[Dict[str, Any]]:
        results_by_case: Dict[str, List[Dict[str, Any]]] = {}
        for turn_result in turn_results:
            metadata = turn_result.get("test_case", {}).get("metadata", {})
            case_id = metadata.get("case_id")
            if not case_id:
                raise ValueError("turn result metadata must include case_id")
            results_by_case.setdefault(case_id, []).append(turn_result)

        grouped_results: List[Dict[str, Any]] = []
        for case in cases:
            case_turn_results = results_by_case.get(case.case_id, [])
            metric_average, overall_average = self._aggregate_turn_scores(case_turn_results)
            dag_node_results = {
                TURN_LEVEL_NODE.key: {
                    "display_name": TURN_LEVEL_NODE.display_name,
                    "dependencies": TURN_LEVEL_NODE.dependencies,
                    "score": overall_average,
                    "passed": all(result.get("overall_passed", False) for result in case_turn_results)
                    if case_turn_results
                    else False,
                }
            }
            dag_node_results.update(conversation_node_results.get(case.case_id, {}))
            dag_overall_average = self._aggregate_dag_node_scores(node_specs, dag_node_results)
            grouped_results.append(
                {
                    "case_id": case.case_id,
                    "scene": case.scene,
                    "tags": case.tags,
                    "turn_count": len(case.turns),
                    "evaluated_turn_count": len(case_turn_results),
                    "overall_passed": all(result.get("overall_passed", False) for result in case_turn_results)
                    if case_turn_results
                    else False,
                    "metric_average": metric_average,
                    "overall_average": overall_average,
                    "dag": {
                        "node_results": dag_node_results,
                        "overall_average": dag_overall_average,
                    },
                    "final_expectation": case.final_expectation,
                    "turn_results": case_turn_results,
                }
            )
        return grouped_results

    def _aggregate_turn_scores(
        self,
        turn_results: List[Dict[str, Any]],
    ) -> tuple[Dict[str, float], float]:
        metric_sum: Dict[str, float] = {}
        metric_count: Dict[str, int] = {}
        turn_averages: List[float] = []

        for turn_result in turn_results:
            turn_total = 0.0
            turn_metric_count = 0
            for metric_name, metric_info in turn_result.get("metrics", {}).items():
                score = metric_info.get("score")
                if score is None:
                    continue
                metric_sum[metric_name] = metric_sum.get(metric_name, 0.0) + score
                metric_count[metric_name] = metric_count.get(metric_name, 0) + 1
                turn_total += score
                turn_metric_count += 1

            if turn_metric_count:
                turn_averages.append(turn_total / turn_metric_count)

        metric_average = {
            metric_name: metric_sum[metric_name] / metric_count[metric_name]
            for metric_name in metric_sum
            if metric_count[metric_name] > 0
        }
        overall_average = sum(turn_averages) / len(turn_averages) if turn_averages else 0.0
        return metric_average, overall_average

    def _aggregate_dag_node_scores(
        self,
        node_specs: List[DAGNodeSpec],
        dag_node_results: Dict[str, Dict[str, Any]],
    ) -> float:
        scores: List[float] = []
        for node_spec in node_specs:
            score = dag_node_results.get(node_spec.key, {}).get("score")
            if score is not None:
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def _build_summary(
        self,
        case_results: List[Dict[str, Any]],
        node_specs: List[DAGNodeSpec],
    ) -> Dict[str, Any]:
        metric_sum: Dict[str, float] = {}
        metric_count: Dict[str, int] = {}
        overall_scores: List[float] = []
        dag_node_sum: Dict[str, float] = {}
        dag_node_count: Dict[str, int] = {}
        dag_overall_scores: List[float] = []

        for case_result in case_results:
            overall_scores.append(case_result["overall_average"])
            dag_overall_scores.append(case_result["dag"]["overall_average"])
            for metric_name, score in case_result["metric_average"].items():
                metric_sum[metric_name] = metric_sum.get(metric_name, 0.0) + score
                metric_count[metric_name] = metric_count.get(metric_name, 0) + 1
            for node_spec in node_specs:
                node_score = case_result["dag"]["node_results"].get(node_spec.key, {}).get("score")
                if node_score is None:
                    continue
                dag_node_sum[node_spec.key] = dag_node_sum.get(node_spec.key, 0.0) + node_score
                dag_node_count[node_spec.key] = dag_node_count.get(node_spec.key, 0) + 1

        return {
            "case_count": len(case_results),
            "passed_case_count": sum(1 for case_result in case_results if case_result["overall_passed"]),
            "overall_average": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "dag_overall_average": sum(dag_overall_scores) / len(dag_overall_scores) if dag_overall_scores else 0.0,
            "metric_average": {
                metric_name: metric_sum[metric_name] / metric_count[metric_name]
                for metric_name in metric_sum
                if metric_count[metric_name] > 0
            },
            "dag_node_average": {
                node_spec.key: dag_node_sum[node_spec.key] / dag_node_count[node_spec.key]
                for node_spec in node_specs
                if dag_node_count.get(node_spec.key, 0) > 0
            },
        }


def _build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Load and inspect multi-turn conversation JSON data.")
    parser.add_argument("--file", required=True, help="Path to the conversation JSON or JSONL file.")
    parser.add_argument(
        "--mode",
        choices=["inspect", "evaluate"],
        default="inspect",
        help="Inspect the conversation file or run evaluation with the existing judge.",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        dest="metric_keys",
        default=[],
        help="Metric registry key to use in evaluate mode. Repeat this option for multiple metrics.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used when building metrics.")
    parser.add_argument("--model", help="Judge model name used in evaluate mode.")
    parser.add_argument("--base-url", dest="base_url", help="Optional model base URL used in evaluate mode.")
    parser.add_argument("--dag-config", dest="dag_config", help="Optional DAG config JSON path.")
    parser.add_argument("--metrics-config", dest="metrics_config", help="Optional turn-level metrics JSON path.")
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    cases = load_conversation_cases(args.file)

    node_specs = load_dag_config(args.dag_config)

    if args.mode == "inspect":
        payload = {
            "conversation_summary": summarize_conversation_cases(cases),
            "dag_nodes": [
                {
                    "key": spec.key,
                    "type": spec.node_type,
                    "display_name": spec.display_name,
                    "dependencies": spec.dependencies,
                }
                for spec in node_specs
            ],
        }
    else:
        if not args.metric_keys and not args.metrics_config:
            parser.error("either --metric-key or --metrics-config is required when --mode evaluate")
        if not args.model:
            parser.error("--model is required when --mode evaluate")

        from ux_evaluator.judge.metric_judge import UXJudge

        evaluator = ConversationDAGEvaluator(
            judge=UXJudge(model=args.model, base_url=args.base_url),
            node_specs=node_specs,
        )
        metric_configs = load_turn_metric_configs(args.metrics_config) if args.metrics_config else None
        turn_metrics = (
            evaluator.build_turn_metrics_from_configs(metric_configs)
            if metric_configs is not None
            else None
        )
        payload = evaluator.evaluate_cases(
            cases=cases,
            metric_keys=args.metric_keys,
            metrics=turn_metrics,
            threshold=args.threshold,
        )

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
