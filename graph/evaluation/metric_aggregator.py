# graph/evaluation/metric_aggregator.py
"""Hybrid Metric Aggregator.

Combines:
  - Deterministic Python evaluators (node/edge coverage, scenario classification,
    workflow path evaluation, grounding)
  - LLM semantic judgments produced by llm_bridge.invoke_llm_for_judgment()

All percentage calculations happen here in pure Python.
The LLM is only used to correct/augment node coverage and scenario classification.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_judgments(judgments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a dict keyed by test case id for O(1) lookup."""
    index: Dict[str, Dict[str, Any]] = {}
    for j in judgments:
        tc_id = str(j.get("id") or "")
        if tc_id:
            index[tc_id] = j
    return index


def _llm_augmented_node_coverage(
    deterministic_result: Dict[str, Any],
    judgments_index: Dict[str, Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use LLM-covered node names to rescue nodes that deterministic matching missed."""
    if not judgments_index:
        return deterministic_result  # no LLM data - return unchanged

    # Collect all node names the LLM said are covered
    llm_covered_names = set()
    for j in judgments_index.values():
        for name in j.get("graph_nodes_covered", []):
            llm_covered_names.add(str(name).lower().strip())

    # Move missed nodes to covered if LLM found them
    still_missed = []
    llm_rescued = []
    for node_info in deterministic_result.get("missed_nodes", []):
        n_name = str(node_info.get("name") or "").lower().strip()
        n_id = str(node_info.get("id") or "").lower().strip()
        if n_name in llm_covered_names or n_id in llm_covered_names:
            llm_rescued.append(node_info)
        else:
            still_missed.append(node_info)

    total = deterministic_result.get("total_nodes", 0)
    newly_covered = len(deterministic_result.get("covered_nodes", [])) + len(llm_rescued)
    node_coverage_pct = round((newly_covered / max(total, 1)) * 100, 1)

    result = dict(deterministic_result)
    result["missed_nodes"] = still_missed
    result["missed_node_count"] = len(still_missed)
    result["covered_node_count"] = newly_covered
    result["node_coverage_pct"] = node_coverage_pct
    result["llm_rescued_node_count"] = len(llm_rescued)
    return result


def _llm_augmented_scenario_coverage(
    scenario_result: Dict[str, Any],
    judgments_index: Dict[str, Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recompute scenario percentages using LLM-corrected scenario_type_correct flags.

    When the LLM says a tc's scenario label is wrong, we still count it under the
    label the LLM classified it as (from the original tc fields). We do NOT change
    the actual tc dict so the UI still shows the original label.
    """
    if not judgments_index:
        return scenario_result

    # per_tc_classification from deterministic evaluator
    per_tc = {
        item["id"]: item
        for item in scenario_result.get("per_tc_classification", [])
        if item.get("id")
    }

    # Re-classify test cases the LLM flagged as incorrect
    covered_counts = {
        "positive": 0, "negative": 0, "edge": 0, "alternate": 0, "failure_retry": 0
    }
    for tc in test_cases:
        tc_id = str(tc.get("id") or "")
        det_class = per_tc.get(tc_id, {}).get("classified_as", "positive")
        j = judgments_index.get(tc_id, {})
        if j.get("scenario_type_correct") is False:
            # LLM says classification is wrong - re-read from tc fields to re-classify
            from graph.evaluation.scenario_evaluator import _classify_test_case
            det_class = _classify_test_case(tc)
        if det_class in covered_counts:
            covered_counts[det_class] += 1
        else:
            covered_counts["positive"] += 1

    expected = scenario_result.get("expected_counts", {})
    percentages = {}
    for cat in covered_counts:
        exp = expected.get(cat, max(1, len(test_cases) // 5))
        cov = covered_counts[cat]
        percentages[cat] = round(min(1.0, cov / max(exp, 1)) * 100, 1)

    result = dict(scenario_result)
    result["covered_counts"] = covered_counts
    result["positive_pct"] = percentages.get("positive", 0.0)
    result["negative_pct"] = percentages.get("negative", 0.0)
    result["edge_pct"] = percentages.get("edge", 0.0)
    result["alternate_pct"] = percentages.get("alternate", 0.0)
    result["failure_retry_pct"] = percentages.get("failure_retry", 0.0)
    return result


def _llm_augmented_flow_correctness(
    workflow_result: Dict[str, Any],
    judgments_index: Dict[str, Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use LLM is_valid_flow flags to override deterministic flow correctness."""
    if not judgments_index:
        return workflow_result

    valid_count = 0
    invalid_ids = []
    for tc in test_cases:
        tc_id = str(tc.get("id") or "")
        j = judgments_index.get(tc_id)
        if j is not None:
            if j.get("is_valid_flow", True):
                valid_count += 1
            else:
                invalid_ids.append(tc_id)
        else:
            # No LLM judgment for this tc - trust deterministic result
            # Check if it appears in deterministic flow_violations
            det_violations = [v.get("test_case_id") for v in workflow_result.get("flow_violations", [])]
            if tc_id not in det_violations:
                valid_count += 1
            else:
                invalid_ids.append(tc_id)

    total = max(len(test_cases), 1)
    flow_correctness_pct = round((valid_count / total) * 100, 1)

    result = dict(workflow_result)
    result["flow_correctness_pct"] = flow_correctness_pct
    result["llm_invalid_flow_ids"] = invalid_ids
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def aggregate_metrics(
    test_cases: List[Dict[str, Any]],
    llm_judgments: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]],
    workflow_model: Optional[Any] = None,
    feature_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all deterministic evaluators then augment with LLM judgments.

    Returns a report dict with the same schema as the legacy evaluate_test_cases()
    so the existing UI code works without changes.
    """
    from graph.evaluation.test_case_deduplicator import deduplicate_test_cases
    from graph.evaluation.node_edge_coverage import evaluate_node_edge_coverage
    from graph.evaluation.workflow_path_evaluator import evaluate_workflow_paths
    from graph.evaluation.grounding_evaluator import evaluate_test_case_grounding
    from graph.evaluation.scenario_evaluator import evaluate_scenario_coverage
    from graph.evaluation.deepeval_evaluator import evaluate_test_cases_qualitative
    from graph.evaluation.test_case_evaluator import (
        generate_test_case_eval_narrative_summary,
        save_test_case_evaluation_run,
    )

    scope_label = (
        feature_name if (feature_name and feature_name != "Entire Graph") else "Entire Graph"
    )

    # 0. Deduplication
    dedup_res = deduplicate_test_cases(test_cases)
    unique_tcs = dedup_res.get("unique_test_cases", []) or test_cases

    # 1. Deterministic evaluators
    node_edge_res = evaluate_node_edge_coverage(unique_tcs, graph_nodes, graph_rels, feature_name=feature_name)
    workflow_res = evaluate_workflow_paths(unique_tcs, workflow_model, graph_nodes, graph_rels)
    grounding_res = evaluate_test_case_grounding(unique_tcs, graph_nodes)
    scenario_res = evaluate_scenario_coverage(unique_tcs, graph_nodes, graph_rels)
    context_text = " ".join([
        f"{n.get('name')}: {n.get('description', '')}"
        for n in graph_nodes[:20]
    ])
    deepeval_res = evaluate_test_cases_qualitative(unique_tcs, context_text)

    # 2. LLM augmentation (pure Python math - no LLM calls here)
    judgments_index = _index_judgments(llm_judgments)
    node_edge_res = _llm_augmented_node_coverage(node_edge_res, judgments_index, graph_nodes)
    scenario_res = _llm_augmented_scenario_coverage(scenario_res, judgments_index, unique_tcs)
    workflow_res = _llm_augmented_flow_correctness(workflow_res, judgments_index, unique_tcs)

    # 3. Final metric values
    node_cov = node_edge_res.get("node_coverage_pct", 0.0)
    edge_cov = node_edge_res.get("edge_coverage_pct", 0.0)
    wf_cov = workflow_res.get("workflow_coverage_pct", 0.0)
    flow_corr = workflow_res.get("flow_correctness_pct", 0.0)
    tc_corr = grounding_res.get("test_case_correctness_pct", 0.0)

    overall_coverage = round(
        0.30 * node_cov
        + 0.25 * edge_cov
        + 0.25 * wf_cov
        + 0.10 * flow_corr
        + 0.10 * tc_corr,
        1,
    )

    # 4. Build report (same schema as legacy evaluate_test_cases)
    report = {
        "run_id": f"tc_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "scope": scope_label,
        "overall_coverage": overall_coverage,
        "evaluation_mode": "hybrid" if llm_judgments else "deterministic_only",
        "llm_judgments_count": len(llm_judgments),
        "deduplication": {
            "total_test_cases": dedup_res.get("total_test_cases", 0),
            "unique_test_cases_count": dedup_res.get("unique_test_cases_count", 0),
            "duplicate_test_cases_count": dedup_res.get("duplicate_test_cases_count", 0),
            "duplication_ratio": dedup_res.get("duplication_ratio", 0.0),
            "duplicate_test_cases": dedup_res.get("duplicate_test_cases", []),
        },
        "summary": {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "workflow_coverage": wf_cov,
            "flow_correctness": flow_corr,
            "test_case_correctness": tc_corr,
            "total_test_cases": dedup_res.get("total_test_cases", 0),
            "unique_test_cases_count": dedup_res.get("unique_test_cases_count", 0),
            "duplicate_test_cases_count": dedup_res.get("duplicate_test_cases_count", 0),
            "total_nodes": node_edge_res.get("total_nodes", 0),
            "total_workflow_edges": node_edge_res.get("total_workflow_edges", 0),
            "llm_rescued_node_count": node_edge_res.get("llm_rescued_node_count", 0),
        },
        "scenario_coverage": {
            "positive": scenario_res.get("positive_pct", 0.0),
            "negative": scenario_res.get("negative_pct", 0.0),
            "edge_cases": scenario_res.get("edge_pct", 0.0),
            "alternate_paths": scenario_res.get("alternate_pct", 0.0),
            "failure_retry": scenario_res.get("failure_retry_pct", 0.0),
        },
        "unique_test_cases": unique_tcs,
        "missed_information": {
            "missed_node_count": len(node_edge_res.get("missed_nodes", [])),
            "missed_nodes": node_edge_res.get("missed_nodes", []),
            "missed_edge_count": len(node_edge_res.get("missed_edges", [])),
            "missed_edges": node_edge_res.get("missed_edges", []),
            "missed_workflow_step_count": len(workflow_res.get("missing_workflow_steps", [])),
            "missed_workflow_steps": workflow_res.get("missing_workflow_steps", []),
        },
        "flow_violations": workflow_res.get("flow_violations", []),
        "unsupported_test_steps": grounding_res.get("unsupported_test_steps", []),
        "deepeval": deepeval_res,
    }

    report["narrative_summary"] = generate_test_case_eval_narrative_summary(report)

    try:
        save_test_case_evaluation_run(report)
    except Exception as exc:
        logger.warning("Could not save evaluation run to Neo4j: %s", exc)

    return report
