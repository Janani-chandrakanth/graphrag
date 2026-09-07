# graph/evaluation/test_case_evaluator.py
"""Test Case Evaluation Engine.
Orchestrates complete evaluation of generated test cases against Neo4j Knowledge Graph and Workflows:
Deduplication, Node Coverage %, Workflow Edge Coverage %, Workflow Coverage %, Flow Correctness %, Test Case Correctness %,
Scenario Coverage breakdown, Missed Information detection, Flow Violations, and Unsupported Step Detection.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from graph.neo4j_manager import run_cypher_query
from graph.evaluation.test_case_deduplicator import deduplicate_test_cases
from graph.evaluation.node_edge_coverage import evaluate_node_edge_coverage
from graph.evaluation.workflow_path_evaluator import evaluate_workflow_paths
from graph.evaluation.grounding_evaluator import evaluate_test_case_grounding
from graph.evaluation.scenario_evaluator import evaluate_scenario_coverage
from graph.evaluation.deepeval_evaluator import evaluate_test_cases_qualitative

logger = logging.getLogger(__name__)


def generate_test_case_eval_narrative_summary(report: Dict[str, Any]) -> str:
    """Generates a clean, human-readable executive summary text for Test Case Evaluation."""
    scope = report.get("scope", "Entire Graph")
    overall = report.get("overall_coverage", 0.0)
    summary = report.get("summary", {})
    dedup = report.get("deduplication", {})
    missed = report.get("missed_information", {})

    total_tcs = summary.get("total_test_cases", 0)
    unique_tcs = summary.get("unique_test_cases_count", 0)
    dup_tcs = summary.get("duplicate_test_cases_count", 0)

    node_cov = summary.get("node_coverage", 0.0)
    edge_cov = summary.get("edge_coverage", 0.0)
    flow_corr = summary.get("flow_correctness", 0.0)

    miss_n = missed.get("missed_node_count", 0)
    miss_e = missed.get("missed_edge_count", 0)

    quality_tier = "High Coverage" if overall >= 80 else ("Moderate Coverage" if overall >= 60 else "Needs Coverage Expansion")

    narrative = (
        f"Executive Test Case Evaluation Summary for scope '{scope}':\n"
        f"Evaluated {total_tcs} generated test cases against the knowledge graph and workflows. "
        f"Out of {total_tcs} total test cases, {unique_tcs} are unique scenarios and {dup_tcs} are duplicate or redundant variations.\n\n"
        f"Graph & Workflow Coverage: The unique test suite achieves {node_cov}% Graph Node Coverage and {edge_cov}% Workflow Edge Coverage. "
        f"A total of {flow_corr}% of test cases follow valid, sequential workflow paths without step order violations. "
        f"{miss_n} nodes and {miss_e} workflow edges remain uncovered by current test cases.\n\n"
        f"Overall Test Coverage Index: {overall}% ({quality_tier})."
    )
    return narrative


def evaluate_test_cases(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_rels: List[Dict[str, Any]],
    workflow_model: Optional[Any] = None,
    feature_name: Optional[str] = None
) -> Dict[str, Any]:
    """Orchestrates test case evaluation for specified scope (Entire Graph or Feature)."""
    scope_label = feature_name if (feature_name and feature_name != "Entire Graph") else "Entire Graph"

    # 0. Test Case Deduplication & Scenario Enrichment
    dedup_res = deduplicate_test_cases(test_cases)
    unique_tcs = dedup_res.get("unique_test_cases", [])

    # Evaluate unique test cases to reward real coverage rather than duplicate counts
    eval_target_tcs = unique_tcs if unique_tcs else test_cases

    # 1. Node & Relationship Coverage
    node_edge_res = evaluate_node_edge_coverage(eval_target_tcs, graph_nodes, graph_rels, feature_name=feature_name)

    # 2. Workflow Path Coverage & Flow Correctness
    workflow_res = evaluate_workflow_paths(eval_target_tcs, workflow_model, graph_nodes, graph_rels)

    # 3. Grounding & Unsupported Step Detection
    grounding_res = evaluate_test_case_grounding(eval_target_tcs, graph_nodes)

    # 4. Scenario Type Coverage Breakdown
    scenario_res = evaluate_scenario_coverage(eval_target_tcs, graph_nodes, graph_rels)

    # 5. DeepEval / LLM Qualitative Evaluation
    context_text = " ".join([f"{n.get('name')}: {n.get('description', '')}" for n in graph_nodes[:20]])
    deepeval_res = evaluate_test_cases_qualitative(eval_target_tcs, context_text)

    # 6. Compute Overall Coverage Index
    node_cov = node_edge_res.get("node_coverage_pct", 0.0)
    edge_cov = node_edge_res.get("edge_coverage_pct", 0.0)
    wf_cov = workflow_res.get("workflow_coverage_pct", 0.0)
    flow_corr = workflow_res.get("flow_correctness_pct", 0.0)
    tc_corr = grounding_res.get("test_case_correctness_pct", 0.0)

    overall_coverage = round(
        0.30 * node_cov +
        0.25 * edge_cov +
        0.25 * wf_cov +
        0.10 * flow_corr +
        0.10 * tc_corr,
        1
    )

    report = {
        "run_id": f"tc_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "scope": scope_label,
        "overall_coverage": overall_coverage,
        "deduplication": {
            "total_test_cases": dedup_res.get("total_test_cases", 0),
            "unique_test_cases_count": dedup_res.get("unique_test_cases_count", 0),
            "duplicate_test_cases_count": dedup_res.get("duplicate_test_cases_count", 0),
            "duplication_ratio": dedup_res.get("duplication_ratio", 0.0),
            "duplicate_test_cases": dedup_res.get("duplicate_test_cases", [])
        },
        "summary": {
            "node_coverage": node_cov,
            "edge_coverage": edge_cov,
            "workflow_coverage": wf_cov,
            "flow_correctness": flow_corr,
            "test_case_correctness": tc_corr,
            
            # New specific dimensions
            "core_workflow_coverage_pct": node_edge_res.get("workflow_node_coverage_pct", 0.0),
            "requirement_coverage_pct": node_edge_res.get("requirement_coverage_pct", 0.0),
            "business_rule_coverage_pct": node_edge_res.get("business_rule_coverage_pct", 0.0),
            "input_coverage_pct": node_edge_res.get("input_coverage_pct", 0.0),

            "total_test_cases": dedup_res.get("total_test_cases", 0),
            "unique_test_cases_count": dedup_res.get("unique_test_cases_count", 0),
            "duplicate_test_cases_count": dedup_res.get("duplicate_test_cases_count", 0),
            "total_nodes": node_edge_res.get("total_nodes", 0),
            "total_workflow_edges": node_edge_res.get("total_workflow_edges", 0)
        },
        "scenario_coverage": {
            "positive": scenario_res.get("positive_pct", 0.0),
            "negative": scenario_res.get("negative_pct", 0.0),
            "edge_cases": scenario_res.get("edge_pct", 0.0),
            "alternate_paths": scenario_res.get("alternate_pct", 0.0),
            "failure_retry": scenario_res.get("failure_retry_pct", 0.0),
            "warnings": scenario_res.get("warnings", [])
        },
        "unique_test_cases": eval_target_tcs,
        "missed_information": {
            "missed_node_count": len(node_edge_res.get("missed_nodes", [])),
            "missed_nodes": node_edge_res.get("missed_nodes", []),
            "missed_edge_count": len(node_edge_res.get("missed_edges", [])),
            "missed_edges": node_edge_res.get("missed_edges", []),
            "missed_workflow_step_count": len(workflow_res.get("missing_workflow_steps", [])),
            "missed_workflow_steps": workflow_res.get("missing_workflow_steps", [])
        },
        "flow_violations": workflow_res.get("flow_violations", []),
        "unsupported_test_steps": grounding_res.get("unsupported_test_steps", []),
        "deepeval": deepeval_res
    }

    report["narrative_summary"] = generate_test_case_eval_narrative_summary(report)

    # Save evaluation run to Neo4j
    try:
        save_test_case_evaluation_run(report)
    except Exception as e:
        logger.warning("Could not save TestCaseEvaluationRun to Neo4j: %s", e)

    return report


def save_test_case_evaluation_run(report: Dict[str, Any]) -> bool:
    """Saves the test case evaluation report to Neo4j under node label `:TestCaseEvaluationRun`."""
    cypher = """
    CREATE (e:TestCaseEvaluationRun {
        id: $run_id,
        timestamp: $timestamp,
        scope: $scope,
        overall_coverage: $overall_coverage,
        node_coverage: $node_coverage,
        edge_coverage: $edge_coverage,
        workflow_coverage: $workflow_coverage,
        flow_correctness: $flow_correctness,
        test_case_correctness: $test_case_correctness,
        details_json: $details_json
    })
    RETURN e.id AS id
    """
    params = {
        "run_id": report.get("run_id"),
        "timestamp": report.get("timestamp"),
        "scope": report.get("scope"),
        "overall_coverage": report.get("overall_coverage"),
        "node_coverage": report.get("summary", {}).get("node_coverage"),
        "edge_coverage": report.get("summary", {}).get("edge_coverage"),
        "workflow_coverage": report.get("summary", {}).get("workflow_coverage"),
        "flow_correctness": report.get("summary", {}).get("flow_correctness"),
        "test_case_correctness": report.get("summary", {}).get("test_case_correctness"),
        "details_json": json.dumps(report)
    }
    try:
        run_cypher_query(cypher, params)
        logger.info("Saved TestCaseEvaluationRun %s to Neo4j.", report.get("run_id"))
        return True
    except Exception as e:
        logger.error("Failed to save TestCaseEvaluationRun to Neo4j: %s", e)
        return False


def get_test_case_evaluation_history() -> List[Dict[str, Any]]:
    """Retrieves historical test case evaluation runs from Neo4j."""
    cypher = """
    MATCH (e:TestCaseEvaluationRun)
    RETURN e.id AS run_id,
           e.timestamp AS timestamp,
           e.scope AS scope,
           e.overall_coverage AS overall_coverage,
           e.node_coverage AS node_coverage,
           e.edge_coverage AS edge_coverage,
           e.workflow_coverage AS workflow_coverage,
           e.flow_correctness AS flow_correctness,
           e.test_case_correctness AS test_case_correctness,
           e.details_json AS details_json
    ORDER BY e.timestamp DESC
    """
    try:
        records = run_cypher_query(cypher)
        history = []
        for rec in records:
            item = dict(rec)
            if item.get("details_json"):
                try:
                    item["details"] = json.loads(item["details_json"])
                except Exception:
                    item["details"] = {}
            history.append(item)
        return history
    except Exception as e:
        logger.error("Error fetching test case evaluation history from Neo4j: %s", e)
        return []
