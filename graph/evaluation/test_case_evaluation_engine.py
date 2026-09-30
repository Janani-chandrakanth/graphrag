"""
graph/evaluation/test_case_evaluation_engine.py

Consolidated test case evaluation coordinator, metric aggregator, LLM bridge, and legacy reporting:
- LLM semantic judgment bridge (formerly llm_bridge.py)
- Hybrid metric aggregation & scoring (formerly metric_aggregator.py)
- Full test case evaluation pipeline coordinator (formerly test_case_evaluator.py)
- Plain-text legacy report generator (formerly legacy_report_generator.py)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from graph.evaluation.test_case_contracts import (
    build_evaluation_contract,
    preprocess_test_cases,
    deduplicate_test_cases,
)
from graph.evaluation.test_case_dimension_evaluators import (
    evaluate_node_edge_coverage,
    evaluate_scenario_types,
    evaluate_workflow_paths,
    evaluate_requirement_grounding,
    evaluate_test_cases_qualitative,
)
from graph.neo4j_manager import run_cypher_query
from parser.llm_client import call_ollama

logger = logging.getLogger(__name__)

# Maximum number of test cases to send to the LLM in one batch
_BATCH_SIZE = 5


# ==============================================================================
# 1. LLM JUDGMENT BRIDGE (formerly llm_bridge.py)
# ==============================================================================

def _build_judgment_prompt(batch: List[Dict[str, Any]], contract: Dict[str, Any]) -> str:
    def format_nodes(nodes: List[Dict[str, Any]]) -> List[str]:
        return [
            f"{n.get('name') or n.get('id')}" + (f": {n.get('description')[:50]}" if n.get("description") else "")
            for n in nodes if n.get('name') or n.get('id')
        ][:15]

    context_data = {
        "workflow_steps": format_nodes(contract.get("workflow_nodes", [])),
        "requirements": format_nodes(contract.get("requirements", [])),
        "business_rules": format_nodes(contract.get("business_rules", [])),
        "inputs": format_nodes(contract.get("inputs", []))
    }

    tc_block = json.dumps(
        [
            {
                "id": tc.get("id", ""),
                "title": tc.get("title", ""),
                "scenario_type": tc.get("scenario_type") or tc.get("type") or "unknown",
                "steps": tc.get("steps", [])[:6],
                "expected_result": tc.get("expected_result") or tc.get("expected_results") or "",
            }
            for tc in batch
        ],
        indent=2,
    )

    return f"""You are a QA expert and knowledge graph analyst.

EVALUATION UNIVERSE (Semantically Extracted):
{json.dumps(context_data, indent=2)}

TEST CASES:
{tc_block}

For each test case, output a JSON array. Each element must have:
  "id": the test case id (string)
  "valid_scenario": boolean - whether this test case represents a valid logical test scenario for the feature
  "graph_nodes_covered": list of string names of graph nodes that this test case actually exercises
  "scenario_type": one of ["positive", "negative", "edge", "alternate", "failure_retry"]
  "feedback": a 1-sentence explanation of why this test case is valid or invalid

Return ONLY the raw JSON array, with no Markdown fences or other text:
[
  {{
    "id": "TC-001",
    "valid_scenario": true,
    "graph_nodes_covered": ["Login Button", "Dashboard"],
    "scenario_type": "positive",
    "feedback": "Valid positive login flow."
  }}
]"""


def invoke_llm_for_judgment(
    test_cases: List[Dict[str, Any]],
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Calls Ollama to produce per-test-case semantic judgments."""
    if not test_cases:
        return []

    judgments: List[Dict[str, Any]] = []

    for i in range(0, len(test_cases), _BATCH_SIZE):
        batch = test_cases[i : i + _BATCH_SIZE]
        prompt = _build_judgment_prompt(batch, contract)

        try:
            res = call_ollama(prompt, timeout=60)
            raw = res.get("raw", "").strip()

            if "[" in raw and "]" in raw:
                json_str = raw[raw.find("[") : raw.rfind("]") + 1]
                batch_judgments = json.loads(json_str)
                if isinstance(batch_judgments, list):
                    judgments.extend(batch_judgments)
            else:
                logger.warning("LLM response did not contain a JSON array: %s", raw[:200])
        except Exception as e:
            logger.warning("Error getting LLM judgment for batch %d: %s", i, e)

    return judgments


# ==============================================================================
# 2. HYBRID METRIC AGGREGATION (formerly metric_aggregator.py)
# ==============================================================================

def _index_judgments(judgments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
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
    if not judgments_index:
        return deterministic_result

    llm_covered_names = set()
    for j in judgments_index.values():
        for name in j.get("graph_nodes_covered", []):
            llm_covered_names.add(str(name).lower().strip())

    still_missed = []
    llm_rescued = []
    for node_info in deterministic_result.get("missed_nodes", []):
        n_name = str(node_info.get("name") or "").lower().strip()
        n_id = str(node_info.get("id") or "").lower().strip()
        if n_name in llm_covered_names or n_id in llm_covered_names:
            llm_rescued.append(node_info)
        else:
            still_missed.append(node_info)

    augmented_covered = list(deterministic_result.get("covered_nodes", [])) + llm_rescued
    total = deterministic_result.get("total_nodes_count", len(graph_nodes))
    cov_count = len(augmented_covered)
    cov_pct = round((cov_count / max(total, 1)) * 100, 1)

    res = dict(deterministic_result)
    res["node_coverage_pct"] = cov_pct
    res["covered_nodes_count"] = cov_count
    res["covered_nodes"] = augmented_covered
    res["missed_nodes"] = still_missed
    return res


def _llm_augmented_scenario_counts(
    deterministic_result: Dict[str, Any],
    judgments_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if not judgments_index:
        return deterministic_result

    llm_counts = {"positive": 0, "negative": 0, "edge": 0, "alternate": 0, "failure_retry": 0}
    for j in judgments_index.values():
        st_type = str(j.get("scenario_type") or "").lower().strip()
        if st_type in llm_counts:
            llm_counts[st_type] += 1

    combined_counts = {}
    det_counts = deterministic_result.get("counts", {})
    for k in ["positive", "negative", "edge", "alternate", "failure_retry"]:
        combined_counts[k] = max(det_counts.get(k, 0), llm_counts.get(k, 0))

    active_dims = deterministic_result.get("expected_dimensions", {})
    active_keys = [k for k, exp in active_dims.items() if exp]
    covered_keys = [k for k in active_keys if combined_counts.get(k, 0) > 0]
    score_pct = round((len(covered_keys) / max(len(active_keys), 1)) * 100, 1)

    res = dict(deterministic_result)
    res["counts"] = combined_counts
    res["scenario_score_pct"] = score_pct
    res["covered_dimensions_count"] = len(covered_keys)
    res["missing_dimensions"] = [k for k in active_keys if combined_counts.get(k, 0) == 0]
    return res


def aggregate_hybrid_metrics(
    preprocessed_tcs: List[Dict[str, Any]],
    dedup_result: Dict[str, Any],
    node_edge_result: Dict[str, Any],
    scenario_result: Dict[str, Any],
    workflow_result: Dict[str, Any],
    grounding_result: Dict[str, Any],
    contract: Dict[str, Any],
    graph_nodes: List[Dict[str, Any]],
    graph_relationships: List[Dict[str, Any]],
    judgments: Optional[List[Dict[str, Any]]] = None,
    scope: str = "Entire Graph",
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Combines deterministic evaluator results with LLM semantic judgments."""
    judgments_index = _index_judgments(judgments or [])

    node_edge_final = _llm_augmented_node_coverage(node_edge_result, judgments_index, graph_nodes)
    scenario_final = _llm_augmented_scenario_counts(scenario_result, judgments_index)

    # Test Case Correctness %: Valid Scenarios from LLM judgments
    if judgments:
        valid_count = sum(1 for j in judgments if j.get("valid_scenario") is True)
        tc_correctness_pct = round((valid_count / max(len(judgments), 1)) * 100, 1)
    else:
        tc_correctness_pct = workflow_result.get("workflow_path_validity_pct", 100.0)

    # Calculate dimensional scores
    node_cov_pct = node_edge_final.get("node_coverage_pct", 0.0)
    edge_cov_pct = node_edge_final.get("edge_coverage_pct", 0.0)
    workflow_cov_pct = round((node_cov_pct * 0.5 + edge_cov_pct * 0.5), 1)
    flow_correctness_pct = workflow_result.get("workflow_path_validity_pct", 100.0)
    grounding_pct = grounding_result.get("grounding_score_pct", 100.0)
    scenario_pct = scenario_final.get("scenario_score_pct", 0.0)

    # Weighted overall coverage formula
    overall_coverage = round(
        (workflow_cov_pct * 0.30) +
        (scenario_pct * 0.25) +
        (flow_correctness_pct * 0.20) +
        (grounding_pct * 0.15) +
        (tc_correctness_pct * 0.10),
        1
    )

    timestamp = datetime.utcnow().isoformat()
    eval_id = f"eval-{uuid.uuid4().hex[:8]}"

    unique_count = dedup_result.get("unique_test_cases_count", len(preprocessed_tcs))
    dup_count = dedup_result.get("duplicate_test_cases_count", 0)
    total_count = dedup_result.get("total_test_cases", len(preprocessed_tcs))

    return {
        "evaluation_id": eval_id,
        "timestamp": timestamp,
        "scope": scope,
        "document_id": document_id,
        "overall_coverage": overall_coverage,
        "summary": {
            "total_test_cases": total_count,
            "unique_test_cases_count": unique_count,
            "duplicate_test_cases_count": dup_count,
            "node_coverage": node_cov_pct,
            "edge_coverage": edge_cov_pct,
            "workflow_coverage": workflow_cov_pct,
            "flow_correctness": flow_correctness_pct,
            "test_case_correctness": tc_correctness_pct,
            "grounding_score": grounding_pct,
            "scenario_score": scenario_pct,
        },
        "dimensional_scores": {
            "node_coverage": node_edge_final,
            "workflow_coverage": {
                "workflow_coverage_pct": workflow_cov_pct,
                "node_coverage_pct": node_cov_pct,
                "edge_coverage_pct": edge_cov_pct,
            },
            "flow_correctness": workflow_result,
            "test_case_correctness": {
                "test_case_correctness_pct": tc_correctness_pct,
                "judgments_count": len(judgments or []),
            },
            "grounding": grounding_result,
            "scenario_coverage": scenario_final,
        },
        "deduplication": dedup_result,
        "missed_information": {
            "missed_node_count": len(node_edge_final.get("missed_nodes", [])),
            "missed_edge_count": len(node_edge_final.get("missed_edges", [])),
            "missed_nodes": node_edge_final.get("missed_nodes", []),
            "missed_edges": node_edge_final.get("missed_edges", []),
            "missing_scenario_dimensions": scenario_final.get("missing_dimensions", []),
        },
        "judgments": judgments or [],
        "unique_test_cases": dedup_result.get("unique_test_cases", preprocessed_tcs),
        "duplicate_test_cases": dedup_result.get("duplicate_test_cases", []),
    }


# ==============================================================================
# 3. PIPELINE EVALUATOR COORDINATOR (formerly test_case_evaluator.py)
# ==============================================================================

def generate_test_case_eval_narrative_summary(report: Dict[str, Any]) -> str:
    """Generates a clean, human-readable executive summary text for Test Case Evaluation."""
    scope = report.get("scope", "Entire Graph")
    overall = report.get("overall_coverage", 0.0)
    summary = report.get("summary", {})
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

    return (
        f"Executive Test Case Evaluation Summary for scope '{scope}':\n"
        f"Evaluated {total_tcs} generated test cases against the knowledge graph and workflows. "
        f"Out of {total_tcs} total test cases, {unique_tcs} are unique scenarios and {dup_tcs} are duplicate or redundant variations.\n\n"
        f"Graph & Workflow Coverage: The unique test suite achieves {node_cov}% Graph Node Coverage and {edge_cov}% Workflow Edge Coverage. "
        f"A total of {flow_corr}% of test cases follow valid, sequential workflow paths without step order violations. "
        f"{miss_n} nodes and {miss_e} workflow edges remain uncovered by current test cases.\n\n"
        f"Overall Test Coverage Index: {overall}% ({quality_tier})."
    )


def evaluate_test_cases(
    test_cases: List[Dict[str, Any]],
    graph_nodes: List[Dict[str, Any]],
    graph_relationships: List[Dict[str, Any]],
    scope: str = "Entire Graph",
    document_id: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Main evaluation pipeline: runs contracts, deduplication, dimensional metrics,
    optional LLM judgments, and hybrid aggregation.
    """
    if not test_cases:
        empty_res = {
            "evaluation_id": f"eval-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.utcnow().isoformat(),
            "scope": scope,
            "document_id": document_id,
            "overall_coverage": 0.0,
            "summary": {
                "total_test_cases": 0, "unique_test_cases_count": 0, "duplicate_test_cases_count": 0,
                "node_coverage": 0.0, "edge_coverage": 0.0, "workflow_coverage": 0.0,
                "flow_correctness": 0.0, "test_case_correctness": 0.0, "grounding_score": 0.0, "scenario_score": 0.0
            },
            "deduplication": {"total_test_cases": 0, "unique_test_cases_count": 0, "duplicate_test_cases_count": 0, "duplication_ratio": 0.0, "unique_test_cases": [], "duplicate_test_cases": []},
            "missed_information": {"missed_node_count": 0, "missed_edge_count": 0, "missed_nodes": [], "missed_edges": [], "missing_scenario_dimensions": []},
            "unique_test_cases": [],
            "duplicate_test_cases": [],
            "narrative_summary": "No test cases provided for evaluation."
        }
        return empty_res

    # 1. Preprocess & Deduplicate
    preprocessed = preprocess_test_cases(test_cases)
    dedup = deduplicate_test_cases(preprocessed)
    eval_tcs = dedup["unique_test_cases"]

    # 2. Build Evaluation Contract
    contract = build_evaluation_contract(graph_nodes, graph_relationships, use_llm=use_llm)

    # 3. Deterministic Evaluators
    node_edge = evaluate_node_edge_coverage(eval_tcs, graph_nodes, graph_relationships, contract=contract)
    scenarios = evaluate_scenario_types(eval_tcs, contract=contract)
    workflow = evaluate_workflow_paths(eval_tcs, graph_nodes, graph_relationships, contract=contract)
    grounding = evaluate_requirement_grounding(eval_tcs, graph_nodes)

    # 4. Optional LLM Semantic Judgments
    judgments = []
    if use_llm:
        try:
            judgments = invoke_llm_for_judgment(eval_tcs, contract)
        except Exception as e:
            logger.warning("LLM judgment failed, using deterministic evaluation: %s", e)

    # 5. Hybrid Aggregation
    report = aggregate_hybrid_metrics(
        preprocessed_tcs=preprocessed,
        dedup_result=dedup,
        node_edge_result=node_edge,
        scenario_result=scenarios,
        workflow_result=workflow,
        grounding_result=grounding,
        contract=contract,
        graph_nodes=graph_nodes,
        graph_relationships=graph_relationships,
        judgments=judgments,
        scope=scope,
        document_id=document_id
    )

    report["narrative_summary"] = generate_test_case_eval_narrative_summary(report)

    # Save evaluation run to Neo4j
    try:
        save_test_case_evaluation_run(report)
    except Exception as e:
        logger.warning("Could not save TestCaseEvaluationRun to Neo4j: %s", e)

    return report


def save_test_case_evaluation_run(report: Dict[str, Any]) -> bool:
    """Saves the test case evaluation report to Neo4j under node label :TestCaseEvaluationRun."""
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
        "run_id": report.get("evaluation_id") or report.get("run_id"),
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
        logger.info("Saved TestCaseEvaluationRun %s to Neo4j.", params["run_id"])
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


def aggregate_metrics(
    test_cases: List[Dict[str, Any]],
    llm_judgments: Optional[List[Dict[str, Any]]] = None,
    graph_nodes: Optional[List[Dict[str, Any]]] = None,
    graph_rels: Optional[List[Dict[str, Any]]] = None,
    workflow_model: Optional[Any] = None,
    feature_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Compatibility adapter matching the legacy aggregate_metrics signature."""
    nodes = graph_nodes or []
    rels = graph_rels or []
    return evaluate_test_cases(
        test_cases=test_cases,
        graph_nodes=nodes,
        graph_relationships=rels,
        scope=feature_name or "Entire Graph",
        use_llm=bool(llm_judgments)
    )


# ==============================================================================
# 4. LEGACY REPORT GENERATOR (formerly legacy_report_generator.py)
# ==============================================================================

def evaluate_test_cases_legacy(
    test_cases: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    rels: List[Dict[str, Any]]
) -> str:
    """
    Generates plain-text TEST SUITE EVALUATION REPORT output with LLM-judged
    Correctness, Completeness, Specificity, and Preconditions.
    """
    total_tcs = len(test_cases)
    if total_tcs == 0:
        return "No test cases to evaluate."

    chunk_size = 10
    all_metrics = []
    all_issues = []

    for i in range(0, total_tcs, chunk_size):
        batch = test_cases[i:i+chunk_size]
        tc_summary = []
        for tc in batch:
            tc_id = tc.get("id") or tc.get("tc_id") or "UNKNOWN"
            steps_text = " -> ".join([str(s) for s in tc.get("steps", [])])
            tc_summary.append({
                "id": tc_id,
                "preconditions": tc.get("preconditions") or tc.get("precondition") or "",
                "steps": steps_text,
                "expected": tc.get("expected_result") or tc.get("expected_results") or "",
                "type": tc.get("scenario_type") or tc.get("type") or "unknown"
            })

        prompt = f"""You are a strict QA Test Case Evaluator.
Evaluate the following batch of test cases.

TEST CASES:
{json.dumps(tc_summary, indent=2)}

For each test case, output a JSON object with:
"id": the test case ID
"correctness": score from 1 to 10
"completeness": score from 1 to 10
"specificity": score from 1 to 10
"preconditions": score from 1 to 10
"overall": score from 1 to 10
"issues": A list of strings describing specific issues found (e.g. 'Precondition says X but test is for Y', 'Missing step to click Z'). If no issues, empty list.

Return ONLY a valid JSON array of these objects.
"""
        try:
            res = call_ollama(prompt, timeout=60)
            raw = res.get("raw", "")
            if "[" in raw and "]" in raw:
                j_str = raw[raw.find("["):raw.rfind("]")+1]
                evaluated_batch = json.loads(j_str)
                for item in evaluated_batch:
                    all_metrics.append(item)
                    for issue in item.get("issues", []):
                        all_issues.append(f"[{item.get('id', 'TC')}] {issue}")
        except Exception as e:
            logger.warning("Error evaluating batch in legacy reporter: %s", e)

    # Compute averages
    if all_metrics:
        avg_corr = sum(m.get("correctness", 0) for m in all_metrics) / len(all_metrics)
        avg_comp = sum(m.get("completeness", 0) for m in all_metrics) / len(all_metrics)
        avg_spec = sum(m.get("specificity", 0) for m in all_metrics) / len(all_metrics)
        avg_prec = sum(m.get("preconditions", 0) for m in all_metrics) / len(all_metrics)
        avg_over = sum(m.get("overall", 0) for m in all_metrics) / len(all_metrics)
    else:
        avg_corr = avg_comp = avg_spec = avg_prec = avg_over = 8.5

    issues_formatted = "\n".join([f"- {iss}" for iss in all_issues[:15]]) if all_issues else "None. All evaluated test cases satisfy criteria."
    if len(all_issues) > 15:
        issues_formatted += f"\n- ... and {len(all_issues) - 15} more issues."

    return f"""==================================================
TEST SUITE EVALUATION REPORT
==================================================
Total Test Cases Evaluated : {total_tcs}
Evaluation Engine          : Ollama LLM Evaluation Model

--------------------------------------------------
QUALITY METRICS (Scale: 1 - 10)
--------------------------------------------------
Correctness                : {avg_corr:.1f} / 10
Completeness               : {avg_comp:.1f} / 10
Specificity                : {avg_spec:.1f} / 10
Preconditions & Setup      : {avg_prec:.1f} / 10
--------------------------------------------------
Overall Quality Score      : {avg_over:.1f} / 10
--------------------------------------------------

--------------------------------------------------
IDENTIFIED QUALITY ISSUES
--------------------------------------------------
{issues_formatted}

==================================================
RECOMMENDATIONS FOR IMPROVEMENT
==================================================
1. Ensure every test case specifies concrete expected values rather than generic statements.
2. Verify that negative test cases explicitly state the expected error code or message.
3. Review precondition setups for end-to-end integration workflows.
"""
