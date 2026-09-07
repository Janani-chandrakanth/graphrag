# graph/evaluation/report.py
"""Evaluation Report Builder & Persistence Module.
Aggregates metrics across Retention, Relationships, Workflow, and Coverage into an overall Graph Quality Index.
Persists evaluation runs to Neo4j under `:EvaluationRun` nodes for historical tracking.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from graph.neo4j_manager import run_cypher_query

logger = logging.getLogger(__name__)

def generate_graph_eval_narrative_summary(report: Dict[str, Any]) -> str:
    """Generates a clean, human-readable executive summary text for Graph Quality Evaluation."""
    doc = report.get("document_name", "Requirement Document")
    overall = report.get("overall_score", 0.0)
    summary = report.get("summary", {})
    metrics = report.get("metrics", {})

    info_ret = metrics.get("information_retention", {})
    rel_q = metrics.get("relationship_quality", {})
    wf_q = metrics.get("workflow_sequence", {})
    nc_q = metrics.get("node_coverage", {})

    total_facts = info_ret.get("total_facts", 0)
    verified = info_ret.get("verified_count", 0)
    partial = info_ret.get("partial_count", 0)
    retention_f1 = round(summary.get("fact_f1", 0.0) * 100, 1)

    nodes_count = summary.get("total_nodes_in_graph", 0)
    rels_count = summary.get("total_relationships", 0)
    orphans = nc_q.get("orphan_nodes", 0)
    connectivity = round(rel_q.get("connectivity_rate", 0.0) * 100, 1)

    steps_count = wf_q.get("total_steps", 0)
    flow_rels = wf_q.get("total_flow_rels", 0)
    wf_acc = round(summary.get("workflow_accuracy", 0.0) * 100, 1)

    quality_tier = "High Quality" if overall >= 80 else ("Moderate Quality" if overall >= 60 else "Needs Improvement")

    narrative = (
        f"Executive Summary for '{doc}':\n"
        f"The Knowledge Graph comprises {nodes_count} nodes and {rels_count} relationships in Neo4j. "
        f"Out of {total_facts} requirement facts extracted from text, {verified} facts were fully verified "
        f"and {partial} facts were partially retained, yielding an Information Retention F1 score of {retention_f1}%.\n\n"
        f"Graph Health & Workflow Structure: Relationship connectivity rate is {connectivity}% with {orphans} orphan nodes. "
        f"The graph covers {steps_count} workflow steps connected across {flow_rels} sequential process edges, "
        f"achieving {wf_acc}% Workflow Sequence Accuracy.\n\n"
        f"Overall Quality Index: {overall}% ({quality_tier})."
    )
    return narrative


def build_evaluation_report(
    retention_results: Dict[str, Any],
    rel_results: Dict[str, Any],
    workflow_results: Dict[str, Any],
    coverage_results: Dict[str, Any],
    doc_name: str = "Requirement Document"
) -> Dict[str, Any]:
    """Combines metrics from all evaluation modules into an overall report."""
    
    fact_f1 = retention_results.get("fact_f1", 0.0)
    rel_f1 = rel_results.get("rel_f1", 0.0)
    workflow_acc = workflow_results.get("sequence_accuracy", 0.0)
    node_cov = coverage_results.get("node_coverage_score", 0.0)
    facts_available = retention_results.get("facts_available", True)

    # Dynamic Weighting depending on fact availability
    if facts_available and retention_results.get("total_facts", 0) > 0:
        w_retention = 0.35
        w_rel = 0.25
        w_wf = 0.25
        w_cov = 0.15
    else:
        w_retention = 0.10
        w_rel = 0.35
        w_wf = 0.35
        w_cov = 0.20

    weighted_score = (
        w_retention * fact_f1 +
        w_rel * rel_f1 +
        w_wf * workflow_acc +
        w_cov * node_cov
    )
    overall_score_pct = round(weighted_score * 100, 2)

    report = {
        "run_id": f"eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "document_name": doc_name,
        "overall_score": overall_score_pct,
        "weights": {
            "retention": int(w_retention * 100),
            "relationship": int(w_rel * 100),
            "workflow": int(w_wf * 100),
            "coverage": int(w_cov * 100)
        },
        "metrics": {
            "information_retention": retention_results,
            "relationship_quality": rel_results,
            "workflow_sequence": workflow_results,
            "node_coverage": coverage_results
        },
        "summary": {
            "fact_f1": fact_f1,
            "rel_f1": rel_f1,
            "workflow_accuracy": workflow_acc,
            "node_coverage": node_cov,
            "total_facts_evaluated": retention_results.get("total_facts", 0),
            "total_nodes_in_graph": coverage_results.get("total_nodes", 0),
            "total_relationships": rel_results.get("total_graph_rels", 0)
        }
    }

    report["narrative_summary"] = generate_graph_eval_narrative_summary(report)
    return report


def save_evaluation_run(report: Dict[str, Any]) -> bool:
    """Saves the evaluation report as an `:EvaluationRun` node in Neo4j."""
    run_id = report.get("run_id")
    timestamp = report.get("timestamp")
    doc_name = report.get("document_name", "Unknown Document")
    overall_score = report.get("overall_score", 0.0)
    summary = report.get("summary", {})

    cypher = """
    CREATE (e:EvaluationRun {
        id: $run_id,
        timestamp: $timestamp,
        document_name: $doc_name,
        overall_score: $overall_score,
        fact_f1: $fact_f1,
        rel_f1: $rel_f1,
        workflow_accuracy: $workflow_accuracy,
        node_coverage: $node_coverage,
        details_json: $details_json
    })
    RETURN e.id AS id
    """
    params = {
        "run_id": run_id,
        "timestamp": timestamp,
        "doc_name": doc_name,
        "overall_score": overall_score,
        "fact_f1": summary.get("fact_f1", 0.0),
        "rel_f1": summary.get("rel_f1", 0.0),
        "workflow_accuracy": summary.get("workflow_accuracy", 0.0),
        "node_coverage": summary.get("node_coverage", 0.0),
        "details_json": json.dumps(report)
    }

    try:
        res = run_cypher_query(cypher, params)
        logger.info("Successfully saved EvaluationRun %s to Neo4j.", run_id)
        return True
    except Exception as e:
        logger.error("Failed to save EvaluationRun to Neo4j: %s", e)
        return False


def get_evaluation_history() -> List[Dict[str, Any]]:
    """Retrieves past evaluation runs from Neo4j."""
    cypher = """
    MATCH (e:EvaluationRun)
    RETURN e.id AS run_id,
           e.timestamp AS timestamp,
           e.document_name AS document_name,
           e.overall_score AS overall_score,
           e.fact_f1 AS fact_f1,
           e.rel_f1 AS rel_f1,
           e.workflow_accuracy AS workflow_accuracy,
           e.node_coverage AS node_coverage,
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
        logger.error("Error fetching evaluation history from Neo4j: %s", e)
        return []
