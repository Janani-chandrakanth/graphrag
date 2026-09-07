# graph/evaluation/workflow_evaluator.py
"""Workflow & Sequence Evaluator for CIG Evaluation Module.
Evaluates sequential business flow accuracy, procedural step order, branching completeness, and step transitions.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

WORKFLOW_NODE_TYPES = {
    "WORKFLOWSTEP", "FEATURE", "STEP", "ACTION", "ACTIVITY",
    "PROCESS", "EVENT", "DECISION", "USECASE", "USERSTORY", "REQUIREMENT"
}

SEQUENCE_REL_TYPES = {
    "NEXT", "HAS_STEP", "LEADS_TO", "TRIGGERS", "PRECEDES",
    "FLOWS_TO", "DEPENDS_ON", "FOLLOWS", "THEN"
}

def evaluate_workflow(workflow_model: Optional[Any], graph_nodes: List[Dict[str, Any]], graph_rels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates business workflow quality and procedural step ordering."""
    total_nodes = len(graph_nodes)
    if total_nodes == 0:
        return {
            "sequence_accuracy": 0.0,
            "step_coverage": 0.0,
            "transition_accuracy": 0.0,
            "total_steps": 0,
            "total_flow_rels": 0,
            "feature_count": 0,
            "total_feature_steps": 0,
            "valid_transitions": 0,
            "orphan_step_count": 0
        }

    # 1. Filter Workflow / Step nodes
    step_nodes = [
        n for n in graph_nodes
        if str(n.get("type", "")).upper() in WORKFLOW_NODE_TYPES or n.get("sequence") is not None
    ]
    
    # Fallback if no specific step types found: use all graph nodes
    if not step_nodes:
        step_nodes = graph_nodes

    # 2. Extract sequence flow relationships
    flow_rels = [
        r for r in graph_rels
        if str(r.get("type") or r.get("relationship") or "").upper() in SEQUENCE_REL_TYPES
    ]

    total_steps = len(step_nodes)
    total_flow_rels = len(flow_rels)

    # 3. Step Coverage & Sequence Completeness
    connected_step_ids = set()
    for rel in flow_rels:
        src = str(rel.get("source") or rel.get("start") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("end") or rel.get("to") or "")
        if src: connected_step_ids.add(src)
        if tgt: connected_step_ids.add(tgt)

    step_coverage_ratio = round(len(connected_step_ids) / max(total_steps, 1), 4)

    # 4. Check Sequence Order Properties (n.sequence)
    sequenced_nodes = [n for n in graph_nodes if n.get("sequence") is not None]
    sequence_prop_ratio = round(len(sequenced_nodes) / max(total_nodes, 1), 4)

    # 5. Workflow Model Feature Analysis
    feature_count = 0
    total_feature_steps = 0
    valid_transitions = 0

    if workflow_model and hasattr(workflow_model, "features"):
        features = getattr(workflow_model, "features", [])
        feature_count = len(features)
        
        for feat in features:
            steps = getattr(feat, "steps", [])
            total_feature_steps += len(steps)

            for i in range(len(steps) - 1):
                curr_id = str(getattr(steps[i], "id", ""))
                next_id = str(getattr(steps[i + 1], "id", ""))

                has_edge = any(
                    (str(r.get("source") or r.get("start") or r.get("from")) == curr_id and
                     str(r.get("target") or r.get("end") or r.get("to")) == next_id)
                    for r in graph_rels
                )
                if has_edge:
                    valid_transitions += 1

    # 6. Calculate Accuracy Scores
    if total_feature_steps > 1:
        transition_accuracy = round(valid_transitions / max(total_feature_steps - feature_count, 1), 4)
    elif total_flow_rels > 0:
        transition_accuracy = round(min(1.0, total_flow_rels / max(total_steps, 1)), 4)
    else:
        transition_accuracy = sequence_prop_ratio

    sequence_accuracy = round(0.5 * step_coverage_ratio + 0.5 * transition_accuracy, 4)

    return {
        "sequence_accuracy": sequence_accuracy,
        "step_coverage": step_coverage_ratio,
        "transition_accuracy": transition_accuracy,
        "total_steps": total_steps,
        "total_flow_rels": total_flow_rels,
        "feature_count": feature_count,
        "total_feature_steps": total_feature_steps,
        "valid_transitions": valid_transitions,
        "orphan_step_count": max(0, total_steps - len(connected_step_ids))
    }
