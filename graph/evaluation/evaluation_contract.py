"""
graph/evaluation/evaluation_contract.py
Phase 5 of the Feature Flow Enhancement.

Builds a semantic evaluation contract directly from the source BRD and Knowledge Graph.
This contract serves as the ground-truth denominator for all test coverage metrics.
By deriving this from the source graph (rather than generated test cases), we prevent
the evaluator from blindly trusting incomplete or hallucinatory test generations.
"""

from typing import Dict, List, Any

from graph.semantic_classifier import classify_graph
from graph.feature_extractor import extract_features

def build_evaluation_contract(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Build the semantic evaluation contract.
    Calls the classifier and feature extractor if needed.
    """
    
    # 1. Classify graph to get the true source coverage universe
    classification = classify_graph(nodes, relationships, use_llm=use_llm)
    universe = classification["coverage_universe"]

    # 2. Extract features to map scenarios
    extraction = extract_features(nodes, relationships, classification, use_llm=use_llm)
    features = extraction.get("features", [])

    # 3. Build scenario applicability for each feature
    scenario_applicability = {}
    for feature in features:
        fname = feature.get("name", "")
        if not fname:
            continue
            
        # Analyze the feature to determine valid scenarios
        has_branches = len(feature.get("branches", [])) > 0
        has_negative = len(feature.get("negative_paths", [])) > 0
        has_alternate = len(feature.get("alternate_paths", [])) > 0
        has_decision = len(feature.get("decision_node_ids", [])) > 0
        
        # Determine failure/retry logic
        has_failure_retry = False
        for end_state in feature.get("end_state_ids", []):
            end_name = end_state.lower()
            if any(term in end_name for term in ["error", "fail", "reject", "invalid"]):
                has_failure_retry = True
        if has_negative:
            has_failure_retry = True

        scenario_applicability[fname] = {
            "positive": True, # Every feature has at least a happy path
            "negative": has_negative or has_failure_retry or has_branches,
            "edge": has_decision, # Decision boundaries often lead to edge cases
            "alternate": has_alternate or has_branches,
            "failure_retry": has_failure_retry
        }

    return {
        "features": features,
        "workflow_nodes": universe.get("workflow_nodes", []),
        "workflow_transitions": universe.get("workflow_transitions", []),
        "requirements": universe.get("requirement_nodes", []),
        "business_rules": universe.get("business_rule_nodes", []),
        "inputs": universe.get("input_nodes", []),
        "scenario_applicability": scenario_applicability
    }
