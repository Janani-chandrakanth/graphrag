import json
from typing import List, Dict, Optional

from .schemas import WorkflowModel, Feature, Step
from .entity_extractor import extract_workflow_sequence


def extract_workflow(document_text: str) -> WorkflowModel:
    """Extract a workflow representation from raw requirement text.

    This high‑level wrapper orchestrates entity extraction (via ``extract_entities``) and
    sequence extraction to produce a ``WorkflowModel`` object containing features and
    ordered steps.
    """
    # 1. Extract entities (nodes) – reuse existing entity_extractor logic.
    from .entity_extractor import extract_entities
    entity_result = extract_entities(document_text)
    nodes = entity_result.get("nodes", [])
    # 2. Detect sequential workflows using the existing sequence extractor.
    seq_result = extract_workflow_sequence(document_text, nodes)
    sequences = seq_result.get("sequences", [])

    # 3. Group sequences by feature. For simplicity we treat each unique "flow_entry_id"
    #    as the start of a feature. In many documents the flow entry corresponds to a
    #    top‑level feature or a major process step.
    features_dict: Dict[str, Feature] = {}
    for seq in sequences:
        entry_id = seq.get("flow_entry_id")
        if not entry_id:
            continue
        # Find the node representing the entry to obtain a name (fallback to ID).
        entry_node = next((n for n in nodes if n["id"] == entry_id), None)
        feature_name = entry_node.get("name") if entry_node else f"Feature_{entry_id}"
        feature = features_dict.setdefault(feature_name, Feature(name=feature_name))
        # Build steps from the sequence edges.
        step_map: Dict[str, Step] = {}
        for edge in seq.get("sequence_edges", []):
            src_id = edge.get("from")
            dst_id = edge.get("to")
            # Ensure source step exists.
            if src_id not in step_map:
                src_node = next((n for n in nodes if n["id"] == src_id), {})
                step_map[src_id] = Step(
                    id=src_id,
                    action=src_node.get("name", src_id),
                    actors=_collect_actors(src_node, nodes),
                )
            # Ensure destination step exists.
            if dst_id not in step_map:
                dst_node = next((n for n in nodes if n["id"] == dst_id), {})
                step_map[dst_id] = Step(
                    id=dst_id,
                    action=dst_node.get("name", dst_id),
                    actors=_collect_actors(dst_node, nodes),
                )
            # Link steps.
            step_map[src_id].next_steps.append(dst_id)
        # Add all steps to the feature preserving insertion order.
        for step_id, step in step_map.items():
            if step not in feature.steps:
                feature.steps.append(step)
    # Convert dict to list.
    workflow = WorkflowModel(features=list(features_dict.values()))
    return workflow


def _collect_actors(node: dict, all_nodes: List[dict]) -> List[str]:
    """Utility to collect actor IDs associated with a node.

    The entity extraction marks ``type == 'Actor'`` nodes. If a node references an actor
    via an ``attributes`` field or a relationship, that information would be supplied
    by the caller. For this simple implementation we only return a list containing the
    node's own ID when its type is ``Actor``.
    """
    if node.get("type") == "Actor":
        return [node.get("id")]
    # Look for an explicit ``actors`` attribute if present.
    return node.get("actors", [])


def extract_workflow_from_graph(nodes: list, relationships: list) -> WorkflowModel:
    """
    Build a WorkflowModel directly from already-extracted nodes and
    relationships (in-memory, zero LLM calls).
    Groups nodes by Feature (is_flow_entry / type=='Feature') and
    walks the LEADS_TO / NEXT / TRIGGERS edges to produce ordered Step lists.
    Actors attached via PERFORMS relationships become step.actors.
    """
    from graph.flow_graph_analysis import FLOW_RELATIONS
    # Index nodes and build adjacency
    node_by_id = {n["id"]: n for n in nodes}
    outgoing: dict = {}   # id -> [to_id, ...]
    performs: dict = {}   # step_id -> [actor_name, ...]
    for r in relationships:
        frm, to, rtype = r.get("from"), r.get("to"), r.get("type", "")
        if not frm or not to:
            continue
        if rtype in FLOW_RELATIONS or rtype in {"LEADS_TO", "NEXT", "TRIGGERS"}:
            outgoing.setdefault(frm, []).append(to)
        if rtype == "PERFORMS":
            actor_node = node_by_id.get(frm)
            actor_name = (actor_node.get("name") if actor_node else None) or frm
            performs.setdefault(to, []).append(actor_name)
    # Identify feature entry points: explicit Feature nodes or is_flow_entry flags
    feature_entries = []
    for n in nodes:
        ntype = n.get("type", "")
        attrs = n.get("attributes") or {}
        if ntype == "Feature" or str(attrs.get("is_flow_entry", "")).lower() == "true":
            feature_entries.append(n)
    # If no explicit features found, or if we want to catch unflagged flows,
    # find all nodes that are the START of a flow (have outgoing flow edges but no incoming ones).
    if not feature_entries:
        has_incoming_flow = set()
        has_outgoing_flow = set()
        for r in relationships:
            if r.get("type") in (FLOW_RELATIONS | {"LEADS_TO", "NEXT", "TRIGGERS"}):
                has_incoming_flow.add(r.get("to"))
                has_outgoing_flow.add(r.get("from"))
        for n in nodes:
            nid = n["id"]
            if nid in has_outgoing_flow and nid not in has_incoming_flow:
                if n.get("type") not in {"Actor", "Attribute", "Constraint"}:
                    feature_entries.append(n)
        # Cap to avoid creating hundreds of single-node features
        feature_entries = feature_entries[:20]
    features = []
    visited_globally: set = set()
    for entry in feature_entries:
        feat_name = entry.get("name") or entry["id"]
        # BFS/DFS walk along flow edges
        steps, queue, seen = [], [entry["id"]], set()
        while queue:
            curr_id = queue.pop(0)
            if curr_id in seen or curr_id not in node_by_id:
                continue
            seen.add(curr_id)
            curr_node = node_by_id[curr_id]
            nexts = outgoing.get(curr_id, [])
            steps.append(Step(
                id=curr_id,
                action=curr_node.get("name") or curr_id,
                actors=performs.get(curr_id, []),
                systems=[],
                data_objects=[],
                conditions=[],
                next_steps=nexts,
            ))
            for nxt in nexts:
                if nxt not in seen:
                    queue.append(nxt)
        visited_globally |= seen
        # Include all features (procedural + structural / single-capability)
        # as requested in the brief, so the dropdown represents the complete feature space.
        if len(steps) >= 1:
            features.append(Feature(name=feat_name, steps=steps))
    return WorkflowModel(features=features)

