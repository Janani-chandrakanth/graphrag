"""
Graph Version Comparison

Diffs two graph states -- each {"nodes": [...], "relationships": [...]},
the shape graph/version_manager.py's get_version_snapshot() /
get_current_graph_as_version() both return -- into added/removed/
modified nodes, added/removed relationships, and a workflow-level
"same step, different next step" view, for app.py's Compare tab.
"""


def _node_signature(node: dict) -> tuple:
    """What counts as 'the same content' -- used to detect a MODIFIED
    node (same id, different substance) vs an untouched one."""
    return (
        node.get("type"),
        node.get("description"),
        tuple(sorted(node.get("aliases") or [])),
    )


def _rel_key(rel: dict) -> tuple:
    return (rel["from"], rel["to"], rel["type"])


def diff_graphs(version_a: dict, version_b: dict) -> dict:
    """
    version_a = older/base version, version_b = newer/comparison version.
    """
    nodes_a = {n["id"]: n for n in version_a.get("nodes", [])}
    nodes_b = {n["id"]: n for n in version_b.get("nodes", [])}

    added_nodes = [nodes_b[nid] for nid in nodes_b if nid not in nodes_a]
    removed_nodes = [nodes_a[nid] for nid in nodes_a if nid not in nodes_b]
    modified_nodes = [
        {"id": nid, "before": nodes_a[nid], "after": nodes_b[nid]}
        for nid in nodes_a
        if nid in nodes_b and _node_signature(nodes_a[nid]) != _node_signature(nodes_b[nid])
    ]

    rels_a = {_rel_key(r): r for r in version_a.get("relationships", [])}
    rels_b = {_rel_key(r): r for r in version_b.get("relationships", [])}

    added_rels = [rels_b[k] for k in rels_b if k not in rels_a]
    removed_rels = [rels_a[k] for k in rels_a if k not in rels_b]

    # Workflow-level view: same (from, type) pair, different target set
    # -- e.g. Payment -[LEADS_TO]-> Receipt in A, -> Invoice in B.
    def outgoing_by_from_type(rels):
        m = {}
        for r in rels:
            m.setdefault((r["from"], r["type"]), set()).add(r["to"])
        return m

    flow_a = outgoing_by_from_type(version_a.get("relationships", []))
    flow_b = outgoing_by_from_type(version_b.get("relationships", []))
    changed_workflow = [
        {"from": key[0], "type": key[1], "before": sorted(flow_a[key]), "after": sorted(flow_b[key])}
        for key in flow_b
        if key in flow_a and flow_a[key] != flow_b[key]
    ]

    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "modified_nodes": modified_nodes,
        "added_relationships": added_rels,
        "removed_relationships": removed_rels,
        "changed_workflow": changed_workflow,
        "summary": {
            "added_nodes": len(added_nodes),
            "removed_nodes": len(removed_nodes),
            "modified_nodes": len(modified_nodes),
            "added_relationships": len(added_rels),
            "removed_relationships": len(removed_rels),
            "changed_workflow": len(changed_workflow),
        },
    }
