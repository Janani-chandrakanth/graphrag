"""
graph/version_manager.py

Manages versioning and snapshots for knowledge graphs, persisting pre-update graph states in Neo4j.
"""

import json
import uuid
from datetime import datetime, timezone

from graph.neo4j_manager import get_all_nodes, get_all_relationships, run_write_query, run_read_query


def new_version_id() -> str:
    return f"v_{uuid.uuid4().hex[:10]}"


def snapshot_current_graph(label: str, description: str = "", source_filename: str = None,
                            version_id: str = None) -> dict:
    """
    Capture the CURRENT live graph as a retained version. Call this
    BEFORE writing an incremental update, so the pre-update state is
    preserved even though the update itself will mutate nodes/edges
    in place.
    """
    nodes = get_all_nodes()
    relationships = get_all_relationships()

    version_id = version_id or new_version_id()
    record = {
        "id": version_id,
        "label": label,
        "description": description or "",
        "source_filename": source_filename or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "nodes_json": json.dumps(nodes),
        "relationships_json": json.dumps(relationships),
    }

    run_write_query(
        """
        CREATE (v:GraphVersion {
            id: $id, label: $label, description: $description,
            source_filename: $source_filename, created_at: $created_at,
            node_count: $node_count, relationship_count: $relationship_count,
            nodes_json: $nodes_json, relationships_json: $relationships_json
        })
        """,
        **record,
    )

    prev = get_latest_version(exclude_id=version_id)
    if prev:
        run_write_query(
            """
            MATCH (a:GraphVersion {id: $prev_id}), (b:GraphVersion {id: $new_id})
            MERGE (a)-[:PRECEDES]->(b)
            """,
            prev_id=prev["id"], new_id=version_id,
        )

    # Don't leak the (potentially large) JSON blobs back to the caller
    # by default -- callers that render a version list only need the
    # metadata, same shape as list_versions() below.
    return {k: v for k, v in record.items() if k not in ("nodes_json", "relationships_json")}


def list_versions() -> list:
    """Metadata only (no node/relationship payload) -- for populating
    version pickers in the UI. Oldest first."""
    return run_read_query(
        """
        MATCH (v:GraphVersion)
        RETURN v.id AS id, v.label AS label, v.description AS description,
               v.source_filename AS source_filename, v.created_at AS created_at,
               v.node_count AS node_count, v.relationship_count AS relationship_count
        ORDER BY v.created_at ASC
        """
    )


def get_latest_version(exclude_id: str = None) -> dict:
    versions = list_versions()
    if exclude_id:
        versions = [v for v in versions if v["id"] != exclude_id]
    return versions[-1] if versions else None


def get_version_snapshot(version_id: str) -> dict:
    """{"nodes": [...], "relationships": [...]} for a stored snapshot."""
    rows = run_read_query(
        "MATCH (v:GraphVersion {id: $id}) RETURN v.nodes_json AS nodes_json, "
        "v.relationships_json AS relationships_json",
        id=version_id,
    )
    if not rows:
        return {"nodes": [], "relationships": []}
    return {
        "nodes": json.loads(rows[0]["nodes_json"] or "[]"),
        "relationships": json.loads(rows[0]["relationships_json"] or "[]"),
    }


def get_current_graph_as_version() -> dict:
    """The live graph, in the same shape as a stored snapshot, so the
    Compare tab can treat 'current' as just another version without a
    real :GraphVersion node existing for it (it's the implicit latest
    one at all times)."""
    return {
        "id": "current",
        "label": "Current (live graph)",
        "nodes": get_all_nodes(),
        "relationships": get_all_relationships(),
    }


# ───────────────────────────────────────────────────────────────
# Graph Version Comparison / Diffing
# ───────────────────────────────────────────────────────────────

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

