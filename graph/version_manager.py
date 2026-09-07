"""
Version Manager

Neo4j's MERGE ... SET (graph/neo4j_manager.py) writes new/changed
properties onto existing nodes IN PLACE -- great for "the graph
evolves instead of restarting," bad for "the original graph should
never be lost" on its own, since a later update can overwrite an
earlier value with nothing left to compare against.

This module closes that gap the simple way: before every incremental
update is applied (see graph/incremental_merge.py + app.py's Update
Graph tab), the CURRENT full graph state is captured into a
:GraphVersion node as a JSON snapshot. "Version N" always means
"exactly what the graph looked like right before update N+1 was
applied"; the live graph in Neo4j itself is always the latest,
current version -- see get_current_graph_as_version().
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
