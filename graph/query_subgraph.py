"""
graph/query_subgraph.py

Answers the gap you found: ask_graph() (graph/langchain_qa.py) returns
a text answer + raw Cypher result rows, but nothing ever turns that
back into a rendered subgraph. Since the Cypher is LLM-generated fresh
per question, we don't control its RETURN shape -- it might return full
node objects, or just scalar properties like a name string. This module
handles both cases defensively rather than assuming one.

Approach: scan the raw result rows for anything that looks like it
identifies a node in the live graph (a dict with an "id"/"name" key, or
a bare string matching a known node's id/name), resolve those against
the full live graph, then pull every relationship directly connecting
the matched nodes (not a full neighborhood expansion, to keep the
rendered picture tightly scoped to what the query actually touched).
"""

from graph.neo4j_manager import get_all_nodes, get_all_relationships


def _candidate_strings_from_result(result_rows: list) -> set:
    """Pulls every plausible node-identifying string out of arbitrary
    Cypher result rows, regardless of whether Neo4j returned full node
    objects, property dicts, or bare scalars for a given column."""
    found = set()

    def walk(value):
        if isinstance(value, str):
            found.add(value.strip().lower())
        elif isinstance(value, dict):
            # Neo4j node objects (via langchain-neo4j) typically surface
            # as a dict of properties -- id/name are the two this
            # project's :Entity nodes always have (see graph/schemas.py).
            for key in ("id", "name"):
                if key in value and isinstance(value[key], str):
                    found.add(value[key].strip().lower())
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    for row in result_rows:
        walk(row)
    return found


def extract_subgraph_from_result(result_rows: list, nodes: list = None, relationships: list = None) -> tuple:
    """
    Args:
        result_rows: ask_graph()'s result["result"] -- raw Cypher rows.
        nodes, relationships: optionally supply the graph directly
            (avoids a live Neo4j round-trip if you already have the
            session's data); pulls the full live graph otherwise.

    Returns:
        (subgraph_nodes, subgraph_relationships) -- ready to pass
        straight into graph/graph_visualizer.py's render_graph().
        Both empty if nothing in the result matched a known node
        (e.g. the query only returned counts/aggregates, not entities).
    """
    if nodes is None or relationships is None:
        nodes = get_all_nodes()
        relationships = get_all_relationships()

    if not result_rows:
        return [], []

    candidates = _candidate_strings_from_result(result_rows)
    if not candidates:
        return [], []

    matched_ids = {
        n["id"] for n in nodes
        if n.get("id", "").strip().lower() in candidates
        or n.get("name", "").strip().lower() in candidates
    }
    if not matched_ids:
        return [], []

    subgraph_nodes = [n for n in nodes if n["id"] in matched_ids]
    subgraph_relationships = [
        r for r in relationships
        if r["from"] in matched_ids and r["to"] in matched_ids
    ]
    return subgraph_nodes, subgraph_relationships