"""
Test Case Graph Writer

Closes a loop this project's architecture review flagged as missing:
generated test cases (graph/graph_test_case_generator.py's output)
existed only as Python dicts — never written back into Neo4j — so
"which KB node inspired this test step" lived only in a `graph_nodes`
name-list inside that dict, not as a real, queryable graph edge.

This adapts the GraphRAG proposal's
    (Test Case) -[VALIDATES]-> (Acceptance Criteria)
to this project's ACTUAL ontology: there's no dedicated
AcceptanceCriteria node type extracted yet, so VALIDATES points at
whichever domain nodes the test case's walk actually touched — those
ARE the acceptance-relevant entities for that scenario. This is the
Traceability Map from the proposal, made real:
    (TestCase) -[VALIDATES]-> (Screen/Action/DataObject/... node)
    (TestCase) -[VERIFIES]->  (the source Requirement node, if resolvable)

Design (deterministic, MERGE-idempotent — same pattern as
graph/neo4j_manager.py's insert_graph; re-running generation + writing
never duplicates, it overwrites):
"""

from graph.neo4j_manager import insert_graph, get_all_nodes


def _tc_node_id(tc_id: str) -> str:
    return f"TESTCASE::{tc_id}"


def build_test_case_graph_elements(test_cases: list, nodes: list) -> tuple:
    """
    Convert generate_test_cases_from_graph()'s output into
    (nodes, relationships) ready for neo4j_manager.insert_graph().

    Kept pure/testable — does NOT touch Neo4j itself, same
    build-then-insert separation graph/graph_builder.py already uses.
    """
    # Resolution source: the FULL LIVE GRAPH (get_all_nodes()), not just
    # `nodes` (the current session's freshly-extracted batch). This was
    # the actual bug -- hybrid_test_case_generator.py's graph half
    # deliberately pulls in cross-series/cross-document neighbor
    # entities via a live 1-hop Cypher traversal (get_neighbor_nodes_db),
    # and the prompt explicitly allows the LLM to put those neighbors'
    # names into entities_used. But those neighbor nodes were written to
    # Neo4j in a PRIOR session/document ingest -- they were never in
    # THIS session's `nodes` list, so a lookup built only from `nodes`
    # silently had zero chance of resolving anything that came from the
    # graph-neighbor half. `nodes` is still merged in on top, so brand
    # new nodes from the current session that haven't round-tripped
    # through Neo4j yet still resolve too.
    #
    # Also: the prompt tells the LLM "its id or its name, either is
    # fine" (kg_extraction catalog format is "id | type | name"), so the
    # lookup must match on EITHER id or name, not name only -- and case-
    # insensitively, matching the lowercasing the generator itself
    # already does for its own ungrounded-entity check (see
    # hybrid_test_case_generator.py's `valid_tokens`).
    try:
        live_nodes = get_all_nodes()
    except Exception:
        # Live graph unreachable (e.g. Neo4j down) -- degrade to
        # session-only resolution rather than failing the whole write.
        live_nodes = []

    token_to_id = {}
    all_ids = set()
    for n in (live_nodes + list(nodes)):
        node_id = n.get("id")
        if not node_id:
            continue
        all_ids.add(node_id)
        token_to_id[node_id.strip().lower()] = node_id
        name = n.get("name")
        if name:
            token_to_id.setdefault(name.strip().lower(), node_id)

    new_nodes = []
    new_rels = []

    for tc in test_cases:
        # Defensive: skip non-dict items (e.g. raw text strings that can
        # appear when a generator mixes text output with dict objects).
        if not isinstance(tc, dict):
            continue
        tc_node_id = _tc_node_id(tc["tc_id"])

        new_nodes.append({
            "id": tc_node_id,
            "name": tc["tc_id"],
            "type": "TestCase",
            "description": tc.get("title", ""),
            "source": tc.get("req_id"),
            "aliases": [],
        })

        # VALIDATES -> every distinct domain node this test case's walk
        # actually touched (dedup by target id — a test case can repeat
        # a node name across steps, e.g. re-visiting the same Screen).
        seen_targets = set()
        for token in tc.get("graph_nodes", []):
            target_id = token_to_id.get(token.strip().lower()) if token else None
            if not target_id or target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            new_rels.append({"from": tc_node_id, "to": target_id, "type": "VALIDATES"})

        # VERIFIES -> source requirement item(s), ONLY if that id
        # actually resolves to a real node already in the graph. Never
        # fabricate an edge to an id that isn't there — same
        # never-silently-disguise convention the rest of this pipeline
        # follows (a dangling MATCH in neo4j_manager would just silently
        # write nothing, which is worse than not trying).
        for req_id in tc.get("source_items", []):
            if req_id in all_ids:
                new_rels.append({"from": tc_node_id, "to": req_id, "type": "VERIFIES"})

    return new_nodes, new_rels


def write_test_cases_to_graph(test_cases: list, nodes: list) -> dict:
    """
    Build + write test case nodes/edges to Neo4j in one call.

    Returns a summary dict for the UI to render — counts only, no
    silent success/failure ambiguity (insert_graph's own retry/
    reconnect handling in neo4j_manager.py surfaces connection errors
    by raising, same as every other write path in this project).
    """
    tc_nodes, tc_rels = build_test_case_graph_elements(test_cases, nodes)
    insert_graph(tc_nodes, tc_rels)

    validates_count = sum(1 for r in tc_rels if r["type"] == "VALIDATES")
    verifies_count = sum(1 for r in tc_rels if r["type"] == "VERIFIES")

    unresolved = [
        tc["tc_id"] for tc in test_cases
        if isinstance(tc, dict) and tc.get("graph_nodes") and not any(
            r["from"] == _tc_node_id(tc["tc_id"]) and r["type"] == "VALIDATES"
            for r in tc_rels
        )
    ]

    return {
        "test_case_nodes_written": len(tc_nodes),
        "validates_edges_written": validates_count,
        "verifies_edges_written": verifies_count,
        # Test cases whose graph_nodes names didn't resolve to any known
        # id at all — visible here instead of just quietly producing
        # zero VALIDATES edges for that test case.
        "test_cases_with_no_resolved_edges": unresolved,
    }