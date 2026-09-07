"""
Structural Linker — tags LLM-extracted entities/relationships with the
requirement item they came from.

Design history (why there's no backbone node builder here anymore):
    An earlier version of this module built a full "requirement
    backbone" — one graph node per valid item (e.g.
    "requirement_br_stay_6"), NEXT_IN_DOCUMENT sequence edges between
    consecutive items, and PART_OF edges from every LLM-extracted node
    back to its backbone node. That made the graph 100% connected with
    zero isolated nodes, but at a real cost: PART_OF fired
    unconditionally on every node (not just ones the LLM genuinely
    couldn't relate to anything), so it became the plurality of all
    edges in the graph — clutter that buried the LLM's own functional
    relationships (TRIGGERS, LEADS_TO, USES, CAUSES...) exactly where
    they mattered most, i.e. seeing how functionality actually flows.

    Requirement-ID nodes also add nothing to test case generation —
    test_case_generator.py works from parser/requirement_linker.py's
    items_with_links / links directly, never touches Neo4j or these
    backbone nodes at all.

    So: no backbone nodes, no PART_OF, no NEXT_IN_DOCUMENT. Traceability
    back to the source requirement is kept via node["source"] / 
    rel["source"] (set below) instead of a graph edge — every extracted
    node/relationship still knows which item it came from, it's just a
    property, not a connection you have to route the graph through.
    Functional sequence between entities is now visible directly via
    whatever real edges the LLM extracted (TRIGGERS, LEADS_TO, USES,
    CAUSES, SHOWS, ...) with nothing competing for attention.

    Accepted tradeoff: an entity the LLM genuinely couldn't relate to
    anything else within its own chunk can now end up as a real
    isolated node in the graph, instead of being artificially connected
    via PART_OF. graph/graph_eval.py's structural checks were updated
    to treat that as an expected, worth-a-look signal rather than a
    wiring bug — see its module docstring.
"""


def tag_extraction_source(graph_data: dict, item_id: str, doc_id: str = None, *args, **kwargs) -> dict:
    """
    Tag every node/relationship extract_entities() returned for one
    chunk with the requirement item it came from (populates the
    "source" field graph/schemas.py's Node/Relationship models already
    define). Optionally tags doc_id for document-level filtering.
    """
    effective_doc_id = doc_id or kwargs.get("doc_id")
    for node in graph_data.get("nodes", []):
        node["source"] = item_id
        if effective_doc_id:
            node["doc_id"] = effective_doc_id

    for rel in graph_data.get("relationships", []):
        rel["source"] = item_id
        if effective_doc_id:
            rel["doc_id"] = effective_doc_id

    return graph_data