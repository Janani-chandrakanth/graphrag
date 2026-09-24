"""
graph/structural_linker.py

Tags LLM-extracted nodes and relationships with their originating requirement item IDs
for source traceability across the knowledge graph.
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