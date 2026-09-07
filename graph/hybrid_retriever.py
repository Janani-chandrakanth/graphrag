"""
Hybrid Retriever

Closes gap #2 from the architecture review: graph/query_engine.py is
named a "GraphRAG" pipeline but only ever did ONE retrieval mode —
semantic search over LLM-written community summary TEXT
(vectorstore/chroma_manager.search_community_summaries). It never
actually walked the live graph. That means:

  - Answers are only as fresh as the last time community summaries
    were regenerated (graph/community_summarizer.py) — if the graph
    changed since (e.g. new documents, or Test Case nodes written by
    graph/test_case_writer.py) and summaries weren't rebuilt, the
    answer is grounded in stale text, not the actual current graph.
  - A summary is a lossy compression of a community's nodes/edges —
    fine for "what is this area about", weak for "what EXACTLY
    connects to X", which is precisely what the proposal's "Graph
    Traversal: Pulls all related Flow Steps and Business Rules
    connected to that feature" step needs.

This module adds the second half of hybrid search WITHOUT touching the
first: same semantic step as before (unchanged,
graph/query_engine.retrieve_relevant_summaries), PLUS a live Cypher
pull of each matched community's actual current nodes/relationships —
reusing graph/neo4j_manager.get_community_nodes_db /
get_community_relationships_db, which already existed but were never
wired into the query pipeline itself, only into the Phase 2 UI's
manual community browser.

Both retrieval modes' output get combined into ONE prompt so the LLM
answers from live graph facts AND the human-written summary framing,
not either alone.
"""

from graph.neo4j_manager import get_community_nodes_db, get_community_relationships_db


def retrieve_graph_facts(community_ids: list, max_nodes_per_community: int = 25) -> dict:
    """
    Live Cypher pull of each matched community's current
    nodes/relationships — the "Graph Traversal" half of hybrid search.

    Args:
        community_ids: community ids surfaced by the semantic search
            step (graph/query_engine.retrieve_relevant_summaries).
        max_nodes_per_community: cap per community so a single
            oversized/hub-heavy community (see the graph_eval System
            hub warning) can't blow out the prompt token budget —
            token efficiency was one of the proposal's own stated
            goals for retrieving sub-graphs instead of whole documents.

    Returns:
        {"nodes": [...], "relationships": [...], "communities_traversed": [...]}
        nodes/relationships deduped across communities (a node can sit
        in more than one community_id passed in, in principle).
    """
    seen_node_ids = set()
    all_nodes = []
    all_rels = []
    traversed = []

    for cid in community_ids:
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            # community_id metadata can come back as non-numeric in
            # edge cases (e.g. missing metadata) — skip rather than
            # let a bad id crash the whole retrieval.
            continue

        nodes = get_community_nodes_db(cid_int)[:max_nodes_per_community]
        rels = get_community_relationships_db(cid_int)

        if not nodes:
            # Semantic search matched a summary for a community that no
            # longer has live nodes (e.g. graph was rebuilt since that
            # summary was generated) — visible via empty traversal for
            # this id, not silently dropped from the count.
            continue

        traversed.append(cid_int)
        kept_ids = {n["id"] for n in nodes}
        for n in nodes:
            if n["id"] not in seen_node_ids:
                seen_node_ids.add(n["id"])
                all_nodes.append(n)
        for r in rels:
            if r["from"] in kept_ids and r["to"] in kept_ids:
                all_rels.append(r)

    return {
        "nodes": all_nodes,
        "relationships": all_rels,
        "communities_traversed": traversed,
    }


def build_graph_facts_text(graph_facts: dict) -> str:
    """
    Render live graph facts as compact "Node -[REL]-> Node" lines for
    prompt inclusion — structured ground truth alongside the summary
    text, not a replacement for it.
    """
    if not graph_facts["nodes"]:
        return ""

    id_to_name = {n["id"]: n.get("name", n["id"]) for n in graph_facts["nodes"]}

    lines = []
    for r in graph_facts["relationships"]:
        src = id_to_name.get(r["from"], r["from"])
        tgt = id_to_name.get(r["to"], r["to"])
        lines.append(f"({src}) -[{r['type']}]-> ({tgt})")

    # Nodes with no edges within the traversed set still carry
    # information (e.g. an isolated Constraint) — list them separately
    # rather than dropping them because they didn't produce a line above.
    edged_ids = {r["from"] for r in graph_facts["relationships"]} | {
        r["to"] for r in graph_facts["relationships"]
    }
    isolated = [n for n in graph_facts["nodes"] if n["id"] not in edged_ids]
    for n in isolated:
        lines.append(f"({n.get('name', n['id'])}) [{n.get('type', '?')}, no traversed edges]")

    return "\n".join(lines)
