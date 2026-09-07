"""
Query Engine Module
Complete GraphRAG query pipeline with level selection:
  ROOT  → broad questions about the entire system
  LOW   → medium questions about feature groups
  HIGH  → specific questions about individual communities
  AUTO  → automatically picks the best level
"""

import requests
from config import OLLAMA_URL
from embeddings.embedding_model import generate_embedding
from vectorstore.chroma_manager import (
    search_community_summaries,
    get_summaries_collection_count,
    get_summaries_count_by_level
)
from graph.hybrid_retriever import retrieve_graph_facts, build_graph_facts_text


QUERY_PROMPT = """
You are an expert requirements analyst assistant.

A user has asked a question about a software requirements document.
You have been provided with relevant community summaries extracted
from the knowledge graph of that document.

Use ONLY the information in the summaries below to answer the question.
Do not invent or assume any information not present in the summaries.
If the summaries do not contain enough information, say so clearly.

---
RELEVANT COMMUNITY SUMMARIES:
{summaries_text}
---

USER QUESTION:
{user_question}

---
ANSWER:
"""

# Hybrid variant: adds a second, structured evidence section built from
# a LIVE Cypher traversal of the matched communities (graph/hybrid_retriever.py)
# rather than relying on the summary text alone. Kept as a separate
# template (not a conditional insert into the one above) so the plain
# summary-only path stays byte-for-byte unchanged when no graph facts
# come back — never risk making an existing, working answer path worse
# to add a new one.
HYBRID_QUERY_PROMPT = """
You are an expert requirements analyst assistant.

A user has asked a question about a software requirements document.
You have been given TWO kinds of evidence from its knowledge graph:

1. Community summaries — human-readable overviews of related areas.
2. Graph facts — exact (Node) -[RELATIONSHIP]-> (Node) edges pulled
   live from the current graph for those same areas. These are the
   ground truth for exact connections; the summaries are framing/
   context. If they conflict, trust the graph facts.

Use ONLY the evidence below. Do not invent or assume anything not
present in it. If it isn't enough to answer, say so clearly.

---
RELEVANT COMMUNITY SUMMARIES:
{summaries_text}
---
RELEVANT GRAPH FACTS (live traversal):
{graph_facts_text}
---

USER QUESTION:
{user_question}

---
ANSWER:
"""


def build_summaries_text(summaries: list) -> str:
    text = ""
    for item in summaries:
        level = item.get("level", "HIGH")
        cid   = item.get("community_id", "?")
        text += f"\n[{level} Level — Community {cid}]\n"
        text += item["summary"]
        text += "\n---\n"
    return text


def retrieve_relevant_summaries(
    question: str,
    n_results: int = 3,
    level: str = None
) -> list:
    """
    Embed the question and search ChromaDB for matching summaries.

    Args:
        question:  user question string
        n_results: how many summaries to retrieve
        level:     "ROOT", "LOW", "HIGH", or None for all levels

    Returns:
        list of dicts with community_id, summary, similarity, level
    """
    query_embedding = generate_embedding(question)

    # Make sure we don't request more than what exists
    total = get_summaries_collection_count()
    if total == 0:
        return []
    n_results = min(n_results, total)

    results = search_community_summaries(
        query_embedding=query_embedding,
        n_results=n_results,
        level=level
    )

    ids       = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else []

    retrieved = []
    for idx, (doc_id, document, distance) in enumerate(
        zip(ids, documents, distances)
    ):
        metadata     = metadatas[idx] if metadatas else {}
        community_id = metadata.get("community_id", idx)
        result_level = metadata.get("level", "HIGH")
        similarity   = round(1 - distance, 4)

        retrieved.append({
            "community_id": community_id,
            "summary":      document,
            "similarity":   similarity,
            "level":        result_level
        })

    return retrieved


def generate_final_answer(
    question: str,
    retrieved_summaries: list,
    graph_facts_text: str = ""
) -> dict:
    """
    Send retrieved summaries (+ optional live graph facts) + question
    to llama3.1. Returns synthesized answer.

    graph_facts_text: output of build_graph_facts_text(); when non-empty,
    uses HYBRID_QUERY_PROMPT (summaries + live traversal). When empty
    (no communities resolved to live nodes, or hybrid mode off), falls
    back to the original QUERY_PROMPT unchanged — same answer this
    function has always produced when there's nothing new to add.
    """
    if not retrieved_summaries:
        return {
            "answer":  "No relevant summaries found to answer this question.",
            "success": False,
            "error":   "No summaries retrieved"
        }

    summaries_text = build_summaries_text(retrieved_summaries)

    if graph_facts_text:
        prompt = HYBRID_QUERY_PROMPT.format(
            summaries_text=summaries_text,
            graph_facts_text=graph_facts_text,
            user_question=question
        )
    else:
        prompt = QUERY_PROMPT.format(
            summaries_text=summaries_text,
            user_question=question
        )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  "llama3.1:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "seed":        42
                }
            },
            proxies={"http": None, "https": None},
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

        return {
            "answer":           answer,
            "communities_used": [s["community_id"] for s in retrieved_summaries],
            "success":          True
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "answer": "", "success": False,
            "error": f"Cannot reach Ollama: {str(e)}"
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "", "success": False,
            "error": "Ollama request timed out"
        }
    except Exception as e:
        return {
            "answer": "", "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def run_graphrag_query(
    question: str,
    level: str = None,
    n_results: int = 3,
    use_graph_traversal: bool = True
) -> dict:
    """
    Complete GraphRAG query pipeline — now genuinely hybrid:
    1. Semantic search over community summaries (unchanged).
    2. NEW: live Cypher traversal of the matched communities'
       actual current nodes/relationships (graph/hybrid_retriever.py).
    3. Both fed into one prompt (or step 1 alone if step 2 finds
       nothing live to traverse — same-quality answer as before, never
       worse).

    Args:
        question:  user question
        level:     "ROOT", "LOW", "HIGH", or None (search all)
        n_results: how many summaries to retrieve
        use_graph_traversal: set False to reproduce the exact
            pre-hybrid behavior (semantic-only) — useful for A/B
            comparison or if Neo4j is unreachable but Chroma isn't.

    Returns:
        {
            "answer": str,
            "retrieved_summaries": list,
            "graph_facts": {"nodes": [...], "relationships": [...], "communities_traversed": [...]},
            "communities_used": list,
            "level_used": str,
            "success": bool,
            "error": str
        }
    """
    try:
        retrieved = retrieve_relevant_summaries(
            question,
            n_results=n_results,
            level=level
        )
    except Exception as e:
        return {
            "answer": "", "retrieved_summaries": [],
            "graph_facts": {"nodes": [], "relationships": [], "communities_traversed": []},
            "communities_used": [], "success": False,
            "error": f"Retrieval failed: {str(e)}"
        }

    if not retrieved:
        return {
            "answer": "No relevant summaries found. Generate community summaries first.",
            "retrieved_summaries": [],
            "graph_facts": {"nodes": [], "relationships": [], "communities_traversed": []},
            "communities_used": [],
            "success": False, "error": "Empty retrieval"
        }

    graph_facts = {"nodes": [], "relationships": [], "communities_traversed": []}
    graph_facts_text = ""
    if use_graph_traversal:
        try:
            community_ids = [s["community_id"] for s in retrieved]
            graph_facts = retrieve_graph_facts(community_ids)
            graph_facts_text = build_graph_facts_text(graph_facts)
        except Exception as e:
            # Traversal failing (e.g. Neo4j asleep/unreachable) must NOT
            # take down the whole query — degrade to summary-only, same
            # as if use_graph_traversal=False, but note it happened
            # rather than silently proceeding as if nothing changed.
            graph_facts_text = ""
            graph_facts["error"] = f"Graph traversal failed, fell back to summary-only: {str(e)}"

    result = generate_final_answer(question, retrieved, graph_facts_text)

    return {
        "answer":              result.get("answer", ""),
        "retrieved_summaries": retrieved,
        "graph_facts":         graph_facts,
        "communities_used":    result.get("communities_used", []),
        "level_used":          level or "ALL",
        "success":             result.get("success", False),
        "error":               result.get("error", None)
    }