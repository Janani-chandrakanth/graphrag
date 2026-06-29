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
    retrieved_summaries: list
) -> dict:
    """
    Send retrieved summaries + question to llama3.1.
    Returns synthesized answer.
    """
    if not retrieved_summaries:
        return {
            "answer":  "No relevant summaries found to answer this question.",
            "success": False,
            "error":   "No summaries retrieved"
        }

    summaries_text = build_summaries_text(retrieved_summaries)

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
    n_results: int = 3
) -> dict:
    """
    Complete GraphRAG query pipeline.

    Args:
        question:  user question
        level:     "ROOT", "LOW", "HIGH", or None (search all)
        n_results: how many summaries to retrieve

    Returns:
        {
            "answer": str,
            "retrieved_summaries": list,
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
            "communities_used": [], "success": False,
            "error": f"Retrieval failed: {str(e)}"
        }

    if not retrieved:
        return {
            "answer": "No relevant summaries found. Generate community summaries first.",
            "retrieved_summaries": [], "communities_used": [],
            "success": False, "error": "Empty retrieval"
        }

    result = generate_final_answer(question, retrieved)

    return {
        "answer":              result.get("answer", ""),
        "retrieved_summaries": retrieved,
        "communities_used":    result.get("communities_used", []),
        "level_used":          level or "ALL",
        "success":             result.get("success", False),
        "error":               result.get("error", None)
    }