"""
ChromaDB Manager
Manages two collections:
  1. requirements_chunks   — raw requirement chunks
  2. community_summaries   — LLM summaries at HIGH, LOW, ROOT levels
"""

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# ── Collection 1: Raw Chunks ───────────────────────────────
chunks_collection = client.get_or_create_collection(
    name="requirements_chunks"
)


def store_chunk(chunk_id, chunk_text, embedding):
    chunks_collection.upsert(
        ids=[chunk_id],
        documents=[chunk_text],
        embeddings=[embedding]
    )


def search_chunks(query_embedding, n_results=5):
    return chunks_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )


def get_collection_count():
    return chunks_collection.count()


# ── Collection 2: Community Summaries ─────────────────────
summaries_collection = client.get_or_create_collection(
    name="community_summaries"
)


def store_community_summary(
    community_id,
    summary_text,
    embedding,
    level="HIGH",
    group_id=None,
    communities_included=None
):
    """
    Store a community summary with level metadata.

    Args:
        community_id: int, "low_0", "low_1", or "root"
        summary_text: prose summary from LLM
        embedding:    768-number vector
        level:        "HIGH", "LOW", or "ROOT"
        group_id:     integer group number (for LOW level)
        communities_included: list of community IDs in this summary
    """
    metadata = {
        "level":      level,
        "community_id": str(community_id)
    }
    if group_id is not None:
        metadata["group_id"] = str(group_id)
    if communities_included is not None:
        metadata["communities_included"] = str(communities_included)

    summaries_collection.upsert(
        ids=[f"community_{community_id}"],
        documents=[summary_text],
        embeddings=[embedding],
        metadatas=[metadata]
    )


def store_community_summaries_batch(summaries_dict):
    """
    Store multiple HIGH level summaries at once.
    summaries_dict: {community_id: {"summary": str, "embedding": list}}
    """
    ids         = []
    documents   = []
    embeddings  = []
    metadatas   = []

    for community_id, data in summaries_dict.items():
        ids.append(f"community_{community_id}")
        documents.append(data["summary"])
        embeddings.append(data["embedding"])
        metadatas.append({
            "level":        "HIGH",
            "community_id": str(community_id)
        })

    if ids:
        summaries_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )


def search_community_summaries(
    query_embedding,
    n_results=5,
    level=None
):
    """
    Search community summaries.
    Optionally filter by level: "HIGH", "LOW", or "ROOT"

    Args:
        query_embedding: 768-number vector of user question
        n_results:       how many to return
        level:           filter by level, None = search all levels

    Returns:
        ChromaDB query result
    """
    where = None
    if level:
        where = {"level": level}

    if where:
        return summaries_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
    else:
        return summaries_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )


def get_summaries_by_level(level: str) -> list:
    """
    Get all stored summaries of a specific level.

    Args:
        level: "HIGH", "LOW", or "ROOT"

    Returns:
        list of dicts with id, document, metadata
    """
    try:
        result = summaries_collection.get(
            where={"level": level}
        )
        summaries = []
        for idx, doc_id in enumerate(result["ids"]):
            summaries.append({
                "id":       doc_id,
                "summary":  result["documents"][idx],
                "metadata": result["metadatas"][idx] if result["metadatas"] else {}
            })
        return summaries
    except Exception:
        return []


def get_summaries_collection_count():
    return summaries_collection.count()


def get_summaries_count_by_level():
    """Return count of summaries at each level."""
    counts = {"HIGH": 0, "LOW": 0, "ROOT": 0}
    for level in counts:
        items = get_summaries_by_level(level)
        counts[level] = len(items)
    return counts


def clear_community_summaries():
    all_data = summaries_collection.get()
    if all_data["ids"]:
        summaries_collection.delete(ids=all_data["ids"])


def get_collection_stats():
    return {
        "chunks": {
            "count": chunks_collection.count(),
            "name":  "requirements_chunks"
        },
        "summaries": {
            "count": summaries_collection.count(),
            "name":  "community_summaries"
        }
    }