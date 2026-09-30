"""
ChromaDB Manager
Manages two collections:
  1. requirements_chunks   — raw requirement chunks
  2. community_summaries   — LLM summaries at HIGH, LOW, ROOT levels
"""

import chromadb
import json
import requests
from config import OLLAMA_EMBED_URL, EMBED_MODEL


def generate_embedding(text):
    """
    Embedding generation via Ollama /api/embeddings.
    """
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }
    response = requests.post(
        OLLAMA_EMBED_URL,
        json=payload,
        proxies={"http": None, "https": None}
    )
    response.raise_for_status()
    return response.json()["embedding"]


client = chromadb.PersistentClient(path="./chroma_db")

# ── Collection 1: Raw Chunks ───────────────────────────────
chunks_collection = client.get_or_create_collection(
    name="requirements_chunks"
)


def store_chunk(chunk_id, chunk_text, embedding, metadata=None):
    """
    Args:
        metadata: optional dict, e.g. output of
                  chunker.create_chunks_from_items(), such as
                  {"item_id": "FR-001", "family": "FR",
                   "linked_ids": ["TC-010", "BR-005"], "is_split": False}.
                  Chroma metadata values must be str/int/float/bool (no
                  lists), so list-valued fields like linked_ids are
                  comma-joined here before storage. Omit entirely for
                  the old call signature — unchanged, still works.
    """
    upsert_kwargs = dict(
        ids=[chunk_id],
        documents=[chunk_text],
        embeddings=[embedding],
    )
    if metadata is not None:
        flat_metadata = {
            k: (",".join(v) if isinstance(v, list) else v)
            for k, v in metadata.items()
            if v is not None
        }
        upsert_kwargs["metadatas"] = [flat_metadata]

    chunks_collection.upsert(**upsert_kwargs)


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


# ── Collection 3: User Stories (User Story → Gherkin flow) ──
# Separate collection so this flow's retrieval (graph/story_hybrid_retriever.py)
# never accidentally mixes with the requirement-chunk-level RAG above —
# same reasoning the summaries collection already keeps its own space.
user_stories_collection = client.get_or_create_collection(
    name="user_stories"
)


def store_user_story_embedding(story_id, summary_text, embedding, metadata=None):
    """
    Args:
        story_id: same id as the UserStory node's storyId in Neo4j
            (graph/story_graph_writer.upsert_user_story) — this is
            what ties a Chroma vector back to its graph node.
        summary_text: the "As a ... I want ... so that ..." summary —
            what similarity search actually compares against.
        metadata: e.g. {"doc_id": ..., "title": ...}. Chroma metadata
            values must be str/int/float/bool (no lists) — list-valued
            fields are comma-joined before storage, same convention as
            store_chunk() above.
    """
    upsert_kwargs = dict(
        ids=[story_id],
        documents=[summary_text],
        embeddings=[embedding],
    )
    if metadata is not None:
        flat_metadata = {
            k: (",".join(v) if isinstance(v, list) else v)
            for k, v in metadata.items()
            if v is not None
        }
        upsert_kwargs["metadatas"] = [flat_metadata]
    user_stories_collection.upsert(**upsert_kwargs)


def search_user_stories(query_embedding, n_results=5):
    total = user_stories_collection.count()
    if total == 0:
        return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
    return user_stories_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total),
    )


def get_user_stories_collection_count():
    return user_stories_collection.count()


def delete_user_story_embedding(story_id):
    try:
        user_stories_collection.delete(ids=[story_id])
    except Exception:
        pass