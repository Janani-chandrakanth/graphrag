"""
Query API Routes

POST /api/query/graphrag   → GraphRAG query with level selection
POST /api/query/chunks     → basic RAG chunk search
GET  /api/query/levels     → available summary levels and counts
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from graph.query_engine import run_graphrag_query
from embeddings.embedding_model import generate_embedding
from vectorstore.chroma_manager import (
    search_chunks,
    get_summaries_count_by_level,
    get_summaries_collection_count
)

router = APIRouter()


# ── Request / Response Models ──────────────────────────────

class GraphRAGRequest(BaseModel):
    question:  str
    level:     Optional[Literal["ROOT", "LOW", "HIGH", "ALL"]] = "ALL"
    n_results: Optional[int] = 3
    use_graph_traversal: Optional[bool] = True


class GraphRAGResponse(BaseModel):
    success:             bool
    question:            str
    answer:              str
    level_used:          str
    communities_used:    list
    retrieved_summaries: list
    graph_facts:         dict = {}
    error:               Optional[str] = None


class ChunkSearchRequest(BaseModel):
    question:  str
    n_results: Optional[int] = 3


class ChunkSearchResponse(BaseModel):
    success:   bool
    question:  str
    chunks:    list
    total:     int


# ── Routes ─────────────────────────────────────────────────

@router.post("/graphrag", response_model=GraphRAGResponse)
def graphrag_query(request: GraphRAGRequest):
    """
    Run a GraphRAG query.

    Embeds the question, retrieves relevant community summaries
    from ChromaDB, and sends them to llama3.1 to generate
    a final synthesized answer.

    level options:
    - ROOT: broad questions about the entire system
    - LOW:  questions about feature groups
    - HIGH: specific questions about individual communities
    - ALL:  search across all levels (default)

    Example request:
    {
        "question": "What does the dashboard show?",
        "level": "HIGH",
        "n_results": 3
    }
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if get_summaries_collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No community summaries found. "
                "Run /community/detect, /community/summarize "
                "and /community/hierarchy first."
            )
        )

    try:
        level  = None if request.level == "ALL" else request.level
        result = run_graphrag_query(
            question=request.question,
            level=level,
            n_results=request.n_results,
            use_graph_traversal=request.use_graph_traversal
        )

        return GraphRAGResponse(
            success=result["success"],
            question=request.question,
            answer=result.get("answer", ""),
            level_used=result.get("level_used", "ALL"),
            communities_used=result.get("communities_used", []),
            retrieved_summaries=[
                {
                    "community_id": s["community_id"],
                    "summary":      s["summary"],
                    "similarity":   s["similarity"],
                    "level":        s.get("level", "HIGH")
                }
                for s in result.get("retrieved_summaries", [])
            ],
            graph_facts=result.get("graph_facts", {}),
            error=result.get("error")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunks", response_model=ChunkSearchResponse)
def chunk_search(request: ChunkSearchRequest):
    """
    Basic RAG search — finds the most similar raw requirement
    chunks to your question using vector similarity.

    Returns matching chunks without LLM synthesis.
    Use /graphrag for a synthesized answer.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        query_embedding = generate_embedding(request.question)
        results         = search_chunks(
            query_embedding=query_embedding,
            n_results=request.n_results
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        chunks = [
            {
                "chunk":      document,
                "similarity": round(1 - distance, 4),
                "rank":       idx + 1
            }
            for idx, (document, distance)
            in enumerate(zip(documents, distances))
        ]

        return ChunkSearchResponse(
            success=True,
            question=request.question,
            chunks=chunks,
            total=len(chunks)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/levels")
def get_available_levels():
    """
    Get available summary levels and their counts.
    Use this to decide which level to query.
    """
    try:
        counts = get_summaries_count_by_level()
        total  = get_summaries_collection_count()

        return {
            "success": True,
            "total_summaries": total,
            "levels": {
                "ROOT": {
                    "count":       counts["ROOT"],
                    "description": "One summary for the entire document",
                    "best_for":    "Broad questions like 'What does this system do?'"
                },
                "LOW": {
                    "count":       counts["LOW"],
                    "description": "Group-level summaries",
                    "best_for":    "Questions about feature groups"
                },
                "HIGH": {
                    "count":       counts["HIGH"],
                    "description": "Individual community summaries",
                    "best_for":    "Specific questions about requirements"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))