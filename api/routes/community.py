"""
Community API Routes

POST /api/community/detect          → run community detection
POST /api/community/summarize       → generate HIGH level summaries
POST /api/community/hierarchy       → build ROOT and LOW levels
GET  /api/community/all             → get all communities
GET  /api/community/{community_id}  → get one community details
GET  /api/community/summaries/{level} → get summaries by level
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from graph.community_detector import (
    run_full_community_detection,
    get_all_communities,
    get_community_nodes,
    get_community_relationships
)
from graph.community_summarizer import summarize_all_communities
from vectorstore.chroma_manager import (
    get_summaries_by_level,
    get_summaries_count_by_level
)

router = APIRouter()


# ── Response Models ────────────────────────────────────────

class DetectionResponse(BaseModel):
    success:           bool
    total_nodes:       int
    total_communities: int
    community_sizes:   dict
    error:             Optional[str] = None


class SummarizeResponse(BaseModel):
    success:           bool
    total_communities: int
    successful:        int
    failed:            int
    errors:            list


class HierarchyResponse(BaseModel):
    success:    bool
    root_built: bool
    low_built:  int
    high_count: int
    error:      Optional[str] = None


# ── Routes ─────────────────────────────────────────────────

@router.post("/detect", response_model=DetectionResponse)
def detect_communities():
    """
    Run Louvain community detection on the Neo4j graph.
    Groups tightly connected nodes into communities.
    Must run after document indexing.
    """
    try:
        stats = run_full_community_detection()

        if not stats.get("success"):
            return DetectionResponse(
                success=False,
                total_nodes=0,
                total_communities=0,
                community_sizes={},
                error=stats.get("error", "Detection failed")
            )

        return DetectionResponse(
            success=True,
            total_nodes=stats["total_nodes"],
            total_communities=stats["total_communities"],
            community_sizes={
                str(k): v
                for k, v in stats["community_sizes"].items()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=SummarizeResponse)
def generate_summaries():
    """
    Generate HIGH level LLM summaries for all detected communities.
    Summaries are stored in ChromaDB.
    Must run after community detection.
    """
    try:
        results = summarize_all_communities()

        return SummarizeResponse(
            success=results["successful"] > 0,
            total_communities=results["total_communities"],
            successful=results["successful"],
            failed=results["failed"],
            errors=results.get("errors", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hierarchy", response_model=HierarchyResponse)
def build_hierarchy():
    """
    Build ROOT and LOW level summaries from existing HIGH level summaries.
    ROOT = one summary for entire document.
    LOW  = group-level summaries.
    Must run after summarize.
    """
    try:
        from graph.community_hierarchy import build_full_hierarchy
        from vectorstore.chroma_manager import get_summaries_by_level

        # Get existing HIGH summaries
        high_items = get_summaries_by_level("HIGH")
        if not high_items:
            raise HTTPException(
                status_code=400,
                detail="No HIGH level summaries found. Run /summarize first."
            )

        high_summaries = {
            int(item["metadata"].get("community_id", idx)): item["summary"]
            for idx, item in enumerate(high_items)
        }

        hierarchy = build_full_hierarchy(high_summaries)

        if "error" in hierarchy:
            return HierarchyResponse(
                success=False,
                root_built=False,
                low_built=0,
                high_count=len(high_summaries),
                error=hierarchy["error"]
            )

        stats = hierarchy.get("stats", {})
        return HierarchyResponse(
            success=True,
            root_built=stats.get("total_root", 0) > 0,
            low_built=stats.get("total_low", 0),
            high_count=stats.get("total_high", 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
def get_communities():
    """
    Get all detected communities with node counts and types.
    """
    try:
        communities = get_all_communities()
        return {
            "success":          True,
            "total_communities": len(communities),
            "communities":      {
                str(k): v for k, v in communities.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summaries/{level}")
def get_summaries(level: str):
    """
    Get all stored summaries at a specific level.
    level: HIGH, LOW, or ROOT
    """
    level = level.upper()
    if level not in ("HIGH", "LOW", "ROOT"):
        raise HTTPException(
            status_code=400,
            detail="Level must be HIGH, LOW, or ROOT"
        )

    try:
        summaries = get_summaries_by_level(level)
        counts    = get_summaries_count_by_level()

        return {
            "success":   True,
            "level":     level,
            "count":     len(summaries),
            "all_counts": counts,
            "summaries": summaries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{community_id}")
def get_community(community_id: int):
    """
    Get full details of a specific community —
    nodes, relationships, and summary if available.
    """
    try:
        nodes = get_community_nodes(community_id)
        if not nodes:
            raise HTTPException(
                status_code=404,
                detail=f"Community {community_id} not found"
            )

        rels = get_community_relationships(community_id)

        # Try to get summary
        from vectorstore.chroma_manager import summaries_collection
        try:
            result  = summaries_collection.get(
                ids=[f"community_{community_id}"]
            )
            summary = result["documents"][0] if result["documents"] else None
        except Exception:
            summary = None

        return {
            "success":       True,
            "community_id":  community_id,
            "node_count":    len(nodes),
            "rel_count":     len(rels),
            "nodes":         nodes,
            "relationships": rels,
            "summary":       summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))