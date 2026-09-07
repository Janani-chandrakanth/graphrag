"""
Document API Routes

POST /api/document/upload   → upload and index a document
GET  /api/document/chunks   → get all stored chunks info
GET  /api/document/stats    → chunk count and collection stats
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from parser.parser import parse_document
from parser.document_type_detector import detect_document_type
from parser.template_normalizer import normalize_document
from parser.normalization_validator import validate_normalized_document
from parser.requirement_linker import link_requirements
from chunking.chunker import create_chunks_from_items
from embeddings.embedding_model import generate_embedding
from graph.entity_extractor import extract_entities
from graph.graph_builder import build_graph
from graph.deduplicator import deduplicate_graph
from graph.confidence_scorer import score_graph
from graph.validator import validate_graph
from graph.structural_linker import tag_extraction_source
from vectorstore.chroma_manager import (
    store_chunk,
    get_collection_count,
    get_collection_stats
)

router = APIRouter()


# ── Response Models ────────────────────────────────────────

class UploadResponse(BaseModel):
    success:            bool
    filename:           str
    doc_type:            str
    doc_type_confidence: float
    doc_type_method:     str
    normalized_items:    int
    normalized_via_regex:int
    normalized_via_llm:  int
    unmatched_blocks:    int
    normalization_warnings: list
    valid_items:        int
    needs_review:       int
    total_links:        int
    broken_references:  int
    total_chunks:       int
    avg_node_confidence: Optional[float] = None
    avg_relationship_confidence: Optional[float] = None
    low_confidence_nodes: int
    low_confidence_relationships: int
    valid_nodes:        int
    valid_relationships:int
    removed_nodes:      int
    removed_rels:       int
    vectors_stored:     int
    warnings:           list
    error:              Optional[str] = None


class StatsResponse(BaseModel):
    chunks_stored:    int
    summaries_stored: int


# ── Routes ─────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT requirement document.
    Parses, chunks, embeds, extracts entities,
    validates, deduplicates and stores in Neo4j + ChromaDB.
    """
    if not file.filename.endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported"
        )

    warnings = []

    try:
        # Step 1: Parse
        content   = await file.read()
        structure = parse_document_from_bytes(content, file.filename)
        text      = structure.to_markdown()

        # Step 2: Document Type Detector + Template Normalizer
        detection  = detect_document_type(structure)
        normalized = normalize_document(structure, detection["doc_type"])
        normalized_via_regex = sum(1 for i in normalized["items"] if i["matched_by"].startswith("regex"))
        normalized_via_llm   = sum(1 for i in normalized["items"] if i["matched_by"] == "llm_gap_fill")
        warnings.extend(normalized["warnings"])

        # Step 3: Normalization Validator - normalized["items"] ->
        # valid_items / review_queue. Never silently drops anything;
        # review_queue items are held for a human, not discarded.
        validated = validate_normalized_document(normalized)
        warnings.extend(validated["document_warnings"])

        # Step 4: Rule-based Requirement Linker - exact-ID cross-links
        # between valid_items, plus broken_references for mentions of
        # IDs that don't resolve within this document.
        #
        # items_for_graph: valid_items PLUS review_queue items — matches
        # app.py's fix. review_queue only means "not grounded in an
        # explicit ID token" or "family unexpected for this doc type,"
        # not "not real content" — excluding it outright was silently
        # starving the extractor of most of the document's actual
        # substance (body sentences, out-of-template business rules,
        # etc.) — they still show up in the Review Queue for
        # visibility, they just aren't discarded from the graph anymore.
        items_for_graph = validated["valid_items"] + [
            r["item"] for r in validated["review_queue"]
        ]
        linked = link_requirements(items_for_graph)

        # Step 5: Structural Chunker (item-aware) - one chunk per item
        # (oversized items split into <id>#1, <id>#2, ...), replacing
        # the old flat-text create_chunks(text) call.
        chunks = create_chunks_from_items(linked["items_with_links"])

        all_nodes, all_relationships = [], []
        prior_chunk_text = None  # one-step lookback only — see
                                 # extract_entities()'s docstring

        # Step 6: Embed and extract for each chunk
        for idx, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk["text"])
            store_chunk(
                chunk_id=f"{file.filename}_{chunk['chunk_id']}",
                chunk_text=chunk["text"],
                embedding=embedding,
                metadata={
                    "item_id":    chunk["item_id"],
                    "family":     chunk["family"],
                    "linked_ids": chunk["linked_ids"],
                    "is_split":   chunk["is_split"],
                }
            )

            graph_data = extract_entities(chunk["text"], prior_context=prior_chunk_text)
            if graph_data.get("error"):
                warnings.append(f"Chunk {idx+1} ({chunk['chunk_id']}): {graph_data['error']}")
            graph_data = tag_extraction_source(graph_data, chunk["item_id"])

            all_nodes.extend(graph_data.get("nodes", []))
            all_relationships.extend(graph_data.get("relationships", []))
            prior_chunk_text = chunk["text"]

        # Step 6b: Structural Linker — no backbone nodes anymore, see
        # graph/structural_linker.py's module docstring. tag_extraction_source()
        # above already sets "source" on every node/relationship, so
        # traceability back to the requirement item doesn't need a
        # graph node/edge.

        # Step 7: Deduplicate
        (
            all_nodes,
            all_relationships,
            type_conflicts,
            direction_conflicts
        ) = deduplicate_graph(all_nodes, all_relationships)

        # Step 8: Confidence Scorer - annotates every deduplicated
        # node/relationship with a "confidence" score + reasons
        # (occurrence counts, dedup conflict signals, Requirement
        # Linker corroboration). Nothing is removed; low-confidence
        # items are flagged for visibility, not dropped.
        scored = score_graph(
            all_nodes, all_relationships,
            type_conflicts, direction_conflicts,
            linked.get("items_with_links", [])
        )

        # Step 9: Validate
        (
            all_nodes,
            all_relationships,
            review_nodes,
            review_rels
        ) = validate_graph(all_nodes, all_relationships)

        # Step 10: Store in Neo4j
        build_graph({
            "nodes":         all_nodes,
            "relationships": all_relationships
        })

        return UploadResponse(
            success=True,
            filename=file.filename,
            doc_type=detection["doc_type"],
            doc_type_confidence=detection["confidence"],
            doc_type_method=detection["method"],
            normalized_items=len(normalized["items"]),
            normalized_via_regex=normalized_via_regex,
            normalized_via_llm=normalized_via_llm,
            unmatched_blocks=len(normalized["unmatched_blocks"]),
            normalization_warnings=normalized["warnings"],
            valid_items=validated["summary"]["valid"],
            needs_review=validated["summary"]["needs_review"],
            total_links=linked["summary"]["total_links"],
            broken_references=linked["summary"]["broken_references"],
            total_chunks=len(chunks),
            avg_node_confidence=scored["summary"]["avg_node_confidence"],
            avg_relationship_confidence=scored["summary"]["avg_relationship_confidence"],
            low_confidence_nodes=scored["summary"]["low_confidence_nodes"],
            low_confidence_relationships=scored["summary"]["low_confidence_relationships"],
            valid_nodes=len(all_nodes),
            valid_relationships=len(all_relationships),
            removed_nodes=len(review_nodes),
            removed_rels=len(review_rels),
            vectors_stored=get_collection_count(),
            warnings=warnings
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """
    Get storage statistics for chunks and summaries.
    """
    stats = get_collection_stats()
    return StatsResponse(
        chunks_stored=stats["chunks"]["count"],
        summaries_stored=stats["summaries"]["count"]
    )


@router.get("/chunks")
def get_chunks_info():
    """
    Get count of stored chunks in ChromaDB.
    """
    return {
        "total_chunks": get_collection_count()
    }


# ── Helper ─────────────────────────────────────────────────

def parse_document_from_bytes(content: bytes, filename: str):
    """
    Parse file bytes into a DocumentStructure via the shared structure-aware
    parser (Structure Preserver). Returns the structure itself (not just
    Markdown) so the caller can derive text AND run the Document Type
    Detector / Template Normalizer from the same parse — no need to parse
    twice, and it keeps the FastAPI upload path in sync with the Streamlit
    app path instead of maintaining its own separate extraction logic.
    """
    import io

    class _NamedBytes(io.BytesIO):
        """io.BytesIO with a .name attribute — parser.parse_document()
        dispatches on uploaded_file.name, same as Streamlit's UploadedFile."""
        def __init__(self, data, name):
            super().__init__(data)
            self.name = name

    named_file = _NamedBytes(content, filename)
    return parse_document(named_file)