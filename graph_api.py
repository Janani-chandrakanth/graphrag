"""
graph_api.py

Graph Visualization API and Interactive UI for Requirement GraphRAG.
Consolidates schemas, service layer, FastAPI routes, and the standalone vis-network frontend.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from graph.graph_visualizer import NODE_COLORS, NODE_SHAPES, DEFAULT_COLOR, DEFAULT_SHAPE
from graph.neo4j_manager import run_cypher_query, get_stored_documents
from logging_config import setup_logging, get_logger

# ─── Initialize Logging ───────────────────────────────────────────────────────
setup_logging()
logger = get_logger(__name__)


# ==============================================================================
# 1. SCHEMAS (Pydantic Models)
# ==============================================================================

class GraphNode(BaseModel):
    """Represents a node formatted for frontend visualization."""
    id: str = Field(..., description="Unique and stable identifier for the node")
    label: str = Field(..., description="Human-readable label for display on the node")
    type: str = Field(..., description="Entity type classification, e.g., Actor, Feature, Requirement")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata and extracted properties")
    color: Optional[str] = Field(None, description="Hex color code based on entity type")
    shape: Optional[str] = Field(None, description="Geometric shape identifier for graph rendering")


class GraphEdge(BaseModel):
    """Represents a directed relationship between two existing nodes."""
    id: str = Field(..., description="Unique and stable edge identifier")
    source: str = Field(..., description="ID of the source/origin node")
    target: str = Field(..., description="ID of the target/destination node")
    label: str = Field(..., description="Human-readable relationship type label")
    type: str = Field(..., description="Relationship type in UPPER_SNAKE_CASE")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties such as doc_id or source")


class GraphMetadata(BaseModel):
    """Metadata describing the graph payload."""
    node_count: int = Field(..., description="Number of nodes returned in this response")
    edge_count: int = Field(..., description="Number of edges returned in this response")
    total_graph_nodes: Optional[int] = Field(None, description="Total nodes available in the full database")
    total_graph_edges: Optional[int] = Field(None, description="Total relationships available in the full database")
    document_id: Optional[str] = Field(None, description="Document filter applied, if any")
    truncated: bool = Field(False, description="Whether the graph results were capped by the limit parameter")
    execution_time_ms: float = Field(..., description="Database query and processing time in milliseconds")


class GraphResponse(BaseModel):
    """Standard graph retrieval response containing nodes, edges, and metadata."""
    nodes: List[GraphNode] = Field(..., description="List of graph nodes")
    edges: List[GraphEdge] = Field(..., description="List of graph edges connecting the returned nodes")
    metadata: GraphMetadata = Field(..., description="Response metadata and execution statistics")


class GraphSummaryResponse(BaseModel):
    """Lightweight summary of graph metrics without returning full nodes and edges."""
    total_nodes: int = Field(..., description="Total entity nodes in Neo4j")
    total_edges: int = Field(..., description="Total relationships in Neo4j")
    node_types: Dict[str, int] = Field(..., description="Counts of nodes grouped by entity type")
    relationship_types: Dict[str, int] = Field(..., description="Counts of relationships grouped by relationship type")
    documents: List[str] = Field(default_factory=list, description="List of distinct source document IDs")
    execution_time_ms: float = Field(..., description="Summary calculation duration in milliseconds")


class NeighborMetadata(GraphMetadata):
    """Metadata specific to neighbor/subgraph traversal queries."""
    focus_node_id: str = Field(..., description="The ID of the central node being explored")
    depth: int = Field(..., description="Traversal depth traversed from the focus node")
    direction: str = Field(..., description="Direction of relationships traversed: all, in, or out")


class NeighborResponse(BaseModel):
    """Response containing a focus node, its neighbors within depth N, and their interconnecting edges."""
    focus_node: Optional[GraphNode] = Field(None, description="The root/focus node explored")
    nodes: List[GraphNode] = Field(..., description="All nodes within the neighborhood (including focus node)")
    edges: List[GraphEdge] = Field(..., description="All interconnecting relationships among returned nodes")
    metadata: NeighborMetadata = Field(..., description="Traversal metadata and execution statistics")


class HealthResponse(BaseModel):
    """System and database connectivity health status."""
    status: str = Field(..., description="Application health status (healthy, degraded, unhealthy)")
    database: str = Field("neo4j", description="Graph database backend name")
    connected: bool = Field(..., description="Whether the Neo4j database is reachable")
    node_count: Optional[int] = Field(None, description="Quick count of nodes if connected")
    message: Optional[str] = Field(None, description="Additional status details or error message")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Short error classification")
    detail: str = Field(..., description="Descriptive explanation of the error")


# ==============================================================================
# 2. SERVICE LAYER (Neo4j Access & Data Transformation)
# ==============================================================================

def _sanitize_value(val: Any) -> Any:
    """Recursively convert Neo4j types (DateTime, etc.) into JSON-serializable primitives."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, list):
        return [_sanitize_value(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_value(v) for k, v in val.items()}
    return str(val)


def _extract_node_properties(record: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and clean properties dictionary from a Neo4j Entity record."""
    props: Dict[str, Any] = {}
    if record.get("description"):
        props["description"] = record["description"]
    if record.get("source"):
        props["source"] = record["source"]
    if record.get("doc_id"):
        props["doc_id"] = record["doc_id"]
    if record.get("aliases"):
        props["aliases"] = _sanitize_value(record["aliases"])
    if record.get("sequence") is not None:
        props["sequence"] = record["sequence"]
    if record.get("is_flow_entry") is not None:
        props["is_flow_entry"] = bool(record["is_flow_entry"])

    attrs_raw = record.get("attributes_json")
    if attrs_raw:
        if isinstance(attrs_raw, str):
            try:
                unpacked = json.loads(attrs_raw)
                if isinstance(unpacked, dict):
                    props.update({k: _sanitize_value(v) for k, v in unpacked.items()})
            except Exception:
                props["raw_attributes"] = attrs_raw
        elif isinstance(attrs_raw, dict):
            props.update({k: _sanitize_value(v) for k, v in attrs_raw.items()})

    return {k: v for k, v in props.items() if not k.lower().startswith("secret")}


def _build_graph_node(record: Dict[str, Any]) -> GraphNode:
    """Transform a raw Neo4j Entity record into a validated GraphNode."""
    node_id = str(record["id"]).strip()
    node_type = str(record.get("type") or "Entity").strip()
    name = record.get("name") or node_id

    color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
    shape = NODE_SHAPES.get(node_type, DEFAULT_SHAPE)
    properties = _extract_node_properties(record)

    return GraphNode(
        id=node_id,
        label=str(name),
        type=node_type,
        properties=properties,
        color=color,
        shape=shape,
    )


def _build_edge_id(source: str, rel_type: str, target: str, index: int = 0) -> str:
    """Generate a stable, deterministic edge identifier."""
    return f"edge:{source}->{rel_type}->{target}:{index}"


def fetch_graph(
    doc_id: Optional[str] = None,
    node_type: Optional[str] = None,
    rel_type: Optional[str] = None,
    limit: int = 300,
    skip: int = 0,
) -> GraphResponse:
    """
    Retrieve nodes and interconnecting relationships with optional filtering and bounded limits.
    Guarantees all returned edges have valid source and target endpoints in the returned node set.
    """
    t0 = time.perf_counter()
    clean_doc = doc_id.strip() if doc_id and doc_id.strip() and doc_id.strip() != "All Documents" else None
    clean_node_type = node_type.strip() if node_type and node_type.strip() and node_type.strip() != "All" else None
    clean_rel_type = rel_type.strip() if rel_type and rel_type.strip() and rel_type.strip() != "All" else None

    # Total counts for metadata context
    total_counts_res = run_cypher_query(
        """
        MATCH (n:Entity)
        OPTIONAL MATCH ()-[r]->()
        RETURN count(DISTINCT n) AS total_nodes, count(r) AS total_edges
        """
    )
    total_nodes = total_counts_res[0]["total_nodes"] if total_counts_res else 0
    total_edges = total_counts_res[0]["total_edges"] if total_counts_res else 0

    # Query filtered nodes
    node_query = """
    MATCH (n:Entity)
    WHERE ($doc_id IS NULL OR n.doc_id = $doc_id OR n.source = $doc_id)
      AND ($node_type IS NULL OR toLower(n.type) = toLower($node_type))
    RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description,
           n.source AS source, n.aliases AS aliases, n.sequence AS sequence,
           n.is_flow_entry AS is_flow_entry, n.attributes_json AS attributes_json,
           n.doc_id AS doc_id
    ORDER BY n.id
    SKIP $skip LIMIT $limit
    """
    node_params = {
        "doc_id": clean_doc,
        "node_type": clean_node_type,
        "skip": skip,
        "limit": limit,
    }
    raw_nodes = run_cypher_query(node_query, node_params)

    nodes: List[GraphNode] = []
    node_ids: Set[str] = set()

    for r in raw_nodes:
        nid = str(r.get("id", "")).strip()
        if not nid or nid in node_ids:
            continue
        node_ids.add(nid)
        nodes.append(_build_graph_node(r))

    # Query relationships strictly connecting the returned nodes
    edges: List[GraphEdge] = []
    if node_ids:
        edge_query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.id IN $node_ids AND b.id IN $node_ids
          AND ($rel_type IS NULL OR type(r) = $rel_type)
        RETURN a.id AS from, b.id AS to, type(r) AS type, r.source AS source,
               r.doc_id AS doc_id, properties(r) AS props
        """
        edge_params = {
            "node_ids": list(node_ids),
            "rel_type": clean_rel_type,
        }
        raw_edges = run_cypher_query(edge_query, edge_params)

        seen_sigs: Dict[Tuple[str, str, str], int] = {}
        for r in raw_edges:
            src = str(r["from"]).strip()
            tgt = str(r["to"]).strip()
            rtype = str(r["type"]).strip()

            if src not in node_ids or tgt not in node_ids:
                continue

            sig = (src, rtype, tgt)
            idx = seen_sigs.get(sig, 0)
            seen_sigs[sig] = idx + 1

            edge_id = _build_edge_id(src, rtype, tgt, idx)
            props: Dict[str, Any] = {}
            if r.get("source"):
                props["source"] = r["source"]
            if r.get("doc_id"):
                props["doc_id"] = r["doc_id"]
            if isinstance(r.get("props"), dict):
                for k, v in r["props"].items():
                    if k not in props:
                        props[k] = _sanitize_value(v)

            edges.append(
                GraphEdge(
                    id=edge_id,
                    source=src,
                    target=tgt,
                    label=rtype,
                    type=rtype,
                    properties=props,
                )
            )

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    truncated = len(nodes) >= limit

    metadata = GraphMetadata(
        node_count=len(nodes),
        edge_count=len(edges),
        total_graph_nodes=total_nodes,
        total_graph_edges=total_edges,
        document_id=clean_doc,
        truncated=truncated,
        execution_time_ms=duration_ms,
    )

    return GraphResponse(nodes=nodes, edges=edges, metadata=metadata)


def fetch_node_neighbors(
    node_id: str,
    depth: int = 1,
    direction: str = "all",
    limit: int = 100,
) -> NeighborResponse:
    """Traverse relationships outwards/inwards from a focus node up to a bounded depth."""
    t0 = time.perf_counter()
    clean_node_id = node_id.strip()
    bounded_depth = max(1, min(depth, 3))
    clean_dir = direction.lower() if direction.lower() in ("in", "out", "all") else "all"

    focus_query = """
    MATCH (n:Entity {id: $node_id})
    RETURN n.id AS id, n.name AS name, n.type AS type, n.description AS description,
           n.source AS source, n.aliases AS aliases, n.sequence AS sequence,
           n.is_flow_entry AS is_flow_entry, n.attributes_json AS attributes_json,
           n.doc_id AS doc_id
    """
    focus_records = run_cypher_query(focus_query, {"node_id": clean_node_id})
    focus_node: Optional[GraphNode] = _build_graph_node(focus_records[0]) if focus_records else None

    if not focus_node:
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        meta = NeighborMetadata(
            node_count=0,
            edge_count=0,
            total_graph_nodes=None,
            total_graph_edges=None,
            document_id=None,
            truncated=False,
            execution_time_ms=duration_ms,
            focus_node_id=clean_node_id,
            depth=bounded_depth,
            direction=clean_dir,
        )
        return NeighborResponse(focus_node=None, nodes=[], edges=[], metadata=meta)

    if clean_dir == "out":
        rel_pattern = f"-[*1..{bounded_depth}]->"
    elif clean_dir == "in":
        rel_pattern = f"<-[*1..{bounded_depth}]-"
    else:
        rel_pattern = f"-[*1..{bounded_depth}]-"

    path_query = f"""
    MATCH path = (start:Entity {{id: $node_id}}){rel_pattern}(m:Entity)
    WITH path LIMIT $limit
    UNWIND nodes(path) AS n
    RETURN DISTINCT n.id AS id, n.name AS name, n.type AS type, n.description AS description,
           n.source AS source, n.aliases AS aliases, n.sequence AS sequence,
           n.is_flow_entry AS is_flow_entry, n.attributes_json AS attributes_json,
           n.doc_id AS doc_id
    """
    neighbor_records = run_cypher_query(path_query, {"node_id": clean_node_id, "limit": limit})

    nodes: List[GraphNode] = [focus_node]
    node_ids: Set[str] = {focus_node.id}

    for r in neighbor_records:
        nid = str(r.get("id", "")).strip()
        if not nid or nid in node_ids:
            continue
        node_ids.add(nid)
        nodes.append(_build_graph_node(r))

    edges: List[GraphEdge] = []
    edge_query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE a.id IN $node_ids AND b.id IN $node_ids
    RETURN a.id AS from, b.id AS to, type(r) AS type, r.source AS source,
           r.doc_id AS doc_id, properties(r) AS props
    """
    raw_edges = run_cypher_query(edge_query, {"node_ids": list(node_ids)})

    seen_sigs: Dict[Tuple[str, str, str], int] = {}
    for r in raw_edges:
        src = str(r["from"]).strip()
        tgt = str(r["to"]).strip()
        rtype = str(r["type"]).strip()

        if src not in node_ids or tgt not in node_ids:
            continue

        sig = (src, rtype, tgt)
        idx = seen_sigs.get(sig, 0)
        seen_sigs[sig] = idx + 1

        edge_id = _build_edge_id(src, rtype, tgt, idx)
        edges.append(
            GraphEdge(
                id=edge_id,
                source=src,
                target=tgt,
                label=rtype,
                type=rtype,
                properties=_sanitize_value(r.get("props") or {}),
            )
        )

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    meta = NeighborMetadata(
        node_count=len(nodes),
        edge_count=len(edges),
        total_graph_nodes=None,
        total_graph_edges=None,
        document_id=focus_node.properties.get("doc_id"),
        truncated=len(nodes) >= limit,
        execution_time_ms=duration_ms,
        focus_node_id=clean_node_id,
        depth=bounded_depth,
        direction=clean_dir,
    )
    return NeighborResponse(focus_node=focus_node, nodes=nodes, edges=edges, metadata=meta)


def fetch_graph_summary() -> GraphSummaryResponse:
    """Compute summary statistics for nodes, relationships, and documents."""
    t0 = time.perf_counter()
    node_type_query = """
    MATCH (n:Entity)
    RETURN coalesce(n.type, 'Unknown') AS type, count(n) AS count
    ORDER BY count DESC
    """
    node_records = run_cypher_query(node_type_query)
    node_types: Dict[str, int] = {r["type"]: int(r["count"]) for r in node_records}
    total_nodes = sum(node_types.values())

    rel_type_query = """
    MATCH ()-[r]->()
    RETURN type(r) AS type, count(r) AS count
    ORDER BY count DESC
    """
    rel_records = run_cypher_query(rel_type_query)
    rel_types: Dict[str, int] = {r["type"]: int(r["count"]) for r in rel_records}
    total_edges = sum(rel_types.values())

    try:
        documents = get_stored_documents()
    except Exception as e:
        logger.warning("Could not fetch stored documents in summary: %s", e)
        documents = []

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return GraphSummaryResponse(
        total_nodes=total_nodes,
        total_edges=total_edges,
        node_types=node_types,
        relationship_types=rel_types,
        documents=documents,
        execution_time_ms=duration_ms,
    )


def check_database_health() -> HealthResponse:
    """Verify live connectivity to Neo4j and report count."""
    try:
        records = run_cypher_query("MATCH (n:Entity) RETURN count(n) AS count")
        count = int(records[0]["count"]) if records else 0
        return HealthResponse(
            status="healthy",
            database="neo4j",
            connected=True,
            node_count=count,
            message="Neo4j connection verified and responsive.",
        )
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return HealthResponse(
            status="unhealthy",
            database="neo4j",
            connected=False,
            node_count=None,
            message=f"Database unreachable: {str(exc)}",
        )


# ==============================================================================
# 3. FASTAPI APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="Requirement GraphRAG — Graph Visualization API",
    description=(
        "Single-file Knowledge Graph Visualization API exposing nodes, relationships, "
        "subgraphs, and summary metrics extracted from Business Requirement Documents (BRDs). "
        "Integrates directly with Neo4j and powers interactive external frontends."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 4. ROUTERS / ENDPOINTS
# ==============================================================================

@app.get(
    "/api/graph",
    response_model=GraphResponse,
    summary="Retrieve Knowledge Graph",
    description="Retrieve graph nodes and interconnecting relationships in a frontend-friendly format.",
    tags=["Graph Visualization"],
    responses={
        status.HTTP_200_OK: {"model": GraphResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def get_graph(
    doc_id: Optional[str] = Query(None, description="Filter by document context."),
    node_type: Optional[str] = Query(None, description="Filter by entity type."),
    rel_type: Optional[str] = Query(None, description="Filter by relationship type."),
    limit: int = Query(default=300, ge=1, le=2000, description="Max nodes to return."),
    skip: int = Query(default=0, ge=0, description="Pagination offset."),
) -> GraphResponse:
    """Fetch nodes and relationships matching filters."""
    try:
        return fetch_graph(
            doc_id=doc_id,
            node_type=node_type,
            rel_type=rel_type,
            limit=limit,
            skip=skip,
        )
    except Exception as exc:
        logger.exception("Failed to fetch graph | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query knowledge graph from database: {str(exc)}",
        )


@app.get(
    "/api/graph/summary",
    response_model=GraphSummaryResponse,
    summary="Get Graph Summary & Metrics",
    description="Returns high-level statistics including total node/edge counts, entity types, and documents.",
    tags=["Graph Visualization"],
    responses={
        status.HTTP_200_OK: {"model": GraphSummaryResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def get_graph_summary() -> GraphSummaryResponse:
    """Retrieve statistical summary of the graph database."""
    try:
        return fetch_graph_summary()
    except Exception as exc:
        logger.exception("Failed to compute graph summary | error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate graph summary: {str(exc)}",
        )


@app.get(
    "/api/graph/nodes/{node_id}/neighbors",
    response_model=NeighborResponse,
    summary="Get Node Neighborhood Subgraph",
    description="Retrieve a focused subgraph containing the specified node, its adjacent neighbors within depth (1 to 3).",
    tags=["Graph Visualization"],
    responses={
        status.HTTP_200_OK: {"model": NeighborResponse},
        status.HTTP_404_NOT_FOUND: {"description": "Focus node not found", "model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def get_node_neighbors(
    node_id: str,
    depth: int = Query(default=1, ge=1, le=3, description="Search depth / hops (1 to 3)."),
    direction: str = Query(default="all", pattern="^(all|in|out)$", description="Relationship direction."),
    limit: int = Query(default=100, ge=1, le=500, description="Max neighbor nodes."),
) -> NeighborResponse:
    """Traverse and return the neighborhood subgraph for a given node."""
    try:
        response = fetch_node_neighbors(
            node_id=node_id,
            depth=depth,
            direction=direction,
            limit=limit,
        )
        if not response.focus_node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node with id '{node_id}' does not exist in the graph.",
            )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch node neighbors for %s | error=%s", node_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to traverse node neighborhood: {str(exc)}",
        )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="System & Database Health Check",
    description="Check connectivity to Neo4j and view active node count.",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Perform health and connectivity verification."""
    return check_database_health()


# ==============================================================================
# 5. EMBEDDED INTERACTIVE VIS-NETWORK FRONTEND
# ==============================================================================

VISUALIZATION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Requirement GraphRAG — Interactive Knowledge Graph</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    :root {
      --bg-dark: #0f172a;
      --bg-panel: rgba(30, 41, 59, 0.88);
      --bg-panel-hover: rgba(51, 65, 85, 0.95);
      --border-color: rgba(148, 163, 184, 0.15);
      --border-active: rgba(56, 189, 248, 0.5);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      overflow: hidden;
      height: 100vh;
      width: 100vw;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--bg-panel);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 260px;
    }
    .brand-logo {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, #0284c7, #38bdf8);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      color: white;
      font-size: 16px;
      box-shadow: 0 2px 8px rgba(56, 189, 248, 0.4);
    }
    .brand-text h1 { font-size: 15px; font-weight: 600; line-height: 1.2; }
    .brand-text span { font-size: 11px; color: var(--text-muted); }
    .toolbar-group { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .control-item {
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 12px;
      color: var(--text-main);
    }
    .control-item label {
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    select, input {
      background: transparent;
      border: none;
      color: var(--text-main);
      font-family: inherit;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }
    select option { background: #1e293b; color: var(--text-main); }
    input[type="text"] { width: 150px; cursor: text; }
    input[type="text"]::placeholder { color: #64748b; }
    .btn {
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #38bdf8;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: rgba(56, 189, 248, 0.22);
      border-color: rgba(56, 189, 248, 0.6);
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      color: var(--text-muted);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text-main);
      border-color: rgba(255, 255, 255, 0.25);
    }
    .badge-pill {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #10b981;
      padding: 4px 8px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 500;
    }
    main { flex: 1; position: relative; overflow: hidden; }
    #graph-container {
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
    }
    .legend-panel {
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--bg-panel);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-width: 480px;
      box-shadow: var(--shadow-lg);
      z-index: 10;
      font-size: 11px;
    }
    .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 8px;
      border-radius: 4px;
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.05);
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .legend-chip:hover { border-color: var(--border-active); }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
    #inspector-drawer {
      position: absolute;
      top: 0;
      right: -420px;
      width: 400px;
      height: 100%;
      background: var(--bg-panel);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-left: 1px solid var(--border-color);
      box-shadow: var(--shadow-lg);
      display: flex;
      flex-direction: column;
      transition: right 0.28s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 30;
    }
    #inspector-drawer.open { right: 0; }
    .drawer-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .drawer-title-group h2 { font-size: 16px; font-weight: 600; line-height: 1.3; word-break: break-word; }
    .drawer-title-group .type-badge {
      display: inline-block;
      margin-top: 6px;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .close-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 18px;
      cursor: pointer;
      padding: 4px;
      line-height: 1;
    }
    .close-btn:hover { color: var(--text-main); }
    .drawer-body {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .property-section {
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 12px;
    }
    .property-section h3 {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .prop-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      font-size: 12px;
    }
    .prop-row:last-child { border-bottom: none; }
    .prop-key { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 11px; }
    .prop-val { color: var(--text-main); text-align: right; word-break: break-word; }
    .drawer-footer { padding: 14px 20px; border-top: 1px solid var(--border-color); display: flex; gap: 10px; }
    .overlay-state {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(4px);
      z-index: 15;
      gap: 14px;
    }
    .spinner {
      width: 36px;
      height: 36px;
      border: 3px solid rgba(56, 189, 248, 0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">GR</div>
      <div class="brand-text">
        <h1>GraphRAG Visualizer</h1>
        <span>Knowledge Graph API Explorer</span>
      </div>
    </div>
    <div class="toolbar-group">
      <div class="control-item">
        <label for="doc-filter">Doc:</label>
        <select id="doc-filter"><option value="">All Documents</option></select>
      </div>
      <div class="control-item">
        <label for="type-filter">Type:</label>
        <select id="type-filter"><option value="">All Types</option></select>
      </div>
      <div class="control-item">
        <label for="search-input">Find:</label>
        <input type="text" id="search-input" placeholder="Search node name..." autocomplete="off">
      </div>
      <div class="control-item">
        <label for="limit-select">Limit:</label>
        <select id="limit-select">
          <option value="150">150</option>
          <option value="300" selected>300</option>
          <option value="500">500</option>
          <option value="1000">1000</option>
        </select>
      </div>
      <button class="btn btn-secondary" id="btn-physics" title="Toggle physics simulation">⚡ Freeze</button>
      <button class="btn btn-secondary" id="btn-fit" title="Zoom to fit graph on screen">⛶ Fit</button>
      <button class="btn" id="btn-refresh" title="Reload graph from API">↻ Refresh</button>
      <a href="/docs" target="_blank" class="btn btn-secondary" title="View OpenAPI / Swagger documentation">API Docs ↗</a>
    </div>
    <div class="toolbar-group">
      <div class="badge-pill" id="badge-stats">Loading...</div>
    </div>
  </header>
  <main>
    <div id="graph-container"></div>
    <div class="legend-panel" id="legend-panel"></div>
    <aside id="inspector-drawer">
      <div class="drawer-header">
        <div class="drawer-title-group">
          <h2 id="inspect-title">Node Details</h2>
          <span class="type-badge" id="inspect-type">Type</span>
        </div>
        <button class="close-btn" id="inspect-close">&times;</button>
      </div>
      <div class="drawer-body">
        <div class="property-section">
          <h3>Identifiers</h3>
          <div class="prop-row"><span class="prop-key">id</span><span class="prop-val" id="inspect-id">-</span></div>
          <div class="prop-row" id="row-doc"><span class="prop-key">document</span><span class="prop-val" id="inspect-doc">-</span></div>
          <div class="prop-row" id="row-source"><span class="prop-key">source_id</span><span class="prop-val" id="inspect-source">-</span></div>
        </div>
        <div class="property-section" id="section-desc">
          <h3>Description</h3>
          <p id="inspect-desc" style="font-size: 13px; line-height: 1.5; color: #cbd5e1;">-</p>
        </div>
        <div class="property-section" id="section-attributes">
          <h3>Properties & Attributes</h3>
          <div id="inspect-props-list"></div>
        </div>
      </div>
      <div class="drawer-footer">
        <button class="btn" id="btn-focus-neighbors" style="width: 100%; justify-content: center;">🔍 View Node Neighbors</button>
      </div>
    </aside>
    <div class="overlay-state" id="loading-overlay">
      <div class="spinner"></div>
      <p style="font-size: 13px; color: #94a3b8;">Loading knowledge graph from API...</p>
    </div>
    <div class="overlay-state" id="empty-overlay" style="display: none;">
      <h3 style="font-size: 16px; font-weight: 600;">No Nodes Found</h3>
      <p style="font-size: 13px; color: #94a3b8;">No entities matched the selected document or type filters.</p>
      <button class="btn" id="btn-reset-filters" style="margin-top: 10px;">Reset Filters</button>
    </div>
  </main>
  <script>
    let network = null;
    let graphData = { nodes: [], edges: [] };
    let selectedNodeId = null;
    let physicsEnabled = true;

    const container = document.getElementById('graph-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const emptyOverlay = document.getElementById('empty-overlay');
    const badgeStats = document.getElementById('badge-stats');
    const docFilter = document.getElementById('doc-filter');
    const typeFilter = document.getElementById('type-filter');
    const limitSelect = document.getElementById('limit-select');
    const searchInput = document.getElementById('search-input');
    const btnPhysics = document.getElementById('btn-physics');
    const btnFit = document.getElementById('btn-fit');
    const btnRefresh = document.getElementById('btn-refresh');
    const drawer = document.getElementById('inspector-drawer');
    const inspectClose = document.getElementById('inspect-close');
    const btnFocusNeighbors = document.getElementById('btn-focus-neighbors');

    async function init() {
      await loadSummary();
      await loadGraph();
      setupEventListeners();
    }

    async function loadSummary() {
      try {
        const res = await fetch('/api/graph/summary');
        if (!res.ok) return;
        const data = await res.json();
        if (data.documents && data.documents.length) {
          docFilter.innerHTML = '<option value="">All Documents</option>';
          data.documents.forEach(doc => {
            const opt = document.createElement('option');
            opt.value = doc;
            opt.textContent = doc;
            docFilter.appendChild(opt);
          });
        }
        if (data.node_types) {
          typeFilter.innerHTML = '<option value="">All Types</option>';
          Object.keys(data.node_types).forEach(type => {
            const opt = document.createElement('option');
            opt.value = type;
            opt.textContent = `${type} (${data.node_types[type]})`;
            typeFilter.appendChild(opt);
          });
        }
      } catch (err) {
        console.warn('Could not load summary:', err);
      }
    }

    async function loadGraph() {
      showLoading(true);
      emptyOverlay.style.display = 'none';

      const doc = docFilter.value;
      const type = typeFilter.value;
      const limit = limitSelect.value || 300;

      const params = new URLSearchParams();
      if (doc) params.set('doc_id', doc);
      if (type) params.set('node_type', type);
      params.set('limit', limit);

      try {
        const res = await fetch(`/api/graph?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();
        graphData = data;
        renderGraph(data);
        badgeStats.textContent = `${data.metadata.node_count} nodes | ${data.metadata.edge_count} edges (${data.metadata.execution_time_ms}ms)`;
        if (!data.nodes || data.nodes.length === 0) {
          emptyOverlay.style.display = 'flex';
        }
      } catch (err) {
        console.error('Failed to load graph:', err);
        badgeStats.textContent = 'Error loading graph';
      } finally {
        showLoading(false);
      }
    }

    function renderGraph(data) {
      const visNodes = data.nodes.map(n => {
        const color = n.color || '#94A3B8';
        return {
          id: n.id,
          label: n.label.length > 20 ? n.label.substring(0, 18) + '...' : n.label,
          title: `<b>${n.label}</b><br/>Type: ${n.type}<br/>ID: ${n.id}`,
          shape: n.shape || 'box',
          color: {
            background: color,
            border: '#ffffff',
            highlight: { background: color, border: '#38bdf8' }
          },
          font: { color: '#ffffff', face: 'Inter', size: 12 },
          rawNode: n
        };
      });

      const visEdges = data.edges.map(e => ({
        id: e.id,
        from: e.source,
        to: e.target,
        label: e.label,
        arrows: 'to',
        font: { color: '#94a3b8', size: 10, align: 'horizontal', background: 'rgba(15, 23, 42, 0.7)' },
        color: { color: 'rgba(148, 163, 184, 0.4)', highlight: '#38bdf8' },
        smooth: { type: 'cubicBezier', roundness: 0.2 }
      }));

      const networkData = {
        nodes: new vis.DataSet(visNodes),
        edges: new vis.DataSet(visEdges)
      };

      const options = {
        physics: {
          enabled: physicsEnabled,
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -40,
            centralGravity: 0.008,
            springLength: 90,
            springConstant: 0.06
          },
          stabilization: { iterations: 120 }
        },
        interaction: {
          hover: true,
          tooltipDelay: 150,
          selectConnectedEdges: true
        }
      };

      if (network) {
        network.destroy();
      }

      network = new vis.Network(container, networkData, options);

      network.on('click', function(params) {
        if (params.nodes.length > 0) {
          const nid = params.nodes[0];
          const nodeItem = data.nodes.find(n => n.id === nid);
          if (nodeItem) openInspector(nodeItem);
        } else {
          closeInspector();
        }
      });

      network.on('doubleClick', function(params) {
        if (params.nodes.length > 0) {
          const nid = params.nodes[0];
          exploreNeighbors(nid);
        }
      });

      updateLegend(data.nodes);
    }

    function updateLegend(nodes) {
      const legendPanel = document.getElementById('legend-panel');
      const types = {};
      nodes.forEach(n => {
        if (!types[n.type]) {
          types[n.type] = { color: n.color, count: 0 };
        }
        types[n.type].count++;
      });

      legendPanel.innerHTML = '';
      Object.keys(types).forEach(type => {
        const chip = document.createElement('div');
        chip.className = 'legend-chip';
        chip.innerHTML = `<span class="legend-dot" style="background:${types[type].color}"></span><span>${type} (${types[type].count})</span>`;
        chip.onclick = () => {
          typeFilter.value = type;
          loadGraph();
        };
        legendPanel.appendChild(chip);
      });
    }

    function openInspector(node) {
      selectedNodeId = node.id;
      document.getElementById('inspect-title').textContent = node.label || node.id;
      const typeBadge = document.getElementById('inspect-type');
      typeBadge.textContent = node.type;
      typeBadge.style.backgroundColor = (node.color || '#38bdf8') + '33';
      typeBadge.style.color = node.color || '#38bdf8';
      typeBadge.style.border = `1px solid ${node.color || '#38bdf8'}`;

      document.getElementById('inspect-id').textContent = node.id;
      document.getElementById('inspect-doc').textContent = node.properties.doc_id || 'N/A';
      document.getElementById('inspect-source').textContent = node.properties.source || 'N/A';

      const desc = node.properties.description;
      const descEl = document.getElementById('inspect-desc');
      if (desc) {
        descEl.textContent = desc;
        document.getElementById('section-desc').style.display = 'block';
      } else {
        document.getElementById('section-desc').style.display = 'none';
      }

      const propsList = document.getElementById('inspect-props-list');
      propsList.innerHTML = '';
      const skipKeys = ['description', 'doc_id', 'source', 'aliases', 'id', 'name', 'type'];
      const extraKeys = Object.keys(node.properties).filter(k => !skipKeys.includes(k));

      if (extraKeys.length) {
        extraKeys.forEach(k => {
          const row = document.createElement('div');
          row.className = 'prop-row';
          row.innerHTML = `<span class="prop-key">${k}</span><span class="prop-val">${node.properties[k]}</span>`;
          propsList.appendChild(row);
        });
        document.getElementById('section-attributes').style.display = 'block';
      } else {
        document.getElementById('section-attributes').style.display = 'none';
      }

      drawer.classList.add('open');
    }

    function closeInspector() {
      drawer.classList.remove('open');
      selectedNodeId = null;
    }

    async function exploreNeighbors(nodeId) {
      showLoading(true);
      try {
        const res = await fetch(`/api/graph/nodes/${encodeURIComponent(nodeId)}/neighbors?depth=1&limit=80`);
        if (!res.ok) throw new Error('Neighborhood traversal failed');
        const data = await res.json();
        renderGraph(data);
        badgeStats.textContent = `Neighborhood of ${nodeId}: ${data.metadata.node_count} nodes (${data.metadata.execution_time_ms}ms)`;
        if (data.focus_node) {
          openInspector(data.focus_node);
        }
      } catch (err) {
        console.error('Failed to explore neighbors:', err);
      } finally {
        showLoading(false);
      }
    }

    function setupEventListeners() {
      docFilter.addEventListener('change', loadGraph);
      typeFilter.addEventListener('change', loadGraph);
      limitSelect.addEventListener('change', loadGraph);
      btnRefresh.addEventListener('click', loadGraph);
      inspectClose.addEventListener('click', closeInspector);

      btnFocusNeighbors.addEventListener('click', () => {
        if (selectedNodeId) exploreNeighbors(selectedNodeId);
      });

      btnFit.addEventListener('click', () => {
        if (network) network.fit({ animation: { duration: 500 } });
      });

      btnPhysics.addEventListener('click', () => {
        physicsEnabled = !physicsEnabled;
        if (network) network.setOptions({ physics: { enabled: physicsEnabled } });
        btnPhysics.textContent = physicsEnabled ? '⚡ Freeze' : '❄ Physics ON';
      });

      document.getElementById('btn-reset-filters').addEventListener('click', () => {
        docFilter.value = '';
        typeFilter.value = '';
        limitSelect.value = '300';
        loadGraph();
      });

      searchInput.addEventListener('input', (e) => {
        const q = e.target.value.trim().toLowerCase();
        if (!q || !network || !graphData.nodes.length) return;
        const match = graphData.nodes.find(n =>
          (n.label && n.label.toLowerCase().includes(q)) || n.id.toLowerCase().includes(q)
        );
        if (match) {
          network.focus(match.id, {
            scale: 1.2,
            animation: { duration: 400 }
          });
          network.selectNodes([match.id]);
          openInspector(match);
        }
      });
    }

    function showLoading(show) {
      loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive graph visualization."""
    return RedirectResponse(url="/visualize")


@app.get(
    "/visualize",
    response_class=HTMLResponse,
    summary="Interactive Graph Visualization Webpage",
    description="Serves the standalone interactive vis-network visualization interface consuming the JSON API.",
    tags=["Frontend Visualization"],
)
def get_visualization_page():
    """Serve standalone interactive HTML/JS visualization client."""
    return HTMLResponse(content=VISUALIZATION_HTML)


# ==============================================================================
# 6. RUNNABLE ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    logger.info("Starting GraphRAG API server on http://%s:%d", API_HOST, API_PORT)
    uvicorn.run("graph_api:app", host=API_HOST, port=API_PORT, reload=True)
