"""
Graph API Routes

GET  /api/graph/nodes              → all nodes
GET  /api/graph/relationships      → all relationships
GET  /api/graph/node/{node_id}     → one node and connections
POST /api/graph/cypher             → run cypher query
GET  /api/graph/stats              → node and relationship counts
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from graph.neo4j_manager import (
    get_all_nodes,
    get_all_relationships,
    run_cypher_query
)

router = APIRouter()


# ── Request / Response Models ──────────────────────────────

class CypherRequest(BaseModel):
    query: str
    limit: Optional[int] = 50


class CypherResponse(BaseModel):
    success:      bool
    records:      list
    total_records:int
    query_used:   str


# ── Routes ─────────────────────────────────────────────────

@router.get("/nodes")
def get_nodes():
    """
    Get all nodes from the knowledge graph.
    Returns id, name, type and community for each node.
    """
    try:
        nodes = get_all_nodes()
        return {
            "success":     True,
            "total_nodes": len(nodes),
            "nodes":       nodes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/relationships")
def get_relationships():
    """
    Get all relationships from the knowledge graph.
    Returns from, to, type for each relationship.
    """
    try:
        rels = get_all_relationships()
        return {
            "success":             True,
            "total_relationships": len(rels),
            "relationships":       rels
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/node/{node_id}")
def get_node(node_id: str):
    """
    Get a specific node and all its connections.
    """
    try:
        query = f"""
        MATCH (n:Entity {{id: '{node_id}'}})
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        """
        records = run_cypher_query(query)

        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"Node '{node_id}' not found"
            )

        return {
            "success": True,
            "node_id": node_id,
            "records": records
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cypher", response_model=CypherResponse)
def execute_cypher(request: CypherRequest):
    """
    Execute a Cypher query on the Neo4j knowledge graph.

    Example queries:
    - MATCH (n:Entity) RETURN n LIMIT 10
    - MATCH (n:Entity {type: 'Actor'}) RETURN n
    - MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 20
    """
    try:
        query = request.query
        if request.limit and "LIMIT" not in query.upper():
            query = f"{query} LIMIT {request.limit}"

        records = run_cypher_query(query)

        # Convert Neo4j objects to serializable dicts
        serializable = []
        for record in records:
            row = {}
            for key, value in record.items():
                row[key] = str(value) if not isinstance(
                    value, (str, int, float, bool, list, dict, type(None))
                ) else value
            serializable.append(row)

        return CypherResponse(
            success=True,
            records=serializable,
            total_records=len(serializable),
            query_used=query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def get_graph_stats():
    """
    Get counts of nodes and relationships in the graph.
    """
    try:
        nodes = get_all_nodes()
        rels  = get_all_relationships()

        # Count by type
        node_type_counts = {}
        for node in nodes:
            t = node.get("type", "Unknown")
            node_type_counts[t] = node_type_counts.get(t, 0) + 1

        rel_type_counts = {}
        for rel in rels:
            t = rel.get("type", "Unknown")
            rel_type_counts[t] = rel_type_counts.get(t, 0) + 1

        return {
            "success":           True,
            "total_nodes":       len(nodes),
            "total_relationships": len(rels),
            "nodes_by_type":     node_type_counts,
            "relationships_by_type": rel_type_counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))