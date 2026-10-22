"""
tests/test_graph_api.py

Unit and integration tests for the Graph Visualization API, covering:
- Node and edge serialization with stable unique IDs.
- Pruning of edges with missing endpoints (endpoint referential integrity).
- Filtering by doc_id, node_type, rel_type, and pagination (limit, skip).
- Neighbor/subgraph traversal with configurable depths.
- Empty graph handling.
- Input validation on bounds (limit, depth).
- Database failure and graceful 500 responses.
- HTML visualization route delivery.
"""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from graph_api import app, GraphResponse, GraphSummaryResponse, NeighborResponse, HealthResponse

client = TestClient(app)


# ─── Mock Data Fixtures ──────────────────────────────────────────────

MOCK_ENTITY_RECORDS = [
    {
        "id": "node_customer",
        "name": "Customer",
        "type": "Actor",
        "description": "Registered retail banking customer",
        "source": "REQ-101",
        "aliases": ["User", "Client"],
        "sequence": 1,
        "is_flow_entry": True,
        "attributes_json": '{"channel": "mobile", "priority": "high"}',
        "doc_id": "BRD_Banking_v1.pdf",
    },
    {
        "id": "node_loan_app",
        "name": "Loan Application",
        "type": "Feature",
        "description": "Digital application workflow",
        "source": "REQ-102",
        "aliases": [],
        "sequence": 2,
        "is_flow_entry": False,
        "attributes_json": '{"risk_level": "medium"}',
        "doc_id": "BRD_Banking_v1.pdf",
    },
    {
        "id": "node_credit_check",
        "name": "Credit Check",
        "type": "Requirement",
        "description": "Automated score calculation",
        "source": "REQ-103",
        "aliases": [],
        "sequence": 3,
        "is_flow_entry": False,
        "attributes_json": None,
        "doc_id": "BRD_Banking_v1.pdf",
    },
]

MOCK_RELATIONSHIP_RECORDS = [
    {
        "from": "node_customer",
        "to": "node_loan_app",
        "type": "USES",
        "source": "REQ-101",
        "doc_id": "BRD_Banking_v1.pdf",
        "props": {"frequency": "daily"},
    },
    {
        "from": "node_loan_app",
        "to": "node_credit_check",
        "type": "TRIGGERS",
        "source": "REQ-102",
        "doc_id": "BRD_Banking_v1.pdf",
        "props": {},
    },
    # Edge with missing target endpoint (should be pruned by the service)
    {
        "from": "node_credit_check",
        "to": "node_missing_system",
        "type": "DEPENDS_ON",
        "source": "REQ-103",
        "doc_id": "BRD_Banking_v1.pdf",
        "props": {},
    },
]


# ─── Tests ───────────────────────────────────────────────────────────

def test_health_check_healthy():
    """Verify GET /api/health reports healthy when Neo4j is reachable."""
    with patch("graph_api.run_cypher_query", return_value=[{"count": 96}]):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["connected"] is True
        assert data["node_count"] == 96


def test_health_check_unhealthy():
    """Verify GET /api/health reports unhealthy when Neo4j fails."""
    with patch("graph_api.run_cypher_query", side_effect=Exception("Connection timed out")):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["connected"] is False
        assert "Connection timed out" in data["message"]


def test_get_graph_successful_and_serialization():
    """Verify GET /api/graph returns valid node/edge models and stable IDs."""
    def mock_query(query, params=None):
        if "count(DISTINCT n)" in query:
            return [{"total_nodes": 3, "total_edges": 2}]
        if "MATCH (n:Entity)" in query:
            return MOCK_ENTITY_RECORDS
        if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
            return MOCK_RELATIONSHIP_RECORDS
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_query):
        response = client.get("/api/graph?limit=100")
        assert response.status_code == 200
        payload = response.json()

        # Validate schema via Pydantic model
        validated = GraphResponse(**payload)
        assert len(validated.nodes) == 3
        assert len(validated.edges) == 2  # The 3rd edge with missing endpoint must be pruned!

        # Check node serialization
        customer_node = next(n for n in validated.nodes if n.id == "node_customer")
        assert customer_node.label == "Customer"
        assert customer_node.type == "Actor"
        assert customer_node.properties["channel"] == "mobile"
        assert customer_node.properties["priority"] == "high"
        assert customer_node.properties["sequence"] == 1
        assert customer_node.properties["is_flow_entry"] is True
        assert customer_node.color is not None
        assert customer_node.shape is not None

        # Check edge serialization and stable ID
        first_edge = validated.edges[0]
        assert first_edge.source == "node_customer"
        assert first_edge.target == "node_loan_app"
        assert first_edge.type == "USES"
        assert first_edge.id == "edge:node_customer->USES->node_loan_app:0"


def test_edge_endpoint_referential_integrity():
    """Ensure no edge references a node ID that is not in the returned node list."""
    def mock_query(query, params=None):
        if "count(DISTINCT n)" in query:
            return [{"total_nodes": 3, "total_edges": 3}]
        if "MATCH (n:Entity)" in query:
            return MOCK_ENTITY_RECORDS
        if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
            return MOCK_RELATIONSHIP_RECORDS
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_query):
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()

        returned_node_ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert edge["source"] in returned_node_ids, f"Edge source {edge['source']} not in nodes!"
            assert edge["target"] in returned_node_ids, f"Edge target {edge['target']} not in nodes!"


def test_get_graph_empty():
    """Verify empty graph response is handled gracefully without error."""
    def mock_empty(query, params=None):
        if "count(DISTINCT n)" in query:
            return [{"total_nodes": 0, "total_edges": 0}]
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_empty):
        response = client.get("/api/graph")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["metadata"]["node_count"] == 0
        assert data["metadata"]["edge_count"] == 0


def test_get_graph_filtering_params():
    """Verify filtering parameters are properly passed to Neo4j queries."""
    captured_params = {}

    def mock_capture(query, params=None):
        nonlocal captured_params
        if params and "doc_id" in params:
            captured_params = params
        if "count(DISTINCT n)" in query:
            return [{"total_nodes": 1, "total_edges": 0}]
        if "MATCH (n:Entity)" in query:
            return [MOCK_ENTITY_RECORDS[0]]
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_capture):
        response = client.get("/api/graph?doc_id=BRD_Banking_v1.pdf&node_type=Actor&limit=50&skip=10")
        assert response.status_code == 200
        assert captured_params.get("doc_id") == "BRD_Banking_v1.pdf"
        assert captured_params.get("node_type") == "Actor"
        assert captured_params.get("limit") == 50
        assert captured_params.get("skip") == 10


def test_invalid_parameters_validation():
    """Verify parameter bounds enforcement (e.g. limit < 1 or limit > 2000)."""
    # Negative limit
    res_neg = client.get("/api/graph?limit=-5")
    assert res_neg.status_code == 422

    # Excess limit
    res_large = client.get("/api/graph?limit=5000")
    assert res_large.status_code == 422

    # Invalid neighbor depth (> 3)
    res_depth = client.get("/api/graph/nodes/node_1/neighbors?depth=10")
    assert res_depth.status_code == 422


def test_get_node_neighbors_success():
    """Verify GET /api/graph/nodes/{node_id}/neighbors traverses and returns neighborhood."""
    def mock_neighbor_queries(query, params=None):
        if "MATCH (n:Entity {id: $node_id})" in query:
            return [MOCK_ENTITY_RECORDS[0]]
        if "MATCH path =" in query:
            return [MOCK_ENTITY_RECORDS[0], MOCK_ENTITY_RECORDS[1]]
        if "MATCH (a:Entity)-[r]->(b:Entity)" in query:
            return [MOCK_RELATIONSHIP_RECORDS[0]]
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_neighbor_queries):
        response = client.get("/api/graph/nodes/node_customer/neighbors?depth=1&direction=all")
        assert response.status_code == 200
        data = response.json()

        validated = NeighborResponse(**data)
        assert validated.focus_node is not None
        assert validated.focus_node.id == "node_customer"
        assert len(validated.nodes) == 2
        assert len(validated.edges) == 1
        assert validated.metadata.depth == 1
        assert validated.metadata.focus_node_id == "node_customer"


def test_get_node_neighbors_not_found():
    """Verify 404 response when querying neighbors of a non-existent node ID."""
    with patch("graph_api.run_cypher_query", return_value=[]):
        response = client.get("/api/graph/nodes/non_existent_id/neighbors")
        assert response.status_code == 404
        assert "does not exist" in response.json()["detail"]


def test_get_graph_summary():
    """Verify GET /api/graph/summary returns node and edge breakdown."""
    def mock_summary(query, params=None):
        if "coalesce(n.type, 'Unknown')" in query:
            return [{"type": "Actor", "count": 10}, {"type": "Feature", "count": 5}]
        if "MATCH ()-[r]->()" in query:
            return [{"type": "USES", "count": 8}, {"type": "TRIGGERS", "count": 4}]
        return []

    with patch("graph_api.run_cypher_query", side_effect=mock_summary), \
         patch("graph_api.get_stored_documents", return_value=["DocA.pdf", "DocB.docx"]):
        response = client.get("/api/graph/summary")
        assert response.status_code == 200
        data = response.json()

        validated = GraphSummaryResponse(**data)
        assert validated.total_nodes == 15
        assert validated.total_edges == 12
        assert validated.node_types["Actor"] == 10
        assert validated.relationship_types["USES"] == 8
        assert "DocA.pdf" in validated.documents


def test_database_failure_returns_500():
    """Verify database exceptions result in clean 500 error responses."""
    with patch("graph_api.run_cypher_query", side_effect=Exception("Neo4j database dead")):
        response = client.get("/api/graph")
        assert response.status_code == 500
        assert "Neo4j database dead" in response.json()["detail"]


def test_visualization_page_returns_html():
    """Verify GET /visualize returns the standalone interactive HTML document."""
    response = client.get("/visualize")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "GraphRAG Visualizer" in response.text
    assert "vis-network" in response.text
