"""
Neo4j Manager
Handles all Neo4j operations with automatic reconnection.
Neo4j AuraDB free tier goes to sleep after inactivity —
SessionExpired errors are caught and the driver is recreated.
"""

from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


# ── Driver with reconnection support ──────────────────────
def _create_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=200,   # recreate connections after 200s
        max_connection_pool_size=10,
        connection_timeout=30
    )

driver = _create_driver()


def _get_driver():
    """
    Returns the driver, recreating it if the connection is dead.
    Called before every session to ensure connection is alive.
    """
    global driver
    try:
        # Quick connectivity check
        driver.verify_connectivity()
        return driver
    except Exception:
        # Connection dead — recreate driver
        driver = _create_driver()
        return driver


def _run_with_retry(fn, retries=2):
    """
    Run a Neo4j function with automatic retry on SessionExpired.

    Args:
        fn: function that takes a session and returns a result
        retries: how many times to retry

    Returns:
        result of fn
    """
    global driver
    for attempt in range(retries):
        try:
            d = _get_driver()
            with d.session() as session:
                return fn(session)
        except (SessionExpired, ServiceUnavailable):
            # Recreate driver and retry
            driver = _create_driver()
            if attempt == retries - 1:
                raise  # re-raise on last attempt
    return None


# ── Write functions ────────────────────────────────────────

def _write_node(tx, node):
    tx.run(
        """
        MERGE (n:Entity {id: $id})
        SET n.name = $name, n.type = $type
        """,
        id=node["id"],
        name=node["name"],
        type=node["type"]
    )


def _write_relationship(tx, rel):
    query = f"""
    MATCH (a:Entity {{id: $from_id}})
    MATCH (b:Entity {{id: $to_id}})
    MERGE (a)-[r:{rel['type']}]->(b)
    """
    tx.run(query, from_id=rel["from"], to_id=rel["to"])


def _write_community(tx, node_id, community_id):
    tx.run(
        """
        MATCH (n:Entity {id: $id})
        SET n.community = $community_id
        """,
        id=node_id,
        community_id=int(community_id)
    )


# ── Public functions ───────────────────────────────────────

def create_node(tx, node):
    _write_node(tx, node)


def create_relationship(tx, rel):
    _write_relationship(tx, rel)


def insert_graph(nodes, relationships):
    def fn(session):
        for node in nodes:
            session.execute_write(_write_node, node)
        for rel in relationships:
            session.execute_write(_write_relationship, rel)

    _run_with_retry(fn)


def run_cypher_query(query: str) -> list:
    """Run any read Cypher query, returns list of dicts."""
    def fn(session):
        result = session.run(query)
        return [dict(record) for record in result]

    return _run_with_retry(fn) or []


def get_all_nodes() -> list:
    """Get all nodes from Neo4j."""
    def fn(session):
        result = session.run(
            "MATCH (n:Entity) RETURN n.id AS id, n.name AS name, n.type AS type"
        )
        return [
            {"id": r["id"], "name": r["name"], "type": r["type"]}
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_all_relationships() -> list:
    """Get all relationships from Neo4j."""
    def fn(session):
        result = session.run(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.id AS from, b.id AS to, type(r) AS rel_type"
        )
        return [
            {"from": r["from"], "to": r["to"], "type": r["rel_type"]}
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_community_nodes_db(community_id: int) -> list:
    """Get all nodes in a specific community."""
    def fn(session):
        result = session.run(
            """
            MATCH (n:Entity {community: $community_id})
            RETURN n.id AS id, n.name AS name,
                   n.type AS type, n.community AS community
            """,
            community_id=community_id
        )
        return [
            {
                "id":        r["id"],
                "name":      r["name"],
                "type":      r["type"],
                "community": r["community"]
            }
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_community_relationships_db(community_id: int) -> list:
    """Get all relationships within a community."""
    def fn(session):
        result = session.run(
            """
            MATCH (a:Entity {community: $community_id})
                  -[r]->
                  (b:Entity {community: $community_id})
            RETURN a.id AS from, b.id AS to, type(r) AS rel_type
            """,
            community_id=community_id
        )
        return [
            {"from": r["from"], "to": r["to"], "type": r["rel_type"]}
            for r in result
        ]
    return _run_with_retry(fn) or []


def get_all_communities_db() -> dict:
    """Get all communities and their info."""
    def fn(session):
        result = session.run(
            """
            MATCH (n:Entity)
            WHERE n.community IS NOT NULL
            RETURN n.community AS community_id,
                   collect(n.id)   AS node_ids,
                   collect(n.type) AS types
            """
        )
        communities = {}
        for record in result:
            community_id = int(record["community_id"])
            types        = record["types"]
            type_counts  = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            primary_types = sorted(
                type_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            communities[community_id] = {
                "node_count":    len(record["node_ids"]),
                "nodes":         record["node_ids"],
                "primary_types": [t[0] for t in primary_types]
            }
        return communities
    return _run_with_retry(fn) or {}


def write_community_ids(partition: dict) -> dict:
    """Write community IDs to Neo4j nodes."""
    def fn(session):
        for node_id, community_id in partition.items():
            session.execute_write(_write_community, node_id, community_id)

    _run_with_retry(fn)

    community_sizes = {}
    for community_id in partition.values():
        community_sizes[int(community_id)] = \
            community_sizes.get(int(community_id), 0) + 1

    return {
        "total_nodes":       len(partition),
        "total_communities": len(set(partition.values())),
        "community_sizes":   community_sizes,
        "success":           True
    }