"""
Community Detection Module
Uses Louvain algorithm to detect communities in the Neo4j graph.
All Neo4j calls go through neo4j_manager for reconnection support.
"""

import networkx as nx
try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from graph.neo4j_manager import (
    get_all_nodes,
    get_all_relationships,
    get_community_nodes_db,
    get_community_relationships_db,
    get_all_communities_db,
    write_community_ids
)


def build_graph_from_neo4j() -> nx.DiGraph:
    """
    Pull all nodes and relationships from Neo4j.
    Build a NetworkX graph for community detection.
    """
    nodes = get_all_nodes()
    rels  = get_all_relationships()

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"], name=node["name"], type=node["type"])
    for rel in rels:
        G.add_edge(rel["from"], rel["to"], relationship_type=rel["type"])

    return G


def detect_communities(G: nx.DiGraph) -> dict:
    """
    Run Louvain community detection.
    Converts directed graph to undirected first.

    Returns:
        {node_id: community_id, ...}
    """
    G_undirected = G.to_undirected()
    if community_louvain is not None:
        partition = community_louvain.best_partition(
            G_undirected,
            resolution=1.0,
            random_state=42
        )
        return partition
    
    # Fallback to networkx community algorithms if python-louvain is not installed
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
    except Exception:
        communities = nx.community.greedy_modularity_communities(G_undirected)

    partition = {}
    for comm_id, node_set in enumerate(communities):
        for node in node_set:
            partition[node] = comm_id
    return partition


def get_community_nodes(community_id: int) -> list:
    """Get all nodes in a community from Neo4j."""
    return get_community_nodes_db(community_id)


def get_community_relationships(community_id: int) -> list:
    """Get all relationships within a community from Neo4j."""
    return get_community_relationships_db(community_id)


def get_all_communities() -> dict:
    """Get all communities and their info from Neo4j."""
    return get_all_communities_db()


def run_full_community_detection() -> dict:
    """
    Complete pipeline:
    1. Pull graph from Neo4j
    2. Run Louvain detection
    3. Write community IDs back to Neo4j
    """
    G = build_graph_from_neo4j()

    if len(G.nodes()) == 0:
        return {
            "success":           False,
            "error":             "No nodes found in Neo4j. Build the index first.",
            "total_communities": 0
        }

    partition = detect_communities(G)
    stats     = write_community_ids(partition)
    return stats