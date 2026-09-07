import streamlit as st
from graph.neo4j_manager import get_all_nodes, get_all_relationships

def render_health_metrics():
    """Render basic health metrics of the Neo4j graph in the dashboard."""
    st.subheader("Graph Health Metrics")
    with st.spinner("Fetching graph stats…"):
        nodes = get_all_nodes()
        rels = get_all_relationships()
        node_count = len(nodes)
        rel_count = len(rels)
    col1, col2 = st.columns(2)
    col1.metric("Nodes", node_count)
    col2.metric("Relationships", rel_count)
