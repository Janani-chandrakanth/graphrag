import streamlit as st
from graph.neo4j_manager import get_all_nodes, get_all_relationships

def render_graph_toolbar():
    """Render toolbar controls for graph visualization tabs."""
    st.sidebar.title("Graph Controls")
    if st.sidebar.button("Refresh Graph"):
        st.experimental_rerun()
    st.sidebar.caption("You can add more controls here, e.g., filters, layout options.")
