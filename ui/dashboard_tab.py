import streamlit as st
import pandas as pd
from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.version_manager import list_versions

def render_dashboard_tab():
    st.header("Dashboard")
    st.caption("A live overview of the knowledge graph and its update history.")

    with st.spinner("Fetching dashboard metrics…"):
        try:
            dash_nodes = get_all_nodes()
            dash_rels = get_all_relationships()
            versions = list_versions()
        except Exception as e:
            st.error(f"Error fetching dashboard metrics: {e}")
            return

    # Metrics
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Total Nodes", len(dash_nodes))
    with d2:
        st.metric("Total Relationships", len(dash_rels))
    with d3:
        st.metric("Versions Retained", len(versions))
    with d4:
        _types = {}
        for n in dash_nodes:
            t = n.get("type") or "Unknown"
            _types[t] = _types.get(t, 0) + 1
        st.metric("Distinct Node Types", len(_types))

    # Node Type Breakdown Expander
    if dash_nodes:
        with st.expander("Node type breakdown", expanded=False):
            df_types = pd.DataFrame(
                [{"Node Type": k, "Count": v} for k, v in sorted(_types.items(), key=lambda x: -x[1])]
            )
            st.dataframe(df_types, use_container_width=True)

    st.write("")
    st.subheader("Version History")
    if versions:
        v_data = []
        for v in versions:
            v_data.append({
                "Label": v.get("label"),
                "Created": str(v.get("created_at"))[:19],
                "Nodes": v.get("node_count"),
                "Relationships": v.get("rel_count"),
                "Source": v.get("source_filename", "N/A"),
            })
        st.dataframe(pd.DataFrame(v_data), use_container_width=True)
    else:
        st.caption("No versions retained yet — snapshots are created automatically when applying incremental updates in the Update Graph tab.")

