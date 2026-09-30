import streamlit as st
import pandas as pd
from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.graph_analysis import evaluate_structure
from graph.flow_graph_analysis import analyze_flow_detailed
from graph.graph_visualizer import render_graph

def render_graph_quality_section():
    # Retrieve from session state or load from database
    nodes = st.session_state.get("nodes")
    rels = st.session_state.get("relationships")
    
    if not nodes or not rels:
        try:
            nodes = get_all_nodes()
            rels = get_all_relationships()
        except Exception as e:
            st.error(f"Error reading graph for quality evaluation: {e}")
            return

    if not nodes:
        st.info("The graph is empty. Upload a document in Build Graph to run quality evaluation.")
        return

    # Extract conflict lists and removed elements
    type_conflicts = st.session_state.get("type_conflicts") or []
    removed_nodes = st.session_state.get("removed_nodes") or []
    removed_rels = st.session_state.get("removed_rels") or []

    # Get structure report
    structure_report = st.session_state.get("structure_report")
    if not structure_report:
        try:
            linked = st.session_state.get("linked_requirements") or {}
            items_with_links = linked.get("items_with_links", [])
            structure_report = evaluate_structure(nodes, rels, items_with_links)
        except Exception:
            structure_report = {}

    # 1. Graph Quality Report
    st.subheader("Graph Quality Report")
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Valid Nodes", len(nodes))
    q2.metric("Valid Relationships", len(rels))
    q3.metric("Removed Nodes", len(removed_nodes))
    q4.metric("Removed Relationships", len(removed_rels))

    if type_conflicts:
        with st.expander(f"Type Conflicts Fixed ({len(type_conflicts)})"):
            df_conflicts = pd.DataFrame(type_conflicts)
            st.dataframe(df_conflicts, use_container_width=True)

    if removed_nodes:
        with st.expander(f"Removed Nodes ({len(removed_nodes)})"):
            df_rem_nodes = pd.DataFrame(removed_nodes)
            st.dataframe(df_rem_nodes, use_container_width=True)

    st.write("")

    # 2. Graph Evaluation (Structural Health)
    st.subheader("Graph Evaluation (Structural Health)")
    st.caption("No ground truth needed — checks the graph is well-formed (connected, not hub-dominated, nothing missing).")

    warnings = [w for w in structure_report.get("flags", []) if w.startswith("WARN:")]
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("Pass: Graph structural health is excellent. No critical issues detected.")

    st.write("")

    # Checkbox to render interactive graphs
    show_graph = st.checkbox(
        f"Show Interactive Knowledge Graphs (currently {len(nodes)} nodes — check this to render)",
        value=False,
        key="chk_show_interactive_graphs"
    )

    if show_graph:
        tab_full, tab_seq = st.tabs(["Full Knowledge Graph", "Sequential Flow"])
        with tab_full:
            try:
                render_graph(nodes, rels, flow_only=False, key="maingraph_quality")
            except Exception as e:
                st.error(f"Failed to render full knowledge graph: {e}")
        with tab_seq:
            try:
                render_graph(nodes, rels, flow_only=True, key="flowgraph_quality")
            except Exception as e:
                st.error(f"Failed to render sequential flow graph: {e}")

