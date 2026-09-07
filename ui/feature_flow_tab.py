import streamlit as st
import traceback

from graph.neo4j_manager import get_all_nodes, get_all_relationships, get_graph_by_document
from graph.feature_extractor import extract_features, get_feature_by_id, get_feature_subgraph
from graph.graph_visualizer import render_graph, render_cypher_graph


@st.cache_data(show_spinner=False)
def get_cached_features(doc_context: str, nodes: list, rels: list):
    """
    Cached wrapper around the feature extractor.
    Since extraction can involve an LLM call, we cache it heavily based on the
    document context and graph structure. The feature_extractor itself also has
    a cache key, but Streamlit's @st.cache_data provides the UI-level persistence.
    """
    # Create deterministic IDs to avoid unhashable type issues in st.cache_data
    # Actually, we can just return the extraction result directly since the inputs
    # (nodes, rels) are lists of dicts which st.cache_data handles via hashing.
    try:
        return extract_features(nodes, rels, use_llm=True)
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        traceback.print_exc()
        return {"features": [], "warnings": [str(e)]}


def render_feature_flow_tab():
    st.header("Feature Flow")
    st.caption("Inspect and navigate the executable workflows extracted from your document. "
               "Features are semantically grouped and ordered.")

    # 1. Fetch graph data
    try:
        doc_context = st.session_state.get("active_doc_context", "All Documents")
        if doc_context and doc_context != "All Documents":
            nodes, rels = get_graph_by_document(doc_context)
        else:
            nodes = get_all_nodes()
            rels = get_all_relationships()
    except Exception as e:
        st.error(f"Failed to load graph from database: {e}")
        return

    if not nodes:
        st.info("No source graph is available yet. Build a graph from a BRD to generate Feature Flows.")
        return

    # 2. Extract Features
    with st.spinner("Extracting semantic features (this may take a moment on first run)..."):
        extraction = get_cached_features(doc_context, nodes, rels)

    features = extraction.get("features", [])
    warnings = extraction.get("warnings", [])

    for w in warnings:
        st.warning(w)

    if not features:
        st.info("No executable workflows could be confidently identified. Showing the relevant source graph instead.")
        render_graph(nodes, rels, key="ff_full_fallback")
        return

    # 3. State Management for Selection
    selected_feature_id = st.session_state.get("selected_feature_id")

    # If a feature is selected, show the focused view
    if selected_feature_id:
        feature = get_feature_by_id(extraction, selected_feature_id)
        if not feature:
            st.session_state.pop("selected_feature_id", None)
            st.rerun()
        
        _render_focused_feature_view(feature, nodes, rels)

    # Otherwise, show the feature cards grid
    else:
        _render_feature_cards_grid(features)


def _render_feature_cards_grid(features: list):
    st.markdown("### Discovered Features")
    
    # Render features in a grid (e.g., 3 columns)
    cols_per_row = 3
    for i in range(0, len(features), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(features):
                feature = features[i + j]
                with cols[j]:
                    _render_feature_card(feature)


def _render_feature_card(feature: dict):
    # Styling to make it look like a card
    st.markdown(f"#### {feature.get('name', 'Unknown Feature')}")
    
    if feature.get("description"):
        st.caption(feature["description"])
    else:
        st.caption("No description available.")

    # Flow Summary
    steps = feature.get("ordered_steps", [])
    if steps:
        flow_summary = " → ".join(steps[:3])
        if len(steps) > 3:
            flow_summary += " → ..."
        st.markdown(f"**Flow:** `{flow_summary}`")

    # Metadata badges
    step_count = len(feature.get("step_node_ids", []))
    branch_count = len(feature.get("branches", []))
    actor_count = len(feature.get("actors", []))
    
    meta_parts = [f"{step_count} Steps"]
    if branch_count > 0:
        meta_parts.append(f"{branch_count} Branches")
    if actor_count > 0:
        meta_parts.append(f"{actor_count} Actors")
        
    st.markdown(f"_{' | '.join(meta_parts)}_")

    # Action button
    if st.button("View Graph", key=f"btn_view_{feature['feature_id']}", use_container_width=True):
        st.session_state["selected_feature_id"] = feature["feature_id"]
        st.rerun()
    
    st.markdown("---")


def _render_focused_feature_view(feature: dict, all_nodes: list, all_rels: list):
    st.markdown(f"### Selected Feature: {feature.get('name', 'Unknown')}")
    if feature.get("description"):
        st.write(feature["description"])

    # Show the full step sequence
    steps = feature.get("ordered_steps", [])
    if steps:
        st.markdown("**Sequence:** " + " ➔ ".join(f"`{s}`" for s in steps))
        
    if feature.get("alternate_paths"):
        st.markdown("**Alternate Paths:** " + ", ".join(feature["alternate_paths"]))
    if feature.get("negative_paths"):
        st.markdown("**Negative Paths:** " + ", ".join(feature["negative_paths"]))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Feature Flows"):
            st.session_state.pop("selected_feature_id", None)
            st.session_state.pop("ff_show_full_graph", None)
            st.rerun()
    with col2:
        show_full = st.session_state.get("ff_show_full_graph", False)
        btn_label = "Show Focused Graph" if show_full else "Show Entire Graph"
        if st.button(btn_label):
            st.session_state["ff_show_full_graph"] = not show_full
            st.rerun()

    st.divider()

    show_full = st.session_state.get("ff_show_full_graph", False)
    if show_full:
        st.markdown("#### Entire Graph (Context)")
        # Show all nodes/rels
        render_graph(all_nodes, all_rels, key="ff_graph_full")
    else:
        st.markdown("#### Focused Feature Graph")
        # Show only this feature's subgraph + 1 hop supporting context
        f_nodes, f_rels = get_feature_subgraph(feature, all_nodes, all_rels, include_supporting=True)
        # Using flow_only=False because we WANT to see the supporting nodes (inputs, requirements)
        # connected to our workflow steps. The workflow steps themselves will still be styled properly.
        render_graph(f_nodes, f_rels, key=f"ff_graph_focused_{feature['feature_id']}")
