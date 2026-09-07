import streamlit as st
from graph.neo4j_manager import get_all_nodes, get_all_relationships, get_graph_by_document
from graph.workflow_extractor import extract_workflow_from_graph
from ui.feature_flow import render_feature_flow
from ui.detailed_feature_graph import render_detailed_feature_graph
from ui.screen_flow_view import render_screen_flow
from ui.test_coverage_view import render_test_coverage
from ui.workflow_overview import render_workflow_overview

def render_workflow_tab():
    st.header("Workflow Viewer")
    st.caption("Inspect and navigate the sequential workflows extracted from your document. Features are identified from the knowledge graph — upload a document in the Build Graph tab first.")

    # Reload from Database button
    if st.button("🔄 Reload from Database", key="btn_reload_wf_db"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Feature Graph Generation")

    try:
        doc_context = st.session_state.get("active_doc_context", "All Documents")
        if doc_context and doc_context != "All Documents":
            nodes, rels = get_graph_by_document(doc_context)
        else:
            nodes = get_all_nodes()
            rels = get_all_relationships()
        wf = extract_workflow_from_graph(nodes, rels)
    except Exception as e:
        st.error(f"Failed to extract workflow from graph: {e}")
        return

    if not wf or not wf.features:
        st.info("No features/workflows detected in the graph yet. Upload a requirement document in the Build Graph tab first.")
        return

    feature_names = [f.name for f in wf.features]
    selected_feature_name = st.selectbox(
        "Select Feature Graph to View",
        feature_names,
        index=0,
        key="sb_workflow_feature_select",
    )

    selected_feature = next((f for f in wf.features if f.name == selected_feature_name), wf.features[0])

    # Tab titles matching Screenshot 10
    tab_flow, tab_screen, tab_graph, tab_test = st.tabs([
        "▶ Feature Flow", "🖥️ Screen Flow", "🔍 Detailed Graph", "✅ Test Coverage"
    ])

    with tab_flow:
        render_feature_flow(selected_feature)

    with tab_screen:
        render_screen_flow(selected_feature)

    with tab_graph:
        render_detailed_feature_graph(selected_feature)

    with tab_test:
        render_test_coverage(selected_feature)


