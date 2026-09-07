import streamlit as st
import pandas as pd
from graph.version_manager import list_versions, get_version_snapshot, get_current_graph_as_version
from graph.graph_diff import diff_graphs
from graph.graph_visualizer import render_diff_graph

def render_compare_tab():
    st.header("Compare Graph Versions")
    st.caption(
        "Every incremental update (Update Graph tab) retains a snapshot of the graph exactly as it was right before that update was applied. "
        "Compare any two versions — or a version against the current live graph — to see what changed."
    )

    try:
        versions = list_versions()
    except Exception as e:
        st.error(f"Error retrieving graph versions: {e}")
        return

    if not versions:
        st.info("No versions retained yet. Versions are created automatically when applying updates in the Update Graph tab.")
        return

    options = [(v["id"], f"{v['label']} — {str(v.get('created_at'))[:19]} ({v.get('node_count', 0)} nodes)") for v in versions]
    options.append(("current", "Current (live graph)"))
    labels = [o[1] for o in options]

    col_a, col_b = st.columns(2)
    with col_a:
        idx_a = st.selectbox("From version (older)", range(len(options)), format_func=lambda i: labels[i], index=0, key="cmp_from")
    with col_b:
        idx_b = st.selectbox("To version (newer)", range(len(options)), format_func=lambda i: labels[i], index=len(options)-1, key="cmp_to")

    # Compare Button
    run_compare = st.button("Compare", type="primary", key="btn_run_compare")

    # We track comparison in session state so it stays visible on interaction
    if run_compare or st.session_state.get("comparison_performed"):
        st.session_state["comparison_performed"] = True
        id_a = options[idx_a][0]
        id_b = options[idx_b][0]

        ver_a = get_current_graph_as_version() if id_a == "current" else get_version_snapshot(id_a)
        ver_b = get_current_graph_as_version() if id_b == "current" else get_version_snapshot(id_b)

        if not ver_a or not ver_b:
            st.error("Failed to load selected version snapshots.")
            return

        diff_res = diff_graphs(ver_a, ver_b)
        summary = diff_res.get("summary", {})

        label_a = labels[idx_a]
        label_b = labels[idx_b]
        st.subheader(f"{label_a} → {label_b}")

        # Metrics matching Screenshot 6
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("+ Nodes", summary.get("added_nodes", 0))
        m2.metric("- Nodes", summary.get("removed_nodes", 0))
        m3.metric("~ Nodes", summary.get("modified_nodes", 0))
        m4.metric("+ Relationships", summary.get("added_relationships", 0))
        m5.metric("- Relationships", summary.get("removed_relationships", 0))
        m6.metric("Workflow changes", len(diff_res.get("changed_workflow", [])))

        st.divider()

        # Diff Graph Section
        st.subheader("Visual Comparison")
        st.caption("Unified diff graph viewer — added elements are green, removed elements are red/dashed, and modified elements are blue. Unchanged elements are drawn in grey.")
        
        show_diff_graph = st.checkbox("Show Interactive Diff Graph", value=True, key="chk_show_diff_graph")
        
        if show_diff_graph:
            try:
                render_diff_graph(diff_res, ver_a, ver_b, key="diff_graph_view")
            except Exception as viz_err:
                st.warning(f"Interactive diff graph rendering notice: {viz_err}")

        # Detail Expanders matching Screenshot 7
        added_nodes = diff_res.get("added_nodes", [])
        with st.expander(f"Added nodes ({len(added_nodes)})"):
            if added_nodes:
                st.dataframe(pd.DataFrame(added_nodes), use_container_width=True)
            else:
                st.caption("No added nodes.")

        modified_nodes = diff_res.get("modified_nodes", [])
        with st.expander(f"Modified nodes ({len(modified_nodes)})"):
            if modified_nodes:
                # flatten the modified nodes
                flat_mod = []
                for mn in modified_nodes:
                    flat_mod.append({
                        "Node ID": mn.get("id"),
                        "Before Type": mn.get("before", {}).get("type"),
                        "After Type": mn.get("after", {}).get("type"),
                        "Before Description": mn.get("before", {}).get("description"),
                        "After Description": mn.get("after", {}).get("description"),
                    })
                st.dataframe(pd.DataFrame(flat_mod), use_container_width=True)
            else:
                st.caption("No modified nodes.")

        added_rels = diff_res.get("added_relationships", [])
        with st.expander(f"Added relationships ({len(added_rels)})"):
            if added_rels:
                st.dataframe(pd.DataFrame(added_rels), use_container_width=True)
            else:
                st.caption("No added relationships.")

        changed_wf = diff_res.get("changed_workflow", [])
        with st.expander(f"Changed workflow ({len(changed_wf)})", expanded=True):
            if changed_wf:
                wf_table = []
                for cw in changed_wf:
                    wf_table.append({
                        "Source Node": cw.get("from"),
                        "Transition": cw.get("type"),
                        "Previous Step(s)": ", ".join(cw.get("before", [])),
                        "Updated Step(s)": ", ".join(cw.get("after", []))
                    })
                st.dataframe(pd.DataFrame(wf_table), use_container_width=True)
            else:
                st.caption("No workflow changes.")

