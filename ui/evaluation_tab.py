# ui/evaluation_tab.py
"""Streamlit UI component for the Graph Quality Evaluation tab.
Renders metric scorecards, detailed diagnostic breakdowns, historical evaluation trends,
and export capabilities without altering existing graph pipelines.
"""

import json
import streamlit as st
from typing import Dict, Any

from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.evaluation.graph_evaluators import generate_evaluation_facts
from graph.evaluation.graph_evaluators import evaluate_information_retention
from graph.evaluation.graph_evaluators import evaluate_relationships
from graph.evaluation.graph_evaluators import evaluate_workflow
from graph.evaluation.graph_evaluators import compute_node_coverage
from graph.evaluation.graph_evaluators import build_evaluation_report, save_evaluation_run, get_evaluation_history


def render_evaluation_tab():
    """Main render function for the Evaluation tab in Streamlit."""
    st.header("Graph Quality Evaluation")
    st.caption(
        "Evaluate Knowledge Graph fidelity, Information Retention (MINE-1 benchmark), "
        "Relationship Quality, Workflow Sequence Accuracy, and Entity Coverage."
    )

    uploaded_doc_name = (
        st.session_state.get("uploaded_filename") or
        st.session_state.get("doc_name") or
        "Uploaded Requirement Document"
    )

    # 1. Header & Controls
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.info(f"**Active Document:** `{uploaded_doc_name}`")
    with col_right:
        run_button = st.button("Run Evaluation", type="primary", use_container_width=True, key="btn_run_graph_quality_eval")

    # Advanced text override expander
    with st.expander("Advanced Document Source Settings", expanded=False):
        doc_name_override = st.text_input(
            "Evaluation Run Title",
            value=uploaded_doc_name,
            key="eval_run_title_override"
        )
        custom_text_override = st.text_area(
            "Override Document Text (Optional)",
            value="",
            height=120,
            placeholder="Only paste here if testing against alternative document text...",
            key="eval_custom_text_override"
        )

    # Trigger Evaluation Execution
    if run_button:
        with st.spinner("Running Knowledge Graph Quality Evaluation..."):
            nodes = get_all_nodes()
            rels = get_all_relationships()

            if not nodes:
                st.warning("The Neo4j Knowledge Graph is empty. Please index or upload a document first.")
                return

            raw_text = (
                custom_text_override.strip() if custom_text_override.strip()
                else st.session_state.get("raw_text") or st.session_state.get("extracted_text")
            )
            normalized_doc = st.session_state.get("normalized_document")
            active_title = doc_name_override if doc_name_override.strip() else uploaded_doc_name

            # 1. Fact Extraction & Retention Evaluation
            facts = generate_evaluation_facts(
                normalized_document=normalized_doc,
                raw_text=raw_text,
                graph_nodes=nodes
            )
            retention_res = evaluate_information_retention(facts, nodes, rels)

            # 2. Relationship Quality Evaluation
            rel_res = evaluate_relationships(facts, nodes, rels)

            # 3. Workflow Evaluation
            wf_model = st.session_state.get("workflow")
            wf_res = evaluate_workflow(wf_model, nodes, rels)

            # 4. Node Coverage Evaluation
            coverage_res = compute_node_coverage(facts, nodes, rels)

            # 5. Build & Persist Report
            report = build_evaluation_report(
                retention_res, rel_res, wf_res, coverage_res, doc_name=active_title
            )
            save_evaluation_run(report)

            st.session_state["latest_evaluation_report"] = report
            st.success(f"Evaluation complete. Overall Quality Index: {report['overall_score']}%")

    # Load current report from session state or Neo4j history
    report = st.session_state.get("latest_evaluation_report")

    if not report:
        history = get_evaluation_history()
        if history:
            report = history[0].get("details")
            if report:
                st.session_state["latest_evaluation_report"] = report

    if not report:
        st.info("No evaluation runs found. Click Run Evaluation above to measure graph quality.")
        return

    # Executive Narrative Summary Text Box
    narrative = report.get("narrative_summary")
    if narrative:
        st.info(narrative)

    # 2. Top-level Scorecard Metrics
    st.divider()
    overall = report.get("overall_score", 0.0)
    summary = report.get("summary", {})
    weights = report.get("weights", {"retention": 35, "relationship": 25, "workflow": 25, "coverage": 15})

    score_status = "[HIGH QUALITY]" if overall >= 80 else ("[MODERATE QUALITY]" if overall >= 60 else "[NEEDS IMPROVEMENT]")
    st.subheader(f"{score_status} Overall Graph Quality Index: {overall}%")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Information Retention F1",
            f"{round(summary.get('fact_f1', 0.0) * 100, 1)}%",
            help=f"MINE-1 benchmark measuring text fact retention in Neo4j (Weight: {weights.get('retention')}%)."
        )
    with m2:
        st.metric(
            "Relationship Quality F1",
            f"{round(summary.get('rel_f1', 0.0) * 100, 1)}%",
            help=f"Precision, Recall, and connectivity of graph relationships (Weight: {weights.get('relationship')}%)."
        )
    with m3:
        st.metric(
            "Workflow Accuracy",
            f"{round(summary.get('workflow_accuracy', 0.0) * 100, 1)}%",
            help=f"Sequential flow accuracy, step order, and valid transitions (Weight: {weights.get('workflow')}%)."
        )
    with m4:
        st.metric(
            "Node Coverage Score",
            f"{round(summary.get('node_coverage', 0.0) * 100, 1)}%",
            help=f"Ratio of document entities covered without orphan node penalty (Weight: {weights.get('coverage')}%)."
        )

    # 3. Detailed Diagnostic Sub-Tabs
    t_overview, t_retention, t_rel, t_workflow, t_coverage, t_history = st.tabs([
        "Overview",
        "Facts & Retention",
        "Relationships",
        "Workflow Flow",
        "Node Coverage",
        "History & Trends"
    ])

    metrics = report.get("metrics", {})

    # Overview Tab
    with t_overview:
        st.markdown("### Evaluation Summary")
        c_left, c_right = st.columns(2)
        with c_left:
            st.json({
                "Document Name": report.get("document_name"),
                "Timestamp": report.get("timestamp"),
                "Total Benchmark Facts Evaluated": summary.get("total_facts_evaluated"),
                "Total Graph Nodes": summary.get("total_nodes_in_graph"),
                "Total Graph Relationships": summary.get("total_relationships")
            })
        with c_right:
            st.markdown("**Metric Weight Distribution**")
            st.markdown(f"- **Information Retention F1**: {weights.get('retention')}%")
            st.markdown(f"- **Relationship Quality F1**: {weights.get('relationship')}%")
            st.markdown(f"- **Workflow Sequence Accuracy**: {weights.get('workflow')}%")
            st.markdown(f"- **Node Coverage Score**: {weights.get('coverage')}%")

        st.divider()
        json_str = json.dumps(report, indent=2)
        st.download_button(
            "Download Evaluation Report (JSON)",
            data=json_str,
            file_name=f"evaluation_report_{report.get('run_id')}.json",
            mime="application/json",
            key="dl_graph_quality_eval_report"
        )

    # Retention Tab
    with t_retention:
        st.markdown("### Information Retention Details (MINE-1 Benchmark)")
        info_ret = metrics.get("information_retention", {})

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Verified Facts", info_ret.get("verified_count", 0))
        col_r2.metric("Partially Retained", info_ret.get("partial_count", 0))
        col_r3.metric("Missing Facts", info_ret.get("missing_count", 0))

        fact_results = info_ret.get("fact_results", [])
        if fact_results:
            st.markdown("**Extracted Facts & Graph Verification Status**")
            status_filter = st.selectbox("Filter Status", ["All", "verified", "partially_verified", "missing"])

            filtered = fact_results
            if status_filter != "All":
                filtered = [f for f in fact_results if f["status"] == status_filter]

            display_data = []
            for f in filtered:
                display_data.append({
                    "Status": f["status"],
                    "Score": f["score"],
                    "Subject": f.get("subject"),
                    "Predicate": f.get("predicate"),
                    "Object": f.get("object"),
                    "Details": f.get("details")
                })
            st.dataframe(display_data, use_container_width=True)

    # Relationships Tab
    with t_rel:
        st.markdown("### Relationship Quality Details")
        rel_q = metrics.get("relationship_quality", {})

        c_rel1, c_rel2, c_rel3, c_rel4 = st.columns(4)
        c_rel1.metric("Connectivity Rate", f"{round(rel_q.get('connectivity_rate', 0.0) * 100, 1)}%")
        c_rel2.metric("Valid Ontology Type Rate", f"{round(rel_q.get('valid_type_rate', 0.0) * 100, 1)}%")
        c_rel3.metric("Matched Fact Triples", rel_q.get("matched_rels", 0))
        c_rel4.metric("Dangling Relationships", rel_q.get("dangling_rels", 0))

        st.markdown("**Relationship Type Distribution in Neo4j**")
        st.json(rel_q.get("rel_type_distribution", {}))

    # Workflow Tab
    with t_workflow:
        st.markdown("### Workflow Sequence & Transition Accuracy")
        wf_q = metrics.get("workflow_sequence", {})

        cw1, cw2, cw3 = st.columns(3)
        cw1.metric("Step Coverage", f"{round(wf_q.get('step_coverage', 0.0) * 100, 1)}%")
        cw2.metric("Transition Accuracy", f"{round(wf_q.get('transition_accuracy', 0.0) * 100, 1)}%")
        cw3.metric("Orphan Procedural Steps", wf_q.get("orphan_step_count", 0))

        st.json({
            "Total Procedural Steps": wf_q.get("total_steps"),
            "Total Sequence Flow Relationships": wf_q.get("total_flow_rels"),
            "Workflow Features Detected": wf_q.get("feature_count"),
            "Valid Flow Transitions": wf_q.get("valid_transitions")
        })

    # Node Coverage Tab
    with t_coverage:
        st.markdown("### Node & Entity Coverage Analysis")
        nc_q = metrics.get("node_coverage", {})

        cn1, cn2, cn3 = st.columns(3)
        cn1.metric("Raw Coverage Ratio", f"{round(nc_q.get('raw_coverage', 0.0) * 100, 1)}%")
        cn2.metric("Orphan Nodes (0 Rels)", nc_q.get("orphan_nodes", 0))
        cn3.metric("Orphan Ratio", f"{round(nc_q.get('orphan_ratio', 0.0) * 100, 1)}%")

        st.markdown("**Node Type Breakdown**")
        st.json(nc_q.get("type_distribution", {}))

    # History Tab
    with t_history:
        st.markdown("### Historical Evaluation Runs")
        history = get_evaluation_history()
        if history:
            hist_table = []
            for h in history:
                hist_table.append({
                    "Run ID": h.get("run_id"),
                    "Document Title": h.get("document_name"),
                    "Date": str(h.get("timestamp"))[:19],
                    "Overall Score": f"{h.get('overall_score')}%",
                    "Fact F1": f"{round((h.get('fact_f1') or 0.0) * 100, 1)}%",
                    "Rel F1": f"{round((h.get('rel_f1') or 0.0) * 100, 1)}%",
                    "Workflow Acc": f"{round((h.get('workflow_accuracy') or 0.0) * 100, 1)}%",
                    "Node Coverage": f"{round((h.get('node_coverage') or 0.0) * 100, 1)}%"
                })
            st.dataframe(hist_table, use_container_width=True)
        else:
            st.caption("No historical evaluation records found in Neo4j.")
