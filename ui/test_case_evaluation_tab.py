# ui/test_case_evaluation_tab.py
"""Streamlit UI component for Test Case Evaluation.
Renders executive narrative summaries, deduplication scorecards, scenario details,
missed information details, flow violations, and unsupported test step reports.
No emojis or decorative elements.
"""

import json
import streamlit as st
from typing import Dict, Any, List

from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.evaluation.test_case_evaluator import get_test_case_evaluation_history
from graph.evaluation.test_case_preprocess import preprocess_test_cases
from graph.evaluation.llm_bridge import invoke_llm_for_judgment
from graph.evaluation.metric_aggregator import aggregate_metrics


def render_test_case_evaluation_tab():
    """Renders the Test Case Evaluation tab in Streamlit."""
    st.header("Test Case Evaluation")
    st.caption(
        "Determine whether generated test cases correctly, uniquely, and sufficiently cover "
        "the information represented in the knowledge graph and workflows."
    )

    # 1. Evaluation Scope & Execution Controls
    col_scope, col_mode, col_btn = st.columns([3, 2, 1])

    all_nodes = get_all_nodes()
    all_rels = get_all_relationships()

    feature_names = []
    for n in all_nodes:
        if str(n.get("type", "")).lower() == "feature":
            f_name = n.get("name") or n.get("id")
            if f_name and f_name not in feature_names:
                feature_names.append(f_name)

    scope_options = ["Entire Graph"] + feature_names

    with col_scope:
        selected_scope = st.selectbox(
            "Evaluation Scope",
            options=scope_options,
            key="test_case_eval_scope_select"
        )

    with col_mode:
        use_llm = st.checkbox(
            "LLM-assisted evaluation (slower, more accurate)",
            value=True,
            key="test_case_eval_use_llm"
        )

    with col_btn:
        st.write("")
        st.write("")
        run_button = st.button(
            "Run Evaluation",
            type="primary",
            use_container_width=True,
            key="btn_run_test_case_eval_main"
        )

    # Trigger Evaluation Execution
    if run_button:
        with st.spinner("Evaluating test cases against knowledge graph and workflows..."):
            if not all_nodes:
                st.warning("The Neo4j Knowledge Graph is empty. Please index or upload a document first.")
                return

            test_cases = (
                st.session_state.get("generated_test_cases")
                or st.session_state.get("test_cases")
                or []
            )

            if not test_cases:
                from graph.neo4j_manager import run_read_query
                records = run_read_query("MATCH (tc:Entity {type: 'TestCase'}) RETURN tc")
                test_cases = [r["tc"] for r in records if "tc" in r]

            if not test_cases:
                st.info("No test cases found to evaluate. Generate test cases first in the Build Graph or Query tab.")
                return

            # Hybrid pipeline: preprocess -> LLM judgments -> aggregate
            preprocessed = preprocess_test_cases(test_cases)

            llm_judgments = []
            if use_llm:
                with st.spinner("Running LLM semantic analysis (this may take a moment)..."):
                    try:
                        llm_judgments = invoke_llm_for_judgment(preprocessed, all_nodes, all_rels)
                    except Exception as llm_err:
                        st.warning(f"LLM judgment step failed, falling back to deterministic evaluation: {llm_err}")

            wf_model = st.session_state.get("workflow")
            report = aggregate_metrics(
                test_cases=preprocessed,
                llm_judgments=llm_judgments,
                graph_nodes=all_nodes,
                graph_rels=all_rels,
                workflow_model=wf_model,
                feature_name=selected_scope,
            )

            st.session_state["latest_test_case_eval_report"] = report
            mode_label = "Hybrid (LLM + Deterministic)" if llm_judgments else "Deterministic Only"
            st.success(f"Evaluation complete. Overall Coverage: {report['overall_coverage']}% [{mode_label}]")

    # Load report from session state or history
    report = st.session_state.get("latest_test_case_eval_report")

    if not report:
        history = get_test_case_evaluation_history()
        if history:
            report = history[0].get("details")
            if report:
                st.session_state["latest_test_case_eval_report"] = report

    if not report:
        st.info("No evaluation runs found. Select a scope and click Run Evaluation above.")
        return

    # 2. Executive Narrative Summary
    narrative = report.get("narrative_summary")
    if narrative:
        st.info(narrative)

    # Show evaluation mode badge
    eval_mode = report.get("evaluation_mode", "deterministic_only")
    llm_count = report.get("llm_judgments_count", 0)
    if eval_mode == "hybrid":
        st.caption(f"Evaluation mode: Hybrid (LLM + Deterministic) | LLM judgments: {llm_count}")
    else:
        st.caption("Evaluation mode: Deterministic Only")

    st.divider()

    # 3. Test Suite Uniqueness & Deduplication
    st.subheader("Test Suite Uniqueness & Deduplication")
    summary = report.get("summary", {})
    dedup = report.get("deduplication", {})

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Test Cases Generated", summary.get("total_test_cases", 0))
    d2.metric("Unique Test Cases", summary.get("unique_test_cases_count", 0))
    d3.metric("Duplicate / Redundant Count", summary.get("duplicate_test_cases_count", 0))

    dup_tcs = dedup.get("duplicate_test_cases", [])
    if dup_tcs:
        with st.expander(f"View Duplicate / Redundant Test Cases ({len(dup_tcs)})", expanded=False):
            st.caption("The following test cases were identified as redundant variations of existing unique test scenarios:")
            st.dataframe(
                [
                    {
                        "Test Case ID": d.get("id"),
                        "Title": d.get("title"),
                        "Duplicate Of": d.get("duplicate_of_id"),
                        "Reason": d.get("reason"),
                    }
                    for d in dup_tcs
                ],
                use_container_width=True,
            )

    st.divider()



    # 6. Unique Test Scenarios & Traceability
    st.subheader("Unique Test Scenarios & Traceability")
    unique_tcs = report.get("unique_test_cases", [])
    if unique_tcs:
        with st.expander(f"View Unique Test Scenarios Detail ({len(unique_tcs)})", expanded=False):
            for idx, tc in enumerate(unique_tcs):
                st.markdown(f"#### {tc.get('id')} - {tc.get('title')}")
                st.write(f"**Scenario Type:** {tc.get('scenario_type') or tc.get('type') or 'positive'}")
                st.write(f"**Objective:** {tc.get('objective')}")
                st.write(f"**Preconditions:** {tc.get('preconditions') or tc.get('precondition')}")

                steps = tc.get("steps", [])
                if steps:
                    st.write("**Test Steps:**")
                    for s_idx, step_item in enumerate(steps, start=1):
                        st.write(f"{s_idx}. {step_item}")

                st.write(f"**Expected Result:** {tc.get('expected_result')}")
                targets = tc.get("target_entities", tc.get("entities_used", []))
                if targets:
                    st.write(f"**Target Graph Entities:** {', '.join([f'{t}' for t in targets])}")
                st.write("---")

    st.divider()

    # 7. Missed Information Section
    st.subheader("Missed Information")
    missed = report.get("missed_information", {})

    miss_n_count = missed.get("missed_node_count", 0)
    miss_e_count = missed.get("missed_edge_count", 0)
    miss_wf_count = missed.get("missed_workflow_step_count", 0)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Uncovered Graph Nodes", miss_n_count)
    col_m2.metric("Uncovered Graph Edges", miss_e_count)
    col_m3.metric("Uncovered Workflow Steps", miss_wf_count)

    with st.expander("View Uncovered Graph Information Detail", expanded=False):
        if miss_n_count > 0:
            st.write("**Uncovered Nodes:**")
            st.dataframe(
                [{"Node ID": n.get("id"), "Node Name": n.get("name"), "Node Type": n.get("type")} for n in missed.get("missed_nodes", [])],
                use_container_width=True,
            )
        if miss_e_count > 0:
            st.write("**Uncovered Relationships:**")
            st.dataframe(
                [{"Source": e.get("source_name"), "Type": e.get("type"), "Target": e.get("target_name")} for e in missed.get("missed_edges", [])],
                use_container_width=True,
            )
        if miss_wf_count > 0:
            st.write("**Uncovered Workflow Steps:**")
            st.dataframe(
                [{"Feature": s.get("feature"), "Step ID": s.get("step_id"), "Action": s.get("action")} for s in missed.get("missed_workflow_steps", [])],
                use_container_width=True,
            )
        if miss_n_count == 0 and miss_e_count == 0 and miss_wf_count == 0:
            st.info("All graph nodes, relationships, and workflow steps are covered by test cases.")

    st.divider()

    # 8. Flow Violations Section
    st.subheader("Flow Violations")
    flow_violations = report.get("flow_violations", [])
    if flow_violations:
        st.warning(f"Detected {len(flow_violations)} test case step order violation(s).")
        with st.expander("View Flow Violation Details", expanded=False):
            for v in flow_violations:
                st.write(f"**Test Case ID:** {v.get('test_case_id')} - {v.get('title')}")
                st.write(f"**Reason:** {v.get('reason')}")
                st.write(f"- Expected transition: {v.get('expected_transition')}")
                st.write(f"- Generated transition: {v.get('generated_transition')}")
                st.write("---")
    else:
        st.success("All generated test cases follow valid graph/workflow paths.")

    st.divider()

    # 9. Unsupported Test Steps Section
    st.subheader("Unsupported Test Steps")
    unsupported = report.get("unsupported_test_steps", [])
    if unsupported:
        st.warning(f"Detected {len(unsupported)} unsupported/hallucinated test step(s).")
        with st.expander("View Unsupported Test Step Details", expanded=False):
            for u in unsupported:
                st.write(f"**Test Case ID:** {u.get('test_case_id')} - {u.get('title')}")
                st.write(f"**Unsupported Step:** {u.get('unsupported_step')}")
                st.write(f"**Referenced Entity:** {u.get('unsupported_entity')}")
                st.write(f"**Reason:** {u.get('reason')}")
                st.write("---")
    else:
        st.success("All test case steps are supported by the source knowledge graph.")

    st.divider()

    # 10. Export Options
    st.subheader("Evaluation Report Export")
    json_str = json.dumps(report, indent=2)
    st.download_button(
        "Download Evaluation Report (JSON)",
        data=json_str,
        file_name=f"test_case_evaluation_report_{report.get('run_id')}.json",
        mime="application/json",
        key="dl_test_case_eval_report_main"
    )

    st.divider()

    # 11. Test Case Quality Evaluation (Restored functionality)
    st.subheader("Test Case Quality Evaluation")
    st.caption("Generate a plain-text evaluation report with LLM-judged Correctness, Completeness, Specificity, and Preconditions metrics.")
    
    if st.button("Generate Text Report", type="secondary", key="btn_legacy_text_report"):
        with st.spinner("Running LLM evaluation (this will take a moment)..."):
            from graph.evaluation.legacy_report_generator import evaluate_test_cases_legacy
            
            # Use original generated test cases from session state if available
            raw_tcs = (
                st.session_state.get("generated_test_cases")
                or st.session_state.get("test_cases")
                or report.get("unique_test_cases", [])
            )
            
            legacy_report_text = evaluate_test_cases_legacy(raw_tcs, all_nodes, all_rels)
            st.session_state["legacy_report_text"] = legacy_report_text

    legacy_text = st.session_state.get("legacy_report_text")
    if legacy_text:
        st.code(legacy_text, language="text")

