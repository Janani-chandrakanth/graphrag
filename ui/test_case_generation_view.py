import streamlit as st
from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.hybrid_test_case_generator import generate_hybrid_test_cases
from graph.graph_test_case_generator import generate_test_cases_from_graph, _render_test_case_text
from graph.gherkin_test_case_generator import generate_bdd_test_suite
from graph.test_case_writer import write_test_cases_to_graph

def render_test_case_generation_section():
    st.subheader("Test Case Generation")
    st.caption("Generate grounded Positive, Negative, and Edge test cases directly from the Knowledge Graph and Requirement workflows.")

    try:
        all_graph_nodes = get_all_nodes()
        rels = get_all_relationships()
    except Exception as e:
        st.error(f"Error querying Neo4j for test case generation: {e}")
        return

    if not all_graph_nodes:
        st.info("The graph is empty. Please index a document in the Build Graph tab first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Generation Strategy**")
        st.markdown("Hybrid (LLM + Vector + Graph)")
        engine_mode = "Hybrid (LLM + Vector + Graph)"
    with col2:
        st.write("")
        st.write("")
        gen_btn = st.button("Generate Test Cases", type="primary", use_container_width=True, key="btn_run_tc_generation")

    if gen_btn:
        with st.spinner("Generating grounded test cases from knowledge graph…"):
            try:
                # Auto-recover items_with_links if session state was cleared
                linked = st.session_state.get("linked_requirements") or {}
                items_with_links = linked.get("items_with_links", [])
                links = linked.get("links", [])
                if not items_with_links:
                    sources = sorted(list({n.get("source") for n in all_graph_nodes if n.get("source") and n.get("source") != "workflow_extractor"}))
                    items_with_links = [{"id": s, "family": None} for s in sources]

                chunks = st.session_state.get("chunks", [])
                if not chunks:
                    try:
                        from vectorstore.chroma_manager import chunks_collection
                        col_data = chunks_collection.get()
                        docs = col_data.get("documents") or []
                        metas = col_data.get("metadatas") or []
                        ids = col_data.get("ids") or []
                        chunks = [
                            {
                                "id": ids[k] if k < len(ids) else f"chunk_{k}",
                                "text": docs[k],
                                "item_id": (metas[k] or {}).get("item_id", "") if k < len(metas) else "",
                                "family": (metas[k] or {}).get("family", "") if k < len(metas) else ""
                            }
                            for k in range(len(docs))
                        ]
                    except Exception:
                        chunks = []

                if engine_mode == "Hybrid (LLM + Vector + Graph)":
                    tc_results = generate_hybrid_test_cases(
                        nodes=all_graph_nodes, relationships=rels,
                        items_with_links=items_with_links,
                        links=links, chunks=chunks,
                    )
                elif engine_mode == "Graph-Only Deterministic":
                    tc_results = generate_test_cases_from_graph(all_graph_nodes, rels, items_with_links, links)
                else: # BDD Gherkin
                    tc_results = generate_bdd_test_suite(all_graph_nodes, rels)

                st.session_state["generated_test_cases_result"] = tc_results
                # Also store under the key the evaluation tab expects
                test_cases_list = tc_results.get("test_cases", [])
                if test_cases_list:
                    st.session_state["generated_test_cases"] = test_cases_list
                st.success(f"Test case generation complete! {len(tc_results.get('test_cases', []))} test case(s) generated.")
            except Exception as gen_err:
                st.error(f"Test case generation failed: {gen_err}")

    res = st.session_state.get("generated_test_cases_result")
    if res:
        st.divider()
        st.subheader("Generated Test Suite Results")

        tcs = res.get("test_cases", [])
        if tcs:
            pos_cnt = sum(1 for c in tcs if c.get("type") == "Positive" or c.get("category") == "Positive")
            neg_cnt = sum(1 for c in tcs if c.get("type") == "Negative" or c.get("category") == "Negative")
            edge_cnt = sum(1 for c in tcs if c.get("type") == "Edge" or c.get("category") == "Edge")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Test Cases", len(tcs))
            m2.metric("Positive Scenarios", pos_cnt)
            m3.metric("Negative Scenarios", neg_cnt)
            m4.metric("Edge Scenarios", edge_cnt)

            for i, tc in enumerate(tcs):
                # Guard: skip non-dict items (e.g. raw strings that slipped in)
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get('tc_id') or tc.get('id', f'TC-{i+1}')
                req_id = tc.get('req_id') or tc.get('req_ids') or 'N/A'
                title = tc.get('title', 'Scenario')
                feature = tc.get('feature', 'General')
                description = tc.get('description') or f"Verifies functionality of {feature} end-to-end."
                tc_type = tc.get('type') or tc.get('category', 'Positive')
                priority = tc.get('priority', 'High')
                actor = tc.get('actor', 'End User')
                precondition = tc.get('preconditions') or tc.get('precondition') or 'None'

                steps = tc.get("steps", [])
                steps_str = ""
                if isinstance(steps, list):
                    for idx, s in enumerate(steps):
                        steps_str += f"  {idx+1}. {s}\n"
                else:
                    steps_str = f"  1. {steps}\n"

                expected = tc.get("expected_result") or tc.get("expected_results") or ""
                tc_graph_nodes = tc.get("graph_nodes") or tc.get("entities_used") or []
                nodes_str = " → ".join(str(n) for n in tc_graph_nodes)

                text = f"""TEST CASE
=========
TC-ID           : {tc_id}
REQ-ID          : {req_id}
TITLE           : {title}
DESCRIPTION     : {description}
TYPE            : {tc_type}
PRIORITY        : {priority}
ACTOR           : {actor}
FEATURE         : {feature}
PRECONDITION    : {precondition}
STEPS
-----
{steps_str.rstrip()}
EXPECTED RESULT : {expected}
GRAPH NODES     : {nodes_str}
"""
                st.code(text.strip(), language="text")

            if st.button("Persist Test Cases to Graph", type="secondary", key="btn_write_tc_neo4j"):
                with st.spinner("Writing test cases as nodes in Neo4j…"):
                    try:
                        # Filter to only valid dict test cases before writing
                        dict_tcs = [tc for tc in tcs if isinstance(tc, dict)]
                        written = write_test_cases_to_graph(dict_tcs, all_graph_nodes)
                        tc_count = written.get("test_case_nodes_written", 0)
                        edge_count = written.get("validates_edges_written", 0) + written.get("verifies_edges_written", 0)
                        st.success(f"Persisted {tc_count} test case node(s) and {edge_count} traceability edge(s) to Neo4j.")
                        unresolved = written.get("test_cases_with_no_resolved_edges", [])
                        if unresolved:
                            st.warning(f"{len(unresolved)} test case(s) had no resolvable graph entity edges: {', '.join(unresolved)}")
                    except Exception as write_err:
                        st.error(f"Failed to persist test cases: {write_err}")

        elif res.get("gherkin_suite"):
            st.code(res["gherkin_suite"], language="gherkin")
