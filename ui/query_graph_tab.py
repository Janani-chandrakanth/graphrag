import streamlit as st
import pandas as pd
from graph.langchain_qa import ask_graph, refresh_schema
from graph.query_subgraph import extract_subgraph_from_result
from graph.neo4j_manager import run_cypher_query
from graph.graph_visualizer import render_graph
from graph.version_manager import list_versions, get_version_snapshot, get_current_graph_as_version
from graph.graph_diff import diff_graphs

def render_query_graph_tab():
    st.header("Search and Query")

    # Search Modes
    search_mode = st.radio(
        "Search Mode",
        ["Natural Language (LangChain + Neo4j + Ollama)", "Cypher Query"],
        horizontal=True,
        key="query_tab_mode",
    )

    if search_mode == "Natural Language (LangChain + Neo4j + Ollama)":
        st.subheader("Ask the Graph")
        st.caption(
            "Replaces the old chunk-vector-search + community-summary query phase. Uses LangChain's GraphCypherQAChain: "
            "an LLM (Ollama) writes the actual Cypher for your question (with fuzzy CONTAINS matching, not exact-string), "
            "langchain-neo4j runs it against the live graph, and a second LLM pass turns the real result rows into an answer. "
            "If nothing matches, the question is reformulated and retried once against the graph; if that still finds nothing, "
            "it falls back to semantic search over the source document text."
        )

        if st.button("Refresh graph schema", key="btn_refresh_schema"):
            with st.spinner("Refreshing schema…"):
                try:
                    refresh_schema()
                    st.success("Graph schema refreshed successfully.")
                except Exception as e:
                    st.error(f"Failed to refresh schema: {e}")

        user_query = st.text_input("Ask a question about the requirements graph", placeholder="e.g., What is the flow for user login?", key="nl_query_input")
        
        if user_query and st.button("Ask Graph", type="primary", key="btn_ask_graph"):
            with st.spinner("Searching graph and generating answer…"):
                res = ask_graph(user_query)
                if res.get("error"):
                    st.error(f"Query error: {res['error']}")
                else:
                    st.subheader("Answer")
                    
                    # Highlight answer inside success block matching Screenshot 9
                    st.success(res.get("answer", "(No answer text returned)"))

                    # Warning / Alert for Fallback
                    if res.get("fallback_used") or res.get("source") == "semantic_fallback":
                        reformulated = res.get("reformulated_query") or user_query
                        st.warning(
                            f"The graph had no matching entities, even after retrying with \"{reformulated}\". "
                            "This answer came from semantic search over the source document text instead of the graph — "
                            "treat it as less structured/verified."
                        )

                    # Source text excerpts expander matching Screenshot 9
                    excerpts = res.get("source_excerpts") or res.get("contexts") or []
                    if excerpts:
                        with st.expander(f"Source text excerpts used ({len(excerpts)})"):
                            for exc in excerpts:
                                st.markdown(f"• {exc}")

                    if res.get("cypher"):
                        with st.expander("Cypher Query Used"):
                            st.code(res["cypher"], language="cypher")

                    if res.get("result"):
                        with st.expander(f"Raw Graph Records ({len(res['result'])})"):
                            st.json(res["result"])

                        sub_nodes, sub_rels = extract_subgraph_from_result(res["result"])
                        if sub_nodes:
                            st.subheader("Graph Touched by this Answer")
                            render_graph(sub_nodes, sub_rels, key="subgraph_nl_view")

    elif search_mode == "Cypher Query":
        st.subheader("Neo4j Cypher Query")
        cypher_q = st.text_area("Cypher Query", value=st.session_state.get("cypher_q_val", "MATCH (a:Entity)-[r]->(b:Entity) RETURN a, r, b LIMIT 50"), height=120, key="cypher_text_area")

        if st.button("Execute Cypher", type="primary", key="btn_exec_cypher"):
            with st.spinner("Executing Cypher query against Neo4j…"):
                try:
                    records = run_cypher_query(cypher_q)
                    st.success(f"Returned {len(records)} record(s).")
                    if records:
                        with st.expander("Raw JSON Results"):
                            st.json(records)
                        sub_nodes, sub_rels = extract_subgraph_from_result(records)
                        if sub_nodes:
                            st.subheader("Query Subgraph Visualization")
                            render_graph(sub_nodes, sub_rels, key="subgraph_cypher_view")
                except Exception as cypher_err:
                    st.error(f"Cypher Execution Error: {cypher_err}")

    # What Changed? section placed at the bottom matching Screenshot 9
    st.divider()
    st.subheader("What Changed? (version-aware queries)")
    st.caption(
        "Answers questions like 'What changed in the latest update?' and 'Which requirements were added?' "
        "by diffing the most recent retained version against the current live graph."
    )
    versions_for_query = list_versions()
    if not versions_for_query:
        st.info("No updates applied yet — use the Update Graph tab, then check update impact here.")
    else:
        latest_version = versions_for_query[-1]
        st.caption(f"Comparing against latest snapshot: **{latest_version.get('label')}** ({str(latest_version.get('created_at'))[:19]})")
        if st.button("Show Impact of the Latest Update", type="secondary", key="btn_show_impact"):
            with st.spinner("Calculating update impact diff…"):
                before = get_version_snapshot(latest_version["id"])
                after = get_current_graph_as_version()
                qdiff = diff_graphs(before, after)
                s = qdiff.get("summary", {})
                st.success(
                    f"Impact summary since **{latest_version.get('label')}**: "
                    f"**{s.get('added_nodes', 0)}** node(s) added, "
                    f"**{s.get('modified_nodes', 0)}** modified, "
                    f"**{s.get('added_relationships', 0)}** relationship(s) added, "
                    f"**{len(qdiff.get('changed_workflow', []))}** workflow change(s)."
                )
                if qdiff.get("added_nodes"):
                    st.write("**Added Nodes:** " + ", ".join(n.get("name") or n.get("id") for n in qdiff["added_nodes"]))
                if qdiff.get("changed_workflow"):
                    st.write("**Workflow Changes:**")
                    for cw in qdiff["changed_workflow"]:
                        st.write(f"- **{cw.get('from')}** now leads to **{', '.join(cw.get('after', []))}** (was {', '.join(cw.get('before', []))})")

