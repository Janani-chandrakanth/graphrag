import streamlit as st
from graph.entity_extractor import extract_entities
from graph.graph_builder import build_graph
from graph.deduplicator import deduplicate_graph
from graph.validator import validate_graph
from graph.neo4j_manager import run_cypher_query
from graph.community_detector import (
    run_full_community_detection,
    get_all_communities,
    get_community_nodes,
    get_community_relationships
)
from graph.community_summarizer import summarize_all_communities
from graph.community_hierarchy import build_full_hierarchy
from graph.query_engine import run_graphrag_query
from chunking.chunker import create_chunks
from embeddings.embedding_model import generate_embedding
from parser.parser import extract_text
from vectorstore.chroma_manager import (
    get_collection_count,
    store_chunk,
    search_chunks,
    search_community_summaries,
    get_summaries_collection_count,
    get_summaries_count_by_level,
    get_collection_stats
)
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Requirement GraphRAG", layout="wide")
st.title("Requirement GraphRAG")

NODE_COLORS = {
    "Actor":                    "#4A90D9",
    "Feature":                  "#7ED321",
    "Requirement":              "#F5A623",
    "NonFunctionalRequirement": "#E67E22",
    "BusinessRule":             "#8E44AD",
    "Condition":                "#9B59B6",
    "Action":                   "#E74C3C",
    "Output":                   "#1ABC9C",
    "SystemComponent":          "#F39C12",
    "DataObject":               "#2ECC71",
    "Constraint":               "#C0392B",
    "Event":                    "#3498DB",
    "State":                    "#95A5A6",
    "Attribute":                "#BDC3C7",
}

def get_node_color(node_type):
    return NODE_COLORS.get(node_type, "#BDC3C7")

def render_graph(nodes, relationships):
    if not nodes:
        st.warning("No nodes to visualize.")
        return
    agraph_nodes = [
        Node(
            id=node["id"],
            label=node["name"],
            color=get_node_color(node.get("type", "")),
            size=20,
            title=f"Type: {node.get('type','Unknown')}\nID: {node['id']}"
        )
        for node in nodes
    ]
    agraph_edges = [
        Edge(source=rel["from"], target=rel["to"], label=rel["type"])
        for rel in relationships
    ]
    config = Config(
        width=900, height=600, directed=True,
        physics=True, hierarchical=False,
        nodeHighlightBehavior=True, highlightColor="#F7A7A6"
    )
    agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

def render_cypher_graph(records):
    from neo4j.graph import Node as NeoNode, Relationship as NeoRel
    seen_nodes = {}
    edges = []
    for record in records:
        for value in record.values():
            if isinstance(value, NeoNode):
                node_id   = str(value.id)
                node_name = value.get("name", node_id)
                node_type = value.get("type", "Unknown")
                if node_id not in seen_nodes:
                    seen_nodes[node_id] = Node(
                        id=node_id, label=node_name,
                        color=get_node_color(node_type), size=20,
                        title=f"Type: {node_type}\nName: {node_name}"
                    )
            elif isinstance(value, NeoRel):
                edges.append(Edge(
                    source=str(value.start_node.id),
                    target=str(value.end_node.id),
                    label=value.type
                ))
    if not seen_nodes:
        return False
    config = Config(width=900, height=500, directed=True,
                    physics=True, hierarchical=False,
                    nodeHighlightBehavior=True, highlightColor="#F7A7A6")
    agraph(nodes=list(seen_nodes.values()), edges=edges, config=config)
    return True


# ─── Clear Session ─────────────────────────────────────────
if st.button("Clear Session"):
    st.session_state.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════
# PHASE 1: INDEXING
# ══════════════════════════════════════════════════════════
uploaded_file = st.file_uploader("Upload Requirement Document", type=["pdf","txt"])

if uploaded_file:
    st.success("File uploaded successfully")
    st.write("File Name:", uploaded_file.name)

    if st.button("Build Index"):
        with st.spinner("Processing document..."):
            try:
                text = extract_text(uploaded_file)
                st.session_state["extracted_text"] = text

                chunks = create_chunks(text)

                # Show document type detection
                from chunking.chunker import get_document_info
                doc_info = get_document_info(text)

                if doc_info["detected_types"]:
                    st.success("Document type detected — pattern-based chunking")
                    for item in doc_info["detected_types"]:
                        st.write(
                            f"  {item['label']}: "
                            f"{item['count']} items found"
                        )
                else:
                    st.info(
                        "No standard ID patterns detected — "
                        "using semantic chunking (LangChain fallback). "
                        "Works for free-form documents, BRDs, user stories."
                    )

                st.write(f"Total Chunks Created: {len(chunks)}")
                st.session_state["chunks"] = chunks

                progress_bar = st.progress(0)
                total_chunks = len(chunks)
                all_nodes, all_relationships = [], []

                for idx, chunk in enumerate(chunks):
                    embedding = generate_embedding(chunk)
                    store_chunk(
                        chunk_id=f"{uploaded_file.name}_chunk_{idx}",
                        chunk_text=chunk,
                        embedding=embedding
                    )
                    graph_data = extract_entities(chunk)
                    if graph_data.get("error"):
                        st.warning(f"Chunk {idx+1} — {graph_data['error']}")
                    all_nodes.extend(graph_data.get("nodes", []))
                    all_relationships.extend(graph_data.get("relationships", []))
                    progress_bar.progress((idx + 1) / total_chunks)

                (all_nodes, all_relationships,
                 type_conflicts, direction_conflicts) = deduplicate_graph(
                    all_nodes, all_relationships)

                (all_nodes, all_relationships,
                 removed_nodes, removed_relationships) = validate_graph(
                    all_nodes, all_relationships)

                build_graph({"nodes": all_nodes, "relationships": all_relationships})

                st.session_state.update({
                    "nodes":              all_nodes,
                    "relationships":      all_relationships,
                    "type_conflicts":     type_conflicts,
                    "direction_conflicts":direction_conflicts,
                    "removed_nodes":      removed_nodes,
                    "removed_rels":       removed_relationships,
                    "indexing_success":   True
                })
                progress_bar.empty()

            except Exception as e:
                st.error(f"Error: {str(e)}")

if "extracted_text" in st.session_state:
    st.subheader("Extracted Text Preview")
    st.text_area("Preview", st.session_state["extracted_text"][:3000],
                 height=250, disabled=True)

if st.session_state.get("indexing_success"):
    st.success("Document Indexed Successfully!")
    st.write(f"Total Vectors Stored: {get_collection_count()}")

    st.divider()
    st.header("Graph Quality Report")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Valid Nodes",          len(st.session_state.get("nodes",[])))
    with col2: st.metric("Valid Relationships",  len(st.session_state.get("relationships",[])))
    with col3: st.metric("Removed Nodes",        len(st.session_state.get("removed_nodes",[])))
    with col4: st.metric("Removed Relationships",len(st.session_state.get("removed_rels",[])))

    if st.session_state.get("type_conflicts"):
        with st.expander(f"Type Conflicts Fixed ({len(st.session_state['type_conflicts'])})"):
            st.json(st.session_state["type_conflicts"])
    if st.session_state.get("removed_nodes"):
        with st.expander(f"Removed Nodes ({len(st.session_state['removed_nodes'])})"):
            st.json(st.session_state["removed_nodes"])
    if st.session_state.get("removed_rels"):
        with st.expander(f"Removed Relationships ({len(st.session_state['removed_rels'])})"):
            st.json(st.session_state["removed_rels"])

if "nodes" in st.session_state and "relationships" in st.session_state:
    st.divider()
    st.header("Knowledge Graph Visualization")
    render_graph(st.session_state["nodes"], st.session_state["relationships"])
    st.divider()
    st.header("Entity Extraction - Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nodes")
        st.json(st.session_state["nodes"])
    with col2:
        st.subheader("Relationships")
        st.json(st.session_state["relationships"])

# ══════════════════════════════════════════════════════════
# PHASE 2: COMMUNITY DETECTION AND SUMMARIZATION
# ══════════════════════════════════════════════════════════
st.divider()
st.header("Phase 2 - Community Detection and Summarization")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Step 1: Detect Communities")
    st.caption("Groups tightly connected nodes using Louvain algorithm.")
    if st.button("Run Community Detection"):
        with st.spinner("Running Louvain community detection..."):
            try:
                stats = run_full_community_detection()
                st.session_state["community_stats"] = stats
                if stats.get("success"):
                    st.success(f"Found {stats['total_communities']} communities")
                    st.json(stats)
                else:
                    st.error(stats.get("error", "Detection failed"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

with col_right:
    st.subheader("Step 2: Generate Summaries (HIGH level)")
    st.caption("LLM summarizes each community. Stored in ChromaDB as HIGH level.")
    communities_ready = (
        "community_stats" in st.session_state and
        st.session_state["community_stats"].get("success")
    )
    if not communities_ready:
        st.info("Run Step 1 first.")
    else:
        if st.button("Generate Community Summaries"):
            with st.spinner("Generating HIGH level summaries..."):
                try:
                    results = summarize_all_communities()
                    st.session_state["summary_results"] = results
                    if results["successful"] > 0:
                        st.success(f"Generated {results['successful']} HIGH level summaries")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: st.metric("Total", results["total_communities"])
                    with col_b: st.metric("Successful", results["successful"])
                    with col_c: st.metric("Failed", results["failed"])
                    if results["errors"]:
                        with st.expander("Errors"):
                            st.json(results["errors"])
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Community Explorer
if "community_stats" in st.session_state and st.session_state["community_stats"].get("success"):
    st.divider()
    st.header("Community Explorer")
    communities = get_all_communities()
    if communities:
        selected = st.selectbox(
            "Select Community",
            options=sorted(communities.keys()),
            format_func=lambda cid: (
                f"Community {cid} "
                f"({communities[cid]['node_count']} nodes | "
                f"Types: {', '.join(communities[cid]['primary_types'])})"
            )
        )
        if selected is not None:
            comm_nodes = get_community_nodes(selected)
            comm_rels  = get_community_relationships(selected)
            render_graph(comm_nodes, comm_rels)
            col1, col2 = st.columns(2)
            with col1:
                with st.expander(f"Nodes ({len(comm_nodes)})"):
                    st.json(comm_nodes)
            with col2:
                with st.expander(f"Relationships ({len(comm_rels)})"):
                    st.json(comm_rels)
            if "summary_results" in st.session_state:
                summaries = st.session_state["summary_results"].get("summaries", {})
                if selected in summaries:
                    st.subheader("Community Summary (HIGH level)")
                    st.info(summaries[selected]["summary"])

# ══════════════════════════════════════════════════════════
# PHASE 3: HIERARCHICAL COMMUNITY STRUCTURE
# ══════════════════════════════════════════════════════════
st.divider()
st.header("Phase 3 - Hierarchical Community Structure")
st.caption("Builds ROOT and LOW level summaries from existing HIGH level summaries.")

level_counts = get_summaries_count_by_level()
col1, col2, col3 = st.columns(3)
with col1: st.metric("ROOT Summaries", level_counts["ROOT"],
                     help="One summary for entire document")
with col2: st.metric("LOW Summaries",  level_counts["LOW"],
                     help="Group-level summaries")
with col3: st.metric("HIGH Summaries", level_counts["HIGH"],
                     help="Individual community summaries")

hierarchy_ready = (
    "summary_results" in st.session_state and
    st.session_state["summary_results"].get("successful", 0) > 0
)

if not hierarchy_ready:
    st.info("Complete Phase 2 first.")
else:
    if st.button("Build Hierarchy (ROOT + LOW levels)"):
        with st.spinner("Building ROOT and LOW level summaries with llama3.1..."):
            try:
                summaries = st.session_state["summary_results"]["summaries"]
                hierarchy = build_full_hierarchy(summaries)
                if "error" in hierarchy:
                    st.error(hierarchy["error"])
                else:
                    st.session_state["hierarchy"] = hierarchy
                    stats = hierarchy["stats"]
                    st.success(
                        f"Hierarchy built — ROOT: {stats['total_root']} | "
                        f"LOW: {stats['total_low']} | HIGH: {stats['total_high']}"
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if "hierarchy" in st.session_state:
        hierarchy = st.session_state["hierarchy"]
        tab1, tab2 = st.tabs(["ROOT Level", "LOW Level"])

        with tab1:
            root = hierarchy.get("root", {})
            if root.get("summary"):
                st.subheader("Document-Level Executive Summary")
                st.info(root["summary"])
                st.caption(
                    f"Covers all {len(root.get('communities_included',[]))} communities"
                )
            else:
                st.warning("ROOT summary not built yet.")

        with tab2:
            low = hierarchy.get("low", {})
            if low:
                for group_id, data in sorted(low.items()):
                    with st.expander(
                        f"Group {group_id} — "
                        f"Communities {data.get('communities_included', [])}"
                    ):
                        if data.get("summary"):
                            st.write(data["summary"])
                        elif data.get("error"):
                            st.error(data["error"])
            else:
                st.warning("LOW summaries not built yet.")

# ══════════════════════════════════════════════════════════
# SEARCH AND QUERY
# ══════════════════════════════════════════════════════════
st.divider()
st.header("Search and Query")

search_mode = st.radio(
    "Search Mode",
    ["Requirement Chunks (Basic RAG)", "Community Summaries (GraphRAG)", "Cypher Query"],
    horizontal=True
)

if search_mode == "Requirement Chunks (Basic RAG)":
    st.subheader("Search - Raw Requirement Chunks")
    query = st.text_input("Ask a question about the requirements")
    if query:
        try:
            query_embedding = generate_embedding(query)
            results = search_chunks(query_embedding=query_embedding, n_results=3)
            documents = results["documents"][0]
            distances = results["distances"][0]
            st.write(f"Found {len(documents)} relevant chunks")
            for idx, (document, distance) in enumerate(zip(documents, distances)):
                with st.expander(f"Chunk {idx+1} — similarity: {round(1-distance, 4)}"):
                    st.write(document)
        except Exception as e:
            st.error(f"Search Error: {str(e)}")

elif search_mode == "Community Summaries (GraphRAG)":
    st.subheader("GraphRAG Query")

    # Level selector
    level_counts = get_summaries_count_by_level()
    available_levels = []
    if level_counts["ROOT"] > 0: available_levels.append("ROOT")
    if level_counts["LOW"]  > 0: available_levels.append("LOW")
    if level_counts["HIGH"] > 0: available_levels.append("HIGH")
    available_levels.append("ALL")

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_level = st.selectbox(
            "Select Community Level",
            options=available_levels,
            help=(
                "ROOT = broad questions about the entire system | "
                "LOW = questions about feature groups | "
                "HIGH = specific requirement questions | "
                "ALL = search across all levels"
            )
        )
    with col2:
        n_results = st.slider("Summaries to retrieve", 1, 5, 3)

    # Level explanation
    level_help = {
        "ROOT": "Best for: 'What does this system do?' — uses the single document-level summary",
        "LOW":  "Best for: 'What are the security features?' — uses group-level summaries",
        "HIGH": "Best for: 'What does the dashboard show?' — uses individual community summaries",
        "ALL":  "Searches across all levels — good when you are unsure"
    }
    st.caption(level_help.get(selected_level, ""))

    summary_count = get_summaries_collection_count()
    st.write(f"Total summaries available: {summary_count}")

    if summary_count == 0:
        st.warning("No summaries found. Complete Phase 2 and Phase 3 first.")
    else:
        query = st.text_input("Ask a question about the system")
        if query:
            with st.spinner(f"Running GraphRAG query at {selected_level} level..."):
                try:
                    result = run_graphrag_query(
                        question=query,
                        level=None if selected_level == "ALL" else selected_level,
                        n_results=n_results
                    )
                    if result["success"]:
                        st.subheader("Answer")
                        st.success(result["answer"])
                        st.caption(
                            f"Level used: {result['level_used']} | "
                            f"Communities: {result['communities_used']}"
                        )
                        st.subheader("Supporting Summaries")
                        for item in result["retrieved_summaries"]:
                            with st.expander(
                                f"{item['level']} — Community {item['community_id']} "
                                f"(similarity: {item['similarity']})"
                            ):
                                st.write(item["summary"])
                    else:
                        st.error(f"Query failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Query Error: {str(e)}")

elif search_mode == "Cypher Query":
    st.subheader("Neo4j Cypher Query")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Full Graph"):
            st.session_state["cypher_query"] = \
                "MATCH (a:Entity)-[r]->(b:Entity) RETURN a, r, b LIMIT 100"
            st.rerun()
    with col2:
        if st.button("All Actors"):
            st.session_state["cypher_query"] = \
                "MATCH (n:Entity {type: 'Actor'}) RETURN n"
            st.rerun()
    with col3:
        if st.button("By Community"):
            st.session_state["cypher_query"] = (
                "MATCH (n:Entity) WHERE n.community IS NOT NULL "
                "RETURN n.community AS community, collect(n.name) AS nodes "
                "ORDER BY community"
            )
            st.rerun()
    with col4:
        if st.button("Count by Type"):
            st.session_state["cypher_query"] = (
                "MATCH (n:Entity) RETURN n.type AS Type, "
                "count(n) AS Count ORDER BY Count DESC"
            )
            st.rerun()

    cypher_input = st.text_area(
        "Cypher Query",
        value=st.session_state.get(
            "cypher_query",
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a, r, b LIMIT 25"
        ),
        height=100
    )
    if st.button("Run Query", type="primary"):
        if cypher_input.strip():
            with st.spinner("Querying Neo4j..."):
                try:
                    records = run_cypher_query(cypher_input.strip())
                    if not records:
                        st.warning("No results returned.")
                    else:
                        st.success(f"{len(records)} record(s) returned")
                        st.subheader("Graph View")
                        if not render_cypher_graph(records):
                            st.info("Use RETURN n, r, m format for graph view.")
                        st.subheader("Table View")
                        import pandas as pd
                        table_rows = []
                        for record in records:
                            row = {}
                            for key, value in record.items():
                                from neo4j.graph import Node as NeoNode, Relationship as NeoRel
                                if isinstance(value, NeoNode):
                                    row[key] = f"{value.get('name','?')} [{value.get('type','?')}]"
                                elif isinstance(value, NeoRel):
                                    row[key] = value.type
                                else:
                                    row[key] = str(value)
                            table_rows.append(row)
                        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Query Error: {str(e)}")
        else:
            st.warning("Please enter a Cypher query.")

# ── Storage Info ───────────────────────────────────────────
st.divider()
st.header("Storage Info")
stats = get_collection_stats()
level_counts = get_summaries_count_by_level()
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Chunks",             stats["chunks"]["count"])
with col2: st.metric("HIGH Summaries",     level_counts["HIGH"])
with col3: st.metric("LOW Summaries",      level_counts["LOW"])
with col4: st.metric("ROOT Summaries",     level_counts["ROOT"])
with col5: st.metric("Total in ChromaDB",  stats["summaries"]["count"])