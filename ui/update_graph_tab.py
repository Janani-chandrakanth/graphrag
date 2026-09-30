import streamlit as st
import pandas as pd
from parser.models import wrap_pasted_text
from parser.parser import parse_document
from chunking.chunker import create_chunks_from_items
from graph.entity_extractor import extract_entities
from graph.schemas import tag_extraction_source
from graph.deduplicator import deduplicate_graph
from graph.validator import validate_graph
from graph.neo4j_manager import get_all_nodes, get_all_relationships
from graph.incremental_merge import build_merge_plan, apply_merge_plan
from graph.version_manager import snapshot_current_graph, new_version_id
from graph.community_engine import run_full_community_detection
from graph.community_engine import summarize_all_communities
from graph.community_engine import build_full_hierarchy
from graph.langchain_qa import refresh_schema
from parser.document_type_detector import detect_document_type
from parser.normalization import normalize_document, validate_normalized_document
from parser.requirement_linker import link_requirements

def render_update_graph_tab():
    st.header("Update Graph (Incremental Ingestion)")
    st.caption(
        "Upload or paste an update — a revised User Story, Test Case, meeting notes, a new BRD/SRS section, anything — "
        "and it goes through the exact same Parser -> Chunking -> Embeddings -> Entity Extraction -> Validation -> Deduplication "
        "pipeline as Build Graph. The only new stage is Graph Merge: the existing graph is fetched, new nodes are matched onto "
        "existing ones (by id, then semantic similarity, then LLM verification), relationship conflicts are flagged for your decision "
        "instead of silently appended, and the graph EVOLVES — it never restarts. The pre-update state is snapshotted as a retained version first, so nothing is ever lost."
    )

    try:
        existing_nodes = get_all_nodes()
        existing_rels = get_all_relationships()
    except Exception as e:
        st.error(f"Failed to query Neo4j: {e}")
        return

    if not existing_nodes:
        st.warning(
            "The graph is currently empty. Please use the Build Graph tab to build the initial graph before performing incremental updates."
        )

    st.write("")
    st.markdown("**Update source**")
    input_mode = st.radio("Update source", ["Upload file", "Paste text"], horizontal=True, key="upd_input_mode_radio", label_visibility="collapsed")
    
    upd_file = None
    if input_mode == "Upload file":
        upd_file = st.file_uploader(
            "Upload the update (BRD, SRS, User Story, Test Case, meeting notes, PDF, DOCX, TXT...)",
            type=["pdf", "txt", "docx"],
            key="upd_uploader",
        )
    else:
        pasted = st.text_area("Paste update content", height=200, key="upd_pasted_text", placeholder="e.g., 'After payment completion, trigger order notification to user.'")
        pasted_name = st.text_input("Label this update (optional)", key="upd_pasted_name", placeholder="update_v2.txt")
        if pasted and pasted.strip():
            upd_file = wrap_pasted_text(pasted, pasted_name or "pasted_update.txt")

    use_llm_verification = st.checkbox("Use LLM verification for ambiguous entity matches", value=True, key="upd_use_llm")

    # Initial merge plan build button
    if upd_file and st.button("Parse & Generate Merge Plan", type="primary", key="btn_build_merge_plan"):
        with st.spinner("Parsing document and extracting update graph…"):
            try:
                structure = parse_document(upd_file)
                detection = detect_document_type(structure)
                normalized = normalize_document(structure, detection["doc_type"])
                validated = validate_normalized_document(normalized)
                items_for_graph = validated["valid_items"] + [r["item"] for r in validated["review_queue"]]
                linked = link_requirements(items_for_graph)
                chunks = create_chunks_from_items(linked["items_with_links"])
                
                # Store linked requirements in session state
                st.session_state["linked_requirements"] = linked
                st.session_state["chunks"] = chunks
                
                all_nodes = []
                all_rels = []
                prior_text = ""
                for chunk in chunks:
                    gdata = extract_entities(chunk["text"], prior_context=prior_text)
                    gdata = tag_extraction_source(gdata, chunk.get("item_id", "upd"))
                    all_nodes.extend(gdata.get("nodes", []))
                    all_rels.extend(gdata.get("relationships", []))
                    prior_text = chunk["text"]

                all_nodes, all_rels, _, _ = deduplicate_graph(all_nodes, all_rels)
                all_nodes, all_rels, _, _ = validate_graph(all_nodes, all_rels)

                st.session_state["upd_source_filename"] = upd_file.name
                st.session_state["upd_new_nodes"] = all_nodes
                st.session_state["upd_new_rels"] = all_rels
                
                # Reset resolutions for a fresh update file
                st.session_state["upd_resolutions"] = {}

                plan = build_merge_plan(
                    all_nodes, all_rels, existing_nodes, existing_rels,
                    use_llm_verification=use_llm_verification,
                    resolutions={}
                )
                st.session_state["upd_merge_plan"] = plan
                st.success(f"Parsed update: {len(all_nodes)} node(s), {len(all_rels)} relationship(s) extracted.")
            except Exception as parse_err:
                st.error(f"Error generating merge plan: {parse_err}")

    # Render plan and metrics if a plan is available
    plan = st.session_state.get("upd_merge_plan")
    if plan:
        st.divider()
        st.subheader("Graph Merge")
        
        stats = plan.get("stats", {})
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched existing", stats.get("matched_existing", 0))
        m2.metric("Genuinely new", stats.get("genuinely_new", 0))
        m3.metric("Conflicts", stats.get("conflicts_detected", 0))
        m4.metric("Relationships to add", stats.get("relationships_added", 0))

        st.write("")

        # 1. Entity Matching
        st.markdown("### 1. Entity Matching")
        st.caption("These new nodes were recognized as EXISTING nodes and will be reused, not duplicated:")
        
        match_report = plan.get("match_report", [])
        if match_report:
            df_match = pd.DataFrame(match_report)
            st.dataframe(df_match, use_container_width=True)
        else:
            st.info("No entity matches detected.")

        unmatched = plan.get("unmatched_new_nodes", [])
        with st.expander(f"Genuinely new nodes ({len(unmatched)})"):
            if unmatched:
                st.dataframe(pd.DataFrame(unmatched), use_container_width=True)
            else:
                st.caption("No new nodes.")

        st.write("")

        # 2. Conflict Detection
        st.markdown("### 2. Conflict Detection")
        conflicts = plan.get("conflicts", [])
        resolutions = st.session_state.setdefault("upd_resolutions", {})
        
        if conflicts:
            st.warning(f"{len(conflicts)} conflict(s) detected — an existing node's downstream flow would change. Choose how to resolve each one below, then recompute the plan.")
            
            resolutions_changed = False
            for conf in conflicts:
                st.markdown(f"**Entity**: `{conf.get('from')}` ({conf.get('type')})")
                st.caption(conf.get("message"))
                
                # Checkbox / Radio choices
                res_key = (conf["from"], conf["type"], conf["new_to"])
                current_res = resolutions.get(res_key, "extend")
                
                selected = st.radio(
                    f"Resolution for {conf['from']} -> {conf['new_to']} ({conf['type']})",
                    options=["extend", "replace", "skip"],
                    format_func=lambda x: {"extend": "Extend (keep both)", "replace": "Replace (old -> new)", "skip": "Skip (discard new)"}[x],
                    index=["extend", "replace", "skip"].index(current_res),
                    key=f"res_opt_{conf['from']}_{conf['type']}_{conf['new_to']}"
                )
                
                if selected != current_res:
                    resolutions[res_key] = selected
                    resolutions_changed = True
                    
            if st.button("Recompute Plan", key="btn_recompute_plan"):
                with st.spinner("Recomputing merge plan with resolutions…"):
                    try:
                        new_plan = build_merge_plan(
                            st.session_state["upd_new_nodes"],
                            st.session_state["upd_new_rels"],
                            existing_nodes,
                            existing_rels,
                            use_llm_verification=use_llm_verification,
                            resolutions=resolutions
                        )
                        st.session_state["upd_merge_plan"] = new_plan
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Recompute error: {ex}")
        else:
            st.success("No downstream flow conflicts detected.")

        st.write("")

        # 3. Relationship Validation
        st.markdown("### 3. Relationship Validation")
        vr = plan.get("validation_report", {})
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Isolated Nodes", len(vr.get("isolated_nodes", [])))
        v2.metric("Duplicate Edges", len(vr.get("duplicate_edges", [])))
        v3.metric("Invalid Direction", len(vr.get("invalid_direction", [])))
        v4.metric("Cycles", len(vr.get("cycles", [])))

        # 4. Workflow Preservation
        st.markdown("### 4. Workflow Preservation")
        st.info(f"Relationships to add: {stats.get('relationships_added', 0)} | Relationships to remove: {stats.get('relationships_removed', 0)}")

        st.divider()
        version_label = st.text_input("Version label for this update", value=st.session_state.get("upd_source_filename", "update_v1"), key="upd_version_label_input")

        if st.button("Apply Update to Graph", type="primary", key="btn_apply_update_graph"):
            with st.spinner("Snapshotting version and applying merge plan to Neo4j…"):
                try:
                    pre_version = snapshot_current_graph(
                        label=f"Before: {version_label}",
                        description="Snapshot taken automatically before applying update.",
                        source_filename=st.session_state.get("upd_source_filename"),
                    )
                    batch_id = new_version_id()
                    apply_merge_plan(plan, batch_id=batch_id)
                    try:
                        refresh_schema()
                    except Exception:
                        pass
                    st.success(f"Graph updated successfully! Retained pre-update snapshot: {pre_version.get('label')}")
                    st.session_state.pop("upd_merge_plan", None)
                    st.session_state.pop("upd_resolutions", None)
                    st.rerun()
                except Exception as apply_err:
                    st.error(f"Failed to apply update: {apply_err}")

