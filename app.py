import os
os.environ["UUID_UTILS_PURE_PYTHON"] = "1"

import streamlit as st
import importlib
import graph.structural_linker
importlib.reload(graph.structural_linker)
from graph.structural_linker import tag_extraction_source
from graph.entity_extractor import extract_entities
from graph.graph_builder import build_graph
from graph.deduplicator import deduplicate_graph
from graph.confidence_scorer import score_graph, LOW_CONFIDENCE_THRESHOLD
from graph.validator import validate_graph
from graph.graph_test_case_generator import _render_test_case_text
from graph.test_case_writer import write_test_cases_to_graph
from graph.failure_trace import trace_failure, format_trace_report
from prompts.prompt_library import list_prompts, get_prompt, save_prompt, validate_hybrid_template
from graph.hybrid_test_case_generator import generate_hybrid_test_cases
from graph.cross_reference_linker import find_cross_references
from graph.graph_eval import evaluate_structure
from graph.neo4j_manager import run_cypher_query, clear_graph
from config import EXTRACTION_MODEL, EXTRACTION_MODEL_URL, CROSS_REF_MODEL, CROSS_REF_MODEL_URL
from graph.community_detector import (
    run_full_community_detection,
    get_all_communities,
    get_community_nodes,
    get_community_relationships
)
from graph.community_summarizer import summarize_all_communities
from graph.community_hierarchy import build_full_hierarchy
from graph.langchain_qa import ask_graph, refresh_schema
from graph.query_subgraph import extract_subgraph_from_result
from chunking.chunker import create_chunks_from_items
from embeddings.embedding_model import generate_embedding
from parser.parser import parse_document
from parser.document_type_detector import detect_document_type
from parser.template_normalizer import normalize_document, normalized_to_markdown
from parser.canonicalizer import (
    canonicalize_document,
    canonical_to_structure,
    AVAILABLE_CANONICALIZER_MODELS,
)
from parser.normalization_validator import validate_normalized_document
from parser.requirement_linker import link_requirements
from vectorstore.chroma_manager import (
    get_collection_count,
    store_chunk,
    get_summaries_collection_count,
    get_summaries_count_by_level,
    get_collection_stats,
    store_user_story_embedding,
    search_user_stories,
    get_user_stories_collection_count,
)
from ingestion.simple_doc_parser import parse_document as simple_parse_document, blocks_to_text
from graph.story_extractor import extract_story_entities
from graph.story_graph_writer import (
    initialize_story_schema, upsert_requirement_doc, upsert_requirement,
    upsert_user_story, upsert_persona, upsert_acceptance_criteria,
    upsert_feature, write_test_case, story_node_counts,
)
from graph.story_hybrid_retriever import retrieve_story_context
from graph.gherkin_test_case_generator import (
    generate_bdd_test_suite, parse_gherkin_to_cases, extract_case_id, now_iso,
)
from graph.story_exports import to_feature_file_bytes, to_excel_bytes
from graph.graph_visualizer import (
    render_graph as _viz_render_graph,
    render_cypher_graph as _viz_render_cypher_graph,
    render_diff_graph as _viz_render_diff_graph,
    get_node_color,
    NODE_COLORS,
)
# ── Incremental Update / Versioning / Comparison (new) ──────
from parser.text_input import wrap_pasted_text
from graph.neo4j_manager import (
    get_all_nodes, get_all_relationships, store_workflow,
    get_stored_documents, get_graph_by_document
)
from graph.incremental_merge import build_merge_plan, apply_merge_plan
from graph.workflow_extractor import extract_workflow, extract_workflow_from_graph
from graph.models import WorkflowModel, Feature, Step
from graph.version_manager import (
    snapshot_current_graph, list_versions, get_version_snapshot,
    get_current_graph_as_version, new_version_id,
)
from graph.graph_diff import diff_graphs
# Import for UI components
from ui.dashboard_tab import render_dashboard_tab
from ui.update_graph_tab import render_update_graph_tab
from ui.compare_tab import render_compare_tab
from ui.query_graph_tab import render_query_graph_tab
from ui.feature_flow_tab import render_feature_flow_tab
from ui.graph_controls import render_graph_toolbar
from ui.graph_evaluation_view import render_graph_quality_section
from ui.test_case_generation_view import render_test_case_generation_section
# Evaluation imports
from graph.evaluation.facts import generate_evaluation_facts
from graph.evaluation.fact_evaluator import evaluate_fact
from graph.evaluation.relationship_evaluator import evaluate_relationships
from graph.evaluation.workflow_evaluator import evaluate_workflow
from graph.evaluation.node_coverage import compute_node_coverage
from graph.evaluation.report import build_evaluation_report
from ui.evaluation_tab import render_evaluation_tab
from ui.test_case_evaluation_tab import render_test_case_evaluation_tab
# ── Caching heavy read operations for Streamlit performance ──
@st.cache_data
def cached_get_all_nodes():
    return get_all_nodes()
@st.cache_data
def cached_get_all_relationships():
    return get_all_relationships()
@st.cache_data
def cached_list_versions():
    return list_versions()
@st.cache_data
def cached_get_version_snapshot(version_id):
    return get_version_snapshot(version_id)
@st.cache_data
def cached_get_collection_stats():
    return get_collection_stats()
@st.cache_data
def cached_get_summaries_count_by_level():
    return get_summaries_count_by_level()
st.set_page_config(page_title="GraphRAG", layout="wide")
st.title("Requirement GraphRAG")
with st.sidebar:
    st.caption("Active models (set via env vars, see config.py)")
    st.code(
        f"Entity extraction : {EXTRACTION_MODEL}\n"
        f"  @ {EXTRACTION_MODEL_URL}\n"
        f"Cross-ref linking : {CROSS_REF_MODEL}\n"
        f"  @ {CROSS_REF_MODEL_URL}",
        language=None,
    )
    st.markdown("---")
    st.subheader("Document Context")
    try:
        stored_docs = ["All Documents"] + get_stored_documents()
    except Exception:
        stored_docs = ["All Documents"]
    
    st.selectbox(
        "Active Document Context",
        options=stored_docs,
        index=0,
        key="active_doc_context",
        on_change=lambda: st.cache_data.clear(),
        help="Select a document context to filter features, workflows, and visual graphs."
    )
def render_graph(nodes, relationships, key="maingraph", flow_only=False):
    _viz_render_graph(nodes, relationships, height=750, key=key, flow_only=flow_only)
def render_cypher_graph(records):
    return _viz_render_cypher_graph(records, height=650, key="cyphergraph")
# ── Workflow helpers ──────────────────────────────────────────────────────────
def load_workflow_from_db() -> WorkflowModel:
    """
    Reconstruct a WorkflowModel from WorkflowStep / Feature :Entity nodes
    that were written by store_workflow() during the build pipeline.
    Uses run_read_query() (supports **params) instead of the bare
    run_cypher_query() which does not accept keyword arguments.
    """
    import json
    from graph.neo4j_manager import run_read_query
    # 1. Load Feature nodes
    feature_records = run_read_query(
        "MATCH (f:Entity {type: 'Feature'}) RETURN f.id AS id, f.name AS name"
    )
    features = []
    for feat_rec in feature_records:
        feat_name = feat_rec.get("name") or ""
        if not feat_name:
            continue
        # 2. Load WorkflowStep nodes that belong to this feature via attributes_json
        step_records = run_read_query(
            """
            MATCH (s:Entity {type: 'WorkflowStep'})
            WHERE s.attributes_json CONTAINS $feat_name
            RETURN s.id AS id, s.name AS name, s.attributes_json AS attr_json
            """,
            feat_name=feat_name,
        )
        # 3. Build Step objects
        steps = []
        for s_rec in step_records:
            attr = {}
            try:
                attr = json.loads(s_rec.get("attr_json") or "{}")
            except Exception:
                pass
            # Double-check the feature_name inside the JSON (CONTAINS is a substring match)
            if attr.get("feature_name") != feat_name:
                continue
            # 4. Follow NEXT edges to find next_steps
            next_records = run_read_query(
                """
                MATCH (s1:Entity {id: $step_id})-[:NEXT]->(s2:Entity {type: 'WorkflowStep'})
                RETURN s2.id AS to
                """,
                step_id=s_rec["id"],
            )
            next_steps = [nr["to"] for nr in next_records if nr.get("to")]
            actors = attr.get("actors") or []
            # actors may be stored as a JSON list or a plain string
            if isinstance(actors, str):
                actors = [actors] if actors else []
            steps.append(Step(
                id=s_rec["id"],
                action=s_rec.get("name") or s_rec["id"],
                actors=actors,
                systems=attr.get("systems") or [],
                data_objects=attr.get("data_objects") or [],
                conditions=attr.get("conditions") or [],
                next_steps=next_steps,
            ))
        # 5. Order steps by following NEXT chain from the HAS_STEP entry point
        entry_records = run_read_query(
            """
            MATCH (f:Entity {type: 'Feature', name: $feat_name})-[:HAS_STEP]->(s:Entity)
            RETURN s.id AS id
            """,
            feat_name=feat_name,
        )
        entry_id = entry_records[0]["id"] if entry_records else None
        step_dict = {s.id: s for s in steps}
        ordered, curr_id, visited = [], entry_id, set()
        while curr_id and curr_id in step_dict and curr_id not in visited:
            visited.add(curr_id)
            curr = step_dict[curr_id]
            ordered.append(curr)
            curr_id = curr.next_steps[0] if curr.next_steps else None
        # Append any steps not reachable from entry (disconnected branches)
        for s in steps:
            if s.id not in visited:
                ordered.append(s)
        features.append(Feature(name=feat_name, steps=ordered))
    return WorkflowModel(features=features)
main_tab_dashboard, main_tab_build, main_tab_update, main_tab_compare, main_tab_query, main_tab_workflow, main_tab_tc_evaluation = st.tabs([
    "Dashboard",
    "Build Graph",
    "Update Graph",
    "Compare",
    "Query Graph",
    "Feature Flow",
    "Test Case Evaluation",
])
with main_tab_build:
    st.caption(
        "The original document-to-graph pipeline -- unchanged. Every new "
        "upload here still creates/extends the SAME graph (Entity nodes "
        "are matched by id via Neo4j's own MERGE), it just doesn't run "
        "Entity Matching, Conflict Detection, or Versioning -- for that, "
        "use the Update Graph tab instead."
    )
    with st.container():
        # ─── Clear Session ─────────────────────────────────────────
        if st.button("Clear Session", key="btn_clear_session_build"):
            st.session_state.clear()
            st.rerun()
        # ─── Clear Graph (Neo4j) ───────────────────────────────────
        if st.button("Clear Graph (Neo4j)", help="Delete all nodes & relationships from Neo4j before re-uploading a document", key="btn_clear_graph_neo4j_build"):
            with st.spinner("Clearing Neo4j graph..."):
                result = clear_graph()
                st.cache_data.clear()
            st.success(f"Graph cleared — {result.get('deleted', 0)} node(s) deleted.")
            st.session_state.clear()
            st.rerun()
        # ══════════════════════════════════════════════════════════
        # PHASE 1: INDEXING
        # ══════════════════════════════════════════════════════════
        uploaded_file = st.file_uploader("Upload Requirement Document", type=["pdf","txt","docx"], key="uploaded_req_doc")
        with st.expander(" Additional sources: Spreadsheets & Flowchart Images (multi-modal)"):
            st.caption(
                "Feeds the same Node/Relationship graph as the pipeline above. No "
                "chunking/canonicalization/cross-ref linking applies here (spreadsheet "
                "rows and diagram steps are already structured) -- but the same "
                "Knowledge Graph Visualization, raw data view, and Test Case Generation "
                "sections below run exactly the same way once this is added."
            )
            mm_file = st.file_uploader(
                "Upload spreadsheet (.xlsx) or flowchart image (.png/.jpg)",
                type=["xlsx", "png", "jpg", "jpeg"], key="multimodal_uploader",
            )
            if mm_file:
                import tempfile
                from pathlib import Path as _Path
                mm_ext = _Path(mm_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=mm_ext) as _tmp:
                    _tmp.write(mm_file.read())
                    mm_path = _tmp.name
                if mm_ext == ".xlsx":
                    from parser.excel_extractor import extract_excel_requirements
                    if st.button("Parse spreadsheet", key="mm_parse_xlsx"):
                        st.session_state["mm_result"] = extract_excel_requirements(mm_path, source=mm_file.name)
                    result = st.session_state.get("mm_result")
                    if result:
                        if result["error"]:
                            st.error(result["error"])
                        else:
                            st.write(f"Parsed {len(result['nodes'])} requirement rows.")
                            if st.button("Add to Knowledge Graph", key="mm_commit_xlsx"):
                                for n in result.get("nodes", []): n["doc_id"] = mm_file.name
                                for r in result.get("relationships", []): r["doc_id"] = mm_file.name
                                build_graph(result)
                                st.cache_data.clear()
                                st.session_state.update({
                                    "nodes": result["nodes"],
                                    "relationships": result["relationships"],
                                })
                                st.success(f"Added {len(result['nodes'])} Requirement nodes to the graph.")
                                st.rerun()
                else:  # .png / .jpg / .jpeg
                    from parser.vision_extractor import extract_flowchart_entities
                    st.image(mm_path, width=400)
                    if st.button("Parse flowchart", key="mm_parse_img"):
                        with st.spinner("Running vision model..."):
                            st.session_state["mm_result"] = extract_flowchart_entities(mm_path, source=mm_file.name)
                    result = st.session_state.get("mm_result")
                    if result:
                        if result["error"]:
                            st.error(result["error"])
                        else:
                            st.write(f"Flow: **{result['flow_name']}** — "
                                     f"{len(result['nodes'])} steps, {len(result['relationships'])} edges.")
                            if st.button("Add to Knowledge Graph", key="mm_commit_img"):
                                for n in result.get("nodes", []): n["doc_id"] = mm_file.name
                                for r in result.get("relationships", []): r["doc_id"] = mm_file.name
                                build_graph(result)
                                st.cache_data.clear()
                                st.session_state.update({
                                    "nodes": result["nodes"],
                                    "relationships": result["relationships"],
                                })
                                st.success(
                                    f"Added {len(result['nodes'])} nodes and "
                                    f"{len(result['relationships'])} relationships to the graph."
                                )
                                st.rerun()
        if uploaded_file:
            st.success("File uploaded successfully")
            st.write("File Name:", uploaded_file.name)
            # Experimental features (canonicalizer, cross-reference linking, multimodal upload) have been removed as per user request.
            if st.button("Build Index", key="btn_build_index_build"):
                with st.spinner("Processing document..."):
                    try:
                        structure = parse_document(uploaded_file)
                        text = structure.to_markdown()
                        st.session_state["extracted_text"] = text
                        st.session_state["raw_text"] = text
                        st.session_state["uploaded_filename"] = uploaded_file.name
                        canonicalized_applied = False
                        # ── Document Type Detector + Template Normalizer ────
                        # Feeds directly into Normalization Validator ->
                        # Requirement Linker -> Structural Chunker below.
                        # so use it directly instead of guessing a type that was
                        # never actually ambiguous.
                        if canonicalized_applied:
                            detection = {
                                "doc_type": "MIXED",
                                "confidence": 1.0,
                                "method": "canonicalizer_fixed_template",
                                "reasoning": (
                                    "Document Type Detection was skipped — the "
                                    "canonicalizer already normalized this document "
                                    "into a fixed template spanning multiple "
                                    "traditional doc types, so MIXED's all-family "
                                    "template is used directly."
                                ),
                            }
                        else:
                            detection = detect_document_type(structure)
                        st.session_state["doc_type_detection"] = detection
                        normalized = normalize_document(structure, detection["doc_type"])
                        st.session_state["normalized_document"] = normalized
                        st.subheader("Document Type Detection")
                        conf_pct = f"{detection['confidence'] * 100:.0f}%"
                        if detection["method"] == "canonicalizer_fixed_template":
                            st.info(f"Skipped — using **MIXED** (all-family) template: {detection['reasoning']}")
                        elif detection["method"] == "regex":
                            st.success(f"Detected **{detection['doc_type']}** — {conf_pct} confidence (regex/keyword)")
                        elif detection["method"] == "llm_fallback":
                            st.info(f"Detected **{detection['doc_type']}** — {conf_pct} confidence (LLM fallback: {detection.get('reasoning', '')})")
                        else:
                            st.warning(f"Could not confidently classify document type ({detection.get('reasoning', 'unknown reason')})")
                        st.subheader("Template Normalization")
                        regex_items = [i for i in normalized["items"] if i["matched_by"].startswith("regex")]
                        llm_items = [i for i in normalized["items"] if i["matched_by"] == "llm_gap_fill"]
                        field_label_items = [i for i in normalized["items"] if i["matched_by"] == "pass_a0_field_label"]
                        st.write(
                            f"**{len(normalized['items'])}** item(s) normalized — "
                            f"{len(field_label_items)} via Title/Description/Acceptance-Criteria grouping, "
                            f"{len(regex_items)} via regex, {len(llm_items)} via LLM gap-fill"
                        )
                        if normalized["unmatched_blocks"]:
                            with st.expander(f"Unmatched blocks ({len(normalized['unmatched_blocks'])})"):
                                st.json(normalized["unmatched_blocks"])
                        if normalized["warnings"]:
                            for w in normalized["warnings"]:
                                st.warning(w)
                        with st.expander("Normalized items"):
                            st.json(normalized["items"])
                        st.subheader("Normalized Document (Template View)")
                        st.caption(
                            "The document as it reads after being fit to the "
                            f"{detection['doc_type']} template — grouped by ID "
                            "family, not the original paragraph order."
                        )
                        with st.container(border=True):
                            st.markdown(normalized_to_markdown(normalized))
                        # ── Normalization Validator ─────────────────────────
                        # normalized["items"] -> valid_items / review_queue.
                        # Nothing is ever silently dropped: failed items are
                        # held in review_queue with their reasons attached.
                        validated = validate_normalized_document(normalized)
                        st.session_state["validated_document"] = validated
                        st.subheader("Normalization Validation")
                        st.write(
                            f"**{validated['summary']['valid']}** valid item(s), "
                            f"**{validated['summary']['needs_review']}** need review "
                            f"out of {validated['summary']['total_items']} total"
                        )
                        for w in validated["document_warnings"]:
                            st.warning(w)
                        if validated["review_queue"]:
                            with st.expander(f"Review queue ({len(validated['review_queue'])})"):
                                st.json(validated["review_queue"])
                        # ── Rule-based Requirement Linker ───────────────────
                        # Feed valid_items PLUS review_queue items into linking
                        # and chunking — review_queue only means "not grounded
                        # in an explicit ID token" or "family unexpected for
                        # this doc type," not "not real content." Excluding
                        # these outright was silently starving the extractor of
                        # most of the document's actual substance (body
                        # sentences, out-of-template business rules, etc.) —
                        # they still show up in the Review Queue expander above
                        # for visibility, they just aren't discarded from the
                        # graph anymore.
                        items_for_graph = validated["valid_items"] + [
                            r["item"] for r in validated["review_queue"]
                        ]
                        linked = link_requirements(items_for_graph)
                        st.session_state["linked_requirements"] = linked
                        st.subheader("Requirement Linking")
                        st.write(
                            f"**{linked['summary']['total_links']}** link(s) found across "
                            f"{linked['summary']['items_with_at_least_one_link']} item(s); "
                            f"**{linked['summary']['broken_references']}** broken reference(s)"
                        )
                        if linked["broken_references"]:
                            with st.expander(f"Broken references ({len(linked['broken_references'])})"):
                                st.json(linked["broken_references"])
                        if linked["links"]:
                            with st.expander(f"Links ({len(linked['links'])})"):
                                st.json(linked["links"])
                        # ── Structural Chunker (item-aware) ─────────────────
                        # One chunk per item (oversized items split into
                        # <id>#1, <id>#2, ...) — replaces the old flat-text
                        # create_chunks(text) call. id/family/linked_ids
                        # metadata now survives into ChromaDB.
                        chunks = create_chunks_from_items(linked["items_with_links"])
                        if not chunks:
                            st.warning(
                                "No valid items to chunk — check the Review Queue above; "
                                "everything normalized from this document needs human review."
                            )
                        st.write(f"Total Chunks Created: {len(chunks)}")
                        st.session_state["chunks"] = chunks
                        progress_bar = st.progress(0)
                        total_chunks = len(chunks)
                        all_nodes, all_relationships = [], []
                        zero_extraction_chunks = []  # diagnostic: chunks that
                                                     # succeeded (no error) but
                                                     # returned zero nodes AND
                                                     # zero relationships — kept
                                                     # with input text + the
                                                     # model's raw response so
                                                     # "thin content" vs "model
                                                     # actually failed silently"
                                                     # can be told apart instead
                                                     # of guessed at.
                        prior_chunk_text = None  # one-step lookback only — see
                                                 # extract_entities()'s docstring
                                                 # for why not full history
                        failed_chunks = []  # chunks where embedding/storage/
                                            # extraction itself raised — kept
                                            # visible with the error, never
                                            # allowed to abort every OTHER
                                            # chunk's already-successful work.
                                            # This is what let one bad chunk
                                            # (an empty-content item producing
                                            # an empty embedding) take down the
                                            # entire graph last time — the
                                            # exception propagated out of this
                                            # loop entirely, so build_graph()
                                            # never ran even though 41/42 chunks
                                            # had already succeeded.
                        for idx, chunk in enumerate(chunks):
                            try:
                                embedding = generate_embedding(chunk["text"])
                                store_chunk(
                                    chunk_id=f"{uploaded_file.name}_{chunk['chunk_id']}",
                                    chunk_text=chunk["text"],
                                    embedding=embedding,
                                    metadata={
                                        "item_id":    chunk["item_id"],
                                        "family":     chunk["family"],
                                        "linked_ids": chunk["linked_ids"],
                                        "is_split":   chunk["is_split"],
                                    }
                                )
                                continue_same_item = (
                                    chunk["is_split"] and idx > 0
                                    and chunks[idx - 1]["item_id"] == chunk["item_id"]
                                )
                                graph_data = extract_entities(
                                    chunk["text"], prior_context=prior_chunk_text,
                                    continue_same_item=continue_same_item,
                                )
                                if graph_data.get("error"):
                                    st.warning(f"Chunk {idx+1} ({chunk['chunk_id']}) — {graph_data['error']}")
                                if not graph_data.get("nodes") and not graph_data.get("relationships"):
                                    zero_extraction_chunks.append({
                                        "item_id":         chunk["item_id"],
                                        "chunk_id":        chunk["chunk_id"],
                                        "input_text":      chunk["text"],
                                        "raw_model_output": graph_data.get("raw_output", ""),
                                        "reported_error":   graph_data.get("error"),
                                    })
                                graph_data = tag_extraction_source(graph_data, chunk["item_id"], doc_id=uploaded_file.name)
                                all_nodes.extend(graph_data.get("nodes", []))
                                all_relationships.extend(graph_data.get("relationships", []))
                                prior_chunk_text = chunk["text"]
                            except Exception as chunk_err:
                                failed_chunks.append({
                                    "item_id":  chunk["item_id"],
                                    "chunk_id": chunk["chunk_id"],
                                    "text_preview": chunk["text"][:200],
                                    "error": str(chunk_err),
                                })
                                st.warning(
                                    f"Chunk {idx+1} ({chunk['chunk_id']}) failed and was "
                                    f"skipped — {chunk_err}"
                                )
                                # prior_chunk_text intentionally NOT updated on
                                # failure — a failed chunk's text shouldn't be
                                # offered as "the previous chunk" continuity
                                # context to the next one.
                            if total_chunks:
                                progress_bar.progress((idx + 1) / total_chunks)
                        st.session_state["zero_extraction_chunks"] = zero_extraction_chunks
                        st.session_state["failed_chunks"] = failed_chunks
                        if failed_chunks:
                            st.error(
                                f"{len(failed_chunks)} chunk(s) failed during indexing and "
                                f"were skipped (see warnings above) — the rest of the "
                                f"document still indexed normally."
                            )
                        # ── Structural Linker ────────────────────────────────
                        # No backbone nodes anymore — see graph/structural_linker.py's
                        # module docstring for why. tag_extraction_source() above
                        # already gives every node/relationship a "source" field
                        # pointing back to its requirement item, so traceability
                        # doesn't need a graph node/edge for it. Functional flow
                        # is now visible directly via the LLM's own TRIGGERS/
                        # LEADS_TO/USES/CAUSES/... edges, with nothing competing
                        # for attention.
                        st.session_state["linked_requirements_summary"] = {
                            "cross_reference_links": linked["summary"]["total_links"],
                            "broken_references": linked["summary"]["broken_references"],
                        }
                        (all_nodes, all_relationships,
                         type_conflicts, direction_conflicts) = deduplicate_graph(
                            all_nodes, all_relationships)
                        # Cross-requirement linking is disabled (feature removed from UI)
                        # ── Confidence Scorer ────────────────────────────────
                        # Annotates every deduplicated node/relationship with a
                        # "confidence" score + reasons, using occurrence counts
                        # (deduplicate_graph's _occurrence_count), dedup conflict
                        # signals, and Requirement Linker corroboration. Nothing
                        # is removed here — low-confidence items are flagged
                        # (low_confidence_nodes/relationships) for visibility,
                        # not dropped; Graph Validator (next) still decides
                        # schema validity independently.
                        scored = score_graph(
                            all_nodes, all_relationships,
                            type_conflicts, direction_conflicts,
                            linked.get("items_with_links", [])
                        )
                        st.session_state["confidence_summary"] = scored["summary"]
                        st.subheader("Confidence Scoring")
                        st.write(
                            f"Avg node confidence: **{scored['summary']['avg_node_confidence']}** — "
                            f"Avg relationship confidence: **{scored['summary']['avg_relationship_confidence']}**"
                        )
                        if scored["low_confidence_nodes"] or scored["low_confidence_relationships"]:
                            st.warning(
                                f"{len(scored['low_confidence_nodes'])} low-confidence node(s), "
                                f"{len(scored['low_confidence_relationships'])} low-confidence relationship(s) "
                                f"(< {LOW_CONFIDENCE_THRESHOLD}) — flagged, not removed."
                            )
                            with st.expander("Low-confidence nodes"):
                                st.json(scored["low_confidence_nodes"])
                            with st.expander("Low-confidence relationships"):
                                st.json(scored["low_confidence_relationships"])
                        (all_nodes, all_relationships,
                         review_nodes, review_relationships) = validate_graph(
                            all_nodes, all_relationships)
                        structure_report = evaluate_structure(
                            all_nodes, all_relationships, linked["items_with_links"]
                        )
                        build_graph({"nodes": all_nodes, "relationships": all_relationships})
                        st.cache_data.clear()
                        # ── Workflow layer (additive, non-breaking) ──────────
                        # Build WorkflowModel from already-extracted nodes/rels
                        # (zero new LLM calls — pure in-memory graph walk) and
                        # persist it to Neo4j so the Workflow tab can read it.
                        with st.spinner("Building workflow layer..."):
                            try:
                                wf = extract_workflow_from_graph(all_nodes, all_relationships)
                                if wf.features:
                                    store_workflow(wf)
                                    st.session_state["workflow"] = wf
                                    st.success(
                                        f"Workflow layer stored — "
                                        f"**{len(wf.features)}** feature(s) detected. "
                                        f"Visit the **Workflow** tab to explore."
                                    )
                                else:
                                    st.info(
                                        "No sequential workflow features detected in this "
                                        "document. Workflow tab will show an empty state."
                                    )
                            except Exception as _wf_err:
                                st.warning(
                                    f"Workflow layer extraction skipped: {_wf_err}\n"
                                    "The main graph is unaffected."
                                )
                        st.session_state.update({
                            "nodes":              all_nodes,
                            "relationships":      all_relationships,
                            "type_conflicts":     type_conflicts,
                            "direction_conflicts":direction_conflicts,
                            "removed_nodes":      review_nodes,
                            "removed_rels":       review_relationships,
                            "structure_report":   structure_report,
                            "indexing_success":   True
                        })
                        # Automatically run community pipeline
                        with st.spinner("Analyzing graph structure (running community detection)..."):
                            try:
                                comm_stats = run_full_community_detection()
                                st.session_state["community_stats"] = comm_stats
                            except Exception as comm_err:
                                st.warning(f"Community detection failed: {comm_err}")
                        if st.session_state.get("community_stats", {}).get("success"):
                            with st.spinner("Generating community summaries..."):
                                try:
                                    summary_results = summarize_all_communities()
                                    st.session_state["summary_results"] = summary_results
                                    if summary_results.get("successful", 0) > 0:
                                        with st.spinner("Building hierarchical summary structure..."):
                                            hierarchy = build_full_hierarchy(summary_results["summaries"])
                                            st.session_state["hierarchy"] = hierarchy
                                except Exception as summ_err:
                                    st.warning(f"Community summarization/hierarchy build failed: {summ_err}")

                        progress_bar.empty()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()
        render_graph_quality_section()
        st.divider()
        render_test_case_generation_section()

with main_tab_dashboard:
    render_dashboard_tab()

with main_tab_update:
    render_update_graph_tab()

with main_tab_compare:
    render_compare_tab()

with main_tab_query:
    render_query_graph_tab()

with main_tab_workflow:
    render_feature_flow_tab()

with main_tab_tc_evaluation:
    render_test_case_evaluation_tab()
