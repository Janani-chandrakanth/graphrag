import re
from pydantic import ValidationError

from prompts.kg_extraction_prompt import KG_EXTRACTION_PROMPT
from graph.schemas import Node, Relationship
from graph.ontology_mapper import apply_ontology_mapping
from parser.llm_client import call_ollama
from config import EXTRACTION_MODEL, EXTRACTION_MODEL_URL
from graph.sequence_extractor import extract_workflow_sequence

METADATA_HEADER_KEYWORDS = {
    "version", "revision", "document control", "change description", "author",
    "reviewed by", "approved by", "document history", "version history", "page number"
}

def is_document_metadata_text(text: str) -> tuple[bool, str]:
    """Detect if chunk text is Document Control/Version History table or repeating header/footer artifact."""
    if not text:
        return False, ""
    text_lower = text.lower().strip()
    
    # Check 1: Document Control / Version History table headers
    match_count = sum(1 for kw in METADATA_HEADER_KEYWORDS if kw in text_lower)
    if match_count >= 2:
        return True, "document_control_table"
        
    # Check 2: Header/footer page artifact patterns
    if re.search(r'^(?:forvis\s+mazars\s+\d+|page\s+\d+(?:\s+of\s+\d+)?|document\s+control|version\s+history)$', text_lower):
        return True, "page_header_footer"
        
    if re.search(r'(?:page\s+\d+\s+of\s+\d+|version\s+\d+\.\d+.*author|change\s+description.*version)', text_lower):
        return True, "page_artifact"

    return False, ""



def extract_entities(requirement_text: str, prior_context: str = None,
                      continue_same_item: bool = False) -> dict:
    """
    Args:
        requirement_text: the current chunk's text — extraction happens
                           from this only.
        prior_context: optional. The immediately preceding chunk's text
                        (nothing further back — see below for why a
                        1-step window, not full history). Purpose: this
                        extractor runs per-chunk with zero visibility
                        into any other chunk, which is fine when a
                        chunk's entities are self-contained, but breaks
                        down for sequential/process documents where step
                        N's text only makes sense as a continuation of
                        step N-1 (e.g. "the dropdown will be enabled" ...
                        "select X from the dropdown" — same dropdown,
                        two different chunks).

                        Mechanism, and why v1 of this made things worse:
                        v1 dropped a loose "do NOT extract from this"
                        instruction inline inside the INPUT REQUIREMENTS
                        text itself — a completely different style than
                        the rest of this prompt's rigid, banner-sectioned
                        structure, right after a long "do not invent /
                        do not infer / only explicit" checklist. That
                        combination measurably made extraction MORE
                        conservative (more isolated nodes, more empty
                        chunks), not better connected — confirmed by
                        re-running against a live document, not assumed.

                        v2 (this version) does two things differently:
                        (1) the continuity context gets its own properly
                        formatted section, in the same voice as every
                        other section in this prompt, kept entirely
                        separate from INPUT REQUIREMENTS (which stays
                        byte-identical to the no-context case); (2) the
                        actual ask is narrower and more mechanical:
                        reuse the same snake_case node id for a
                        continuing entity, not "recognize continuity"
                        in the abstract. graph/deduplicator.py already
                        merges nodes across chunks by normalized id —
                        so this leans on a merge mechanism that already
                        works, rather than asking the model to invent a
                        new cross-chunk relationship from scratch.

                        Omit entirely (default) for the original
                        single-chunk behavior — fully backward compatible,
                        INPUT REQUIREMENTS section is untouched either way.

        continue_same_item: True only when this chunk and prior_context
                        are two halves of the SAME requirement/User
                        Story/Test Case item, split apart purely because
                        the combined text exceeded MAX_CHUNK_SIZE (see
                        chunking/chunker.py's create_chunks_from_items —
                        pass chunk["is_split"] plus "same item_id as the
                        previous chunk" from the caller's loop). This is
                        deliberately NOT the default/general case: v1's
                        mistake (see above) was loosening the "do not
                        extract from prior_context" rule broadly, which
                        made extraction worse across the board. This is
                        the one narrow, well-justified exception —
                        threading a sequence back together that would
                        have been ONE LLM call had the character limit
                        not cut it apart — not a general invitation to
                        infer relationships from context.

                        When True, the model is allowed to emit EXACTLY
                        ONE relationship whose "from" is the last node
                        described in prior_context and whose "to" is the
                        first node in this chunk (continuing the same
                        sequence number's step forward, per the SEQUENCE
                        / FLOW EXTRACTION RULES already in the prompt),
                        and nothing else from prior_context. Every other
                        rule (no new nodes from prior_context, no other
                        relationships from it) still applies unchanged.
    """
    if prior_context and continue_same_item:
        continuity_section = (
            "\n=========================================================\n"
            "CONTINUITY CONTEXT — SAME ITEM, SPLIT ONLY BY LENGTH\n"
            "=========================================================\n\n"
            "The following is the immediately preceding part of the SAME "
            "requirement/User Story/Test Case as the text in INPUT "
            "REQUIREMENTS below — it was split into two separate chunks "
            "purely because the combined text was too long for one call, "
            "not because it describes a different item. Treat INPUT "
            "REQUIREMENTS as a direct continuation of this text.\n\n"
            f"{prior_context}\n\n"
            "Rules for using this section:\n"
            "- Do NOT create any node from this section — every node "
            "must still come from INPUT REQUIREMENTS only.\n"
            "- IF an entity in INPUT REQUIREMENTS refers to the same "
            "real-world thing as something named here, reuse the exact "
            "same lowercase snake_case id it would get from this text, "
            "so it is recognized as the same entity.\n"
            "- You MAY create AT MOST ONE relationship whose \"from\" is "
            "the id of the LAST step/action/node described in this "
            "section and whose \"to\" is the id of the FIRST step/action/"
            "node in INPUT REQUIREMENTS, using the appropriate FLOW "
            "relationship type (LEADS_TO/TRIGGERS/CAUSES/...) — ONLY if "
            "this section clearly described a procedure that continues "
            "into INPUT REQUIREMENTS. This is the one exception to the "
            "\"no relationships from context\" rule; do not create any "
            "OTHER relationship referencing a node from this section.\n"
            "- If you add this one continuation relationship, and the "
            "step it points TO has a \"sequence\" attribute, continue the "
            "SAME numbering forward (i.e. if this section's last visible "
            "step was sequence N, INPUT REQUIREMENTS' first step is "
            "N+1) instead of restarting the count at 1.\n"
        )
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        ).replace(
            "=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================",
            continuity_section +
            "\n=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================\n"
        )
    elif prior_context:
        continuity_section = (
            "\n=========================================================\n"
            "CONTINUITY CONTEXT (OPTIONAL — READ, DO NOT EXTRACT FROM)\n"
            "=========================================================\n\n"
            "The following is the immediately preceding requirement in "
            "this document. It is provided ONLY so you can recognize "
            "whether the requirement below (in INPUT REQUIREMENTS) refers "
            "to the same real-world entity as something already named "
            "here — e.g. \"the dropdown\" in both.\n\n"
            f"{prior_context}\n\n"
            "Rules for using this section:\n"
            "- Do NOT create any node or relationship from this section.\n"
            "- Do NOT extract anything that appears only here and not in "
            "INPUT REQUIREMENTS.\n"
            "- IF an entity in INPUT REQUIREMENTS clearly refers to the "
            "same real-world thing as something named here, reuse the "
            "exact same lowercase snake_case id for it that you would "
            "assign based on this text, so it is recognized as the same "
            "entity.\n"
            "- If nothing in INPUT REQUIREMENTS continues anything here, "
            "ignore this section completely.\n"
        )
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        ).replace(
            "=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "=========================================================",
            continuity_section +
            "\n=========================================================\n"
            "INPUT REQUIREMENTS\n"
            "========================================================="
        )
    else:
        prompt = KG_EXTRACTION_PROMPT.replace(
            "{requirements_text}",
            requirement_text
        )

    # Model + host are now config-driven (config.EXTRACTION_MODEL /
    # EXTRACTION_MODEL_URL) instead of hardcoded here — set the
    # EXTRACTION_MODEL / EXTRACTION_MODEL_URL env vars to test a
    # different model (e.g. a coder-tuned one) for entity/id-naming
    # consistency without touching this file. Same deterministic
    # decoding options (temperature 0, seed 42, top_p/top_k 1) and same
    # 8192 num_ctx as before — only the model/host changed.
    result = call_ollama(
        prompt,
        timeout=120,
        num_ctx=8192,
        model=EXTRACTION_MODEL,
        base_url=EXTRACTION_MODEL_URL,
    )
    if result["error"]:
        return {
            "nodes": [], "relationships": [],
            "error": result["error"],
            "raw_output": ""
        }
    output = result["raw"]

    try:
        start = output.find("{")

        if start == -1:
            return {
                "nodes": [], "relationships": [],
                "error": "LLM did not return any JSON block",
                "raw_output": output
            }

        # Use raw_decode instead of find("{")/rfind("}") slicing: rfind("}")
        # grabs the LAST closing brace in the whole output, so if the model
        # appends any trailing text/commentary after the JSON (even a
        # stray brace in an explanation), the slice includes that extra
        # data and json.loads blows up with "Extra data: line N column 1".
        # raw_decode parses exactly one JSON value starting at `start` and
        # tells us where it ended — everything after is just ignored.
        decoder = json.JSONDecoder()
        raw_data, idx = decoder.raw_decode(output, start)
        json_str = output[start:idx]

        # Check if current chunk text is document metadata
        is_meta, meta_reason = is_document_metadata_text(requirement_text)

        # Validate NODES one by one
        valid_nodes   = []
        skipped_nodes = []
        for node in raw_data.get("nodes", []):
            try:
                validated = Node(**node)
                ndict = validated.model_dump(exclude_none=True)
                nname_lower = (ndict.get("name") or "").lower().strip()
                nid_lower = (ndict.get("id") or "").lower().strip()
                
                # Tag metadata nodes if chunk is metadata or node matches metadata patterns
                if is_meta or nname_lower in ("parichita", "forvis mazars 5", "document control", "version history") or re.search(r'^(page\s+\d+|forvis\s+mazars|\d+\.\d+.*author)', nname_lower):
                    if "author" in nname_lower or "parichita" in nname_lower:
                        ndict["type"] = "Author"
                    elif "page" in nname_lower or "mazars" in nname_lower:
                        ndict["type"] = "PageArtifact"
                    else:
                        ndict["type"] = "DocumentMetadata"

                valid_nodes.append(ndict)
            except ValidationError as e:
                skipped_nodes.append({"input": node, "reason": str(e.errors())})


        # Validate RELATIONSHIPS one by one
        valid_relationships   = []
        skipped_relationships = []
        for rel in raw_data.get("relationships", []):
            try:
                validated = Relationship(**rel)
                valid_relationships.append(
                    validated.model_dump(by_alias=True, exclude_none=True)
                )
            except ValidationError:
                skipped_relationships.append(rel)

        # ── Ontology Mapping ──────────────────────────────
        # Fold literal value nodes (Euro, USD, GBP, English, ...)
        # into canonical concept nodes (Currency Preference, ...)
        # before this chunk's data flows into deduplication.
        mapping_result = apply_ontology_mapping(
            valid_nodes, valid_relationships
        )
        # ── Phase 2: Workflow Sequence Extraction (multi-actor aware) ─
        # Runs a second LLM pass to detect sequential workflows in this
        # chunk's text. Returns one entry per actor group that has a real
        # sequential journey (e.g., User chain and Restaurant Owner chain
        # separately). Independent requirement lists (FR-001, Admin tasks,
        # etc.) correctly produce zero sequences, so nothing is touched.
        sequence_data = extract_workflow_sequence(requirement_text, valid_nodes)
        sequences = sequence_data.get("sequences", [])

        if sequences:
            actor_ids = {n["id"] for n in valid_nodes if n.get("type") == "Actor"}

            # Collect all sequence node IDs and per-actor entry points
            all_sequence_node_ids = set()
            actor_to_entry = {}   # actor_id -> flow_entry_id for that sequence

            for seq in sequences:
                entry = seq.get("flow_entry_id")
                edges = seq.get("sequence_edges", [])
                actor_id = seq.get("actor_id")

                # Mark flow entry on the node itself
                if entry:
                    for node in valid_nodes:
                        if node["id"] == entry:
                            if "attributes" not in node or node["attributes"] is None:
                                node["attributes"] = {}
                            node["attributes"]["is_flow_entry"] = "true"
                            break

                # Add sequence edges to the graph
                valid_relationships.extend(edges)

                # Track which nodes are in a sequence chain
                for edge in edges:
                    all_sequence_node_ids.add(edge["from"])
                    all_sequence_node_ids.add(edge["to"])

                if actor_id and entry:
                    actor_to_entry[actor_id] = entry

            # Hub-and-spoke cleanup:
            # For each actor, keep only the edge pointing to THEIR flow entry.
            # Drop any actor→step edges where the step is inside a sequence
            # chain but is NOT that actor's own entry point.
            filtered_relationships = []
            for rel in valid_relationships:
                frm = rel.get("from")
                to  = rel.get("to")

                if frm in actor_ids and to in all_sequence_node_ids:
                    # Which entry point does this actor own?
                    this_actors_entry = actor_to_entry.get(frm)
                    if this_actors_entry is None:
                        # Actor has no detected sequence — find the global
                        # entry that this step belongs to, if any.
                        # Keep only edges to declared entries; drop all other
                        # actor→sequential-step edges.
                        is_any_entry = any(
                            s.get("flow_entry_id") == to for s in sequences
                        )
                        if not is_any_entry:
                            continue   # drop: actor→mid-chain step
                    else:
                        if to != this_actors_entry:
                            continue   # drop: actor→non-entry step in their own chain

                filtered_relationships.append(rel)

            valid_relationships = filtered_relationships

        warnings = []
        if sequence_data.get("error"):
            warnings.append(f"Workflow sequence extraction failed: {sequence_data['error']}")


        if skipped_nodes:
            warnings.append(
                f"{len(skipped_nodes)} node(s) skipped: "
                f"{[s['input'].get('id','?') for s in skipped_nodes]}"
            )
        if skipped_relationships:
            invalid_types = list({r.get('type','?') for r in skipped_relationships})
            warnings.append(
                f"{len(skipped_relationships)} relationship(s) skipped "
                f"with invalid types: {invalid_types}"
            )
        if mapping_result["mappings_applied"]:
            warnings.append(
                f"{len(mapping_result['mappings_applied'])} node(s) "
                f"mapped to ontology concepts"
            )

        return {
            "nodes":                 valid_nodes,
            "relationships":         valid_relationships,
            "raw_output":            json_str,
            "skipped_nodes":         skipped_nodes,
            "skipped_relationships": skipped_relationships,
            "ontology_mappings":     mapping_result["mappings_applied"],
            "error": "; ".join(warnings) if warnings else None
        }

    except json.JSONDecodeError as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"JSON parse error: {str(e)}",
            "raw_output": output
        }
    except Exception as e:
        return {
            "nodes": [], "relationships": [],
            "error": f"Unexpected error: {str(e)}",
            "raw_output": output
        }