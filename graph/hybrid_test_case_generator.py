import hashlib
import json
from collections import defaultdict

from parser.llm_client import call_ollama, extract_json_block
from prompts.prompt_library import get_prompt as _get_library_prompt
from graph.graph_test_case_generator import (
    generate_test_cases_from_graph,
    _render_test_case_text,
    validate_and_align_precondition,
)
from graph.neo4j_manager import get_neighbor_nodes_db, insert_graph, get_all_nodes
from vectorstore.chroma_manager import generate_embedding, search_chunks

_SOURCE_TEXT_MAX_CHARS = 4000
_RELATED_CONTEXT_MAX_CHARS = 2500
_MAX_GRAPH_NEIGHBORS = 30
_VECTOR_TOP_K = 6          # over-fetched; own-series chunks filtered out after
_VECTOR_MAX_CHUNKS_KEPT = 4
_CATALOG_MAX_CHARS = 3000

_HYBRID_PROMPT = """You are a senior QA engineer producing a COMPLETE, high-coverage test-case set for one requirement flow. You have THREE sources of grounding — use all of them together, they are not interchangeable:

REQUIREMENT SERIES: {req_ids}

1) ORIGINAL REQUIREMENT TEXT (this series' own source text — the primary account of what happens, in what order):
{source_text}

2) RELATED CONTEXT FROM ELSEWHERE IN THE KNOWLEDGE BASE (retrieved by semantic similarity — may describe similar flows, shared screens, or referenced rules; NARRATIVE CONTEXT ONLY, to help you phrase realistic and complete scenarios — do NOT copy entities from here into "entities_used" unless that same entity also appears in the catalog below):
{related_context}

3) GRAPH-DERIVED DRAFT (deterministic walk of the extracted graph for this series — steps may be thin or oddly ordered; reorder/merge/rephrase freely using sources 1 and 2, but every step must stay grounded in the catalog below):
  Actor: {actor}
  Feature: {feature}
  Precondition: {precondition}
  Draft Steps: {steps}
  Draft Expected Result: {expected_result}

ENTITIES YOU MAY REFERENCE (id | type | name) — you may ONLY use entities from this list in "entities_used". This list includes BOTH this series' own extracted entities AND entities one hop away in the live graph (shared screens, cross-referenced rules, shared data objects, etc. — marked accordingly). Do NOT invent, assume, or reference anything not in this list. No external systems, no generic infrastructure ("the database", "the network", "the API") unless it is literally in the list below:
{catalog}

Produce a COMPLETE, non-redundant test-case set for this flow — do not stop at a fixed quota. Cover every DISTINCT scenario the sources above actually support, drawing from these categories as far as the grounding allows (skip a category entirely if nothing above genuinely supports it — never pad with an invented scenario just to fill a category):
  - Positive: the real, correctly-ordered happy-path journey, phrased as a human would actually perform it. Include a second Positive case ONLY if the catalog/context genuinely supports a distinct valid path (e.g. an alternate valid input, an alternate route to the same outcome) — do not invent one.
  - Negative: realistic failures of the entities above (missing, invalid, unauthorized, rejected, mismatched).
  - Edge: boundary/limit conditions of the entities above (empty, minimum, maximum, exceeds limit, unexpected state, timeout, expired) — use type "Edge" for these, distinct from "Negative", so boundary conditions aren't hidden inside error-path cases.

Every case must be grounded ENTIRELY in the entities listed in the catalog — never in something only mentioned in the related-context section. Every entity name in "entities_used" must be copied EXACTLY as it appears in the catalog (its id or its name, either is fine).

Return ONLY a JSON array, no other text, no markdown fences, no comments. Order: Positive case(s) first, then Negative, then Edge. Each entry:
{{"title": "<short scenario title>", "type": "Positive" or "Negative" or "Edge", "steps": ["<step 1>", "<step 2>", "..."], "expected_result": "<expected outcome>", "entities_used": ["<exact id or name from catalog>", "..."]}}
"""

_NEGATIVE_PROMPT = """You are a QA engineer identifying NEGATIVE and EDGE-CASE test scenarios for one part of a system, based ONLY on entities already extracted from its requirements.

REQUIREMENT SERIES: {req_ids}

POSITIVE TEST CASE ALREADY BUILT FOR THIS SERIES:
  Actor: {actor}
  Feature: {feature}
  Steps: {steps}
  Expected Result: {expected_result}

ENTITIES YOU MAY REFERENCE (id | type | name) — you may ONLY use entities from this list. Do NOT invent, assume, or reference anything not in this list. No external systems, no generic infrastructure ("the database", "the network", "the API") unless it is literally in the list below:
{catalog}

Propose 1 to 3 realistic NEGATIVE or EDGE-CASE scenarios that stress, negate, or find the boundary of the entities above (e.g. an entity's value is missing, invalid, unmatched, empty, out of range, or in an unexpected state) — grounded ENTIRELY in the entities listed. Every entity name in "entities_used" must be copied EXACTLY as it appears in the catalog above. If you cannot construct a grounded negative scenario from this list, return an empty array — do not fabricate one just to have something to say.

Return ONLY a JSON array, no other text, no markdown fences. Each entry:
{{"title": "<short scenario title>", "steps": ["<step 1>", "<step 2>", "..."], "expected_result": "<expected outcome>", "entities_used": ["<exact name from catalog>", "..."]}}
"""


def _resolve_prompt_template(prompt_name: str) -> str:
    try:
        return _get_library_prompt(prompt_name)["template"]
    except KeyError:
        return _HYBRID_PROMPT


def _resolve_negative_prompt_template(prompt_name: str) -> str:
    try:
        return _get_library_prompt(prompt_name)["template"]
    except KeyError:
        return _NEGATIVE_PROMPT


def _catalog_for_series(source_items: list, nodes: list) -> tuple:
    """
    Entities already extracted for this series (node["source"] in the
    series' item ids). Returns (catalog_text, valid_tokens).
    """
    entries = [
        {"id": n["id"], "type": n.get("type", "?"), "name": n.get("name", n["id"])}
        for n in nodes
        if n.get("source") in source_items
    ]
    text = ""
    for e in entries:
        line = f"{e['id']} | {e['type']} | {e['name']}\n"
        if text and len(text) + len(line) > _CATALOG_MAX_CHARS:
            break
        text += line
    valid_tokens = set()
    for e in entries:
        valid_tokens.add(e["id"].strip().lower())
        name_lower = e["name"].strip().lower()
        valid_tokens.add(name_lower)
        valid_tokens.add(name_lower.replace(" ", "_"))
        valid_tokens.add(name_lower.replace("_", " "))
    return text, valid_tokens


def _escape_curly(text: str) -> str:
    """Escape curly braces so str.format() treats them as literals."""
    return text.replace("{", "{{").replace("}", "}}") if text else text


def _source_text_for_series(source_items: list, chunks: list) -> str:
    parts = []
    total = 0
    for c in chunks or []:
        if c.get("item_id") not in source_items:
            continue
        t = c.get("text", "")
        if total and total + len(t) > _SOURCE_TEXT_MAX_CHARS:
            break
        parts.append(t)
        total += len(t)
    return "\n---\n".join(parts) if parts else "(no source chunk text available for this series)"


def _vector_related_context(query_text: str, source_items: list) -> tuple:
    if not query_text.strip():
        return "(no related context available)", []

    try:
        embedding = generate_embedding(query_text)
        results = search_chunks(embedding, n_results=_VECTOR_TOP_K)
    except Exception as e:
        return f"(related-context retrieval failed: {e})", []

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    parts = []
    used = []
    total = 0
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        item_id = meta.get("item_id")
        if item_id in source_items:
            continue
        if not doc or not doc.strip():
            continue
        snippet = doc.strip()
        if total and total + len(snippet) > _RELATED_CONTEXT_MAX_CHARS:
            continue
        if len(used) >= _VECTOR_MAX_CHUNKS_KEPT:
            break
        label = f"[from {item_id or 'unknown item'}, similarity {round(1 - dist, 3) if dist is not None else '?'}]"
        parts.append(f"{label}\n{snippet}")
        used.append({"item_id": item_id, "distance": dist})
        total += len(snippet)

    if not parts:
        return "(no related context found elsewhere in the knowledge base)", []
    return "\n---\n".join(parts), used


def _graph_neighbor_context(local_node_ids: list) -> tuple:
    if not local_node_ids:
        return [], None
    try:
        result = get_neighbor_nodes_db(local_node_ids, max_neighbors=_MAX_GRAPH_NEIGHBORS)
        return result.get("nodes", []), None
    except Exception as e:
        return [], f"graph neighbor traversal failed: {e}"


def _build_expanded_catalog(local_catalog_text: str, local_valid_tokens: set, neighbors: list) -> tuple:
    lines = [local_catalog_text] if local_catalog_text.strip() else []
    valid_tokens = set(local_valid_tokens)
    neighbor_tokens = set()

    for n in neighbors:
        nid = n.get("id", "")
        name = n.get("name", nid)
        ntype = n.get("type", "?")
        reached = n.get("reached_from", "?")
        lines.append(f"{nid} | {ntype} | {name}  (connected via graph, reached from {reached})")
        for tok in (nid.strip().lower(), name.strip().lower(),
                    name.strip().lower().replace(" ", "_"), name.strip().lower().replace("_", " ")):
            valid_tokens.add(tok)
            neighbor_tokens.add(tok)

    catalog_text = "\n".join(lines) if lines else "(no entities available)"
    return catalog_text, valid_tokens, neighbor_tokens


# ── Standalone Negative Scenario Generation ─────────────────────────────────

def generate_negative_scenarios(
    positive_test_cases: list,
    nodes: list,
    model: str = None,
    prompt_name: str = "edge_case_focused",
) -> dict:
    scenarios = []
    warnings = []
    seq_by_req = defaultdict(int)
    active_template = _resolve_negative_prompt_template(prompt_name)

    for tc in positive_test_cases:
        req_ids = tc.get("source_items", [tc.get("req_id", "")])
        catalog_text, valid_tokens = _catalog_for_series(req_ids, nodes)

        if not catalog_text.strip():
            warnings.append(
                f"{tc.get('req_id')}: no entity catalog available for this "
                f"series — skipped (nothing grounded to reason from)."
            )
            continue

        prompt = active_template.format(
            req_ids=", ".join(req_ids),
            actor=_escape_curly(tc.get("actor", "?")),
            feature=_escape_curly(tc.get("feature", "?")),
            steps=_escape_curly("; ".join(tc.get("steps", []))),
            expected_result=_escape_curly(tc.get("expected_result", "?")),
            catalog=_escape_curly(catalog_text),
        )

        result = call_ollama(prompt, model=model)
        if result["error"]:
            warnings.append(f"{tc.get('req_id')}: LLM call failed — {result['error']}")
            continue

        try:
            parsed = extract_json_block(result["raw"])
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(f"{tc.get('req_id')}: response could not be parsed — {e}")
            continue

        for entry in parsed:
            title = (entry.get("title") or "").strip()
            steps = entry.get("steps") or []
            expected = (entry.get("expected_result") or "").strip()
            entities_used = entry.get("entities_used") or []

            if not title or not steps or not expected:
                warnings.append(
                    f"{tc.get('req_id')}: dropped a scenario missing "
                    f"title/steps/expected_result."
                )
                continue

            ungrounded = [
                e for e in entities_used
                if e.strip().lower() not in valid_tokens
            ]
            if ungrounded:
                warnings.append(
                    f"{tc.get('req_id')}: dropped scenario '{title}' — "
                    f"referenced entities not in the graph (hallucinated): "
                    f"{ungrounded}"
                )
                continue

            seq_by_req[tc.get("req_id", "")] += 1
            seq = seq_by_req[tc.get("req_id", "")]

            scenarios.append({
                "tc_id": f"TC-NEG-{tc.get('req_id', 'X')}-{seq:03d}",
                "based_on_tc_id": tc.get("tc_id"),
                "req_id": tc.get("req_id"),
                "title": title,
                "type": "Negative",
                "priority": tc.get("priority", "Medium"),
                "actor": tc.get("actor", "User"),
                "feature": tc.get("feature", "?"),
                "precondition": tc.get("precondition", ""),
                "steps": steps,
                "expected_result": expected,
                "graph_nodes": entities_used,
                "source_items": req_ids,
                "fallbacks": [],
                "llm_derived": True,
            })

    return {"scenarios": scenarios, "warnings": warnings, "prompt_used": prompt_name}


# ── Test Case Graph Writing / Persistence ────────────────────────────────────

def _tc_node_id(tc_id: str) -> str:
    return f"TESTCASE::{tc_id}"


def build_test_case_graph_elements(test_cases: list, nodes: list) -> tuple:
    """
    Convert test case list into (nodes, relationships) ready for neo4j_manager.insert_graph().
    """
    try:
        live_nodes = get_all_nodes()
    except Exception:
        live_nodes = []

    token_to_id = {}
    all_ids = set()
    for n in (live_nodes + list(nodes)):
        node_id = n.get("id")
        if not node_id:
            continue
        all_ids.add(node_id)
        token_to_id[node_id.strip().lower()] = node_id
        name = n.get("name")
        if name:
            token_to_id.setdefault(name.strip().lower(), node_id)

    new_nodes = []
    new_rels = []

    for tc in test_cases:
        if not isinstance(tc, dict):
            continue
        tc_node_id = _tc_node_id(tc["tc_id"])

        new_nodes.append({
            "id": tc_node_id,
            "name": tc["tc_id"],
            "type": "TestCase",
            "description": tc.get("title", ""),
            "source": tc.get("req_id"),
            "aliases": [],
        })

        seen_targets = set()
        for token in tc.get("graph_nodes", []):
            target_id = token_to_id.get(token.strip().lower()) if token else None
            if not target_id or target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            new_rels.append({"from": tc_node_id, "to": target_id, "type": "VALIDATES"})

        for req_id in tc.get("source_items", []):
            if req_id in all_ids:
                new_rels.append({"from": tc_node_id, "to": req_id, "type": "VERIFIES"})

    return new_nodes, new_rels


def write_test_cases_to_graph(test_cases: list, nodes: list) -> dict:
    """
    Build + write test case nodes/edges to Neo4j in one call.
    """
    tc_nodes, tc_rels = build_test_case_graph_elements(test_cases, nodes)
    insert_graph(tc_nodes, tc_rels)

    validates_count = sum(1 for r in tc_rels if r["type"] == "VALIDATES")
    verifies_count = sum(1 for r in tc_rels if r["type"] == "VERIFIES")

    unresolved = [
        tc["tc_id"] for tc in test_cases
        if isinstance(tc, dict) and tc.get("graph_nodes") and not any(
            r["from"] == _tc_node_id(tc["tc_id"]) and r["type"] == "VALIDATES"
            for r in tc_rels
        )
    ]

    return {
        "test_case_nodes_written": len(tc_nodes),
        "validates_edges_written": validates_count,
        "verifies_edges_written": verifies_count,
        "test_cases_with_no_resolved_edges": unresolved,
    }


# ── Hybrid Test Case Generator Main ──────────────────────────────────────────

def generate_hybrid_test_cases(
    nodes: list,
    relationships: list,
    items_with_links: list = None,
    links: list = None,
    chunks: list = None,
    model: str = None,
    prompt_name: str = "hybrid_flow",
) -> dict:
    links = links or []
    chunks = chunks or []
    if not items_with_links:
        sources = sorted(list({n.get("source") for n in nodes if n.get("source") and n.get("source") != "workflow_extractor"}))
        items_with_links = [{"id": s, "family": None} for s in sources]
    try:
        skeleton_result = generate_test_cases_from_graph(nodes, relationships, items_with_links, links)
    except Exception as e:
        return {
            "test_cases": [],
            "text": "",
            "warnings": [f"generate_test_cases_from_graph failed: {e}"],
            "summary": {
                "total_test_cases": 0, "positive": 0, "negative": 0, "edge": 0,
                "llm_derived": 0, "graph_only_fallback": 0,
            },
            "prompt_used": prompt_name,
        }

    active_template = _resolve_prompt_template(prompt_name)

    import requests as _requests
    from config import OLLAMA_URL as _OLLAMA_URL
    _llm_available = False
    _probe_urls = list(dict.fromkeys([_OLLAMA_URL, "http://localhost:11434"]))
    for _purl in _probe_urls:
        try:
            _pr = _requests.get(f"{_purl}/api/tags", timeout=3)
            if _pr.status_code == 200:
                _llm_available = True
                break
        except Exception:
            pass

    if not _llm_available:
        _all_tcs = skeleton_result.get("test_cases", [])
        return {
            "test_cases": [{**tc, "llm_derived": False} for tc in _all_tcs],
            "text": skeleton_result.get("text", ""),
            "warnings": [
                "LLM server is unreachable — returned graph-only test cases without hybrid enrichment. "
                "Start Ollama locally or ensure the remote server is reachable to enable LLM refinement."
            ],
            "summary": {
                "total_test_cases": len(_all_tcs),
                "positive": sum(1 for tc in _all_tcs if tc.get("type") == "Positive"),
                "negative": sum(1 for tc in _all_tcs if tc.get("type") == "Negative"),
                "edge": sum(1 for tc in _all_tcs if tc.get("type") == "Edge"),
                "llm_derived": 0,
                "graph_only_fallback": len(_all_tcs),
            },
            "prompt_used": prompt_name,
        }

    test_cases = []
    warnings = []
    seq_by_req = defaultdict(int)
    seen_tc_keys = set()
    seen_tc_ids: set = set()

    for skeleton in skeleton_result["test_cases"]:
        tc_id = skeleton.get("tc_id", "")
        is_variant = any(tc_id.endswith(suffix) for suffix in ("-NEG", "-DEP", "-EDGE", "-ALT", "-EXC")) or tc_id.startswith("TC-FEAT-")
        
        req_ids = skeleton.get("source_items", [skeleton.get("req_id", "")])
        local_catalog_text, local_valid_tokens = _catalog_for_series(req_ids, nodes)
        local_node_ids = [n["id"] for n in nodes if n.get("source") in req_ids]

        if is_variant or not local_catalog_text.strip():
            if not is_variant and not local_catalog_text.strip():
                warnings.append(
                    f"{skeleton.get('req_id')}: no entity catalog available — "
                    f"kept the graph-only draft as-is (no hybrid refinement)."
                )
            test_cases.append({**skeleton, "llm_derived": False})
            continue

        neighbors, neighbor_error = _graph_neighbor_context(local_node_ids)
        if neighbor_error:
            warnings.append(f"{skeleton.get('req_id')}: {neighbor_error} — proceeding with local catalog only.")

        catalog_text, valid_tokens, neighbor_tokens = _build_expanded_catalog(
            local_catalog_text, local_valid_tokens, neighbors
        )

        source_text = _source_text_for_series(req_ids, chunks)

        vector_query = " ".join(filter(None, [
            skeleton.get("feature", ""), skeleton.get("actor", ""),
            "; ".join(skeleton.get("steps", [])),
        ]))
        related_context, vector_chunks_used = _vector_related_context(vector_query, req_ids)

        prompt = active_template.format(
            req_ids=", ".join(req_ids),
            source_text=_escape_curly(source_text),
            related_context=_escape_curly(related_context),
            actor=_escape_curly(skeleton.get("actor", "?")),
            feature=_escape_curly(skeleton.get("feature", "?")),
            precondition=_escape_curly(skeleton.get("precondition", "?")),
            steps=_escape_curly("; ".join(skeleton.get("steps", []))),
            expected_result=_escape_curly(skeleton.get("expected_result", "?")),
            catalog=_escape_curly(catalog_text),
        )

        result = call_ollama(prompt, model=model, num_ctx=8192)
        if result["error"]:
            warnings.append(
                f"{skeleton.get('req_id')}: LLM call failed ({result['error']}) — "
                f"kept the graph-only draft as-is."
            )
            test_cases.append({**skeleton, "llm_derived": False})
            continue

        try:
            parsed = extract_json_block(result["raw"])
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("expected a non-empty JSON array")
        except (ValueError, json.JSONDecodeError) as e:
            warnings.append(
                f"{skeleton.get('req_id')}: response could not be parsed ({e}) — "
                f"kept the graph-only draft as-is."
            )
            test_cases.append({**skeleton, "llm_derived": False})
            continue

        kept_any = False
        for entry in parsed:
            title = (entry.get("title") or "").strip()
            case_type = (entry.get("type") or "").strip().capitalize()
            steps = entry.get("steps") or []
            expected = (entry.get("expected_result") or "").strip()
            entities_used = entry.get("entities_used") or []

            if not title or not steps or not expected or case_type not in ("Positive", "Negative", "Edge"):
                warnings.append(
                    f"{skeleton.get('req_id')}: dropped a scenario missing "
                    f"title/type/steps/expected_result (or an unrecognized type)."
                )
                continue

            ungrounded = [e for e in entities_used if e.strip().lower() not in valid_tokens]
            if ungrounded:
                warnings.append(
                    f"{skeleton.get('req_id')}: dropped scenario '{title}' — "
                    f"referenced entities not in the expanded catalog (hallucinated): {ungrounded}"
                )
                continue

            title_lower = title.strip().lower()
            steps_hash = hash(tuple(s.strip().lower() for s in steps))
            tc_key = (skeleton.get("req_id"), case_type, title_lower, steps_hash)
            if tc_key in seen_tc_keys:
                warnings.append(f"{skeleton.get('req_id')}: dropped duplicate scenario '{title}'.")
                continue
            seen_tc_keys.add(tc_key)

            local_used = [e for e in entities_used if e.strip().lower() not in neighbor_tokens]
            neighbor_used = [e for e in entities_used if e.strip().lower() in neighbor_tokens]

            tc_num = seq_by_req[skeleton.get("req_id")] + 1
            seq_by_req[skeleton.get("req_id")] = tc_num
            req_id = skeleton.get("req_id")
            is_combined = len(req_ids) > 1
            title_hash = hashlib.sha256(title.encode()).hexdigest()[:4]
            tc_id = (
                f"TC-COMBINED-{req_id}-HYB-{tc_num:03d}-{title_hash}" if is_combined
                else f"TC-{req_id}-HYB-{tc_num:03d}-{title_hash}"
            )
            if tc_id in seen_tc_ids:
                raise RuntimeError(
                    f"TC-ID collision detected: '{tc_id}' was already generated in this batch. "
                    f"This is a pipeline invariant violation — check req_id/title uniqueness."
                )
            seen_tc_ids.add(tc_id)

            tc_candidate = {
                "tc_id": tc_id,
                "req_id": req_id,
                "title": title,
                "type": case_type,
                "priority": skeleton.get("priority"),
                "actor": skeleton.get("actor"),
                "feature": skeleton.get("feature"),
                "precondition": skeleton.get("precondition"),
                "steps": steps,
                "expected_result": expected,
                "graph_nodes": entities_used,
                "source_items": skeleton.get("source_items", req_ids),
                "fallbacks": [],
                "llm_derived": True,
                "graph_context": {"local": local_used, "neighbor": neighbor_used},
                "vector_context_used": vector_chunks_used,
                "grounding_evidence": f"Requirement Series: {req_id} | Graph Nodes: {len(entities_used)}"
            }
            tc_aligned, ok = validate_and_align_precondition(tc_candidate)
            if ok:
                test_cases.append(tc_aligned)
            else:
                warnings.append(f"{req_id}: precondition mismatch after LLM generation — kept graph‑only draft.")

            kept_any = True

        if not kept_any:
            warnings.append(
                f"{skeleton.get('req_id')}: hybrid pass returned no scenario that passed "
                f"grounding/shape checks — kept the graph-only draft as-is."
            )
            test_cases.append({**skeleton, "llm_derived": False})

    text = "\n\n".join(_render_test_case_text(tc) for tc in test_cases)

    return {
        "test_cases": test_cases,
        "text": text,
        "warnings": warnings,
        "summary": {
            "total_test_cases": len(test_cases),
            "positive": sum(1 for tc in test_cases if tc["type"] == "Positive"),
            "negative": sum(1 for tc in test_cases if tc["type"] == "Negative"),
            "edge": sum(1 for tc in test_cases if tc["type"] == "Edge"),
            "llm_derived": sum(1 for tc in test_cases if tc.get("llm_derived")),
            "graph_only_fallback": sum(1 for tc in test_cases if not tc.get("llm_derived")),
            "used_graph_neighbors": sum(
                1 for tc in test_cases if tc.get("graph_context", {}).get("neighbor")
            ),
            "used_vector_context": sum(
                1 for tc in test_cases if tc.get("vector_context_used")
            ),
        },
        "prompt_used": prompt_name,
    }