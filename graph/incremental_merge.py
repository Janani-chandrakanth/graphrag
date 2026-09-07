"""
Incremental Graph Merge

The one new stage the "Update Graph" pipeline adds on top of the exact
same Parser -> Chunking -> Embeddings -> Entity Extraction -> Validation
-> Deduplication pipeline the main "Build Graph" pipeline already uses
(see app.py's Update Graph tab). Everything upstream of this module is
100% reused, unchanged.

Four responsibilities, run in order:

  1. Entity Matching     -- match_entities()
     Decide whether a newly-extracted node is actually an EXISTING
     node under the graph's own MERGE-by-id semantics, and only
     genuinely un-matchable, un-remapped nodes look like the graph is
     restarting from empty.

  2. Conflict Detection   -- detect_conflicts()
     "Payment used to lead to Receipt, the new upload says Payment
     leads to Invoice" -- flag it as a modification instead of
     blindly appending a second, silently-contradictory edge.

  3. Relationship Validation -- validate_relationships_no_cycles()
     Cycles, isolated nodes, invalid direction (dangling endpoints),
     duplicate edges, run against the FULL merged graph (existing +
     new), not just the new upload in isolation.

  4. Workflow Preservation -- preserve_workflow()
     Applies the user's resolution for each conflict (extend/replace/
     skip) and defaults every non-conflicting edge to "extend" (append
     -- never silently deletes an existing edge without an explicit
     "replace" decision), so the existing sequential flow can only be
     extended or deliberately modified, never accidentally broken.

build_merge_plan() runs all four and returns a plan for the UI to
review; apply_merge_plan() does the actual Neo4j write, only once the
user has confirmed it (and only after graph/version_manager.py has
snapshotted the pre-merge state, so nothing is ever lost -- see
app.py's Update Graph tab for the exact sequencing).
"""

import math

from graph.deduplicator import normalize_id, TYPE_PRIORITY
from graph.neo4j_manager import insert_graph_with_batch, delete_relationships
from embeddings.embedding_model import generate_embedding
from parser.llm_client import call_ollama
from config import EXTRACTION_MODEL, EXTRACTION_MODEL_URL

# Above this cosine similarity, two nodes are treated as the same
# entity automatically -- no human/LLM review needed.
SEMANTIC_MATCH_THRESHOLD = 0.90
# Between this and SEMANTIC_MATCH_THRESHOLD, it's ambiguous enough to
# ask the extraction LLM to verify before auto-merging (or to flag for
# human review, if use_llm_verification=False).
SEMANTIC_REVIEW_THRESHOLD = 0.80


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _node_text(node: dict) -> str:
    return f"{node.get('name', '')}: {node.get('description') or ''}".strip()


def _llm_verify_same_entity(new_node: dict, existing_node: dict) -> bool:
    """Tie-breaker for borderline semantic matches. Same call pattern
    (model/base_url/deterministic decoding) as the rest of the
    pipeline -- see parser/llm_client.py."""
    prompt = (
        "You are verifying whether two entities extracted from a "
        "requirements knowledge graph refer to the SAME real-world "
        "business concept, just described slightly differently across "
        "two different uploaded documents.\n\n"
        f"Entity A: name=\"{new_node.get('name')}\" type={new_node.get('type')} "
        f"description=\"{new_node.get('description') or ''}\"\n"
        f"Entity B: name=\"{existing_node.get('name')}\" type={existing_node.get('type')} "
        f"description=\"{existing_node.get('description') or ''}\"\n\n"
        "Answer with exactly one word: YES if they refer to the same "
        "concept, NO otherwise."
    )
    result = call_ollama(prompt, model=EXTRACTION_MODEL, base_url=EXTRACTION_MODEL_URL, timeout=60)
    if result.get("error"):
        return False
    return result.get("raw", "").strip().upper().startswith("Y")


def match_entities(new_nodes: list, existing_nodes: list, use_llm_verification: bool = True):
    """
    Try to map every new node onto an existing node it actually refers
    to, in three escalating passes: exact normalized ID, semantic
    embedding similarity, LLM verification for the ambiguous middle
    ground.

    Returns:
        id_remap: {new_node_id: existing_node_id} for every match found
                  (including exact-id matches, which the graph's own
                  MERGE would have deduped anyway -- included here so
                  the UI can still SHOW "Payment already exists,
                  reusing that node" instead of it happening silently).
        match_report: list of {new_id, matched_to, method, score}.
        unmatched: new nodes with no existing counterpart -- genuinely
                   new concepts this upload introduced.
    """
    id_remap = {}
    match_report = []
    unmatched = []

    existing_by_id = {normalize_id(n["id"]): n for n in existing_nodes}
    existing_embeddings = {}  # cached across the whole call

    for node in new_nodes:
        nid = normalize_id(node["id"])
        node["id"] = nid

        if nid in existing_by_id:
            id_remap[nid] = nid
            match_report.append({"new_id": nid, "matched_to": nid, "method": "exact_id", "score": 1.0})
            continue

        best_id, best_score = None, 0.0
        try:
            new_emb = generate_embedding(_node_text(node))
        except Exception:
            new_emb = None

        if new_emb:
            for eid, enode in existing_by_id.items():
                if eid not in existing_embeddings:
                    try:
                        existing_embeddings[eid] = generate_embedding(_node_text(enode))
                    except Exception:
                        existing_embeddings[eid] = None
                eemb = existing_embeddings[eid]
                if not eemb:
                    continue
                score = _cosine(new_emb, eemb)
                if score > best_score:
                    best_score, best_id = score, eid

        if best_id and best_score >= SEMANTIC_MATCH_THRESHOLD:
            id_remap[nid] = best_id
            match_report.append({
                "new_id": nid, "matched_to": best_id,
                "method": "semantic", "score": round(best_score, 3),
            })
            continue

        if best_id and best_score >= SEMANTIC_REVIEW_THRESHOLD and use_llm_verification:
            if _llm_verify_same_entity(node, existing_by_id[best_id]):
                id_remap[nid] = best_id
                match_report.append({
                    "new_id": nid, "matched_to": best_id,
                    "method": "llm_verified", "score": round(best_score, 3),
                })
                continue

        unmatched.append(node)

    return id_remap, match_report, unmatched


def apply_id_remap(nodes: list, relationships: list, id_remap: dict):
    """Rewrite node ids and every from/to reference through id_remap
    (identity for anything not remapped). If two new nodes collapse
    onto the same existing id, keep the higher-priority type -- same
    TYPE_PRIORITY rule graph/deduplicator.py already uses within a
    single upload."""
    remapped_nodes = {}
    for node in nodes:
        nid = id_remap.get(node["id"], node["id"])
        node["id"] = nid
        if nid in remapped_nodes:
            existing_type = remapped_nodes[nid].get("type", "")
            incoming_type = node.get("type", "")
            if TYPE_PRIORITY.get(incoming_type, 0) > TYPE_PRIORITY.get(existing_type, 0):
                remapped_nodes[nid] = node
        else:
            remapped_nodes[nid] = node

    remapped_rels = []
    for rel in relationships:
        rel["from"] = id_remap.get(rel["from"], rel["from"])
        rel["to"] = id_remap.get(rel["to"], rel["to"])
        remapped_rels.append(rel)

    return list(remapped_nodes.values()), remapped_rels


def detect_conflicts(new_relationships: list, existing_relationships: list) -> list:
    """
    Flag "this is a modification, not a blind append" situations:

        Old:  Payment -[LEADS_TO]-> Receipt
        New:  Payment -[LEADS_TO]-> Invoice
        =>    conflict: Payment's LEADS_TO target changed.

    A brand-new (from, type) pair, or one whose target already
    matches, is NOT a conflict -- only an existing (from, type) whose
    target set is about to gain a genuinely different target.
    """
    existing_targets = {}
    for rel in existing_relationships:
        existing_targets.setdefault((rel["from"], rel["type"]), set()).add(rel["to"])

    conflicts = []
    for rel in new_relationships:
        key = (rel["from"], rel["type"])
        prior_targets = existing_targets.get(key)
        if not prior_targets or rel["to"] in prior_targets:
            continue
        conflicts.append({
            "from": rel["from"],
            "type": rel["type"],
            "existing_to": sorted(prior_targets),
            "new_to": rel["to"],
            "message": (
                f"{rel['from']} already has a {rel['type']} relationship "
                f"to {', '.join(sorted(prior_targets))}. The new upload "
                f"points it to {rel['to']} instead."
            ),
        })
    return conflicts


def preserve_workflow(new_relationships: list, existing_relationships: list, resolutions: dict = None):
    """
    Turn conflict resolutions into concrete add/remove relationship
    lists.

    resolutions: {(from, type, new_to): "extend" | "replace" | "skip"}
        extend  -> keep BOTH the old and new edge (a branch -- e.g.
                   Payment now leads to both Receipt and Invoice)
        replace -> the old edge for that (from, type) is removed, the
                   new edge is added (old step is superseded)
        skip    -> drop the new edge, graph stays exactly as it was

    Unresolved conflicts, and every non-conflicting new edge, default
    to "extend" (append-only) -- the safest default: nothing existing
    is ever silently removed without an explicit "replace" decision.

    Returns (relationships_to_add, relationships_to_remove).
    """
    resolutions = resolutions or {}

    existing_targets = {}
    for rel in existing_relationships:
        existing_targets.setdefault((rel["from"], rel["type"]), set()).add(rel["to"])

    to_add, to_remove = [], []
    for rel in new_relationships:
        key = (rel["from"], rel["type"])
        prior_targets = existing_targets.get(key, set())
        if rel["to"] in prior_targets:
            continue  # no-op, already there

        decision = resolutions.get((rel["from"], rel["type"], rel["to"]), "extend")
        if decision == "skip":
            continue
        if decision == "replace":
            for old_to in prior_targets:
                to_remove.append({"from": rel["from"], "to": old_to, "type": rel["type"]})
        to_add.append(rel)

    return to_add, to_remove


def validate_relationships_no_cycles(nodes: list, relationships: list) -> dict:
    """
    Cycle / isolated-node / invalid-direction / duplicate-edge checks
    against the MERGED graph (existing + new) -- on top of, not
    instead of, graph/validator.py's per-node/per-type schema
    validation that already ran earlier in the pipeline. Never raises
    or blocks the merge; issues are surfaced for the user to see, the
    same "flag, don't silently drop" philosophy graph/deduplicator.py
    and graph/confidence_scorer.py already use.
    """
    node_ids = {n["id"] for n in nodes}
    adjacency = {}
    seen_edges = set()
    duplicate_edges = []
    invalid_direction = []

    for rel in relationships:
        if rel["from"] not in node_ids or rel["to"] not in node_ids:
            invalid_direction.append(rel)
            continue
        edge_key = (rel["from"], rel["to"], rel["type"])
        if edge_key in seen_edges:
            duplicate_edges.append(rel)
            continue
        seen_edges.add(edge_key)
        adjacency.setdefault(rel["from"], []).append(rel["to"])

    connected = set()
    for rel in relationships:
        connected.add(rel["from"])
        connected.add(rel["to"])
    isolated_nodes = [n["id"] for n in nodes if n["id"] not in connected]

    # Cycle detection (DFS, white/gray/black) -- reported, never
    # blocked: some domains legitimately have feedback loops (retry,
    # rejection-and-resubmit), so this is visibility, not a hard rule.
    cycles = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in node_ids}

    def dfs(u, path):
        color[u] = GRAY
        path.append(u)
        for v in adjacency.get(u, []):
            if color.get(v) == GRAY:
                cyc_start = path.index(v)
                cycles.append(path[cyc_start:] + [v])
            elif color.get(v) == WHITE:
                dfs(v, path)
        path.pop()
        color[u] = BLACK

    for nid in list(node_ids):
        if color[nid] == WHITE:
            dfs(nid, [])

    return {
        "isolated_nodes": isolated_nodes,
        "duplicate_edges": duplicate_edges,
        "invalid_direction": invalid_direction,
        "cycles": cycles,
    }


def build_merge_plan(new_nodes: list, new_relationships: list,
                      existing_nodes: list, existing_relationships: list,
                      use_llm_verification: bool = True, resolutions: dict = None) -> dict:
    """
    Runs Entity Matching -> Conflict Detection -> Workflow Preservation
    -> Relationship Validation, in order, and packages the result into
    one plan for the UI to review. Nothing is written to Neo4j here --
    see apply_merge_plan().
    """
    id_remap, match_report, unmatched_new = match_entities(
        new_nodes, existing_nodes, use_llm_verification=use_llm_verification
    )
    remapped_nodes, remapped_rels = apply_id_remap(new_nodes, new_relationships, id_remap)

    conflicts = detect_conflicts(remapped_rels, existing_relationships)
    rels_to_add, rels_to_remove = preserve_workflow(remapped_rels, existing_relationships, resolutions)

    merged_nodes_preview = {n["id"]: n for n in existing_nodes}
    for n in remapped_nodes:
        merged_nodes_preview[n["id"]] = n
    remove_keys = {(r["from"], r["to"], r["type"]) for r in rels_to_remove}
    merged_rels_preview = [r for r in existing_relationships
                            if (r["from"], r["to"], r["type"]) not in remove_keys]
    merged_rels_preview.extend(rels_to_add)

    validation_report = validate_relationships_no_cycles(
        list(merged_nodes_preview.values()), merged_rels_preview
    )

    return {
        "match_report": match_report,
        "unmatched_new_nodes": unmatched_new,
        "nodes_to_upsert": remapped_nodes,
        "conflicts": conflicts,
        "relationships_to_add": rels_to_add,
        "relationships_to_remove": rels_to_remove,
        "validation_report": validation_report,
        "stats": {
            "new_nodes_total": len(new_nodes),
            "matched_existing": len(match_report),
            "genuinely_new": len(unmatched_new),
            "relationships_added": len(rels_to_add),
            "relationships_removed": len(rels_to_remove),
            "conflicts_detected": len(conflicts),
        },
    }


def apply_merge_plan(plan: dict, batch_id: str):
    """
    Actually writes a reviewed/confirmed merge plan to Neo4j:
      - upserts nodes_to_upsert + relationships_to_add, tagged with
        batch_id (graph/neo4j_manager.py's insert_graph_with_batch --
        the same mechanism graph/batch_upload-style flows already use,
        now reused here so every incremental update is attributable to
        the version it came from -- see graph/version_manager.py).
      - deletes relationships_to_remove (only ever populated by an
        explicit "replace" conflict resolution -- see
        preserve_workflow()'s docstring).
    Caller (app.py's Update Graph tab) is responsible for calling
    graph/version_manager.snapshot_current_graph() BEFORE this, so the
    pre-merge state is retained as a version first.
    """
    insert_graph_with_batch(plan["nodes_to_upsert"], plan["relationships_to_add"], batch_id)
    if plan["relationships_to_remove"]:
        delete_relationships(plan["relationships_to_remove"])
