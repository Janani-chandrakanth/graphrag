"""
graph/feature_extractor.py

Extracts structured business feature models from graph data and semantic
classifications, establishing ordered steps, entry points, branches, and supporting context.
"""

import json
import hashlib
import logging
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Set, Tuple

from graph.flow_graph_analysis import FLOW_RELATIONS
from graph.semantic_classifier import (
    classify_graph,
    build_classification_cache_key,
    _CLASSIFIER_VERSION,
)
from parser.llm_client import call_ollama, extract_json_block
from config import EXTRACTION_MODEL, EXTRACTION_MODEL_URL

logger = logging.getLogger(__name__)

# ── Extractor version — bump when prompt/algo changes to bust caches ──
_EXTRACTOR_VERSION = "v1.2"

# How many candidate features to send to LLM in one batch
_LLM_FEATURE_BATCH = 20

# Min workflow steps for a component to be its own feature
_MIN_STEPS_FOR_FEATURE = 1

# How many hops of supporting context to collect per feature
_SUPPORT_HOPS = 1


# =============================================================================
# Public API
# =============================================================================

def extract_features(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    classification: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point.  Returns:

        {
            "features":            [Feature Model, ...],
            "cache_key":           str,
            "classification":      {...},   # re-exported for downstream use
            "coverage_universe":   {...},
            "warnings":            [str, ...],
        }

    If classification is None, calls classify_graph() first.
    Pass in a cached classification to avoid re-running the classifier.
    """
    warnings: List[str] = []

    # ── Step 0: Classification ────────────────────────────────────────
    if classification is None:
        classification = classify_graph(nodes, relationships, use_llm=use_llm)

    node_cls  = classification["node_classifications"]
    edge_cls  = classification["edge_classifications"]
    universe  = classification["coverage_universe"]

    cache_key = _build_extractor_cache_key(classification["cache_key"])

    # Index structures
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}

    # ── Step 1: Build flow adjacency (workflow transitions only) ──────
    flow_adj:  Dict[str, List[str]] = defaultdict(list)   # id → [successor_id]
    flow_adj_r:Dict[str, List[str]] = defaultdict(list)   # id → [predecessor_id]
    for rel in relationships:
        key = (rel.get("from", ""), rel.get("to", ""), rel.get("type", ""))
        if edge_cls.get(key, {}).get("role") == "workflow_transition":
            frm = key[0]
            to  = key[1]
            if frm and to:
                flow_adj[frm].append(to)
                flow_adj_r[to].append(frm)

    # ── Step 2: Collect workflow node IDs ─────────────────────────────
    workflow_node_ids: Set[str] = {
        nid for nid, cls in node_cls.items()
        if cls.get("role") in ("workflow_step", "workflow_state", "actor")
        and nid in node_by_id
    }

    if not workflow_node_ids:
        warnings.append("No workflow nodes found in the graph. "
                         "Upload a BRD and build the graph first.")
        return {
            "features": [],
            "cache_key": cache_key,
            "classification": classification,
            "coverage_universe": universe,
            "warnings": warnings,
        }

    # ── Step 3: Connected components (undirected) of workflow nodes ───
    components = _find_connected_components(workflow_node_ids, flow_adj, flow_adj_r)

    # ── Step 4: Build candidate feature structures ────────────────────
    candidates: List[Dict[str, Any]] = []
    for comp in components:
        if len(comp) < _MIN_STEPS_FOR_FEATURE:
            continue
        candidate = _build_candidate_feature(
            comp, node_by_id, node_cls, edge_cls, relationships,
            flow_adj, flow_adj_r,
        )
        if candidate:
            candidates.append(candidate)

    if not candidates:
        warnings.append("Graph has workflow nodes but no connected workflow "
                         "transitions were found. The workflow may need more "
                         "LEADS_TO / TRIGGERS relationships.")
        # Fall back: treat each workflow node as its own 1-step feature
        for nid in list(workflow_node_ids)[:20]:
            n = node_by_id[nid]
            candidates.append(_single_node_feature(n, node_cls, node_by_id, relationships, edge_cls))

    # ── Step 5: LLM semantic enrichment (single batch call) ───────────
    if use_llm and candidates:
        _enrich_with_llm(candidates, node_by_id, warnings)
    else:
        for c in candidates:
            if not c.get("name"):
                c["name"] = _derive_name_heuristic(c, node_by_id)
            c["confidence"] = 0.60

    # ── Step 6: Attach supporting context ────────────────────────────
    for c in candidates:
        support = _collect_supporting_context(
            c["step_node_ids"], node_by_id, node_cls, edge_cls, relationships, _SUPPORT_HOPS
        )
        c["supporting_node_ids"]  = support["node_ids"]
        c["supporting_edge_ids"]  = support["edge_ids"]
        c["business_rule_ids"]    = support["business_rule_ids"]
        c["requirement_ids"]      = support["requirement_ids"]

    # ── Step 7: Stable feature IDs ────────────────────────────────────
    for i, c in enumerate(candidates):
        c["feature_id"] = _stable_feature_id(c["name"], i)

    return {
        "features":          candidates,
        "cache_key":         cache_key,
        "classification":    classification,
        "coverage_universe": universe,
        "warnings":          warnings,
    }


# =============================================================================
# Connected Component Discovery
# =============================================================================

def _find_connected_components(
    node_ids: Set[str],
    adj:   Dict[str, List[str]],
    adj_r: Dict[str, List[str]],
) -> List[Set[str]]:
    """
    Undirected connected components over the workflow node set,
    using the union of forward + reverse flow edges.
    """
    visited: Set[str] = set()
    components: List[Set[str]] = []

    def bfs(start: str) -> Set[str]:
        component: Set[str] = set()
        queue = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in visited or nid not in node_ids:
                continue
            visited.add(nid)
            component.add(nid)
            for nb in adj.get(nid, []):
                if nb not in visited and nb in node_ids:
                    queue.append(nb)
            for nb in adj_r.get(nid, []):
                if nb not in visited and nb in node_ids:
                    queue.append(nb)
        return component

    for nid in node_ids:
        if nid not in visited:
            comp = bfs(nid)
            if comp:
                components.append(comp)

    # Sort largest component first
    return sorted(components, key=len, reverse=True)


def _expand_step_sequence(
    step_ids: List[str],
    relationships: List[Dict],
    node_by_id: Dict[str, Dict]
) -> Tuple[List[str], List[str]]:
    """
    Expands high-level step IDs with their procedural flow:
    Screens -> Inputs -> Actions / Step -> Next Steps.
    Prevents consecutive duplicate names and excludes Feature container nodes.
    """
    sequence_nodes = []
    seen_ids = set()

    for sid in step_ids:
        screens = []
        inputs = []
        actions = []
        next_steps = []

        for r in relationships:
            frm, to, rtype = r.get("from"), r.get("to"), r.get("type")
            if frm == sid:
                tgt = node_by_id.get(to, {})
                ttype = tgt.get("type", "")
                tname = tgt.get("name") or to
                if ttype == "Screen" and to not in seen_ids:
                    screens.append(to)
                elif ttype == "UIElement" and to not in seen_ids:
                    if any(w in tname.lower() for w in ("button", "submit", "click", "select", "press", "action")):
                        actions.append(to)
                    else:
                        inputs.append(to)
                elif rtype in ("NEXT", "LEADS_TO", "NOTIFIES", "TRIGGERS", "CONTAINS") and to not in seen_ids:
                    if ttype not in ("Feature", "System", "SoftwareSystem"):
                        next_steps.append(to)
            elif to == sid:
                src = node_by_id.get(frm, {})
                stype = src.get("type", "")
                sname = src.get("name") or frm
                if stype == "Screen" and frm not in seen_ids:
                    screens.append(frm)
                elif rtype in ("CONTAINS", "TRIGGERS") and frm not in seen_ids:
                    if stype not in ("Feature", "System", "SoftwareSystem"):
                        next_steps.append(frm)

        for nid in screens:
            if nid not in seen_ids:
                seen_ids.add(nid)
                sequence_nodes.append(nid)
        for nid in inputs:
            if nid not in seen_ids:
                seen_ids.add(nid)
                sequence_nodes.append(nid)
        for nid in actions:
            if nid not in seen_ids:
                seen_ids.add(nid)
                sequence_nodes.append(nid)

        if not screens and not inputs and not actions:
            if sid not in seen_ids:
                seen_ids.add(sid)
                sequence_nodes.append(sid)
        else:
            s_node = node_by_id.get(sid) or {}
            s_name = s_node.get("name") or sid
            existing_names = [(node_by_id.get(x) or {}).get("name") for x in sequence_nodes]
            if s_name not in existing_names and s_node.get("type") != "Feature":
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    sequence_nodes.append(sid)

        for nid in next_steps:
            if nid not in seen_ids:
                seen_ids.add(nid)
                sequence_nodes.append(nid)

    cleaned_ids = []
    cleaned_names = []
    last_name = None
    for nid in sequence_nodes:
        n_obj = node_by_id.get(nid) or {}
        if n_obj.get("type") == "Feature":
            continue
        name = n_obj.get("name") or nid
        if name != last_name:
            cleaned_ids.append(nid)
            cleaned_names.append(name)
            last_name = name

    return cleaned_ids, cleaned_names


# =============================================================================
# Candidate Feature Builder
# =============================================================================

def _build_candidate_feature(
    comp: Set[str],
    node_by_id: Dict[str, Dict],
    node_cls:   Dict[str, Dict],
    edge_cls:   Dict[Tuple, Dict],
    relationships: List[Dict],
    flow_adj:   Dict[str, List[str]],
    flow_adj_r: Dict[str, List[str]],
) -> Optional[Dict[str, Any]]:
    """Build a candidate feature dict from a connected component."""
    # Only include non-actor workflow nodes in the primary walk
    step_ids = [
        nid for nid in comp
        if node_cls.get(nid, {}).get("role") in ("workflow_step", "workflow_state")
    ]
    actor_ids = [
        nid for nid in comp
        if node_cls.get(nid, {}).get("role") == "actor"
    ]

    if not step_ids:
        return None

    # Find entry point: step node(s) with no predecessor inside the component
    comp_set = set(comp)
    entry_candidates = [
        nid for nid in step_ids
        if not any(p in comp_set for p in flow_adj_r.get(nid, []))
    ]
    if not entry_candidates:
        # Cyclic component — pick the node with the most outgoing edges
        entry_candidates = sorted(
            step_ids,
            key=lambda n: len(flow_adj.get(n, [])),
            reverse=True,
        )
    entry_id = entry_candidates[0] if entry_candidates else step_ids[0]

    # Walk ordered steps (DFS from entry, cycle-safe)
    ordered_step_ids = _dfs_ordered_walk(entry_id, step_ids, flow_adj, comp_set)

    # Workflow transitions within this component
    transitions = []
    workflow_edge_ids = []
    for rel in relationships:
        key = (rel.get("from", ""), rel.get("to", ""), rel.get("type", ""))
        if (key[0] in comp_set and key[1] in comp_set
                and edge_cls.get(key, {}).get("role") == "workflow_transition"):
            transitions.append({"from": key[0], "to": key[1], "type": key[2]})
            workflow_edge_ids.append(key)

    # Decision nodes: steps with multiple outgoing transitions
    successors_in_comp = {
        nid: [s for s in flow_adj.get(nid, []) if s in comp_set]
        for nid in step_ids
    }
    decision_node_ids = [nid for nid, succs in successors_in_comp.items() if len(succs) > 1]
    branches = [
        {"from": nid, "to_list": succs}
        for nid, succs in successors_in_comp.items()
        if len(succs) > 1
    ]

    # End states: steps with no outgoing transitions inside the component
    end_state_ids = [
        nid for nid in step_ids
        if not any(s in comp_set for s in flow_adj.get(nid, []))
    ]

    # Expand procedural workflow sequence: Screen -> Inputs -> Actions / Step -> Next Steps
    expanded_step_ids, expanded_step_names = _expand_step_sequence(
        ordered_step_ids, relationships, node_by_id
    )

    if expanded_step_ids:
        final_step_ids = expanded_step_ids
        final_step_names = expanded_step_names
    else:
        # Fallback: exclude any Feature container nodes and deduplicate consecutive identical names
        final_step_ids = []
        final_step_names = []
        last_name = None
        for nid in ordered_step_ids:
            n_obj = node_by_id.get(nid) or {}
            if n_obj.get("type") == "Feature":
                continue
            name = n_obj.get("name") or nid
            if name != last_name:
                final_step_ids.append(nid)
                final_step_names.append(name)
                last_name = name

    if not final_step_ids:
        final_step_ids = ordered_step_ids
        final_step_names = [
            (node_by_id.get(nid) or {}).get("name") or nid
            for nid in ordered_step_ids
        ]

    actor_names = [
        (node_by_id.get(aid) or {}).get("name") or aid
        for aid in actor_ids
    ]

    return {
        "feature_id":             "",          # assigned in step 7
        "name":                   "",          # filled by LLM or heuristic
        "description":            "",
        "actors":                 actor_names,
        "entry_point_id":         entry_id,
        "ordered_steps":          final_step_names,
        "step_node_ids":          final_step_ids,
        "workflow_edge_ids":      workflow_edge_ids,
        "transitions":            transitions,
        "decision_node_ids":      decision_node_ids,
        "branches":               branches,
        "end_state_ids":          end_state_ids,
        "supporting_node_ids":    [],          # filled in step 6
        "supporting_edge_ids":    [],
        "business_rule_ids":      [],
        "requirement_ids":        [],
        "alternate_paths":        [],
        "negative_paths":         [],
        "failure_recovery_paths": [],
        "source_evidence":        [nid for nid in ordered_step_ids],
        "confidence":             0.70,
    }


def _single_node_feature(
    node: Dict,
    node_cls: Dict,
    node_by_id: Dict,
    relationships: List[Dict],
    edge_cls: Dict,
) -> Dict[str, Any]:
    """Fallback: wrap a single workflow node as a trivial feature."""
    nid = node.get("id", "")
    return {
        "feature_id":             "",
        "name":                   node.get("name") or nid,
        "description":            node.get("description") or "",
        "actors":                 [],
        "entry_point_id":         nid,
        "ordered_steps":          [node.get("name") or nid],
        "step_node_ids":          [nid],
        "workflow_edge_ids":      [],
        "transitions":            [],
        "decision_node_ids":      [],
        "branches":               [],
        "end_state_ids":          [nid],
        "supporting_node_ids":    [],
        "supporting_edge_ids":    [],
        "business_rule_ids":      [],
        "requirement_ids":        [],
        "alternate_paths":        [],
        "negative_paths":         [],
        "failure_recovery_paths": [],
        "source_evidence":        [nid],
        "confidence":             0.40,
    }


# =============================================================================
# DFS Ordered Walk (cycle-safe)
# =============================================================================

def _dfs_ordered_walk(
    entry: str,
    step_ids: List[str],
    flow_adj: Dict[str, List[str]],
    comp_set: Set[str],
) -> List[str]:
    """
    DFS from `entry` over the workflow step nodes in the component.
    Handles cycles by tracking visited nodes (never revisit).
    Appends any step nodes unreachable from entry at the end.
    """
    step_set = set(step_ids)
    ordered: List[str] = []
    visited: Set[str] = set()

    def dfs(nid: str):
        if nid in visited or nid not in step_set:
            return
        visited.add(nid)
        ordered.append(nid)
        for succ in flow_adj.get(nid, []):
            if succ in comp_set:
                dfs(succ)

    dfs(entry)
    # Append disconnected islands within the component
    for nid in step_ids:
        if nid not in visited:
            dfs(nid)

    return ordered


# =============================================================================
# LLM Semantic Enrichment
# =============================================================================

def _enrich_with_llm(
    candidates: List[Dict[str, Any]],
    node_by_id: Dict[str, Dict],
    warnings: List[str],
) -> None:
    """
    Single batched LLM call to assign names/descriptions/alternate_paths/etc.
    Modifies candidates in-place.
    """
    # Build a compact summary for each candidate
    summaries = []
    for i, c in enumerate(candidates):
        step_names = c.get("ordered_steps", [])
        actor_names = c.get("actors", [])
        summaries.append({
            "index":        i,
            "step_count":   len(step_names),
            "steps":        step_names[:12],    # cap to keep prompt manageable
            "actors":       actor_names[:4],
            "entry":        (node_by_id.get(c.get("entry_point_id") or "") or {}).get("name", ""),
            "end_states":   [
                (node_by_id.get(eid) or {}).get("name", eid)
                for eid in c.get("end_state_ids", [])[:3]
            ],
            "has_branches": bool(c.get("branches")),
        })

    prompt = f"""You are a senior business analyst.
Below are candidate application features extracted from a Requirements Knowledge Graph.
Each candidate has an ordered list of workflow steps, actors, and end states.

Your task: assign a concise, professional feature name and a one-sentence description
to each candidate. Also identify:
  - alternate_paths: list of brief descriptions of valid alternative flows (if any)
  - negative_paths:  list of brief descriptions of expected failure/rejection flows (if any)

IMPORTANT rules:
1. Use the actual step names — do NOT invent steps or business behaviour not shown.
2. Feature names must be short (1-4 words), meaningful business names.
3. Do NOT use generic names like "Feature 1" or "Unknown Flow".
4. If two candidates describe the same high-level feature (e.g. both are about login),
   name them differently by scope (e.g. "Login Flow" vs "Login Recovery").
5. Alternate and negative paths may be empty lists if not supported by the steps shown.

Candidates:
{json.dumps(summaries, indent=2)}

Return ONLY a valid JSON array with one object per candidate, in the same order.
Each object must have:
  "index":          int   (same index as the input)
  "name":           str   (concise feature name)
  "description":    str   (one sentence)
  "alternate_paths":  [str, ...]
  "negative_paths":   [str, ...]
  "confidence":     float 0.0–1.0

Example:
[
  {{
    "index": 0,
    "name": "User Login",
    "description": "Allows a registered user to authenticate and access the dashboard.",
    "alternate_paths": ["Login via SSO"],
    "negative_paths": ["Invalid credentials → error message"],
    "confidence": 0.92
  }}
]
"""

    result = call_ollama(prompt, timeout=120, num_ctx=6144,
                          model=EXTRACTION_MODEL, base_url=EXTRACTION_MODEL_URL)

    if result.get("error"):
        warnings.append(f"LLM feature enrichment failed: {result['error']}. "
                         "Using heuristic names instead.")
        for c in candidates:
            c["name"]        = _derive_name_heuristic(c, node_by_id)
            c["confidence"]  = 0.55
        return

    try:
        enrichments = extract_json_block(result["raw"])
        if not isinstance(enrichments, list):
            raise ValueError("LLM returned non-list")

        idx_map = {item.get("index"): item for item in enrichments
                   if isinstance(item, dict) and "index" in item}

        for i, c in enumerate(candidates):
            item = idx_map.get(i)
            if not item:
                c["name"]       = _derive_name_heuristic(c, node_by_id)
                c["confidence"] = 0.55
                continue
            c["name"]             = str(item.get("name") or "").strip() or \
                                     _derive_name_heuristic(c, node_by_id)
            c["description"]      = str(item.get("description") or "").strip()
            c["alternate_paths"]  = item.get("alternate_paths") or []
            c["negative_paths"]   = item.get("negative_paths") or []
            c["confidence"]       = float(item.get("confidence") or 0.7)

    except (ValueError, json.JSONDecodeError, TypeError) as e:
        warnings.append(f"LLM returned malformed feature enrichment JSON ({e}). "
                         "Using heuristic names.")
        for c in candidates:
            c["name"]       = _derive_name_heuristic(c, node_by_id)
            c["confidence"] = 0.55


# =============================================================================
# Supporting Context Collector
# =============================================================================


# Issue 4: Supporting context attachment excludes flow transitions and DEPENDS_ON.
# Structural, semantic, and metadata relation edges (USES, SUPPORTS, PART_OF, REALIZES, CONSTRAINS, TRACKS, etc.) are included.
_EXCLUDED_SUPPORT_EDGE_TYPES: frozenset = frozenset(
    set(FLOW_RELATIONS) | {"DEPENDS_ON", "NEXT_IN_DOCUMENT"}
)


def _collect_supporting_context(
    step_node_ids: List[str],
    node_by_id:    Dict[str, Dict],
    node_cls:      Dict[str, Dict],
    edge_cls:      Dict[Tuple, Dict],
    relationships: List[Dict],
    hops:          int = 1,
) -> Dict[str, Any]:
    """
    Collect non-workflow neighbours of the feature's workflow steps, reachable
    only via non-procedural edges (excluding FLOW_RELATIONS and DEPENDS_ON).

    Returns node/edge IDs for supporting context, business rules, and requirements.
    """
    step_set = set(step_node_ids)
    support_node_ids:  Set[str] = set()
    support_edge_ids:  List[Tuple] = []
    business_rule_ids: Set[str] = set()
    requirement_ids:   Set[str] = set()

    # Build adjacency restricted to non-flow, non-DEPENDS_ON edge types
    structural_adj: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for rel in relationships:
        rtype = rel.get("type", "")
        if rtype in _EXCLUDED_SUPPORT_EDGE_TYPES:
            continue
        frm = rel.get("from", "")
        to  = rel.get("to", "")
        if frm and to:
            structural_adj[frm].append((to, frm, rtype))
            structural_adj[to].append((frm, frm, rtype))   # undirected walk

    frontier = set(step_node_ids)
    for _ in range(hops):
        next_frontier: Set[str] = set()
        for nid in frontier:
            for (nb, orig_from, rtype) in structural_adj.get(nid, []):
                if nb in step_set or nb in support_node_ids or nb in frontier:
                    continue
                role = node_cls.get(nb, {}).get("role", "other")
                if role == "artifact":
                    continue
                support_node_ids.add(nb)
                next_frontier.add(nb)
                edge_key = (orig_from, nb, rtype) if orig_from != nb else (nid, nb, rtype)
                support_edge_ids.append(edge_key)
                if role == "business_rule":
                    business_rule_ids.add(nb)
                elif role == "requirement":
                    requirement_ids.add(nb)
        frontier = next_frontier

    return {
        "node_ids":          list(support_node_ids),
        "edge_ids":          support_edge_ids,
        "business_rule_ids": list(business_rule_ids),
        "requirement_ids":   list(requirement_ids),
    }



# =============================================================================
# Helpers
# =============================================================================

def _derive_name_heuristic(candidate: Dict, node_by_id: Dict) -> str:
    """
    Deterministic feature name when LLM is unavailable or failed.
    Uses the entry node name + the first distinct action step after it.
    """
    steps = candidate.get("ordered_steps", [])
    if not steps:
        entry_id = candidate.get("entry_point_id") or ""
        return (node_by_id.get(entry_id) or {}).get("name") or "Unknown Feature"
    if len(steps) == 1:
        return steps[0]
    # "Login → Dashboard" style — cap at 2 key nodes
    entry_name = steps[0]
    second_name = steps[1] if len(steps) > 1 else ""
    if second_name and second_name != entry_name:
        return f"{entry_name} → {second_name}"
    return entry_name


def _stable_feature_id(name: str, index: int) -> str:
    """
    URL-safe, collision-resistant feature ID derived from name + index.
    e.g. "User Login" → "feature_user_login_3a2f"
    """
    slug = (name or f"feature_{index}").lower().replace(" ", "_")
    slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
    slug = slug[:30].rstrip("_")
    suffix = hashlib.sha256(f"{name}:{index}".encode()).hexdigest()[:4]
    return f"{slug}_{suffix}"


def _build_extractor_cache_key(classifier_cache_key: str) -> str:
    """Cache key for the full extraction result."""
    raw = f"{classifier_cache_key}:{_EXTRACTOR_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# =============================================================================
# Convenience: get a feature by ID
# =============================================================================

def get_feature_by_id(
    feature_result: Dict[str, Any],
    feature_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve one feature from the extract_features() result by its ID."""
    for f in feature_result.get("features", []):
        if f.get("feature_id") == feature_id:
            return f
    return None


def get_feature_node_ids(feature: Dict[str, Any]) -> Set[str]:
    """All node IDs relevant to a feature (workflow + supporting)."""
    return set(feature.get("step_node_ids", [])) | set(feature.get("supporting_node_ids", []))


def get_feature_subgraph(
    feature: Dict[str, Any],
    nodes: List[Dict],
    relationships: List[Dict],
    include_supporting: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter the full node/relationship lists to only those relevant
    to the given feature.  Used by the UI to focus the graph.

    include_supporting=True  → includes 1-hop support context
    include_supporting=False → workflow-only (primary steps)
    """
    if include_supporting:
        relevant_ids = get_feature_node_ids(feature)
    else:
        relevant_ids = set(feature.get("step_node_ids", []))

    filtered_nodes = [n for n in nodes if n.get("id") in relevant_ids]
    filtered_rels  = [
        r for r in relationships
        if r.get("from") in relevant_ids and r.get("to") in relevant_ids
    ]
    return filtered_nodes, filtered_rels
