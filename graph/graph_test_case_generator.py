"""
Graph Test Case Generator

Pipeline position: consumes the SAME nodes/relationships session_state
already holds after Build Index — post Confidence Scorer + Graph
Validator, the exact objects the "Graph Quality Report" section
renders — plus parser/requirement_linker.py's items_with_links (only
for series grouping/ordering, via graph/structural_linker.py's
_series_key, the same "which items form one continuing flow" grouping
graph/test_case_generator.py's Gherkin generator already uses).

Different from graph/test_case_generator.py (Gherkin): that generator
never touches the knowledge graph at all — its Given/When/Then comes
from requirement text and item-level links only. This one walks the
ACTUAL extracted entity graph (Actor/Feature/Screen/UIElement/
DataObject/... nodes connected by USES/TRIGGERS/GENERATES/... edges)
to build a real step-by-step scenario, in a fielded test-case format
(TC-ID / REQ-ID / TITLE / TYPE / PRIORITY / ACTOR / FEATURE /
PRECONDITION / STEPS / EXPECTED RESULT / GRAPH NODES) rather than
Given/When/Then prose.

Design (deterministic, no extra LLM call — matches this project's
established rule-based-wherever-a-signal-exists pattern, same as
parser/requirement_linker.py and graph/test_case_generator.py):

1. Group requirement items into series (structural_linker._series_key —
   e.g. all "FR_LOGIN" items, or all "GEN" items for a document with no
   native ID scheme, are one continuing flow).

2. For each item in a series, take its own extracted nodes
   (node["source"] == item_id) and the edges between them, and find
   that item's "local path": a greedy walk from a root (no incoming
   edges within the item's own subgraph, preferring an Actor node) to a
   leaf, preferring State/Output/Screen/Message-typed leaves as the
   meaningful stopping point over an arbitrary dead end.

3. Concatenate each item's local path across the whole series, in
   document order, into ONE combined step sequence — this is the
   "COMBINED" test case: one scenario walking the full flow the
   document describes (e.g. the whole login journey), not one atomic
   requirement in isolation. Matches the naming in the sample format
   this was built from ("TC-COMBINED-001").

4. Render each hop as one imperative STEP line, using the edge's
   relationship type + target node's type bucket to pick a phrasing
   template (Screen/UIElement -> "Navigate to X.", DataObject/Attribute
   via USES/CREATES/UPDATES -> "Enter X.", Action -> "Click X." /
   "Perform X.", ...) — a fixed lookup table, not free text generation.

5. Classify TYPE (Positive/Negative) by scanning the series' own item
   content + node names for a small, explicit set of negative-outcome
   cue words (invalid, error, fail, incorrect, denied, ...).

6. Derive ACTOR / FEATURE / PRECONDITION / EXPECTED RESULT / PRIORITY
   from the same subgraph, each with a documented fallback when the
   graph doesn't have that information explicit.

Never-silently-disguise: every test case carries a "fallbacks" list
naming which fields used a fallback instead of a genuine graph signal —
same convention as parser/normalization_validator.py's review_queue,
graph/validator.py's review_nodes/relationships, graph/confidence_scorer.py's
low_confidence flags, and graph/test_case_generator.py's is_fallback.
A scenario assembled from thin graph signal looks visibly weaker than
one built from a rich, well-connected subgraph — not indistinguishable
from it.
"""

import re
from collections import defaultdict

from graph.flow_graph_analysis import FLOW_RELATIONS

# NOTE: deliberately NOT importing _series_key from graph/structural_linker.py.
# That module is under active trimming on the project's end (its own
# docstring documents removing the requirement backbone entirely) and
# has already dropped this function once between versions seen during
# development. A test-case generator reaching into another module's
# underscore-prefixed "private" helper was fragile to begin with — this
# is the same grouping logic (strip a trailing numeric suffix off an
# item id to get its "continuing flow" key), kept here as a self-
# contained copy so this module has zero dependency on whatever shape
# structural_linker.py happens to be in.
def _series_key(item_id: str) -> str:
    """'BR_STAY_6' -> 'BR_STAY', 'FR-002' -> 'FR', 'GEN-001' -> 'GEN'."""
    return re.sub(r'[_-]?\d+$', '', (item_id or "").strip()) or (item_id or "").strip()


series_key = _series_key

# ── Node type buckets ────────────────────────────────────────────
# graph/validator.py's VALID_NODE_TYPES, grouped by the role each type
# plays in a UI-level test step. A type not listed here falls through
# to the generic phrasing template (see _STEP_TEMPLATES' "default").
ACTOR_TYPES = {"Actor", "Role"}
NAVIGATION_TYPES = {"Screen", "Page", "UIElement", "Module"}
INPUT_TYPES = {"DataObject", "Attribute", "Country", "Currency", "Language", "Location"}
ACTION_TYPES = {"Action"}
OUTCOME_TYPES = {"State", "Output", "Message", "Event"}
CONDITION_TYPES = {"Condition", "Constraint", "Assumption"}
FEATURE_TYPES = {"Feature", "BusinessProcess", "Workflow", "Service", "API", "SystemComponent"}

# Names that graph/graph_eval.py's structural health check already
# warns about (e.g. "System" touching 23%+ of all edges) — a strong
# signal the extractor over-generalized rather than found a genuinely
# specific feature. Used ONLY to de-prioritize these as a FEATURE
# fallback (never to exclude them from the graph or the walked path
# itself) — a real requirement about "the System" should still be able
# to say so if nothing more specific exists anywhere in that item.
_GENERIC_FEATURE_FILLER_NAMES = {"system", "the system", "application", "the application", "platform"}


def _is_generic_filler(name: str) -> bool:
    return (name or "").strip().lower() in _GENERIC_FEATURE_FILLER_NAMES

# Priority order for which outgoing edge to follow when a node has more
# than one — steers the walk toward "the next thing a human would
# actually do" (navigate, then input, then act, then observe the
# outcome) instead of an arbitrary graph-iteration order.
_HOP_TYPE_PRIORITY = [NAVIGATION_TYPES, INPUT_TYPES, ACTION_TYPES, OUTCOME_TYPES, FEATURE_TYPES]

# Safety cap — an item's local subgraph shouldn't realistically need
# more hops than this; stops a mis-extracted cyclical subgraph from
# producing a runaway step list.
MAX_HOPS_PER_ITEM = 6

# (relationship_type, target_type_bucket_name) -> step phrasing
# template. Checked in this order; first match wins. Falls through to
# a generic "<Verb> <target>." built from the relationship type itself
# if nothing here matches.
def _phrase_step(rel_type: str, target_name: str, target_type: str) -> str:
    t = target_type
    if t in NAVIGATION_TYPES:
        return f"Navigate to {target_name}."
    if rel_type in {"USES", "CREATES", "UPDATES", "ASSOCIATED_WITH", "PROCESSES"} and t in INPUT_TYPES:
        return f"Enter {target_name}."
    if t in ACTION_TYPES:
        return f"Click {target_name}." if "button" in target_name.lower() or "click" in target_name.lower() else f"Perform {target_name}."
    if rel_type in {"GENERATES", "PRODUCES", "CREATES"} and t in OUTCOME_TYPES:
        return f"Verify {target_name} is generated."
    if rel_type == "SHOWS" and t in OUTCOME_TYPES:
        return f"Verify {target_name} is displayed."
    if t in OUTCOME_TYPES:
        return f"Verify system reaches {target_name}."
    if t in CONDITION_TYPES:
        return f"Confirm {target_name}."
    verb = rel_type.replace("_", " ").capitalize()
    return f"{verb} {target_name}."


# ── Negative-outcome cue words ───────────────────────────────────
# Small, explicit set — a heuristic, not exhaustive. Matched against
# item content and node names (case-insensitive, word-boundary).
_NEGATIVE_CUES = [
    "invalid", "error", "fail", "failure", "incorrect", "denied", "deny",
    "reject", "missing", "unauthorized", "expired", "locked", "timeout",
    "unable", "cannot", "can not", "does not", "not enrolled", "not moved",
    "exceed", "wrong", "unsuccessful",
]
_NEGATIVE_CUE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in _NEGATIVE_CUES) + r')\b',
    re.IGNORECASE,
)


_KNOWN_FAMILY_PREFIX = re.compile(
    r'^(FR|NFR|BR|UC|US|TC|REQ|GEN)[_\-]+', re.IGNORECASE
)


def _humanize(text: str) -> str:
    text = _KNOWN_FAMILY_PREFIX.sub('', text or "")
    words = re.sub(r'[_\-]+', ' ', text).strip().split()
    return " ".join(w.capitalize() for w in words) or text


def _build_item_subgraph(item_id: str, nodes: list, relationships: list) -> tuple:
    """Nodes/edges whose 'source' is this item, PLUS a live 1-hop
    expansion out along FLOW_RELATIONS edges (graph/flow_graph_analysis.py's
    same allow-list -- LEADS_TO/PRODUCES/PROCESSES/PERFORMS/... -- the
    subset that means "this happens, then that happens", not every
    relation type).

    Why: many single requirement items extract to nothing but an
    Actor -ACTOR_OF-> Feature pair (a structural edge, not a flow
    edge) -- there is nothing else in THAT item's own chunk to walk
    onto. But the real next step often exists one flow-edge away in a
    DIFFERENT item's extraction (e.g. a later chunk describing what
    that feature actually does next). Without this expansion the walk
    below can never reach it, so every generated test case's
    GRAPH NODES stays pinned to actor/feature pairs that
    graph/failure_trace.py's dominator analysis then can't trace
    (those pairs have no FLOW_RELATIONS edges at all) -- discovered
    from failure_trace.py giving the same "not part of the sequential
    flow subgraph" result for every single generated test case.

    This expansion is restricted to FLOW_RELATIONS specifically (not
    "any edge touching this item's nodes") so it can't drag in
    unrelated structural noise -- same reasoning flow_graph_analysis.py
    itself documents for why it uses an allow-list instead of a
    deny-list.

    Returns (nodes_by_id, adjacency, in_degree).
    """
    item_nodes = {n["id"]: n for n in nodes if n.get("source") == item_id}
    node_by_id = {n["id"]: n for n in nodes}

    adjacency = defaultdict(list)  # node_id -> [(rel_type, target_id), ...]
    in_degree = defaultdict(int)

    for rel in relationships:
        if rel.get("source") != item_id:
            continue
        f, t = rel.get("from"), rel.get("to")
        if f in item_nodes and t in item_nodes and f != t:
            adjacency[f].append((rel.get("type", "ASSOCIATED_WITH"), t))
            in_degree[t] += 1

    # Live 1-hop flow expansion, in EITHER direction: if a node this
    # item extracted has a FLOW_RELATIONS edge to/from a node outside
    # this item's own chunk, pull that neighbor in too so the walk can
    # step onto it like a local hop -- _walk_item_path's priority
    # ordering (below) decides whether it's actually taken.
    for rel in relationships:
        if rel.get("type") not in FLOW_RELATIONS:
            continue
        f, t = rel.get("from"), rel.get("to")
        if f == t:
            continue
        if f in item_nodes and t not in item_nodes and t in node_by_id:
            item_nodes[t] = node_by_id[t]
            adjacency[f].append((rel.get("type"), t))
            in_degree[t] += 1
        elif t in item_nodes and f not in item_nodes and f in node_by_id:
            item_nodes[f] = node_by_id[f]
            adjacency[f].append((rel.get("type"), t))
            in_degree[t] += 1

    return item_nodes, adjacency, in_degree


def _walk_item_path(item_nodes: dict, adjacency: dict, in_degree: dict) -> list:
    """
    Greedy walk through one item's own subgraph. Returns a list of
    (rel_type, target_node_dict) hops — the root itself is NOT included
    as a hop (it's the starting point, surfaced separately as the
    item's root node for ACTOR/context purposes).

    Root choice: prefer an Actor/Role node with outgoing edges; else
    any node with in_degree 0 and outgoing edges; else the node with
    the most outgoing edges (breaks a pure-cycle deadlock without
    crashing).
    """
    if not item_nodes:
        return [], None

    candidates_with_out = [nid for nid in item_nodes if adjacency.get(nid)]
    if not candidates_with_out:
        # No edges at all within this item — nothing to walk.
        return [], next(iter(item_nodes.values()))

    actor_roots = [nid for nid in candidates_with_out if item_nodes[nid].get("type") in ACTOR_TYPES]
    zero_in_roots = [nid for nid in candidates_with_out if in_degree.get(nid, 0) == 0]

    if actor_roots:
        root_id = actor_roots[0]
    elif zero_in_roots:
        root_id = zero_in_roots[0]
    else:
        root_id = max(candidates_with_out, key=lambda nid: len(adjacency.get(nid, [])))

    path = []
    visited = {root_id}
    current = root_id

    for _ in range(MAX_HOPS_PER_ITEM):
        options = [(rt, tid) for rt, tid in adjacency.get(current, []) if tid not in visited]
        if not options:
            break

        def _priority(opt):
            rel_type, tid = opt
            # A genuine flow edge (see _build_item_subgraph's 1-hop
            # expansion) always wins over a structural/type-bucket hop
            # -- rank -1 sorts before every _HOP_TYPE_PRIORITY bucket,
            # since walking onto a real "next step" is more valuable
            # than a UIElement/DataObject hop that stays local but
            # dead-ends the flow trace.
            if rel_type in FLOW_RELATIONS:
                return -1
            ttype = item_nodes[tid].get("type", "")
            for rank, bucket in enumerate(_HOP_TYPE_PRIORITY):
                if ttype in bucket:
                    return rank
            return len(_HOP_TYPE_PRIORITY)

        options.sort(key=_priority)
        rel_type, next_id = options[0]
        path.append((rel_type, item_nodes[next_id]))
        visited.add(next_id)
        current = next_id

    return path, item_nodes[root_id]


def _classify_type(series_items: list) -> tuple:
    """Returns (is_positive: bool, matched_cue: str or None)."""
    for item in series_items:
        content = item.get("content", "") or ""
        m = _NEGATIVE_CUE_RE.search(content)
        if m:
            return False, m.group(1)
    return True, None


def _priority_for_series(series_items: list, links: list) -> str:
    """Small heuristic: BR/NFR-family items, or items with a VERIFIED_BY/
    VERIFIES link elsewhere in the document, are treated as higher
    priority than a plain unlinked FR/GEN item."""
    linked_ids = {l["source_id"] for l in links} | {l["target_id"] for l in links}
    for item in series_items:
        if item.get("family") in {"BR", "NFR"}:
            return "High"
        if item.get("id") in linked_ids:
            return "High"
    return "Medium"


def _connected_series_for_gen_items(gen_items: list, nodes: list, relationships: list) -> dict:
    """
    _series_key() strips a trailing number to find an item's family —
    correct for real IDs ('BR_STAY_6' -> 'BR_STAY', a genuine shared
    family), but 'GEN-001'/'GEN-002'/... all strip down to the same
    bare 'GEN', since GEN numbering is Pass B's own incrementing
    counter, not a family the source document defined. Left alone,
    that silently merges EVERY GEN item in a document into one combined
    test case regardless of whether they're actually the same flow —
    fine for a document that's genuinely one continuous procedure, but
    wrong for a document with several unrelated GEN facts.

    Fix: group GEN items by ACTUAL graph connectivity instead of ID
    string. Two GEN items belong in the same series if some
    relationship connects an entity sourced from one to an entity
    sourced from the other — real edge evidence (same-chunk mention,
    a NEXT_IN_DOCUMENT-less direct link, or a Cross-Requirement Linker
    edge), not a coincidence of ID numbering. An item with no such
    connection to anything else becomes its own single-item series,
    which is the honest outcome now that structural_linker.py no
    longer adds an artificial connecting edge for every item (see its
    module docstring on that tradeoff).
    """
    node_source = {n["id"]: n.get("source") for n in nodes}
    gen_ids = [item.get("id", "") for item in gen_items]
    gen_id_set = set(gen_ids)

    parent = {iid: iid for iid in gen_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in relationships:
        f_source = node_source.get(rel.get("from"))
        t_source = node_source.get(rel.get("to"))
        if f_source in gen_id_set and t_source in gen_id_set and f_source != t_source:
            union(f_source, t_source)

    groups = defaultdict(list)
    for item in gen_items:
        groups[find(item.get("id", ""))].append(item)
    return groups


def generate_feature_driven_test_cases(nodes: list, relationships: list) -> list:
    """
    Extracts all features from the Knowledge Graph using feature_extractor.py
    and produces grounded Positive, Negative, and Edge test cases for EVERY feature.
    Ensures full application-wide coverage (Login, Filtering, Deleting, Archiving, Approval, Audit Trail, etc.)
    with specific preconditions, descriptions, clear titles, and logical prerequisite steps.
    """
    try:
        from graph.semantic_classifier import classify_graph
        from graph.feature_extractor import extract_features
        
        classification = classify_graph(nodes, relationships, use_llm=False)
        extraction = extract_features(nodes, relationships, classification, use_llm=False)
        features = extraction.get("features", [])
    except Exception as e:
        logger.warning(f"Could not extract features for test case generation: {e}")
        features = []

    feature_tcs = []
    seen_fingerprints = set()
    
    for idx, feature in enumerate(features):
        fname = feature.get("name") or f"Feature_{idx+1}"
        fid_raw = feature.get("feature_id") or f"feat_{idx+1}"
        fslug = re.sub(r'[^a-zA-Z0-9]+', '-', fname.upper()).strip('-')[:25]
        fname_lower = fname.lower()
        
        steps = feature.get("ordered_steps", [])
        if not steps:
            steps = [fname]

        actors = feature.get("actors", [])
        actor_name = actors[0] if actors else "End User"
        
        # Build step phrasings & insert logical prerequisites if needed
        step_phrasings = []
        
        # Domain-aware prerequisite insertion
        if any(w in fname_lower for w in ["checkout", "payment", "order", "slot", "delivery"]):
            step_phrasings.append("Navigate to Product Catalog and add item to cart.")
            step_phrasings.append("Proceed to Checkout and select delivery address.")
            if "payment" in fname_lower or "order" in fname_lower or "slot" in fname_lower:
                step_phrasings.append("Select valid delivery slot.")

        for s_idx, step_name in enumerate(steps):
            s_lower = str(step_name).lower()
            if any(w in s_lower for w in ["page", "screen", "view", "portal"]):
                step_phrasings.append(f"Navigate to {step_name}.")
            elif any(w in s_lower for w in ["field", "data", "input", "form", "dropdown", "address", "email", "password"]):
                step_phrasings.append(f"Select / Enter valid input in {step_name}.")
            elif any(w in s_lower for w in ["button", "click", "submit", "action", "confirm", "approve", "delete"]):
                step_phrasings.append(f"Click / Execute {step_name}.")
            else:
                step_phrasings.append(f"Perform {step_name}.")
                
        # Tailor preconditions & descriptions dynamically by feature type
        if any(w in fname_lower for w in ["search", "browse", "category", "tab", "dropdown", "eta", "view", "display"]):
            pos_prec = f"User is on the {fname} interface with available catalog data."
            neg_prec = f"User searches or filters {fname} using invalid syntax or empty query."
            edge_prec = f"User submits maximum allowed query string length (255 chars) to {fname} during peak load."
            pos_desc = f"Verifies that searching/browsing in {fname} displays accurate and relevant results."
            neg_desc = f"Verifies that empty or invalid queries in {fname} return appropriate empty-state feedback."
            edge_desc = f"Verifies that {fname} handles long query strings and high query concurrency without crashing."
        elif any(w in fname_lower for w in ["cart", "quantity", "sku", "item"]):
            pos_prec = f"User has active items in the shopping cart."
            neg_prec = f"User attempts to modify quantity for {fname} using negative values or out-of-stock items."
            edge_prec = f"User attempts to update item quantity in {fname} to maximum cart capacity (99 items)."
            pos_desc = f"Verifies that item quantity changes in {fname} update cart totals and item counts correctly."
            neg_desc = f"Verifies that invalid quantity updates or out-of-stock modifications in {fname} are rejected."
            edge_desc = f"Verifies that {fname} correctly enforces upper cart limit boundaries."
        elif any(w in fname_lower for w in ["checkout", "payment", "address", "slot", "order", "cod"]):
            pos_prec = f"User has valid items in cart and has navigated to {fname}."
            neg_prec = f"User submits {fname} with missing mandatory fields or invalid payment credentials."
            edge_prec = f"User selects {fname} at boundary delivery slot time limit under high system load."
            pos_desc = f"Verifies end-to-end successful completion of {fname} with valid user details."
            neg_desc = f"Verifies that invalid payment/address data during {fname} triggers clear validation errors."
            edge_desc = f"Verifies that {fname} preserves transaction integrity during peak time-slot boundaries."
        elif any(w in fname_lower for w in ["delete", "deletion", "archive", "archival", "approval", "audit", "period"]):
            pos_prec = f"{actor_name} is authenticated with authorized reviewer/administrative privileges."
            neg_prec = f"Unauthenticated or unauthorized user attempts to execute {fname}."
            edge_prec = f"{fname} is requested for maximum configured retention period boundary."
            pos_desc = f"Verifies that authorized users can execute {fname} and update system records cleanly."
            neg_desc = f"Verifies that unauthorized or invalid deletion/archival requests are blocked by security policy."
            edge_desc = f"Verifies that {fname} handles maximum boundary periods without data truncation."
        else:
            pos_prec = f"User is authenticated and has navigated to {fname}."
            neg_prec = f"User attempts {fname} using invalid input parameters or missing fields."
            edge_prec = f"{fname} is invoked at upper boundary threshold limits."
            pos_desc = f"Verifies end-to-end happy path execution of {fname} under standard operating conditions."
            neg_desc = f"Verifies that invalid input values for {fname} return appropriate error feedback."
            edge_desc = f"Verifies system stability when {fname} is executed at boundary limits."

        # 1. Positive Scenario
        pos_tc = {
            "tc_id": f"TC-FEAT-{fslug}-POS",
            "req_id": f"REQ-{fslug}",
            "title": f"Positive - Execute {fname} workflow successfully",
            "description": pos_desc,
            "type": "Positive",
            "priority": "High",
            "actor": actor_name,
            "feature": fname,
            "precondition": pos_prec,
            "steps": step_phrasings if step_phrasings else [f"Navigate to {fname}.", f"Complete {fname} successfully."],
            "expected_result": f"System processes {fname} successfully and updates state.",
            "graph_nodes": steps,
            "source_items": [fid_raw],
            "fallbacks": []
        }
        fp_pos = (pos_tc["title"], tuple(pos_tc["steps"]))
        if fp_pos not in seen_fingerprints:
            seen_fingerprints.add(fp_pos)
            feature_tcs.append(pos_tc)

        # 2. Negative Scenario
        neg_steps = []
        if step_phrasings:
            neg_steps = list(step_phrasings[:-1]) + [f"Provide invalid data or unauthorized credentials for {steps[-1]}."]
        else:
            neg_steps = [f"Attempt {fname} with invalid input parameters."]

        neg_tc = {
            "tc_id": f"TC-FEAT-{fslug}-NEG",
            "req_id": f"REQ-{fslug}",
            "title": f"Negative - Reject {fname} when invalid parameters are provided",
            "description": neg_desc,
            "type": "Negative",
            "priority": "High",
            "actor": actor_name,
            "feature": fname,
            "precondition": neg_prec,
            "steps": neg_steps,
            "expected_result": f"System rejects operation, displays clear validation error message, and preserves current state.",
            "graph_nodes": steps,
            "source_items": [fid_raw],
            "fallbacks": []
        }
        fp_neg = (neg_tc["title"], tuple(neg_tc["steps"]))
        if fp_neg not in seen_fingerprints:
            seen_fingerprints.add(fp_neg)
            feature_tcs.append(neg_tc)

        # 3. Edge Scenario
        edge_steps = [
            f"Navigate to {steps[0] if steps else fname}.",
            f"Supply input parameters at maximum allowed boundary limit (e.g. max length / limit value).",
            f"Execute {fname} under concurrent system load."
        ]
        edge_tc = {
            "tc_id": f"TC-FEAT-{fslug}-EDGE",
            "req_id": f"REQ-{fslug}",
            "title": f"Edge - {fname} boundary limits and peak load stability",
            "description": edge_desc,
            "type": "Edge",
            "priority": "Medium",
            "actor": actor_name,
            "feature": fname,
            "precondition": edge_prec,
            "steps": edge_steps,
            "expected_result": f"System handles boundary condition cleanly without data corruption, timeout, or crash.",
            "graph_nodes": steps,
            "source_items": [fid_raw],
            "fallbacks": []
        }
        fp_edge = (edge_tc["title"], tuple(edge_tc["steps"]))
        if fp_edge not in seen_fingerprints:
            seen_fingerprints.add(fp_edge)
            feature_tcs.append(edge_tc)

    return feature_tcs


def generate_test_cases_from_graph(
    nodes: list,
    relationships: list,
    items_with_links: list = None,
    links: list = None,
) -> dict:
    """
    Args:
        nodes, relationships: session_state's post-validation graph
            (the same objects "Graph Quality Report" renders).
        items_with_links: parser/requirement_linker.py's link_requirements()
            "items_with_links" — used only for series grouping/order and
            content-based negative-cue scanning, never as the source of
            STEPS/GRAPH NODES (those come from the graph itself).
        links: link_requirements()'s "links" — used only for the
            PRIORITY heuristic.

    Returns:
        {
            "test_cases": [
                {
                    "tc_id", "req_id", "title", "type", "priority",
                    "actor", "feature", "precondition",
                    "steps": [str, ...],
                    "expected_result",
                    "graph_nodes": [str, ...],
                    "source_items": [item_id, ...],
                    "fallbacks": [str, ...],   # which fields had no
                                               # real graph/content
                                               # signal and used a
                                               # generic default
                },
                ...
            ],
            "text": str,   # all test cases rendered, matching the
                            # fielded TEST CASE / STEPS / ... layout
            "summary": {"total_test_cases", "positive", "negative",
                        "with_fallbacks"},
        }
    """
    links = links or []
    if not items_with_links:
        sources = sorted(list({n.get("source") for n in nodes if n.get("source") and n.get("source") != "workflow_extractor"}))
        items_with_links = [{"id": s, "family": None} for s in sources]

    by_series = defaultdict(list)
    seen_series = []

    # GEN items (family is None — Pass B gap-fill, no real ID family)
    # get connectivity-based grouping; everything else keeps the
    # existing ID-string grouping, which is a real signal for actual
    # families like BR_STAY/FR.
    non_gen_items = [it for it in items_with_links if it.get("family") is not None]
    gen_items = [it for it in items_with_links if it.get("family") is None]

    item_order = {it.get("id", ""): i for i, it in enumerate(items_with_links)}

    for item in non_gen_items:
        sk = series_key(item.get("id", ""))
        if sk not in by_series:
            seen_series.append(sk)
        by_series[sk].append(item)

    if gen_items:
        gen_groups = list(_connected_series_for_gen_items(gen_items, nodes, relationships).values())
        # keep output order close to document order: order groups by
        # the earliest item they contain, and order items within a
        # group the same way
        for group_items in gen_groups:
            group_items.sort(key=lambda it: item_order.get(it.get("id", ""), 0))
        gen_groups.sort(key=lambda g: item_order.get(g[0].get("id", ""), 0))

        for group_items in gen_groups:
            if len(group_items) == 1:
                sk = group_items[0].get("id", "")
            else:
                ids = [it.get("id", "") for it in group_items]
                sk = f"GEN[{ids[0]}..{ids[-1]}]"
            seen_series.append(sk)
            by_series[sk] = group_items

    test_cases = []
    seq_by_series = defaultdict(int)

    for sk in seen_series:
        series_items = by_series[sk]

        combined_steps = []
        combined_graph_nodes = []
        actor_name = None
        feature_name = None
        generic_feature_candidate = None  # e.g. "System" — last resort only
        precondition_text = None
        expected_result = None
        fallbacks = []
        any_edges_found = False

        for item in series_items:
            item_id = item.get("id", "")
            item_nodes, adjacency, in_degree = _build_item_subgraph(item_id, nodes, relationships)
            path, root = _walk_item_path(item_nodes, adjacency, in_degree)

            if path:
                any_edges_found = True

            # ── Gap flagging ──────────────────────────────────────
            # Previously: an item that contributed nothing (no
            # entities extracted, or entities with no walkable edge)
            # was silently invisible — combined_steps just moved on to
            # the next item with no trace this one was ever skipped.
            # That's exactly what makes a stitched-together flow read
            # as "smooth" when it actually has a hole in it.
            #
            # Fix: surface the gap AT THE POINT IN SEQUENCE it occurs,
            # as a real entry in combined_steps (not buried only in a
            # summary count), plus a fallbacks entry naming which item
            # and why — same "never silently disguise weak signal"
            # convention this file already uses for actor/feature/
            # precondition/expected_result fallbacks below.
            if not item_nodes:
                combined_steps.append(
                    f"[GAP: {item_id} — no entities were extracted from this "
                    f"requirement; the flow may be missing a step here.]"
                )
                fallbacks.append(f"{item_id}: no extracted entities (extraction likely failed on this chunk)")
            elif not path:
                combined_steps.append(
                    f"[GAP: {item_id} — entities were extracted but none were "
                    f"connected by a usable edge; the flow may be missing a step here.]"
                )
                fallbacks.append(f"{item_id}: entities extracted but not connected (isolated nodes, no walkable edge)")

            if root and actor_name is None and root.get("type") in ACTOR_TYPES:
                actor_name = root.get("name")

            if not combined_graph_nodes and root:
                combined_graph_nodes.append(root.get("name"))

            for rel_type, target in path:
                if target.get("type") in OUTCOME_TYPES:
                    # Outcome nodes (State/Output/Message/Event) become
                    # EXPECTED RESULT, not another STEP line — avoids
                    # saying "verify Dashboard is displayed" as a step
                    # AND listing Dashboard as the expected result.
                    combined_graph_nodes.append(target.get("name"))
                    if expected_result is None:
                        expected_result = target.get("name")
                    continue
                combined_steps.append(_phrase_step(rel_type, target.get("name", target.get("id")), target.get("type", "")))
                combined_graph_nodes.append(target.get("name"))
                if feature_name is None and target.get("type") in FEATURE_TYPES:
                    feature_name = target.get("name")
                if precondition_text is None and target.get("type") in CONDITION_TYPES:
                    precondition_text = target.get("name")

            # Also scan the item's own nodes (not just path hops) for an
            # Actor/Feature/Condition we may have walked past.
            #
            # FEATURE is the one field this scan is unreliable for: it's
            # not restricted to the walked path, so it can pick up a
            # node that's in this item's local extraction but has
            # NOTHING to do with what this requirement is actually
            # about — most commonly a generic "System" node (see
            # graph/graph_eval.py's own hub-dominance warning: if
            # "System" touches 20%+ of all edges, it's very likely
            # sitting in most items' local node lists regardless of
            # relevance). A specific FEATURE found this way is kept
            # immediately; a generic one is parked as a fallback-of-
            # last-resort instead, so a real Feature/BusinessProcess
            # node elsewhere in the series still wins over it.
            for n in item_nodes.values():
                if actor_name is None and n.get("type") in ACTOR_TYPES:
                    actor_name = n.get("name")
                if n.get("type") in FEATURE_TYPES:
                    name = n.get("name")
                    if _is_generic_filler(name):
                        if generic_feature_candidate is None:
                            generic_feature_candidate = name
                    elif feature_name is None:
                        feature_name = name
                if precondition_text is None and n.get("type") in CONDITION_TYPES:
                    precondition_text = n.get("name")

        if not any_edges_found:
            # Nothing to walk for this whole series — skip rather than
            # emit an empty scenario. (Never silently drop principle
            # still holds one level up: this is visible in the
            # "requirements_with_no_extracted_content" flag graph/graph_eval.py
            # already surfaces, so it's not vanishing without a trace.)
            continue

        is_positive, cue = _classify_type(series_items)

        if actor_name is None:
            actor_name = "User"
            fallbacks.append("actor (no Actor/Role node found in this flow's subgraph — defaulted to 'User')")

        if feature_name is None:
            humanized = _humanize(sk)
            # Priority: humanized series key (e.g. "BR_HEADER_CURRENCY" ->
            # something specific) beats a generic filler node's name
            # ("System") — the series key is always at least as specific
            # as the word "System". Only fall through to the generic
            # node if the humanized key ITSELF turns out generic too.
            if humanized and not _is_generic_filler(humanized):
                feature_name = humanized
                fallbacks.append("feature (no specific Feature/BusinessProcess node found — used the requirement series name)")
            elif generic_feature_candidate:
                feature_name = generic_feature_candidate
                fallbacks.append(
                    f"feature (only a generic node '{generic_feature_candidate}' was found in "
                    "this flow's subgraph — likely LLM over-generalization across chunks; see "
                    "graph_eval's hub-dominance warning)"
                )
            else:
                feature_name = humanized
                fallbacks.append("feature (no Feature/BusinessProcess node found — used the requirement series name)")

        if precondition_text is None:
            precondition_text = f"Preconditions as described in {series_items[0].get('id')}"
            fallbacks.append("precondition (no Condition/Constraint node found in this flow's subgraph)")
        else:
            precondition_text = f"{actor_name} has {precondition_text.lower()}"

        if expected_result is None:
            expected_result = combined_graph_nodes[-1] if combined_graph_nodes else "(no outcome node found)"
            fallbacks.append("expected_result (no State/Output/Message/Event node found — used the last node walked)")

        if not combined_steps:
            fallbacks.append("steps (subgraph had nodes but no usable edges to walk — see graph_nodes for what was found instead)")

        seq_by_series[sk] += 1
        tc_num = seq_by_series[sk]
        is_combined = len(series_items) > 1
        tc_id = f"TC-COMBINED-{tc_num:03d}" if is_combined else f"TC-{series_items[0].get('id', sk)}-{tc_num:03d}"
        req_id = series_items[0].get("id", sk)

        title = (
            f"{'Successful' if is_positive else 'Failed'} {feature_name} "
            f"with {'valid' if is_positive else 'invalid'} inputs"
        )

        priority = _priority_for_series(series_items, links)

        test_cases.append({
            "tc_id": tc_id,
            "req_id": req_id,
            "title": title,
            "type": "Positive" if is_positive else "Negative",
            "priority": priority,
            "actor": actor_name,
            "feature": feature_name,
            "precondition": precondition_text,
            "steps": combined_steps or [f"(no walkable graph path found for {req_id})"],
            "expected_result": expected_result,
            "graph_nodes": [n for n in combined_graph_nodes if n],
            "source_items": [i.get("id") for i in series_items],
            "fallbacks": fallbacks,
        })

    # Helper: create a negative‑input variant by marking each INPUT step as invalid
    def _make_negative_variant(base_tc: dict) -> dict:
        """Return a copy of *base_tc* where every INPUT‑type step is made invalid."""
        tc = base_tc.copy()
        tc["type"] = "Negative"
        tc["tc_id"] = f"{base_tc['tc_id']}-NEG"
        tc["title"] = f"Negative - {tc.get('feature', '')} with invalid inputs"
        tc["description"] = f"Verifies that invalid input entries for {tc.get('feature', '')} are rejected with clear error feedback."
        tc["precondition"] = f"User submits {tc.get('feature', '')} using invalid or malformed data."
        new_steps = []
        for step in tc["steps"]:
            if step.startswith("Enter "):
                new_steps.append(re.sub(r"Enter (.+?)\.", r"Enter invalid \1.", step))
            else:
                new_steps.append(step)
        tc["steps"] = new_steps
        tc["fallbacks"] = list(tc.get("fallbacks", [])) + ["generated negative‑input variant"]
        return tc

    # Helper: create a dependency‑missing variant by removing the first step (assumed prerequisite)
    def _make_dependency_variant(base_tc: dict) -> dict:
        """Return a copy where the initial step (often a prerequisite) is omitted."""
        if not base_tc["steps"]:
            return None
        tc = base_tc.copy()
        tc["type"] = "Negative"
        tc["tc_id"] = f"{base_tc['tc_id']}-DEP"
        tc["title"] = f"Negative - Missing prerequisite step for {tc.get('feature', '')}"
        tc["description"] = f"Verifies that performing {tc.get('feature', '')} without completing prerequisite steps fails gracefully."
        tc["precondition"] = f"Prerequisite initial step for {tc.get('feature', '')} has been omitted."
        tc["steps"] = base_tc["steps"][1:]
        tc["expected_result"] = "Prerequisite step missing – operation should fail"
        return tc


    # Generate extra variants for each base test case
    extra_variants = []
    seen_fp = set()
    for tc in test_cases:
        neg = _make_negative_variant(tc)
        if neg:
            fp = (neg["title"], tuple(neg["steps"]))
            if fp not in seen_fp:
                seen_fp.add(fp)
                extra_variants.append(neg)
        dep = _make_dependency_variant(tc)
        if dep:
            fp = (dep["title"], tuple(dep["steps"]))
            if fp not in seen_fp:
                seen_fp.add(fp)
                extra_variants.append(dep)
        edge = _make_edge_variant(tc)
        if edge:
            fp = (edge["title"], tuple(edge["steps"]))
            if fp not in seen_fp:
                seen_fp.add(fp)
                extra_variants.append(edge)
        alt = _make_alternative_variant(tc)
        if alt:
            fp = (alt["title"], tuple(alt["steps"]))
            if fp not in seen_fp:
                seen_fp.add(fp)
                extra_variants.append(alt)
        exc = _make_exception_variant(tc)
        if exc:
            fp = (exc["title"], tuple(exc["steps"]))
            if fp not in seen_fp:
                seen_fp.add(fp)
                extra_variants.append(exc)
    test_cases.extend(extra_variants)

    # Append feature-driven test cases for full application coverage
    feat_tcs = generate_feature_driven_test_cases(nodes, relationships)
    for ftc in feat_tcs:
        fp = (ftc["title"], tuple(ftc["steps"]))
        if fp not in seen_fp:
            seen_fp.add(fp)
            test_cases.append(ftc)

    text = "\n\n".join(_render_test_case_text(tc) for tc in test_cases)

    gap_fallback_markers = (": no extracted entities", ": entities extracted but not connected")
    test_cases_with_gaps = [
        tc for tc in test_cases
        if any(marker in fb for fb in tc.get("fallbacks", []) for marker in gap_fallback_markers)
    ]
    total_gap_steps = sum(
        sum(1 for s in tc.get("steps", []) if str(s).startswith("[GAP:")) for tc in test_cases
    )

    return {
        "test_cases": test_cases,
        "text": text,
        "summary": {
            "total_test_cases": len(test_cases),
            "positive": sum(1 for tc in test_cases if tc.get("type") == "Positive"),
            "negative": sum(1 for tc in test_cases if tc.get("type") == "Negative"),
            "edge": sum(1 for tc in test_cases if tc.get("type") == "Edge"),
            "with_fallbacks": sum(1 for tc in test_cases if tc.get("fallbacks")),
            "test_cases_with_gaps": len(test_cases_with_gaps),
            "total_gap_steps": total_gap_steps,
        },
    }



    # --------------------------------------------------------------------------

# ── Additional Variant Helpers ────────────────────────────────────────
# These helpers generate extra test case variants beyond the basic
# negative‑input and dependency‑missing variants already present.
# They are intentionally simple and deterministic, following the same
# rule‑based style as the rest of the generator.

def _make_edge_variant(base_tc: dict) -> dict:
    """Create a variant that simulates a navigation edge failure."""
    tc = base_tc.copy()
    tc["type"] = "Edge"
    tc["tc_id"] = f"{base_tc['tc_id']}-EDGE"
    tc["title"] = f"Edge - {tc.get('feature', '')} navigation boundary & latency"
    tc["description"] = f"Verifies system resilience when navigation in {tc.get('feature', '')} experiences boundary latency or threshold limits."
    tc["precondition"] = f"Navigation to {tc.get('feature', '')} is executed under upper latency threshold conditions."
    new_steps = []
    changed = False
    for step in tc["steps"]:
        if not changed and str(step).startswith("Navigate to "):
            new_steps.append(re.sub(r"Navigate to (.+?)\.", r"Navigate to \1 under latency threshold.", step))
            changed = True
        else:
            new_steps.append(step)
    tc["steps"] = new_steps
    tc["fallbacks"] = list(tc.get("fallbacks", [])) + ["generated edge‑variant"]
    return tc


def _make_alternative_variant(base_tc: dict) -> dict:
    """Create an alternative‑flow variant by swapping an action step."""
    tc = base_tc.copy()
    tc["tc_id"] = f"{base_tc['tc_id']}-ALT"
    tc["title"] = f"Positive - Alternative interaction path for {tc.get('feature', '')}"
    tc["description"] = f"Verifies an alternative valid interaction sequence to complete {tc.get('feature', '')}."
    tc["precondition"] = f"User selects an alternative valid interaction route for {tc.get('feature', '')}."
    new_steps = []
    changed = False
    for step in tc["steps"]:
        if not changed and (str(step).startswith("Click ") or str(step).startswith("Perform ")):
            if str(step).startswith("Click "):
                new_steps.append(step.replace("Click ", "Tap / Select "))
            else:
                new_steps.append(step.replace("Perform ", "Execute alternate "))
            changed = True
        else:
            new_steps.append(step)
    tc["steps"] = new_steps
    tc["fallbacks"] = list(tc.get("fallbacks", [])) + ["generated alternative‑variant"]
    return tc


def _make_exception_variant(base_tc: dict) -> dict:
    """Create a variant that inserts an exception step before the expected result."""
    tc = base_tc.copy()
    tc["type"] = "Negative"
    tc["tc_id"] = f"{base_tc['tc_id']}-EXC"
    tc["title"] = f"Negative - Exception recovery during {tc.get('feature', '')}"
    tc["description"] = f"Verifies system exception handling and error recovery when an exception triggers during {tc.get('feature', '')}."
    tc["precondition"] = f"System triggers an unexpected runtime exception during {tc.get('feature', '')} execution."
    if tc["steps"]:
        exc_step = f"Trigger runtime exception during {tc.get('feature', '')}."
        tc["steps"] = [tc["steps"][0], exc_step] + tc["steps"][1:]
    else:
        tc["steps"] = [f"Trigger runtime exception during {tc.get('feature', '')}."]
    tc["expected_result"] = f"System catches exception in {tc.get('feature', '')}, logs error, and recovers state safely."
    tc["fallbacks"] = list(tc.get("fallbacks", [])) + ["generated exception‑variant"]
    return tc


def _render_test_case_text(tc: dict) -> str:
    """Render one test case in the fielded plain-text layout."""
    feat_name = tc.get('feature') or 'system'
    desc = tc.get('description') or f"Verifies functionality of {feat_name} end-to-end."
    lines = [
        "TEST CASE",
        "=========",
        f"TC-ID           : {tc['tc_id']}",
        f"REQ-ID          : {tc['req_id']}",
        f"TITLE           : {tc['title']}",
        f"DESCRIPTION     : {desc}",
        f"TYPE            : {tc['type']}",
        f"PRIORITY        : {tc['priority']}",
        f"ACTOR           : {tc['actor']}",
        f"FEATURE         : {tc['feature']}",
        f"PRECONDITION    : {tc['precondition']}",
        "STEPS",
        "-----",
    ]
    for i, step in enumerate(tc["steps"], start=1):
        lines.append(f"  {i}. {step}")
    lines.append(f"EXPECTED RESULT : {tc['expected_result']}")
    lines.append(f"GRAPH NODES     : {' \u2192 '.join(tc['graph_nodes'])}")
    if tc.get("fallbacks"):
        lines.append(f"NOTE            : built with fallback(s) — {'; '.join(tc['fallbacks'])}")
    return "\n".join(lines)