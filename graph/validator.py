# graph/validator.py

ALLOWED_RELATIONS = {
    "USES",
    "REQUIRED_FOR",
    "TRIGGERS",
    "SHOWS",
    "LEADS_TO",
    "CAUSES",
    "CONTRIBUTES_TO",
    "PART_OF",
    "CREATES",
    "UPDATES",
    "GENERATES",
    "TRACKS",
    "CONTAINS",
    "OWNS",
    "AUTHENTICATES",
    "VALIDATES",
    "MONITORS",
    "NOTIFIES",
    "DEPENDS_ON",
    "ASSOCIATED_WITH",
    "PROCESSES",
    "STORES",
    "RETRIEVES",
    "EXECUTES",
    # ── Deterministic types produced by graph/structural_linker.py,
    # not the LLM — mirrors parser/requirement_linker.py's RELATION_RULES
    # so its exact-ID cross-references can become real graph edges. ──
    "VERIFIED_BY",
    "VERIFIES",
    "REALIZES",
    "REALIZED_BY",
    "RELATED_TO",
    "RELATES_TO",
    # ── Added to cover richer BR/NFR/Risk/Glossary documents ──
    "ACTOR_OF",
    "PRODUCES",
    "VIEWS",
    "PERFORMS",
    "REFERENCES",
    "SUPPORTS",
    "CONSTRAINS",
    "THREATENS",
    "DEFINES",
    # Document-order backbone edge — explicitly NOT the same claim as
    # LEADS_TO above. LEADS_TO (from the LLM) means "this genuinely
    # causes/triggers that, per the requirement text." NEXT_IN_DOCUMENT
    # means only "these two requirements were adjacent in the document" —
    # them as separate types is deliberate; don't merge them.
    "NEXT_IN_DOCUMENT",
    "NAVIGATES_TO",
}

# =========================================================
# VALID NODE TYPES
# Synced with the allowed node types listed in
# prompts/kg_extraction_prompt.py — keep these two in sync
# whenever the prompt's ontology is extended.
# =========================================================

VALID_NODE_TYPES = {
    "Actor",
    "Feature",
    "Requirement",
    "BusinessRule",
    "Condition",
    "Action",
    "Output",
    "State",
    "SystemComponent",
    "SoftwareSystem",
    "Application",
    "Component",
    "Subsystem",
    "Microservice",
    "DataObject",
    "Constraint",
    "Event",
    "NonFunctionalRequirement",
    "Attribute",
    "Country",
    "Currency",
    "Language",
    "Location",
    "Service",
    "UIElement",
    # ── Added to match prompt's allowed node types ────────
    "BusinessProcess",
    "Module",
    "Workflow",
    "API",
    "Screen",
    "Page",
    # ── Written deterministically by graph/test_case_writer.py, not the
    # LLM extractor — a generated test case becomes a real graph node so
    # the Traceability Map (which KB node inspired which test step) is a
    # queryable edge (VALIDATES), not just a list inside a Python dict. ──
    "TestCase",
    "Role",
    "Message",
    "Validation",
    "SearchOption",
    "Calendar",
    "BookingOption",
    # ── Added to cover non-BR document context: assumptions,
    # dependencies, risks, glossary entries, acronyms ──
    "Risk",
    "Assumption",
    "Definition",
}


def _check_node(node: dict) -> tuple:
    """Run all node rules. Returns (reasons, normalized_type) — empty
    reasons means the node is valid. normalized_type is returned even
    for invalid nodes so the review entry shows what was actually seen."""
    reasons = []
    node_type = (node.get("type", "") or "").replace(" ", "").strip()

    if node_type not in VALID_NODE_TYPES:
        reasons.append(f"Invalid node type '{node.get('type')}'")

    return reasons, node_type


# Narrow, deliberately conservative semantic guards — NOT a full
# node-type-compatibility matrix (that would need certainty about every
# type pairing this project's ontology allows, which isn't grounded
# clearly enough in prompts/kg_extraction_prompt.py to build safely
# without risking false rejections on legitimate data). These two are
# safe because they hold regardless of finer ontology details:
#
# 1. An Actor is a person/role, not an artifact — nothing "creates,"
#    "generates," "produces," or "updates" an Actor as an output. If a
#    relationship's TARGET is an Actor and its type is one of these
#    production verbs, the direction is backwards.
_ACTOR_TARGET_FORBIDDEN = {"CREATES", "GENERATES", "PRODUCES", "UPDATES"}

# 2. AUTHENTICATES is the one allowed relationship type
#    prompts/kg_extraction_prompt.py lists with zero usage guidance
#    anywhere else in the prompt (unlike every other type, which gets
#    an explicit directional example) — so the model has no signal for
#    what a plausible source looks like. A DataObject/Requirement/
#    Output authenticating someone is never sensible; restrict to
#    node types that could plausibly perform authentication.
_AUTHENTICATES_VALID_SOURCES = {"Actor", "SystemComponent", "Service", "Module", "API"}

# 3. SHOWS is the SECOND allowed relationship type (after AUTHENTICATES)
#    with zero directional guidance anywhere in
#    prompts/kg_extraction_prompt.py — same bug class, same fix shape.
#    A Feature/System/Screen/UIElement shows something TO an Actor; an
#    Actor showing something (e.g. "User SHOWS Currency Symbol") is
#    backwards in every real case we've seen. Actor VIEWS things (that
#    direction is fine, already used elsewhere) — Actor never SHOWS
#    things.
_SHOWS_INVALID_SOURCE_TYPES = {"Actor"}

# Relation types where the same pair legitimately CAN point both ways,
# or where "both directions" isn't really a contradiction (e.g. two
# features mutually supporting each other) — excluded from the
# reciprocal-contradiction check below. Everything else in
# ALLOWED_RELATIONS is treated as directional: A->B and B->A with the
# SAME type between the SAME two nodes can't both be correct, and we
# have no reliable way to tell which one is — flag both for review
# rather than silently keeping one that might be backwards.
_SYMMETRIC_OK_RELATIONS = {
    "ASSOCIATED_WITH", "RELATED_TO", "RELATES_TO", "SUPPORTS",
    "CONTRIBUTES_TO", "REFERENCES", "NEXT_IN_DOCUMENT",
}


RELATIONSHIP_SYNONYMS = {
    "STARTS_WITH": "TRIGGERS",
    "STARTS": "TRIGGERS",
    "BEGINS_WITH": "TRIGGERS",
    "DEPENDS_UPON": "DEPENDS_ON",
    "REQUIRES": "REQUIRED_FOR",
    "INCLUDES": "CONTAINS",
    "HAS": "CONTAINS",
    "USES_DATA": "USES",
    "CALLS": "EXECUTES",
    "INVOKES": "EXECUTES",
    "OPENS": "NAVIGATES_TO",
    "GOES_TO": "NAVIGATES_TO",
    "DISPLAYS": "SHOWS",
    "MODIFIES": "UPDATES",
    "ALERTS": "NOTIFIES",
}

BLOCKED_REL_TYPES = {"", "NULL", "UNDEFINED", "NONE", "AND", "OR", "THE", "A", "AN"}

def normalize_relationship_type(rel_type: str) -> str:
    """Normalize custom/non-standard relationship types to ontology or valid UPPER_SNAKE_CASE."""
    if not rel_type:
        return "RELATED_TO"
    clean = str(rel_type).strip().upper().replace(" ", "_").replace("-", "_")
    # 1. Direct match
    if clean in ALLOWED_RELATIONS:
        return clean
    # 2. Synonym mapping
    if clean in RELATIONSHIP_SYNONYMS:
        return RELATIONSHIP_SYNONYMS[clean]
    # 3. Custom type check (accept any non-blocked relationship string)
    if clean not in BLOCKED_REL_TYPES and len(clean) >= 1:
        # Strip any invalid non-alphanumeric chars except underscore
        import re
        clean_id = re.sub(r'[^A-Z0-9_]', '', clean)
        return clean_id if clean_id else "RELATED_TO"
    return "RELATED_TO"


def _check_relationship(rel: dict, node_ids: set, node_type_by_id: dict = None) -> list:
    """Run all relationship rules. Returns a list of failure reason
    strings — empty list means the relationship is valid."""
    reasons = []
    raw_rel_type = (rel.get("type", "") or "").strip()
    norm_rel_type = normalize_relationship_type(raw_rel_type)

    if not norm_rel_type:
        reasons.append(f"Invalid relationship type '{raw_rel_type}'")
    else:
        # Save normalized type back to relationship dict
        rel["type"] = norm_rel_type

    rel_type = norm_rel_type or raw_rel_type

    if rel.get("from") not in node_ids:
        reasons.append(f"Source node '{rel.get('from')}' not found")

    if rel.get("to") not in node_ids:
        reasons.append(f"Target node '{rel.get('to')}' not found")

    if node_type_by_id and not reasons:
        from_type = node_type_by_id.get(rel.get("from"))
        to_type = node_type_by_id.get(rel.get("to"))

        if rel_type in _ACTOR_TARGET_FORBIDDEN and to_type == "Actor":
            reasons.append(
                f"Implausible direction: '{rel_type}' shouldn't target an Actor — "
                f"an Actor isn't created/generated/produced/updated by "
                f"'{rel.get('from')}' ({from_type}); check whether this edge is backwards."
            )
        if rel_type == "AUTHENTICATES" and from_type not in _AUTHENTICATES_VALID_SOURCES:
            reasons.append(
                f"Implausible source: AUTHENTICATES should originate from an Actor or "
                f"system-like node, not '{rel.get('from')}' ({from_type})."
            )
        if rel_type == "SHOWS" and from_type in _SHOWS_INVALID_SOURCE_TYPES:
            reasons.append(
                f"Implausible direction: SHOWS shouldn't originate from an Actor — "
                f"an Actor is normally shown something, not the one showing it. "
                f"'{rel.get('from')}' ({from_type}) -> '{rel.get('to')}' looks backwards; "
                f"check whether this should be reversed."
            )

    return reasons


def validate_graph(nodes, relationships):
    """
    Split nodes/relationships into valid vs needs-review buckets.

    Matches the Normalization Validator's never-silently-drop principle
    (see parser/normalization_validator.py): nothing that fails a rule
    is discarded here. It's held in the review bucket with every reason
    it failed for attached, ready for a human review flow to act on
    later (approve as-is, edit, or reject) — same as this pipeline's
    review_queue / broken_references elsewhere. Previously this function
    only reported a single reason and a bare removed-count; the shape
    below is a superset, so callers unpacking the same 4-tuple
    positionally still work unchanged.

    Returns:
        valid_nodes, valid_relationships, review_nodes, review_relationships

        review_nodes:         [{"node": node, "reasons": [str, ...]}, ...]
        review_relationships: [{"relationship": rel, "reasons": [str, ...]}, ...]
    """

    valid_nodes = []
    review_nodes = []

    for node in nodes:
        reasons, node_type = _check_node(node)
        if reasons:
            review_nodes.append({"node": node, "reasons": reasons})
            continue

        node["type"] = node_type
        valid_nodes.append(node)

    node_ids = {node["id"] for node in valid_nodes}
    node_type_by_id = {node["id"]: node["type"] for node in valid_nodes}

    valid_relationships = []
    review_relationships = []

    for rel in relationships:
        reasons = _check_relationship(rel, node_ids, node_type_by_id)
        if reasons:
            review_relationships.append({"relationship": rel, "reasons": reasons})
            continue

        valid_relationships.append(rel)

    # Reciprocal-contradiction pass: for directional relation types (see
    # _SYMMETRIC_OK_RELATIONS above), A->B and B->A with the SAME type
    # between the SAME two nodes can't both be correct — e.g. "Warehouse
    # Manager CREATES Purchase Order" AND "Purchase Order CREATES
    # Warehouse Manager" both present. There's no reliable way to tell
    # which direction is the real one, so both are flagged for review
    # rather than arbitrarily keeping one.
    by_type_pair = {}
    for rel in valid_relationships:
        by_type_pair.setdefault(rel["type"], {})[(rel["from"], rel["to"])] = rel

    contradictory = set()
    for rel_type, pairs in by_type_pair.items():
        if rel_type in _SYMMETRIC_OK_RELATIONS:
            continue
        for (f, t), rel in pairs.items():
            if (t, f) in pairs:
                contradictory.add(id(rel))

    if contradictory:
        still_valid = []
        for rel in valid_relationships:
            if id(rel) in contradictory:
                review_relationships.append({
                    "relationship": rel,
                    "reasons": [
                        f"Contradictory: '{rel['type']}' exists in both directions "
                        f"between '{rel['from']}' and '{rel['to']}' — at most one "
                        f"direction can be correct; flagged rather than guessing "
                        f"which one to drop."
                    ],
                })
            else:
                still_valid.append(rel)
        valid_relationships = still_valid

    return valid_nodes, valid_relationships, review_nodes, review_relationships