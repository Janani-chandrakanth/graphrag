# graph/deduplicator.py

# =========================================================
# TYPE PRIORITY — fixes Problem 4
# When same node ID appears with different types across chunks
# we pick the type with the highest priority
#
# Example:
#   chunk 1 → purchase_order: "Feature"
#   chunk 2 → purchase_order: "DataObject"
#   Result  → purchase_order: "DataObject" (higher priority)
#
# Logic: more specific types win over generic ones
# =========================================================

TYPE_PRIORITY = {
    "Actor":                    10,
    "DataObject":               9,
    "SystemComponent":          8,
    "Feature":                  7,
    "Requirement":              6,
    "NonFunctionalRequirement": 5,
    "BusinessRule":             4,
    "Action":                   3,
    "Condition":                3,
    "Output":                   3,
    "State":                    2,
    "Constraint":               2,
    "Event":                    2,
    "Attribute":                1,
}


def normalize_id(node_id: str) -> str:
    """
    Normalize a node ID to consistent snake_case.
    Handles: spaces, hyphens, mixed case, extra whitespace.

    Examples:
        "User-Login"  → "user_login"
        "User Login"  → "user_login"
        "UserLogin"   → "userlogin"
        "USER_LOGIN"  → "user_login"
    """
    return (
        node_id.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .strip()
    )


def deduplicate_graph(nodes: list, relationships: list):
    """
    Remove duplicate nodes and relationships across all chunks.

    Fixes:
        Problem 4 — same node ID with different types
                    → keep the type with highest priority
        Problem 5 — relationship direction inconsistency
                    → keep only one direction per node pair + type

    Returns:
        unique_nodes, unique_relationships, type_conflicts
    """

    # =========================================================
    # Step 1: Deduplicate Nodes — fix Problem 4
    # =========================================================

    unique_nodes = {}
    node_occurrences = {}    # id -> how many chunk-level extractions mentioned it
    type_conflicts = []     # track conflicts for debugging

    for node in nodes:

        node["id"] = normalize_id(node["id"])
        node_id = node["id"]
        node_occurrences[node_id] = node_occurrences.get(node_id, 0) + 1

        if node_id not in unique_nodes:
            # First time seeing this ID — store it
            unique_nodes[node_id] = node

        else:
            # Already seen this ID — check if types conflict
            existing_type = unique_nodes[node_id].get("type", "")
            incoming_type = node.get("type", "")

            if existing_type != incoming_type:

                # Log the conflict so you can see it in Streamlit
                type_conflicts.append({
                    "id":            node_id,
                    "kept_type":     existing_type,
                    "ignored_type":  incoming_type,
                    "reason":        "Type conflict — higher priority type kept"
                })

                # Compare priority scores — higher score wins
                existing_priority = TYPE_PRIORITY.get(existing_type, 0)
                incoming_priority = TYPE_PRIORITY.get(incoming_type, 0)

                if incoming_priority > existing_priority:
                    # Incoming type is more specific — replace
                    type_conflicts[-1]["kept_type"]    = incoming_type
                    type_conflicts[-1]["ignored_type"] = existing_type
                    unique_nodes[node_id] = node

                # else: keep existing (already stored), do nothing

    # =========================================================
    # Step 2: Deduplicate Relationships — fix Problem 5
    #
    # Problem 5 example:
    #   chunk 1: warehouse_manager → USES → user_authentication
    #   chunk 2: user_authentication → USES → warehouse_manager
    #
    # These are different tuples so normal dedup misses them.
    #
    # Fix: use a CANONICAL key where from/to are always sorted
    # alphabetically — so both directions map to the same key.
    # Keep whichever direction we see first.
    # =========================================================

    unique_relationships = []
    seen_exact = set()          # catches exact duplicates
    seen_canonical = set()      # catches reversed duplicates
    rel_occurrences = {}         # exact_key -> how many chunk-level extractions produced it

    direction_conflicts = []    # track for debugging

    for rel in relationships:

        rel["from"] = normalize_id(rel["from"])
        rel["to"]   = normalize_id(rel["to"])

        rel_type = rel.get("type", "")

        # Key for exact duplicate check (same direction)
        exact_key = (rel["from"], rel["to"], rel_type)
        rel_occurrences[exact_key] = rel_occurrences.get(exact_key, 0) + 1

        # Key for direction conflict check (either direction)
        # sorted() ensures (A, B) and (B, A) produce the same key
        canonical_key = (
            tuple(sorted([rel["from"], rel["to"]])),
            rel_type
        )

        if exact_key in seen_exact:
            # Pure duplicate — skip silently
            continue

        if canonical_key in seen_canonical:
            # Reverse direction exists — log and skip
            direction_conflicts.append({
                "from":   rel["from"],
                "to":     rel["to"],
                "type":   rel_type,
                "reason": "Reverse direction already exists — removed"
            })
            continue

        # New unique relationship — keep it
        seen_exact.add(exact_key)
        seen_canonical.add(canonical_key)
        unique_relationships.append(rel)

    # Stamp final occurrence counts onto the surviving objects — an
    # additive field (leading underscore, internal-use marker) that
    # doesn't change this function's return signature. Consumed by
    # graph/confidence_scorer.py as its main deterministic signal:
    # something independently extracted from more chunks is more
    # likely to be a real, corroborated fact rather than a one-off
    # LLM extraction quirk.
    for node in unique_nodes.values():
        node["_occurrence_count"] = node_occurrences[node["id"]]
    for rel in unique_relationships:
        key = (rel["from"], rel["to"], rel.get("type", ""))
        rel["_occurrence_count"] = rel_occurrences.get(key, 1)

    return (
        list(unique_nodes.values()),
        unique_relationships,
        type_conflicts,
        direction_conflicts
    )