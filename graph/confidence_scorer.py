"""
Confidence Scorer

Pipeline position: Entity Resolution (deduplicate_graph) -> HERE ->
Graph Validator. Scores each deduplicated node and relationship for how
much to trust it, so low-confidence ones can be flagged for manual
review downstream — same never-silently-drop principle as the rest of
this pipeline (parser/normalization_validator.py,
parser/requirement_linker.py, graph/validator.py): nothing is removed
here, everything is annotated and, if it scores low, also surfaced in
low_confidence_nodes / low_confidence_relationships for visibility.

Design decision (was an open question in the handoff doc): deterministic
heuristic scoring, not LLM self-reported confidence. Matches this
project's established hybrid rule — regex/rule-based wherever a
signal is available without an extra LLM call. Asking the same LLM
call that produced an extraction to also grade its own confidence is
weaker signal anyway (a hallucinated entity is just as likely to be
reported "confident" as a real one); independent corroboration across
chunks is a stronger and free-to-compute signal, since it's already
sitting in the dedup step's occurrence counts.

Signals used (all deterministic, no LLM):

Nodes:
  - Occurrence count (graph/deduplicator.py's _occurrence_count) — the
    entity/relationship was independently extracted from more than one
    chunk. This is the main signal: something the LLM reconstructed the
    same way from different source text is much less likely to be a
    one-off hallucination than something seen exactly once.
  - Has a non-empty description — richer, more grounded extractions
    tend to come with a description; empty-description nodes are more
    often a bare label with weaker grounding.
  - Was involved in a node type conflict during dedup (graph/deduplicator.py's
    type_conflicts) — the LLM described this entity inconsistently
    across chunks, which is itself a (mild) signal of shakier grounding.

Relationships:
  - Occurrence count, same rationale as nodes.
  - Endpoint node confidence — a relationship can't be more trustworthy
    than the shakier of the two entities it connects.
  - Was involved in a direction conflict during dedup (graph/deduplicator.py's
    direction_conflicts) — the LLM disagreed with itself about which way
    the relationship points.
  - Corroborated by the Rule-based Requirement Linker (parser/requirement_linker.py):
    if the relationship's source requirement item is one that the
    (independent, non-LLM) linker also found to have at least one
    exact-ID cross-reference, that's a second, unrelated signal pointing
    the same direction — a modest confidence boost.
"""

LOW_CONFIDENCE_THRESHOLD = 0.5

# Score contribution constants — deliberately small integer-ish steps so
# the reasoning stays legible (each reason maps to a fixed, inspectable
# delta) rather than an opaque weighted formula.
_BASELINE = 0.5
_OCCURRENCE_BONUS = {1: 0.0, 2: 0.15, 3: 0.25}   # 3+ chunks caps at the 3 value
_OCCURRENCE_BONUS_CAP = 0.25
_DESCRIPTION_BONUS = 0.10
_TYPE_CONFLICT_PENALTY = 0.20
_DIRECTION_CONFLICT_PENALTY = 0.15
_LINKER_CORROBORATION_BONUS = 0.15
_ENDPOINT_PROPAGATION_WEIGHT = 0.3   # how much endpoint confidence pulls the score


def _occurrence_bonus(count: int) -> float:
    return _OCCURRENCE_BONUS.get(count, _OCCURRENCE_BONUS_CAP)


def _score_node(node: dict, type_conflict_ids: set) -> tuple:
    """Returns (score, reasons)."""
    reasons = []
    score = _BASELINE

    occ = node.get("_occurrence_count", 1)
    bonus = _occurrence_bonus(occ)
    if bonus:
        score += bonus
        reasons.append(f"independently extracted from {occ} chunk(s)")
    else:
        reasons.append("extracted from only 1 chunk")

    if node.get("description"):
        score += _DESCRIPTION_BONUS
        reasons.append("has a description")

    if node.get("id") in type_conflict_ids:
        score -= _TYPE_CONFLICT_PENALTY
        reasons.append(
            f"type was inconsistent across chunks before resolving to '{node.get('type')}'"
        )

    return max(0.0, min(1.0, round(score, 2))), reasons


def _score_relationship(
    rel: dict,
    node_confidence: dict,
    direction_conflict_keys: set,
    linked_source_ids: set,
) -> tuple:
    """Returns (score, reasons)."""
    reasons = []
    score = _BASELINE

    occ = rel.get("_occurrence_count", 1)
    bonus = _occurrence_bonus(occ)
    if bonus:
        score += bonus
        reasons.append(f"independently extracted from {occ} chunk(s)")
    else:
        reasons.append("extracted from only 1 chunk")

    from_conf = node_confidence.get(rel.get("from"), _BASELINE)
    to_conf = node_confidence.get(rel.get("to"), _BASELINE)
    endpoint_avg = (from_conf + to_conf) / 2
    score += (endpoint_avg - _BASELINE) * _ENDPOINT_PROPAGATION_WEIGHT
    reasons.append(f"endpoint node confidence averages {endpoint_avg:.2f}")

    key = (tuple(sorted([rel.get("from"), rel.get("to")])), rel.get("type"))
    if key in direction_conflict_keys:
        score -= _DIRECTION_CONFLICT_PENALTY
        reasons.append("direction was inconsistent across chunks before resolving")

    source = rel.get("source")
    if source and source in linked_source_ids:
        score += _LINKER_CORROBORATION_BONUS
        reasons.append(
            f"source item '{source}' also has a rule-based requirement "
            f"cross-reference (Requirement Linker), an independent signal"
        )

    return max(0.0, min(1.0, round(score, 2))), reasons


def score_graph(
    nodes: list,
    relationships: list,
    type_conflicts: list = None,
    direction_conflicts: list = None,
    linked_items: list = None,
) -> dict:
    """
    Score a deduplicated candidate graph.

    Args:
        nodes, relationships: deduplicate_graph()'s first two return
            values — each node/relationship is annotated IN PLACE with
            "confidence" (float 0-1) and "confidence_reasons" (list of
            str). Nothing is removed or reordered.
        type_conflicts, direction_conflicts: deduplicate_graph()'s 3rd
            and 4th return values (optional — pass [] or omit if not
            available; scoring degrades gracefully, just without those
            two signals).
        linked_items: requirement_linker.link_requirements()'s
            "items_with_links" (optional) — used to check whether a
            relationship's source requirement item also has at least
            one rule-based cross-reference link, as corroboration.

    Returns:
        {
            "nodes": nodes,                 # same list, now annotated
            "relationships": relationships, # same list, now annotated
            "low_confidence_nodes": [node, ...],         # score < 0.5
            "low_confidence_relationships": [rel, ...],  # score < 0.5
            "summary": {
                "total_nodes", "total_relationships",
                "low_confidence_nodes", "low_confidence_relationships",
                "avg_node_confidence", "avg_relationship_confidence",
            },
        }
    """
    type_conflicts = type_conflicts or []
    direction_conflicts = direction_conflicts or []
    linked_items = linked_items or []

    type_conflict_ids = {c.get("id") for c in type_conflicts}
    direction_conflict_keys = {
        (tuple(sorted([c.get("from"), c.get("to")])), c.get("type"))
        for c in direction_conflicts
    }
    linked_source_ids = {
        item.get("id") for item in linked_items if item.get("linked_ids")
    }

    node_confidence = {}
    for node in nodes:
        score, reasons = _score_node(node, type_conflict_ids)
        node["confidence"] = score
        node["confidence_reasons"] = reasons
        node_confidence[node.get("id")] = score

    for rel in relationships:
        score, reasons = _score_relationship(
            rel, node_confidence, direction_conflict_keys, linked_source_ids
        )
        rel["confidence"] = score
        rel["confidence_reasons"] = reasons

    low_confidence_nodes = [n for n in nodes if n["confidence"] < LOW_CONFIDENCE_THRESHOLD]
    low_confidence_relationships = [
        r for r in relationships if r["confidence"] < LOW_CONFIDENCE_THRESHOLD
    ]

    avg_node = round(sum(n["confidence"] for n in nodes) / len(nodes), 2) if nodes else None
    avg_rel = (
        round(sum(r["confidence"] for r in relationships) / len(relationships), 2)
        if relationships else None
    )

    return {
        "nodes": nodes,
        "relationships": relationships,
        "low_confidence_nodes": low_confidence_nodes,
        "low_confidence_relationships": low_confidence_relationships,
        "summary": {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "low_confidence_nodes": len(low_confidence_nodes),
            "low_confidence_relationships": len(low_confidence_relationships),
            "avg_node_confidence": avg_node,
            "avg_relationship_confidence": avg_rel,
        },
    }
