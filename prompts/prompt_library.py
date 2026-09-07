"""
Prompt Library

Closes gap #3 from the architecture review: graph/negative_scenario_generator.py
had exactly ONE hardcoded style baked directly into the module — no way
to pick "Gherkin style" vs "regression suite" vs "edge-case focused"
without editing source code, and nothing saved across sessions.

Deliberately a small file-backed JSON store, not a database table —
this is app CONFIG (prompt styles), not knowledge-graph data. Neo4j/
Chroma stay reserved for the actual KB, matching how the rest of this
project separates "extracted knowledge" from "pipeline configuration"
(e.g. config.py's plain constants for model names / URLs).

Every entry uses the SAME placeholders
graph/negative_scenario_generator.py's original _NEGATIVE_PROMPT
always used ({req_ids}, {actor}, {feature}, {steps}, {expected_result},
{catalog}) — swapping the ACTIVE template never changes the call site's
shape, only which wording/style gets sent to the LLM.

Versioning: saving under a name that already exists APPENDS a new
version rather than overwriting — "Save as New Template" with a name
you've used before means "here's a revision", and no prior version is
ever silently lost.
"""

import json
import os
import shutil
import time
from datetime import datetime, timezone

_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "prompt_library_store.json")

REQUIRED_PLACEHOLDERS = {"req_ids", "actor", "feature", "steps", "expected_result", "catalog"}

# graph/combined_test_case_generator.py's call site needs two more
# placeholders than the older negative-only generator did: {source_text}
# (the original chunk text, so steps can be phrased as a real journey
# instead of a raw graph-edge readout) and {precondition} (the
# deterministic walk's precondition, carried through for context).
# Kept as its own set/validator rather than folding into
# REQUIRED_PLACEHOLDERS above, since that constant is still used as-is
# by the older negative-only call site.
REQUIRED_PLACEHOLDERS_COMBINED = REQUIRED_PLACEHOLDERS | {"source_text", "precondition"}

# graph/hybrid_test_case_generator.py's call site (replaces
# graph/combined_test_case_generator.py) needs one more placeholder
# than the combined generator did: {related_context} — the vector-
# retrieval half of hybrid search (semantically similar chunks from
# elsewhere in the KB), kept separate from {source_text} (this
# series' own chunk text) since the prompt treats them differently
# (source_text is authoritative sequence/content; related_context is
# narrative-only, never a source of entities_used).
REQUIRED_PLACEHOLDERS_HYBRID = REQUIRED_PLACEHOLDERS_COMBINED | {"related_context"}

# ── Seed styles ─────────────────────────────────────────────
# "edge_case_focused" is byte-for-byte the original _NEGATIVE_PROMPT
# that lived in graph/negative_scenario_generator.py before this
# library existed — so the default behavior of an unmodified install
# is unchanged; the library only adds NEW options on top.

_EDGE_CASE_FOCUSED = """You are a QA engineer identifying NEGATIVE and EDGE-CASE test scenarios for one part of a system, based ONLY on entities already extracted from its requirements.

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

_GHERKIN_STYLE = """You are a QA engineer writing Gherkin-style (Given/When/Then) NEGATIVE and EDGE-CASE scenarios for one part of a system, based ONLY on entities already extracted from its requirements.

REQUIREMENT SERIES: {req_ids}

POSITIVE TEST CASE ALREADY BUILT FOR THIS SERIES:
  Actor: {actor}
  Feature: {feature}
  Steps: {steps}
  Expected Result: {expected_result}

ENTITIES YOU MAY REFERENCE (id | type | name) — you may ONLY use entities from this list. Do NOT invent, assume, or reference anything not in this list. No external systems, no generic infrastructure ("the database", "the network", "the API") unless it is literally in the list below:
{catalog}

Propose 1 to 3 realistic NEGATIVE or EDGE-CASE scenarios, each expressed as Given/When/Then steps (e.g. "Given the {actor} is on the Login screen", "When an invalid password is submitted", "Then an error message is displayed") instead of plain imperative steps — grounded ENTIRELY in the entities listed. Every entity name in "entities_used" must be copied EXACTLY as it appears in the catalog above. If you cannot construct a grounded scenario from this list, return an empty array.

Return ONLY a JSON array, no other text, no markdown fences. Each entry:
{{"title": "<short scenario title>", "steps": ["Given ...", "When ...", "Then ..."], "expected_result": "<expected outcome>", "entities_used": ["<exact name from catalog>", "..."]}}
"""

_REGRESSION_SUITE = """You are a QA engineer building REGRESSION-focused NEGATIVE and EDGE-CASE scenarios for one part of a system, based ONLY on entities already extracted from its requirements. Regression framing: assume this flow worked correctly in a prior release — scenarios should target the specific entities/states most likely to break silently after a future change (boundary values, previously-fixed edge conditions, state transitions), not novel/exotic failures.

REQUIREMENT SERIES: {req_ids}

POSITIVE TEST CASE ALREADY BUILT FOR THIS SERIES:
  Actor: {actor}
  Feature: {feature}
  Steps: {steps}
  Expected Result: {expected_result}

ENTITIES YOU MAY REFERENCE (id | type | name) — you may ONLY use entities from this list. Do NOT invent, assume, or reference anything not in this list. No external systems, no generic infrastructure ("the database", "the network", "the API") unless it is literally in the list below:
{catalog}

Propose 1 to 3 realistic regression-worthy NEGATIVE or EDGE-CASE scenarios grounded ENTIRELY in the entities listed — prioritize the entities most central to this flow (the ones a future change is most likely to touch) over peripheral ones. Every entity name in "entities_used" must be copied EXACTLY as it appears in the catalog above. If you cannot construct a grounded scenario from this list, return an empty array.

Return ONLY a JSON array, no other text, no markdown fences. Each entry:
{{"title": "<short scenario title>", "steps": ["<step 1>", "<step 2>", "..."], "expected_result": "<expected outcome>", "entities_used": ["<exact name from catalog>", "..."]}}
"""


_HYBRID_FLOW = """You are a senior QA engineer producing a COMPLETE, high-coverage test-case set for one requirement flow. You have THREE sources of grounding — use all of them together, they are not interchangeable:

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


def _seed_defaults() -> dict:
    now = datetime.now(timezone.utc).isoformat()

    def entry(label, template, category="negative_scenarios"):
        return {
            "label": label,
            "category": category,
            "versions": [{"template": template, "created_at": now}],
        }

    return {
        "edge_case_focused": entry("Edge Case Focused (default)", _EDGE_CASE_FOCUSED),
        "gherkin_style": entry("Gherkin Style", _GHERKIN_STYLE),
        "regression_suite": entry("Regression Suite", _REGRESSION_SUITE),
        "hybrid_flow": entry("Hybrid Vector+Graph Flow (default)", _HYBRID_FLOW, category="hybrid_test_cases"),
    }


def _load() -> dict:
    if not os.path.exists(_LIBRARY_PATH):
        library = _seed_defaults()
        _save_all(library)
        return library
    try:
        with open(_LIBRARY_PATH, "r", encoding="utf-8") as f:
            library = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Never let a corrupt/unreadable config file crash generation —
        # re-seed rather than propagate. This IS a visible behavior
        # change (custom templates would be gone) but a KeyError on
        # every single call would be worse; callers that care can
        # check list_prompts() length before/after.
        library = _seed_defaults()
        _save_all(library)
        return library

    # Merge-in-missing: a store.json written before a new seed name
    # existed (e.g. "hybrid_flow", added when
    # graph/hybrid_test_case_generator.py replaced the old two-step
    # flow) would otherwise never pick up that new template — _load()
    # previously only ever seeded on a totally absent file. Add any
    # seed keys the existing file doesn't already have, without
    # touching or overwriting anything saved under an existing key.
    seeds = _seed_defaults()
    missing = {name: entry for name, entry in seeds.items() if name not in library}
    if missing:
        library.update(missing)
        _save_all(library)
    return library


def _save_all(library: dict) -> None:
    """
    Atomic write (temp file + replace) so a crash mid-write never
    leaves a half-written, unparseable JSON file behind.

    On Windows, a stale .tmp file from a previous crash prevents
    os.replace() — clean it up proactively before writing.
    """
    tmp_path = _LIBRARY_PATH + ".tmp"

    # Clean up any stale .tmp file from a prior interrupted write
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(library, f, indent=2)
    except OSError:
        # If we can't even write the temp file, don't bother trying to
        # replace — the original file is still intact, and the caller
        # still gets usable data from _load()'s fallback path.
        return

    # Windows can temporarily lock the target file.
    # Retry the replacement a few times before failing.
    for attempt in range(5):
        try:
            # shutil.move is more reliable on Windows than os.replace
            # when file locks are involved
            shutil.move(tmp_path, _LIBRARY_PATH)
            return
        except PermissionError:
            if attempt == 4:
                # Clean up the orphaned .tmp file so the next startup
                # doesn't hit the same PermissionError from a stale
                # .tmp it can't overwrite itself.
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                return  # Degrade gracefully — the text was written to
                        # .tmp but couldn't replace the live file. The
                        # in-memory library is still usable this session.

            time.sleep(0.2)


def list_prompts(category: str = None) -> list:
    """Returns [{"name","label","category","version_count"}], sorted by name."""
    library = _load()
    out = []
    for name, entry in library.items():
        if category and entry.get("category") != category:
            continue
        out.append({
            "name": name,
            "label": entry.get("label", name),
            "category": entry.get("category", "general"),
            "version_count": len(entry["versions"]),
        })
    return sorted(out, key=lambda x: x["name"])


def get_prompt(name: str, version: int = None) -> dict:
    """
    Returns {"name","label","category","template","version"} for the
    given (or latest) version.

    Raises KeyError if name/version is unknown — callers decide the
    fallback (e.g. negative_scenario_generator.py falls back to its
    own baked-in copy rather than crashing generation over a missing
    library entry).
    """
    library = _load()
    if name not in library:
        raise KeyError(f"No prompt template named '{name}' in the library.")
    entry = library[name]
    versions = entry["versions"]
    idx = (version - 1) if version else len(versions) - 1
    if idx < 0 or idx >= len(versions):
        raise KeyError(f"'{name}' has no version {version} (has {len(versions)}).")
    return {
        "name": name,
        "label": entry.get("label", name),
        "category": entry.get("category", "general"),
        "template": versions[idx]["template"],
        "version": idx + 1,
    }


def validate_template(template: str) -> list:
    """
    Returns the list of MISSING required placeholders (empty = fully
    compatible with the negative-scenario call site). Advisory only —
    save_prompt() does not block on this; a deliberately narrower
    template is the user's call, not an error this library enforces.
    """
    return [p for p in REQUIRED_PLACEHOLDERS if ("{" + p + "}") not in template]


def validate_combined_template(template: str) -> list:
    """Same as validate_template(), but against
    REQUIRED_PLACEHOLDERS_COMBINED — kept for any pre-existing saved
    templates in the old "combined_test_cases" category. The active
    generator is now graph/hybrid_test_case_generator.py; see
    validate_hybrid_template() below for its category."""
    return [p for p in REQUIRED_PLACEHOLDERS_COMBINED if ("{" + p + "}") not in template]


def validate_hybrid_template(template: str) -> list:
    """Same as validate_template(), but against
    REQUIRED_PLACEHOLDERS_HYBRID — for templates in the
    "hybrid_test_cases" category (graph/hybrid_test_case_generator.py's
    call site), which also needs {source_text}, {precondition}, and
    {related_context} (the vector-retrieval half of hybrid search)."""
    return [p for p in REQUIRED_PLACEHOLDERS_HYBRID if ("{" + p + "}") not in template]


def save_prompt(name: str, template: str, label: str = None, category: str = "negative_scenarios") -> dict:
    """
    Save `template` under `name`. If `name` already exists, appends a
    NEW VERSION — never overwrites, so "Save as New Template" reusing
    an existing name is a revision, not data loss.
    """
    library = _load()
    now = datetime.now(timezone.utc).isoformat()
    if name in library:
        library[name]["versions"].append({"template": template, "created_at": now})
        if label:
            library[name]["label"] = label
    else:
        library[name] = {
            "label": label or name,
            "category": category,
            "versions": [{"template": template, "created_at": now}],
        }
    _save_all(library)
    return get_prompt(name)


def delete_prompt(name: str) -> bool:
    """Removes a template entirely (all versions). Returns False if it didn't exist."""
    library = _load()
    if name not in library:
        return False
    del library[name]
    _save_all(library)
    return True
