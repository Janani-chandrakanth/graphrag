"""
graph/story_hybrid_retriever.py

Retrieves User Story and Gherkin flow context using vector similarity search
combined with Cypher graph traversals for upstream/downstream entity retrieval.
"""

from embeddings.embedding_model import generate_embedding
from vectorstore.chroma_manager import search_user_stories, get_user_stories_collection_count
from graph.neo4j_manager import run_read_query
from parser.llm_client import call_ollama, extract_json_block

SIMILARITY_FLOOR = 0.55  # below this, treat the vector match as noise, not signal
VECTOR_TOP_K = 5
GRAPH_CONTEXT_LIMIT = 25

_TRAVERSAL_QUERY = """
MATCH (s:UserStory) WHERE s.storyId IN $story_ids
OPTIONAL MATCH (s)-[:REQUIRES]->(f:Feature)-[:HAS_FLOW]->(fs:FlowStep)
OPTIONAL MATCH (r:Requirement)-[:REFINES]->(s)
OPTIONAL MATCH (s)-[:VALIDATES]->(ac:AcceptanceCriteria)
RETURN
    collect(DISTINCT f.name)                    AS feature_names,
    collect(DISTINCT f.featureId)               AS feature_ids,
    collect(DISTINCT fs.action)                 AS flow_steps,
    collect(DISTINCT r.description)             AS upstream_rules,
    collect(DISTINCT ac.text)                   AS known_acceptance_criteria,
    collect(DISTINCT ac.acId)                   AS known_acceptance_criteria_ids
LIMIT $limit
"""

_FEATURE_FALLBACK_QUERY = """
MATCH (f:Feature) WHERE toLower(f.name) CONTAINS toLower($keyword)
OPTIONAL MATCH (f)-[:HAS_FLOW]->(fs:FlowStep)
OPTIONAL MATCH (r:Requirement) WHERE toLower(r.description) CONTAINS toLower(f.name)
RETURN f.name AS feature_name, f.featureId AS feature_id,
       collect(DISTINCT fs.action) AS flow_steps,
       collect(DISTINCT r.description) AS upstream_rules
LIMIT $limit
"""


def _vector_search_similar_stories(story_summary: str, top_k: int) -> list:
    if get_user_stories_collection_count() == 0:
        return []
    embedding = generate_embedding(story_summary)
    results = search_user_stories(embedding, n_results=top_k)
    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    out = []
    for story_id, dist in zip(ids, distances):
        similarity = 1 - dist if dist is not None else 0
        if similarity >= SIMILARITY_FLOOR:
            out.append(story_id)
    return out


def _extract_keywords(story_text: str, model: str = None) -> list:
    prompt = (
        "You extract short noun-phrase keywords from a user story for keyword search. "
        "Respond with ONLY a JSON array of 1-4 short strings, nothing else.\n\n"
        f"User story: {story_text}"
    )
    result = call_ollama(prompt, model=model)
    if result["error"]:
        return [story_text.split()[0]] if story_text.split() else []
    try:
        parsed = extract_json_block(result["raw"])
        if isinstance(parsed, list):
            return [str(k) for k in parsed]
    except Exception:
        pass
    return [story_text.split()[0]] if story_text.split() else []


def retrieve_story_context(story_text: str, model: str = None) -> dict:
    """
    Main entry point used by graph/gherkin_test_case_generator.py.

    Returns:
        {
          "mode": "vector+graph" | "keyword_fallback" | "none",
          "matched_story_ids": [str, ...],
          "features": [str, ...],
          "feature_ids": [str, ...],
          "flow_steps": [str, ...],
          "upstream_rules": [str, ...],
          "acceptance_criteria": [str, ...],
          "acceptance_criteria_ids": [str, ...],
        }
    """
    try:
        story_ids = _vector_search_similar_stories(story_text, VECTOR_TOP_K)
    except Exception:
        story_ids = []

    if story_ids:
        rows = run_read_query(_TRAVERSAL_QUERY, story_ids=story_ids, limit=GRAPH_CONTEXT_LIMIT)
        if rows and rows[0]["feature_names"]:
            r = rows[0]
            return {
                "mode": "vector+graph",
                "matched_story_ids": story_ids,
                "features": [f for f in r["feature_names"] if f],
                "feature_ids": [f for f in r["feature_ids"] if f],
                "flow_steps": [f for f in r["flow_steps"] if f],
                "upstream_rules": [f for f in r["upstream_rules"] if f],
                "acceptance_criteria": [f for f in r["known_acceptance_criteria"] if f],
                "acceptance_criteria_ids": [f for f in r["known_acceptance_criteria_ids"] if f],
            }

    # cold-start / no-match fallback: keyword search over Feature names
    keywords = _extract_keywords(story_text, model=model)
    features, feature_ids, flow_steps, rules = [], [], [], []
    for kw in keywords:
        rows = run_read_query(_FEATURE_FALLBACK_QUERY, keyword=kw, limit=GRAPH_CONTEXT_LIMIT)
        for r in rows:
            if r["feature_name"] and r["feature_name"] not in features:
                features.append(r["feature_name"])
                feature_ids.append(r["feature_id"])
            flow_steps.extend([s for s in r["flow_steps"] if s])
            rules.extend([u for u in r["upstream_rules"] if u])

    if features:
        return {
            "mode": "keyword_fallback",
            "matched_story_ids": [],
            "features": features,
            "feature_ids": feature_ids,
            "flow_steps": list(dict.fromkeys(flow_steps)),
            "upstream_rules": list(dict.fromkeys(rules)),
            "acceptance_criteria": [],
            "acceptance_criteria_ids": [],
        }

    return {
        "mode": "none",
        "matched_story_ids": [],
        "features": [],
        "feature_ids": [],
        "flow_steps": [],
        "upstream_rules": [],
        "acceptance_criteria": [],
        "acceptance_criteria_ids": [],
    }
