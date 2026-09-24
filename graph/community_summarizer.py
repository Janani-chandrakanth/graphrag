"""
Community Summarization Module
Generates prose summaries for each detected community using llama3.1,
then embeds and stores them in ChromaDB for semantic search.
Uses direct requests.post to avoid Ollama SDK proxy issue.
"""

import json
import requests
from config import OLLAMA_URL
from embeddings.embedding_model import generate_embedding
from graph.community_detector import (
    get_community_nodes,
    get_community_relationships,
    get_all_communities
)
from vectorstore.chroma_manager import (
    store_community_summary,
    search_community_summaries
)


COMMUNITY_SUMMARY_PROMPT = """
You are an expert requirements analyst.

Analyze the following group of requirement entities and relationships.
Generate a concise professional summary describing what this group is about.

Include:
- What business capability or feature this group represents
- Key entities involved (actors, data objects, features)
- Primary interactions or workflows
- Business value or purpose
- Any constraints or special considerations

Keep the summary to 2-3 paragraphs, clear and actionable.

Entities:
{entities_text}

Relationships:
{relationships_text}

SUMMARY:
"""


def format_community_content(nodes: list, relationships: list) -> tuple:
    """
    Format nodes and relationships into readable text for the LLM.
    Returns (entities_text, relationships_text)
    """
    entities_text = ""
    for node in nodes:
        entities_text += f"- {node['name']} [{node['type']}]\n"

    relationships_by_type = {}
    for rel in relationships:
        rel_type = rel["type"]
        if rel_type not in relationships_by_type:
            relationships_by_type[rel_type] = []
        relationships_by_type[rel_type].append(
            f"{rel['from']} -> {rel['to']}"
        )

    relationships_text = ""
    for rel_type, rels in relationships_by_type.items():
        relationships_text += f"\n{rel_type}:\n"
        for rel in rels:
            relationships_text += f"  {rel}\n"

    return entities_text, relationships_text


def summarize_community(community_id: int) -> dict:
    """
    Generate an LLM summary for a specific community
    and store it in ChromaDB immediately.

    Returns:
        {
            "community_id": int,
            "summary": str,
            "node_count": int,
            "rel_count": int,
            "success": bool,
            "error": str (only if failed)
        }
    """
    nodes         = get_community_nodes(community_id)
    relationships = get_community_relationships(community_id)

    if not nodes:
        return {
            "community_id": community_id,
            "success":      False,
            "error":        "Community has no nodes"
        }

    entities_text, relationships_text = format_community_content(nodes, relationships)

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities_text=entities_text,
        relationships_text=relationships_text if relationships_text
                           else "(No internal relationships)"
    )

    try:
        # Direct HTTP call — avoids Ollama SDK proxy issue
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  "llama3.1:latest",
                "prompt": prompt,
                "stream": False
            },
            proxies={"http": None, "https": None},
            timeout=120
        )
        response.raise_for_status()
        summary = response.json()["response"].strip()

        # Embed the summary
        embedding = generate_embedding(summary)

        # Store in ChromaDB immediately
        store_community_summary(community_id, summary, embedding)

        return {
            "community_id": community_id,
            "summary":      summary,
            "node_count":   len(nodes),
            "rel_count":    len(relationships),
            "success":      True
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "community_id": community_id,
            "success":      False,
            "error":        f"Cannot reach Ollama server: {str(e)}"
        }
    except requests.exceptions.Timeout:
        return {
            "community_id": community_id,
            "success":      False,
            "error":        "Ollama request timed out (120s)"
        }
    except Exception as e:
        return {
            "community_id": community_id,
            "success":      False,
            "error":        str(e)
        }


def summarize_all_communities() -> dict:
    """
    Generate and store summaries for ALL detected communities.

    Returns:
        {
            "total_communities": int,
            "successful": int,
            "failed": int,
            "summaries": {community_id: {"summary": str, "node_count": int, "rel_count": int}},
            "errors": [list of error dicts]
        }
    """
    all_communities = get_all_communities()

    if not all_communities:
        return {
            "total_communities": 0,
            "successful":        0,
            "failed":            0,
            "summaries":         {},
            "errors":            ["No communities found. Run community detection first."]
        }

    summaries  = {}
    errors     = []
    successful = 0
    failed     = 0

    for community_id in sorted(all_communities.keys()):
        result = summarize_community(community_id)

        if result["success"]:
            summaries[community_id] = {
                "summary":   result["summary"],
                "node_count": result["node_count"],
                "rel_count":  result["rel_count"]
            }
            successful += 1
        else:
            errors.append({
                "community_id": community_id,
                "error":        result.get("error", "Unknown error")
            })
            failed += 1

    return {
        "total_communities": len(all_communities),
        "successful":        successful,
        "failed":            failed,
        "summaries":         summaries,
        "errors":            errors
    }


def get_community_summary_from_db(community_id: int) -> str:
    """
    Retrieve a stored community summary from ChromaDB by community ID.

    Returns:
        Summary text or empty string if not found
    """
    try:
        from vectorstore.chroma_manager import summaries_collection
        result = summaries_collection.get(
            ids=[f"community_{community_id}"]
        )
        if result["documents"]:
            return result["documents"][0]
        return ""
    except Exception:
        return ""