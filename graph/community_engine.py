"""
graph/community_engine.py

Consolidated Community Engine:
1. Community Detection (Louvain algorithm over Neo4j)
2. Community Summarization (LLM summaries of detected communities)
3. Community Hierarchy (Multi-level ROOT / LOW / HIGH hierarchical summaries)
"""

import json
import math
import requests
import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

from config import OLLAMA_URL
from vectorstore.chroma_manager import (
    generate_embedding,
    store_community_summary,
    search_community_summaries,
    summaries_collection
)
from graph.neo4j_manager import (
    get_all_nodes,
    get_all_relationships,
    get_community_nodes_db,
    get_community_relationships_db,
    get_all_communities_db,
    write_community_ids
)


# =========================================================
# 1. COMMUNITY DETECTION
# =========================================================

def build_graph_from_neo4j() -> nx.DiGraph:
    """
    Pull all nodes and relationships from Neo4j.
    Build a NetworkX graph for community detection.
    """
    nodes = get_all_nodes()
    rels  = get_all_relationships()

    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node["id"], name=node["name"], type=node["type"])
    for rel in rels:
        G.add_edge(rel["from"], rel["to"], relationship_type=rel["type"])

    return G


def detect_communities(G: nx.DiGraph) -> dict:
    """
    Run Louvain community detection.
    Converts directed graph to undirected first.

    Returns:
        {node_id: community_id, ...}
    """
    G_undirected = G.to_undirected()
    if community_louvain is not None:
        partition = community_louvain.best_partition(
            G_undirected,
            resolution=1.0,
            random_state=42
        )
        return partition
    
    # Fallback to networkx community algorithms if python-louvain is not installed
    try:
        communities = nx.community.louvain_communities(G_undirected, seed=42)
    except Exception:
        communities = nx.community.greedy_modularity_communities(G_undirected)

    partition = {}
    for comm_id, node_set in enumerate(communities):
        for node in node_set:
            partition[node] = comm_id
    return partition


def get_community_nodes(community_id: int) -> list:
    """Get all nodes in a community from Neo4j."""
    return get_community_nodes_db(community_id)


def get_community_relationships(community_id: int) -> list:
    """Get all relationships within a community from Neo4j."""
    return get_community_relationships_db(community_id)


def get_all_communities() -> dict:
    """Get all communities and their info from Neo4j."""
    return get_all_communities_db()


def run_full_community_detection() -> dict:
    """
    Complete pipeline:
    1. Pull graph from Neo4j
    2. Run Louvain detection
    3. Write community IDs back to Neo4j
    """
    G = build_graph_from_neo4j()

    if len(G.nodes()) == 0:
        return {
            "success":           False,
            "error":             "No nodes found in Neo4j. Build the index first.",
            "total_communities": 0
        }

    partition = detect_communities(G)
    stats     = write_community_ids(partition)
    return stats


# =========================================================
# 2. COMMUNITY SUMMARIZATION
# =========================================================

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
        result = summaries_collection.get(
            ids=[f"community_{community_id}"]
        )
        if result["documents"]:
            return result["documents"][0]
        return ""
    except Exception:
        return ""


# =========================================================
# 3. COMMUNITY HIERARCHY
# =========================================================

MERGE_SUMMARY_PROMPT = """
You are an expert requirements analyst.

Below are summaries from multiple related requirement groups.
Generate a single consolidated summary that captures the overall
business capability these groups represent together.

Include:
- The overall business domain or system purpose
- Key actors and their roles
- Major features and workflows
- Important constraints or business rules
- How these groups relate to each other

Keep the summary to 3-4 paragraphs, clear and comprehensive.

COMMUNITY SUMMARIES TO MERGE:
{summaries_text}

CONSOLIDATED SUMMARY:
"""

ROOT_SUMMARY_PROMPT = """
You are an expert requirements analyst.

Below are all the requirement group summaries from an entire
software requirements document.

Generate a single high-level executive summary of the entire system.

Include:
- What the system is and its primary purpose
- Who the main users/actors are
- Core features and capabilities
- Key business rules and constraints
- Overall system scope

Keep the summary to 4-5 paragraphs, comprehensive yet concise.

ALL COMMUNITY SUMMARIES:
{summaries_text}

EXECUTIVE SUMMARY:
"""


def call_llm(prompt: str) -> str:
    """
    Call llama3.1 via direct HTTP.
    Returns the response text.
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model":  "llama3.1:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "seed":        42
            }
        },
        proxies={"http": None, "https": None},
        timeout=180
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def get_high_level_summaries(summary_results: dict) -> dict:
    """
    Get all existing HIGH level summaries from session state.

    Args:
        summary_results: the summary_results from session state
                        {community_id: {"summary": str, ...}}

    Returns:
        {community_id: summary_text}
    """
    high_summaries = {}
    for community_id, data in summary_results.items():
        high_summaries[int(community_id)] = data["summary"]
    return high_summaries


def build_low_level(
    high_summaries: dict,
    target_groups: int = None
) -> dict:
    """
    Build LOW level by merging HIGH level communities into groups.
    """
    if not high_summaries:
        return {}

    community_ids = sorted(high_summaries.keys())
    num_communities = len(community_ids)

    if target_groups is None:
        target_groups = max(2, math.ceil(math.sqrt(num_communities)))

    target_groups = min(target_groups, num_communities)

    group_size = math.ceil(num_communities / target_groups)
    groups = []
    for i in range(0, num_communities, group_size):
        groups.append(community_ids[i:i + group_size])

    low_level = {}

    for group_idx, group_ids in enumerate(groups):
        summaries_text = ""
        for cid in group_ids:
            summaries_text += f"\n[Community {cid}]\n"
            summaries_text += high_summaries[cid]
            summaries_text += "\n---\n"

        prompt = MERGE_SUMMARY_PROMPT.format(
            summaries_text=summaries_text
        )

        try:
            summary   = call_llm(prompt)
            embedding = generate_embedding(summary)

            store_community_summary(
                community_id=f"low_{group_idx}",
                summary_text=summary,
                embedding=embedding,
                level="LOW",
                group_id=group_idx,
                communities_included=group_ids
            )

            low_level[group_idx] = {
                "summary":               summary,
                "embedding":             embedding,
                "communities_included":  group_ids,
                "level":                 "LOW"
            }

        except Exception as e:
            low_level[group_idx] = {
                "summary":               "",
                "communities_included":  group_ids,
                "level":                 "LOW",
                "error":                 str(e)
            }

    return low_level


def build_root_level(high_summaries: dict) -> dict:
    """
    Build ROOT level — one single summary of the entire document.
    """
    if not high_summaries:
        return {}

    summaries_text = ""
    for community_id in sorted(high_summaries.keys()):
        summaries_text += f"\n[Community {community_id}]\n"
        summaries_text += high_summaries[community_id]
        summaries_text += "\n---\n"

    prompt = ROOT_SUMMARY_PROMPT.format(
        summaries_text=summaries_text
    )

    try:
        summary   = call_llm(prompt)
        embedding = generate_embedding(summary)

        all_ids = sorted(high_summaries.keys())

        store_community_summary(
            community_id="root",
            summary_text=summary,
            embedding=embedding,
            level="ROOT",
            group_id=0,
            communities_included=all_ids
        )

        return {
            "summary":              summary,
            "embedding":            embedding,
            "communities_included": all_ids,
            "level":                "ROOT"
        }

    except Exception as e:
        return {
            "summary": "",
            "level":   "ROOT",
            "error":   str(e)
        }


def build_full_hierarchy(summary_results: dict) -> dict:
    """
    Build complete hierarchy from existing HIGH level summaries.
    """
    high_summaries = get_high_level_summaries(summary_results)

    if not high_summaries:
        return {
            "error": "No HIGH level summaries found. Generate community summaries first."
        }

    low_level = build_low_level(high_summaries)
    root_level = build_root_level(high_summaries)

    return {
        "root":  root_level,
        "low":   low_level,
        "high":  high_summaries,
        "stats": {
            "total_high": len(high_summaries),
            "total_low":  len(low_level),
            "total_root": 1 if root_level.get("summary") else 0
        }
    }
