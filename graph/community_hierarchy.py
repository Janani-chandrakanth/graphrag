"""
Hierarchical Community Structure Module

Builds three levels of community summaries:

ROOT (Level 0) — one summary for the entire document
LOW  (Level 1) — groups of communities merged together
HIGH (Level 2) — individual communities (already built)

Flow:
1. HIGH level already exists from community_summarizer.py
2. LOW level  — merge communities into N groups, summarize each group
3. ROOT level — merge ALL communities into one, summarize everything
"""

import json
import requests
import math
from config import OLLAMA_URL
from embeddings.embedding_model import generate_embedding
from graph.community_detector import get_all_communities, get_community_nodes, get_community_relationships
from vectorstore.chroma_manager import store_community_summary


# ── Prompt for merging multiple community summaries ────────
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

    If you have 6 communities, LOW level might have 2-3 groups.
    Each group merges 2-3 related communities into one summary.

    How grouping works:
    - Sort communities by ID
    - Divide into roughly equal groups
    - Merge each group into one summary

    Args:
        high_summaries: {community_id: summary_text}
        target_groups: how many LOW groups to create
                      defaults to sqrt(num_communities)

    Returns:
        {
            low_group_id: {
                "summary": str,
                "embedding": list,
                "communities_included": list,
                "level": "LOW"
            }
        }
    """
    if not high_summaries:
        return {}

    community_ids = sorted(high_summaries.keys())
    num_communities = len(community_ids)

    # Default: square root gives a sensible number of groups
    # 4 communities → 2 groups
    # 9 communities → 3 groups
    # 16 communities → 4 groups
    if target_groups is None:
        target_groups = max(2, math.ceil(math.sqrt(num_communities)))

    # If fewer communities than target groups, just use num_communities
    target_groups = min(target_groups, num_communities)

    # Split community IDs into groups
    group_size = math.ceil(num_communities / target_groups)
    groups = []
    for i in range(0, num_communities, group_size):
        groups.append(community_ids[i:i + group_size])

    low_level = {}

    for group_idx, group_ids in enumerate(groups):
        # Build merged text from all communities in this group
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

            # Store in ChromaDB with level prefix
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

    Merges ALL community summaries into one executive summary.

    Args:
        high_summaries: {community_id: summary_text}

    Returns:
        {
            "summary": str,
            "embedding": list,
            "communities_included": list,
            "level": "ROOT"
        }
    """
    if not high_summaries:
        return {}

    # Build text with all summaries
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

        # Store in ChromaDB
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

    Args:
        summary_results: st.session_state["summary_results"]["summaries"]
                        {community_id: {"summary": str, ...}}

    Returns:
        {
            "root": {...},
            "low":  {0: {...}, 1: {...}},
            "high": {0: {...}, 1: {...}, ...},  # already exists
            "stats": {
                "total_high": int,
                "total_low":  int,
                "total_root": int
            }
        }
    """
    # Get existing HIGH summaries
    high_summaries = get_high_level_summaries(summary_results)

    if not high_summaries:
        return {
            "error": "No HIGH level summaries found. Generate community summaries first."
        }

    # Build LOW level
    low_level = build_low_level(high_summaries)

    # Build ROOT level
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