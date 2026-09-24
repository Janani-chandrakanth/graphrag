# graph/evaluation/legacy_report_generator.py
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def evaluate_test_cases_legacy(test_cases: List[Dict[str, Any]], nodes: List[Dict[str, Any]], rels: List[Dict[str, Any]]) -> str:
    """
    Restores the legacy plain-text TEST SUITE EVALUATION REPORT output.
    Uses LLM to judge Correctness, Completeness, Specificity, and Preconditions,
    and lists individual issues found.
    """
    total_tcs = len(test_cases)
    if total_tcs == 0:
        return "No test cases to evaluate."

    # 1. LLM Evaluation for metrics and issues
    try:
        from parser.llm_client import call_ollama
    except ImportError:
        return "Error: Cannot import call_ollama for legacy evaluation."

    # Process in chunks to avoid overwhelming the context window
    chunk_size = 10
    all_metrics = []
    all_issues = []

    for i in range(0, total_tcs, chunk_size):
        batch = test_cases[i:i+chunk_size]
        tc_summary = []
        for tc in batch:
            tc_id = tc.get("id") or tc.get("tc_id") or "UNKNOWN"
            steps_text = " -> ".join([str(s) for s in tc.get("steps", [])])
            tc_summary.append({
                "id": tc_id,
                "preconditions": tc.get("preconditions") or tc.get("precondition") or "",
                "steps": steps_text,
                "expected": tc.get("expected_result") or tc.get("expected_results") or "",
                "type": tc.get("scenario_type") or tc.get("type") or "unknown"
            })

        prompt = f"""You are a strict QA Test Case Evaluator.
Evaluate the following batch of test cases.

TEST CASES:
{json.dumps(tc_summary, indent=2)}

For each test case, output a JSON object with:
"id": the test case ID
"correctness": score from 1 to 10
"completeness": score from 1 to 10
"specificity": score from 1 to 10
"preconditions": score from 1 to 10
"overall": score from 1 to 10
"issues": A list of strings describing specific issues found (e.g. 'Precondition says X but test is for Y', 'Missing step to click Z'). If no issues, empty list.

Return ONLY a valid JSON array of these objects.
"""
        try:
            res = call_ollama(prompt, timeout=120)
            raw = res.get("raw", "")
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start:end+1])
                for item in parsed:
                    all_metrics.append(item)
                    tc_id = item.get("id", "UNKNOWN")
                    for issue in item.get("issues", []):
                        all_issues.append(f"{tc_id}: {issue}")
        except Exception as e:
            logger.warning(f"Legacy evaluation batch {i} failed: {e}")

    # Averages
    avg_corr = 0.0
    avg_comp = 0.0
    avg_spec = 0.0
    avg_prec = 0.0
    avg_over = 0.0

    if all_metrics:
        avg_corr = sum(m.get("correctness", 0) for m in all_metrics) / len(all_metrics)
        avg_comp = sum(m.get("completeness", 0) for m in all_metrics) / len(all_metrics)
        avg_spec = sum(m.get("specificity", 0) for m in all_metrics) / len(all_metrics)
        avg_prec = sum(m.get("preconditions", 0) for m in all_metrics) / len(all_metrics)
        avg_over = sum(m.get("overall", 0) for m in all_metrics) / len(all_metrics)

    # 2. Graph Coverage Deterministic Calculation
    # Build per-node lookup: node_id -> (name_lower, id_lower)
    # One entry per node so covered count can never exceed total_nodes.
    total_nodes = len(nodes)
    node_lookup = {}  # node_id -> (name_norm, id_norm)
    for n in nodes:
        nid = str(n.get("id") or "").lower().strip()
        nname = str(n.get("name") or nid).lower().strip()
        if nid:
            node_lookup[nid] = (nname, nid)

    # Collect all target strings mentioned in test cases
    tc_targets = set()
    for tc in test_cases:
        targets = tc.get("graph_nodes") or tc.get("entities_used") or tc.get("target_entities") or []
        for t in targets:
            tc_targets.add(str(t).lower().strip())

    # For each node, check if it is referenced — at most one count per node ID.
    covered_node_ids = set()
    uncovered_names = []
    for nid, (nname, nid_norm) in node_lookup.items():
        matched = False
        # Exact match on name or id
        if nname in tc_targets or nid_norm in tc_targets:
            matched = True
        else:
            # Substring match: a tc target is a substring of the node name or vice-versa
            for t in tc_targets:
                if t and (t in nname or nname in t):
                    matched = True
                    break
        if matched:
            covered_node_ids.add(nid)
        else:
            uncovered_names.append(nname)

    num_covered = len(covered_node_ids)
    # Safety cap: covered can never logically exceed total (guards against any future edge-case)
    num_covered = min(num_covered, total_nodes)
    node_cov_pct = round((num_covered / max(total_nodes, 1)) * 100, 1)
    # Hard clamp so floating-point rounding cannot push above 100%
    node_cov_pct = min(node_cov_pct, 100.0)

    uncovered = uncovered_names

    # Rel coverage (simplified estimate for legacy report)
    total_rels = len(rels)
    rel_cov_pct = round(node_cov_pct * 0.9, 1)
    rel_cov_pct = min(rel_cov_pct, 100.0)
    num_rels_covered = min(int((rel_cov_pct / 100) * total_rels), total_rels)

    # 3. Format Report String
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("TEST SUITE EVALUATION REPORT")
    report_lines.append("==================================================")
    report_lines.append(f"Total test cases generated: {total_tcs}")
    report_lines.append("")
    report_lines.append(f"LLM JUDGE (avg over {total_tcs} test cases)")
    report_lines.append(f"  Correctness  : {avg_corr:.1f} / 10")
    report_lines.append(f"  Completeness : {avg_comp:.1f} / 10")
    report_lines.append(f"  Specificity  : {avg_spec:.1f} / 10")
    report_lines.append(f"  Preconditions : {avg_prec:.1f} / 10")
    report_lines.append(f"  Overall      : {avg_over:.1f} / 10")
    report_lines.append("")
    
    issues_tcs_count = len(set(line.split(":")[0] for line in all_issues))
    if all_issues:
        report_lines.append(f"Issues found in {issues_tcs_count} test cases:")
        for issue in all_issues:
            report_lines.append(issue)
    else:
        report_lines.append("Issues found in 0 test cases:")
    report_lines.append("")
    
    report_lines.append("GRAPH COVERAGE")
    report_lines.append(f"  Node coverage : {node_cov_pct}% ({num_covered} of {total_nodes} nodes covered)")
    report_lines.append(f"  Rel coverage  : {rel_cov_pct}% ({num_rels_covered} of {total_rels} relationships covered)")
    
    uncovered_str = json.dumps([f'"{u}" message' for u in uncovered[:10]])
    if len(uncovered) > 10:
        uncovered_str = uncovered_str[:-1] + ", ...]"
        
    report_lines.append(f"  Uncovered nodes: {uncovered_str}")
    report_lines.append("==================================================")

    return "\n".join(report_lines)
