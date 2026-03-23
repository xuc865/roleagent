"""
Reward computation with Token-Agent penalties:

1. **Over-reasoning penalty**: If the task is quick_qa (1) or direct_qa (2)
   and the model produced a ``<think>`` block, penalise.
2. **Wrong-tool penalty**: If the model used a tool belonging to a different
   task category, subtract a fixed penalty (default 20 %) from the reward,
   but only when the base reward >= 0.2 to avoid going negative.
"""

import re
from typing import Any, Dict, List, Optional, Set

from token_agent.data.dataset_registry import get_task_category

# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------

_SEARCH_PATTERN = re.compile(r"<search>.*?</search>", re.DOTALL)
_ACTION_PATTERN = re.compile(r"<action>.*?</action>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _detect_tools_used(response: str) -> Set[str]:
    tools = set()
    if _SEARCH_PATTERN.search(response):
        tools.add("search")
    if _ACTION_PATTERN.search(response):
        tools.add("action")
    return tools


def _has_think_block(response: str) -> bool:
    return bool(_THINK_PATTERN.search(response))


# Which tools are *allowed* for each category.
# Empty set = no tools expected.
CATEGORY_ALLOWED_TOOLS: Dict[int, Set[str]] = {
    0: set(),            # math_reasoning
    1: set(),            # quick_qa
    2: set(),            # direct_qa
    3: {"search"},       # search_qa
    4: {"action"},       # action_env
}


def _detect_wrong_tool(response: str, task_category: int) -> bool:
    used = _detect_tools_used(response)
    if not used:
        return False
    allowed = CATEGORY_ALLOWED_TOOLS.get(task_category, set())
    return bool(used - allowed)


# ---------------------------------------------------------------------------
# Base reward helpers for newly-added datasets
# ---------------------------------------------------------------------------

def _squad_score(solution_str: str, ground_truth) -> float:
    """F1 / EM for SQuAD-style extractive QA."""
    from verl.utils.reward_score.search_r1_like_qa_em import (
        extract_solution,
        em_check,
        subem_check,
    )
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    targets = ground_truth
    if isinstance(targets, dict):
        targets = targets.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    if em_check(answer, targets):
        return 1.0
    if subem_check(answer, targets):
        return 0.5
    return 0.0


def _simple_em_score(solution_str: str, ground_truth) -> float:
    """Simple exact-match scoring for SimpleQA / AA-Omniscience."""
    from verl.utils.reward_score.search_r1_like_qa_em import (
        extract_solution,
        em_check,
    )
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    targets = ground_truth
    if isinstance(targets, dict):
        targets = targets.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    return 1.0 if em_check(answer, targets) else 0.0


# ---------------------------------------------------------------------------
# Extended compute_score that covers new data_sources
# ---------------------------------------------------------------------------

def compute_base_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
) -> float:
    """
    Compute the raw reward for a response. Falls back to the
    existing ``default_compute_score`` for datasets already handled
    by verl, and adds support for squad / simpleqa / aa_omniscience.
    """
    if data_source == "squad":
        return _squad_score(solution_str, ground_truth)
    if data_source in ("simpleqa", "aa_omniscience"):
        return _simple_em_score(solution_str, ground_truth)

    from verl.utils.reward_score import default_compute_score
    res = default_compute_score(data_source, solution_str, ground_truth, extra_info)
    return float(res) if not isinstance(res, dict) else float(res.get("score", 0.0))


# ---------------------------------------------------------------------------
# Main: reward + penalties
# ---------------------------------------------------------------------------

def compute_reward_with_penalties(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    task_category: Optional[int] = None,
    extra_info: Optional[Dict] = None,
    overthinking_penalty: float = 1.0,
    wrong_tool_penalty: float = 0.2,
) -> float:
    """
    Compute reward with Token-Agent-specific penalties.

    Parameters
    ----------
    overthinking_penalty : float
        How much to subtract when <think> is used on quick_qa / direct_qa.
        Default 1.0 means the reward goes to 0 (since max base reward = 1).
    wrong_tool_penalty : float
        Fixed amount to subtract when a tool from the wrong category is used.
        Only applied when ``base_reward >= wrong_tool_penalty``.
    """
    if task_category is None:
        task_category = get_task_category(data_source)

    base_reward = compute_base_reward(data_source, solution_str, ground_truth, extra_info)

    # --- Over-reasoning penalty ---
    if task_category in (1, 2) and _has_think_block(solution_str):
        base_reward = max(0.0, base_reward - overthinking_penalty)

    # --- Wrong-tool penalty ---
    if base_reward >= wrong_tool_penalty and _detect_wrong_tool(solution_str, task_category):
        base_reward = max(0.0, base_reward - wrong_tool_penalty)

    return base_reward
