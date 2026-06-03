import re
from typing import Any, Dict, Optional, Set

from token_agent.data.dataset_registry import get_task_category

_SEARCH_PATTERN = re.compile(r"<search>.*?</search>", re.DOTALL)
_ACTION_PATTERN = re.compile(r"<action>.*?</action>", re.DOTALL)
_THINK_PATTERN = re.compile(r".*? ", re.DOTALL)

CATEGORY_ALLOWED_TOOLS: Dict[int, Set[str]] = {
    0: set(),
    1: set(),
    2: set(),
    3: {"search"},
    4: {"action"},
}


def _detect_tools_used(response: str) -> Set[str]:
    tools = set()
    if _SEARCH_PATTERN.search(response):
        tools.add("search")
    if _ACTION_PATTERN.search(response):
        tools.add("action")
    return tools


def _has_think_block(response: str) -> bool:
    return bool(_THINK_PATTERN.search(response))


def _detect_wrong_tool(response: str, task_category: int) -> bool:
    used = _detect_tools_used(response)
    if not used:
        return False
    allowed = CATEGORY_ALLOWED_TOOLS.get(task_category, set())
    return bool(used - allowed)


def _squad_score(solution_str: str, ground_truth) -> float:
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


# SearchR1 data_source names as they appear in the raw search_qa parquet.
# The parquet preserves the original data_source from SearchR1 preprocessing
# (e.g. "nq", "triviaqa") rather than the unified "search_qa" label.
_SEARCH_R1_DATA_SOURCES = {
    "nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique",
    "bamboogle", "search_qa",
    # Also handle the "searchR1_*" prefix variants just in case.
    "searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle",
}


def compute_base_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
) -> float:
    if data_source == "squad":
        return _squad_score(solution_str, ground_truth)

    if data_source in ("simpleqa",):
        return _simple_em_score(solution_str, ground_truth)

    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        return float(gsm8k.compute_score(solution_str, ground_truth, method="flexible"))

    # SearchR1-style QA datasets: use the same EM scoring as the original SearchR1.
    # The raw search_qa parquet preserves the original data_source names ("nq",
    # "triviaqa", etc.) rather than the unified "search_qa" label, so we must
    # handle them explicitly here instead of falling through to default_compute_score.
    if data_source in _SEARCH_R1_DATA_SOURCES:
        return _simple_em_score(solution_str, ground_truth)

    from verl.utils.reward_score import default_compute_score
    res = default_compute_score(data_source, solution_str, ground_truth, extra_info)
    return float(res) if not isinstance(res, dict) else float(res.get("score", 0.0))


def compute_reward_with_penalties(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    task_category: Optional[int] = None,
    extra_info: Optional[Dict] = None,
    overthinking_penalty: float = 1.0,
    wrong_tool_penalty: float = 0.2,
) -> float:
    if task_category is None:
        task_category = get_task_category(data_source)

    base_reward = compute_base_reward(data_source, solution_str, ground_truth, extra_info)

    if task_category in (1, 2) and _has_think_block(solution_str):
        base_reward = max(0.0, base_reward - overthinking_penalty)

    if base_reward >= wrong_tool_penalty and _detect_wrong_tool(solution_str, task_category):
        base_reward = max(0.0, base_reward - wrong_tool_penalty)

    return base_reward
