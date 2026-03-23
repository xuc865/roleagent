"""
Failure mode analysis for baselines (e.g. GRPO) on the mixed benchmark.

This module examines whether a baseline model collapses into a single
reasoning mode when trained/evaluated on a heterogeneous mix of task types.

Key failure modes detected:
  1. **Mode Collapse**: The model converges toward a single reasoning style
     (e.g. always producing long chain-of-thought or always being terse).
  2. **Over-Reasoning**: The model uses deep thinking (<think> blocks) on
     tasks that only require short-chain or direct answers.
  3. **Under-Reasoning**: The model skips reasoning on tasks that need it.
  4. **Wrong Tool Usage**: The model calls tools from a different category.
  5. **Performance Degradation per Category**: Drop from single-task specialist.

Usage
-----
From a checkpoint or evaluation log directory::

    python -m token_agent.analysis.failure_mode_analysis \
        --log_dir ./results/grpo_mixed/ \
        --output_dir ./analysis_output/

Or programmatically::

    from token_agent.analysis import FailureModeAnalyzer
    analyzer = FailureModeAnalyzer(records)
    report = analyzer.full_report()
"""

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from token_agent.data.dataset_registry import (
    DATASET_REGISTRY,
    TASK_CATEGORIES,
    get_task_category,
)
from token_agent.rewards.penalty_reward import (
    CATEGORY_ALLOWED_TOOLS,
    _detect_tools_used,
    _has_think_block,
)

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
    0: "math_reasoning",
    1: "quick_qa",
    2: "direct_qa",
    3: "search_qa",
    4: "action_env",
    5: "game",
    6: "multimodal",
}

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """A single evaluation sample with model response and metadata."""
    data_source: str
    task_category: int
    question: str
    response: str
    ground_truth: Any = None
    reward: float = 0.0
    episode_length: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryStats:
    name: str
    category_id: int
    count: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0

    think_block_rate: float = 0.0
    mean_think_length: float = 0.0

    mean_response_length: float = 0.0
    std_response_length: float = 0.0

    wrong_tool_rate: float = 0.0
    tool_usage: Dict[str, float] = field(default_factory=dict)

    per_dataset: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ModeCollapseMetrics:
    """Metrics quantifying how much the model's reasoning style varies."""
    response_length_cv: float = 0.0
    think_rate_range: float = 0.0
    think_rate_per_category: Dict[str, float] = field(default_factory=dict)
    response_length_per_category: Dict[str, float] = field(default_factory=dict)
    entropy_of_think_usage: float = 0.0
    collapse_score: float = 0.0


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class FailureModeAnalyzer:
    """Analyzes failure modes from a collection of evaluation records."""

    def __init__(self, records: List[EvalRecord]):
        self.records = records
        self._by_category: Dict[int, List[EvalRecord]] = defaultdict(list)
        self._by_dataset: Dict[str, List[EvalRecord]] = defaultdict(list)
        for r in records:
            self._by_category[r.task_category].append(r)
            self._by_dataset[r.data_source].append(r)

    # ------------------------------------------------------------------
    # Per-category statistics
    # ------------------------------------------------------------------

    def category_stats(self, cat_id: int) -> CategoryStats:
        recs = self._by_category.get(cat_id, [])
        if not recs:
            return CategoryStats(name=CATEGORY_NAMES.get(cat_id, "unknown"), category_id=cat_id)

        rewards = np.array([r.reward for r in recs])
        resp_lens = np.array([len(r.response) for r in recs])

        think_flags = [_has_think_block(r.response) for r in recs]
        think_lengths = []
        for r in recs:
            blocks = _THINK_PATTERN.findall(r.response)
            think_lengths.append(sum(len(b) for b in blocks))

        wrong_tool_flags = [
            bool(_detect_tools_used(r.response) - CATEGORY_ALLOWED_TOOLS.get(cat_id, set()))
            for r in recs
        ]
        tool_counter: Counter = Counter()
        for r in recs:
            for t in _detect_tools_used(r.response):
                tool_counter[t] += 1

        per_ds: Dict[str, Dict[str, float]] = {}
        for ds_name, ds_recs in self._by_dataset.items():
            ds_cat_recs = [r for r in ds_recs if r.task_category == cat_id]
            if ds_cat_recs:
                ds_rewards = [r.reward for r in ds_cat_recs]
                per_ds[ds_name] = {
                    "count": len(ds_cat_recs),
                    "mean_reward": float(np.mean(ds_rewards)),
                    "think_rate": float(np.mean([_has_think_block(r.response) for r in ds_cat_recs])),
                }

        return CategoryStats(
            name=CATEGORY_NAMES.get(cat_id, "unknown"),
            category_id=cat_id,
            count=len(recs),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            think_block_rate=float(np.mean(think_flags)),
            mean_think_length=float(np.mean(think_lengths)) if think_lengths else 0.0,
            mean_response_length=float(np.mean(resp_lens)),
            std_response_length=float(np.std(resp_lens)),
            wrong_tool_rate=float(np.mean(wrong_tool_flags)),
            tool_usage={t: c / len(recs) for t, c in tool_counter.items()},
            per_dataset=per_ds,
        )

    def all_category_stats(self) -> Dict[int, CategoryStats]:
        return {cat: self.category_stats(cat) for cat in sorted(self._by_category)}

    # ------------------------------------------------------------------
    # Mode collapse detection
    # ------------------------------------------------------------------

    def mode_collapse_metrics(self) -> ModeCollapseMetrics:
        """
        Compute metrics that measure whether the model collapses into
        a single reasoning mode across different task categories.

        A well-adapted model should:
        - Use <think> blocks rarely on quick_qa/direct_qa, but often on math
        - Have short responses for quick_qa, long for math/search
        - Not use search tools for math, nor action tools for QA

        A collapsed model:
        - Uniform <think> usage across all categories
        - Similar response lengths regardless of task type
        """
        think_rates = {}
        resp_lengths = {}
        all_resp_lens = []

        for cat_id, recs in self._by_category.items():
            cat_name = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
            tr = np.mean([_has_think_block(r.response) for r in recs]) if recs else 0.0
            rl = np.mean([len(r.response) for r in recs]) if recs else 0.0
            think_rates[cat_name] = float(tr)
            resp_lengths[cat_name] = float(rl)
            all_resp_lens.extend([len(r.response) for r in recs])

        all_resp_lens = np.array(all_resp_lens) if all_resp_lens else np.array([0])
        cv = float(np.std(all_resp_lens) / max(np.mean(all_resp_lens), 1e-8))

        tr_values = list(think_rates.values())
        tr_range = max(tr_values) - min(tr_values) if tr_values else 0.0

        # entropy of think usage: if the model uses <think> uniformly across
        # categories, entropy is high (bad). If it's selective, entropy is lower.
        probs = np.array(tr_values)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        entropy = -float(np.mean(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs)))

        # Collapse score: 0 = perfectly adaptive, 1 = fully collapsed
        # Based on: low think_rate_range + low response_length CV + high entropy
        think_range_penalty = max(0, 1.0 - tr_range / 0.5)  # ideally range > 0.5
        resp_len_values = list(resp_lengths.values())
        if len(resp_len_values) >= 2:
            resp_cv = float(np.std(resp_len_values) / max(np.mean(resp_len_values), 1e-8))
            resp_cv_penalty = max(0, 1.0 - resp_cv / 0.5)
        else:
            resp_cv_penalty = 1.0
        collapse_score = float(np.mean([think_range_penalty, resp_cv_penalty, entropy / 1.0]))

        return ModeCollapseMetrics(
            response_length_cv=cv,
            think_rate_range=tr_range,
            think_rate_per_category=think_rates,
            response_length_per_category=resp_lengths,
            entropy_of_think_usage=entropy,
            collapse_score=collapse_score,
        )

    # ------------------------------------------------------------------
    # Specific failure pattern detectors
    # ------------------------------------------------------------------

    def over_reasoning_analysis(self) -> Dict[str, Any]:
        """
        For quick_qa (1) and direct_qa (2): how often does the model
        produce <think> blocks, and what is the cost?
        """
        results = {}
        for cat_id in [1, 2]:
            recs = self._by_category.get(cat_id, [])
            if not recs:
                continue
            with_think = [r for r in recs if _has_think_block(r.response)]
            without_think = [r for r in recs if not _has_think_block(r.response)]

            results[CATEGORY_NAMES.get(cat_id, str(cat_id))] = {
                "total": len(recs),
                "over_reasoning_count": len(with_think),
                "over_reasoning_rate": len(with_think) / len(recs),
                "reward_with_think": float(np.mean([r.reward for r in with_think])) if with_think else None,
                "reward_without_think": float(np.mean([r.reward for r in without_think])) if without_think else None,
                "mean_think_length": float(np.mean([
                    sum(len(b) for b in _THINK_PATTERN.findall(r.response))
                    for r in with_think
                ])) if with_think else 0.0,
            }
        return results

    def under_reasoning_analysis(self) -> Dict[str, Any]:
        """
        For math_reasoning (0): how often does the model skip reasoning?
        """
        recs = self._by_category.get(0, [])
        if not recs:
            return {}
        with_think = [r for r in recs if _has_think_block(r.response)]
        without_think = [r for r in recs if not _has_think_block(r.response)]
        return {
            "total": len(recs),
            "skipped_reasoning_count": len(without_think),
            "skipped_reasoning_rate": len(without_think) / len(recs),
            "reward_with_think": float(np.mean([r.reward for r in with_think])) if with_think else None,
            "reward_without_think": float(np.mean([r.reward for r in without_think])) if without_think else None,
        }

    def wrong_tool_analysis(self) -> Dict[str, Any]:
        """Per-category breakdown of wrong tool usage."""
        results = {}
        for cat_id, recs in self._by_category.items():
            cat_name = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
            allowed = CATEGORY_ALLOWED_TOOLS.get(cat_id, set())
            wrong_usage_details: Counter = Counter()
            wrong_count = 0
            for r in recs:
                used = _detect_tools_used(r.response)
                wrong = used - allowed
                if wrong:
                    wrong_count += 1
                    for t in wrong:
                        wrong_usage_details[t] += 1
            results[cat_name] = {
                "total": len(recs),
                "wrong_tool_count": wrong_count,
                "wrong_tool_rate": wrong_count / len(recs) if recs else 0.0,
                "wrong_tools_detail": dict(wrong_usage_details),
            }
        return results

    def response_style_distribution(self) -> Dict[str, Any]:
        """
        Compute response length and style distributions to detect
        whether the model produces uniform-style outputs.
        """
        cat_distributions = {}
        for cat_id, recs in self._by_category.items():
            cat_name = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
            lens = [len(r.response) for r in recs]
            cat_distributions[cat_name] = {
                "count": len(recs),
                "mean_length": float(np.mean(lens)),
                "median_length": float(np.median(lens)),
                "std_length": float(np.std(lens)),
                "min_length": int(np.min(lens)) if lens else 0,
                "max_length": int(np.max(lens)) if lens else 0,
                "p10_length": float(np.percentile(lens, 10)) if lens else 0,
                "p90_length": float(np.percentile(lens, 90)) if lens else 0,
            }
        return cat_distributions

    def training_dynamics(self, checkpoint_records: Dict[int, List[EvalRecord]]) -> Dict[str, Any]:
        """
        Given records at different training steps, track how metrics evolve.

        Parameters
        ----------
        checkpoint_records : dict mapping step_number -> list of EvalRecords
        """
        dynamics: Dict[str, list] = defaultdict(list)
        for step in sorted(checkpoint_records.keys()):
            analyzer = FailureModeAnalyzer(checkpoint_records[step])
            mc = analyzer.mode_collapse_metrics()
            stats = analyzer.all_category_stats()

            entry = {
                "step": step,
                "collapse_score": mc.collapse_score,
                "think_rate_range": mc.think_rate_range,
            }
            for cat_id, s in stats.items():
                cat_name = CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}")
                entry[f"{cat_name}_reward"] = s.mean_reward
                entry[f"{cat_name}_think_rate"] = s.think_block_rate
                entry[f"{cat_name}_resp_len"] = s.mean_response_length
            dynamics["per_step"].append(entry)
        return dict(dynamics)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive failure mode report."""
        stats = self.all_category_stats()
        mc = self.mode_collapse_metrics()

        report = {
            "summary": {
                "total_samples": len(self.records),
                "categories_present": list(stats.keys()),
                "overall_mean_reward": float(np.mean([r.reward for r in self.records])) if self.records else 0.0,
                "collapse_score": mc.collapse_score,
                "collapse_interpretation": _interpret_collapse(mc.collapse_score),
            },
            "mode_collapse": {
                "collapse_score": mc.collapse_score,
                "think_rate_range": mc.think_rate_range,
                "think_rate_per_category": mc.think_rate_per_category,
                "response_length_per_category": mc.response_length_per_category,
                "response_length_cv": mc.response_length_cv,
                "entropy_of_think_usage": mc.entropy_of_think_usage,
            },
            "per_category": {
                CATEGORY_NAMES.get(cat_id, f"cat_{cat_id}"): _stats_to_dict(s)
                for cat_id, s in stats.items()
            },
            "over_reasoning": self.over_reasoning_analysis(),
            "under_reasoning": self.under_reasoning_analysis(),
            "wrong_tool_usage": self.wrong_tool_analysis(),
            "response_style": self.response_style_distribution(),
        }
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpret_collapse(score: float) -> str:
    if score < 0.3:
        return "LOW: Model adapts its reasoning style well across task categories."
    if score < 0.6:
        return "MODERATE: Some mode collapse detected – the model partially differentiates task types."
    return "HIGH: Severe mode collapse – the model uses a near-uniform reasoning style regardless of task type."


def _stats_to_dict(s: CategoryStats) -> dict:
    return {
        "count": s.count,
        "mean_reward": s.mean_reward,
        "std_reward": s.std_reward,
        "think_block_rate": s.think_block_rate,
        "mean_think_length": s.mean_think_length,
        "mean_response_length": s.mean_response_length,
        "std_response_length": s.std_response_length,
        "wrong_tool_rate": s.wrong_tool_rate,
        "tool_usage": s.tool_usage,
        "per_dataset": s.per_dataset,
    }


# ---------------------------------------------------------------------------
# Loading from various formats
# ---------------------------------------------------------------------------

def load_records_from_jsonl(path: str) -> List[EvalRecord]:
    """Load records from a JSONL file (one JSON per line)."""
    records = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            records.append(EvalRecord(
                data_source=d.get("data_source", "unknown"),
                task_category=d.get("task_category", get_task_category(d.get("data_source", "unknown"))),
                question=d.get("question", ""),
                response=d.get("response", ""),
                ground_truth=d.get("ground_truth"),
                reward=float(d.get("reward", d.get("score", 0.0))),
                episode_length=int(d.get("episode_length", 1)),
                extra=d.get("extra", {}),
            ))
    return records


def load_records_from_rollout_dir(rollout_dir: str) -> List[EvalRecord]:
    """
    Load records from a rollout output directory.
    Expects per-sample JSON files or a single results.jsonl.
    """
    rollout_path = Path(rollout_dir)
    records = []

    jsonl_files = list(rollout_path.glob("*.jsonl"))
    for jf in jsonl_files:
        records.extend(load_records_from_jsonl(str(jf)))

    json_files = list(rollout_path.glob("*.json"))
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list):
            for d in data:
                records.append(EvalRecord(
                    data_source=d.get("data_source", "unknown"),
                    task_category=d.get("task_category", get_task_category(d.get("data_source", "unknown"))),
                    question=d.get("question", ""),
                    response=d.get("response", ""),
                    ground_truth=d.get("ground_truth"),
                    reward=float(d.get("reward", d.get("score", 0.0))),
                    episode_length=int(d.get("episode_length", 1)),
                    extra=d.get("extra", {}),
                ))
    return records


def load_records_from_wandb_table(table_path: str) -> List[EvalRecord]:
    """
    Load from a W&B exported table (JSON with 'data' and 'columns' keys).
    """
    with open(table_path) as f:
        table = json.load(f)
    columns = table["columns"]
    col_idx = {c: i for i, c in enumerate(columns)}
    records = []
    for row in table["data"]:
        def _get(key, default=None):
            return row[col_idx[key]] if key in col_idx else default
        ds = _get("data_source", "unknown") or "unknown"
        cat = _get("task_category", None)
        cat = int(cat) if cat is not None else get_task_category(ds)
        records.append(EvalRecord(
            data_source=ds,
            task_category=cat,
            question=_get("question", "") or "",
            response=_get("response", "") or "",
            ground_truth=_get("ground_truth"),
            reward=float(_get("reward", _get("score", 0.0)) or 0.0),
        ))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze GRPO/baseline failure modes on the mixed benchmark."
    )
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory with evaluation results (JSONL/JSON files)")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Where to save the analysis report")
    parser.add_argument("--format", choices=["jsonl", "dir", "wandb"], default="dir",
                        help="Format of input data")
    args = parser.parse_args()

    if args.format == "jsonl":
        records = load_records_from_jsonl(args.log_dir)
    elif args.format == "wandb":
        records = load_records_from_wandb_table(args.log_dir)
    else:
        records = load_records_from_rollout_dir(args.log_dir)

    if not records:
        print("No records found. Check your --log_dir path.")
        return

    print(f"Loaded {len(records)} evaluation records.")

    analyzer = FailureModeAnalyzer(records)
    report = analyzer.full_report()

    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, "failure_mode_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")

    _print_summary(report)


def _print_summary(report: dict):
    """Pretty-print the key findings."""
    s = report["summary"]
    print("\n" + "=" * 72)
    print("FAILURE MODE ANALYSIS REPORT")
    print("=" * 72)
    print(f"Total samples: {s['total_samples']}")
    print(f"Overall mean reward: {s['overall_mean_reward']:.4f}")
    print(f"Collapse score: {s['collapse_score']:.3f}")
    print(f"  → {s['collapse_interpretation']}")

    mc = report["mode_collapse"]
    print(f"\nThink-block usage rate per category:")
    for cat, rate in mc["think_rate_per_category"].items():
        print(f"  {cat:20s}: {rate:.1%}")
    print(f"  Range: {mc['think_rate_range']:.1%}")

    print(f"\nMean response length per category:")
    for cat, length in mc["response_length_per_category"].items():
        print(f"  {cat:20s}: {length:.0f} chars")

    print(f"\nPer-category reward:")
    for cat_name, stats in report["per_category"].items():
        print(f"  {cat_name:20s}: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}  (n={stats['count']})")
        if stats["wrong_tool_rate"] > 0:
            print(f"    ⚠ Wrong tool rate: {stats['wrong_tool_rate']:.1%}")

    if report["over_reasoning"]:
        print(f"\nOver-reasoning analysis:")
        for cat, info in report["over_reasoning"].items():
            print(f"  {cat}: {info['over_reasoning_rate']:.1%} of samples have <think> blocks")
            if info["reward_with_think"] is not None and info["reward_without_think"] is not None:
                delta = info["reward_without_think"] - info["reward_with_think"]
                print(f"    Reward WITHOUT think: {info['reward_without_think']:.4f}")
                print(f"    Reward WITH think:    {info['reward_with_think']:.4f}")
                print(f"    Delta (penalty):      {delta:+.4f}")

    if report["under_reasoning"]:
        ur = report["under_reasoning"]
        print(f"\nUnder-reasoning (math):")
        print(f"  {ur['skipped_reasoning_rate']:.1%} of math samples lack <think> blocks")
        if ur["reward_with_think"] is not None and ur["reward_without_think"] is not None:
            print(f"  Reward WITH think:    {ur['reward_with_think']:.4f}")
            print(f"  Reward WITHOUT think: {ur['reward_without_think']:.4f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
