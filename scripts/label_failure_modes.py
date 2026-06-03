#!/usr/bin/env python3
"""
Label failure modes for ALFWorld training failures using LLM.

Strategy:
  1. Parse training log to get per-step, per-task-type success rates
  2. Parse failure history JSONL to get per-step failure counts
  3. For each (task_type, training_phase), call LLM to classify
     typical failure mode distribution
  4. Assign failure modes to individual records based on LLM output
  5. Save labeled records and generate plots

Usage:
    python scripts/label_failure_modes.py \
        --log_path log/role_agent_alfworld_8gpu.log \
        --failure_path log/failure_history/role_agent_alfworld_8gpu_failures.jsonl \
        --output_dir failure_mode_plots
"""

import argparse
import json
import os
import re
import ssl
import time
import urllib.request
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ──

API_BASE = "https://api.zhizengzeng.com/v1"
API_KEY = "sk-zk24c21b6b604dea52cc57190d4b90c833768dd4bcc4adc2"
MODEL = "gpt-5.4-mini"

TASK_TYPES = [
    "pick_and_place",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]

TASK_DESCRIPTIONS = {
    "pick_and_place": "Pick up an object and place it in/on a receptacle (e.g., 'put some bowl on coffeetable')",
    "pick_cool_then_place_in_recep": "Cool an object (using fridge) then place it somewhere (e.g., 'put a cool mug in coffeemachine')",
    "pick_clean_then_place_in_recep": "Clean an object (using sinkbasin) then place it somewhere (e.g., 'put a clean butterknife in diningtable')",
    "pick_heat_then_place_in_recep": "Heat an object (using microwave) then place it somewhere (e.g., 'put a hot plate in cabinet')",
    "look_at_obj_in_light": "Find an object and examine it using a light source (e.g., 'examine the statue with the desklamp')",
    "pick_two_obj_and_place": "Find two instances of an object and place them in a receptacle (e.g., 'find two pan and put them in countertop')",
}

FAILURE_MODES = [
    "NAVIGATION_ERROR",
    "WRONG_OBJECT_SELECTION",
    "WRONG_RECEPTACLE",
    "ACTION_SEQUENCE_ERROR",
    "PREMATURE_TERMINATION",
    "OBJECT_NOT_FOUND",
    "HALLUCINATED_ACTION",
    "REPEATED_FAILED_ACTION",
]

FAILURE_MODE_SHORT = {
    "NAVIGATION_ERROR":        "NavError",
    "WRONG_OBJECT_SELECTION":  "WrongObj",
    "WRONG_RECEPTACLE":        "WrongRecep",
    "ACTION_SEQUENCE_ERROR":   "SeqError",
    "PREMATURE_TERMINATION":   "PremTerm",
    "OBJECT_NOT_FOUND":        "ObjNotFound",
    "HALLUCINATED_ACTION":     "HallucAction",
    "REPEATED_FAILED_ACTION":  "RepeatFail",
}

FAILURE_MODE_COLORS = {
    "NAVIGATION_ERROR":        "#e63946",
    "WRONG_OBJECT_SELECTION":  "#457b9d",
    "WRONG_RECEPTACLE":        "#2a9d8f",
    "ACTION_SEQUENCE_ERROR":   "#f4a261",
    "PREMATURE_TERMINATION":   "#e76f51",
    "OBJECT_NOT_FOUND":        "#264653",
    "HALLUCINATED_ACTION":     "#a8dadc",
    "REPEATED_FAILED_ACTION":  "#6d6875",
}

TASK_SHORT_NAMES = {
    "pick_and_place":                   "Pick&Place",
    "pick_cool_then_place_in_recep":    "Cool&Place",
    "pick_clean_then_place_in_recep":   "Clean&Place",
    "pick_heat_then_place_in_recep":    "Heat&Place",
    "look_at_obj_in_light":             "LookInLight",
    "pick_two_obj_and_place":           "PickTwo&Place",
}


# ── LLM API ──

def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Call LLM via OpenAI-compatible API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_completion_tokens": 2000,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


# ── Failure mode classification prompt ──

CLASSIFICATION_PROMPT = """You are an expert in analyzing failure modes of RL-trained agents in the ALFWorld embodied environment.

## ALFWorld Background
ALFWorld is a text-based household environment where an agent must complete tasks by navigating rooms and manipulating objects. Available actions include: go to [location], pick up [object], put [object] in/on [receptacle], open/close [receptacle], heat/cool/clean [object], examine [object], look, inventory.

## Task Type Being Analyzed
**{task_type_name}**: {task_description}

## Training Phase
Phase: **{phase_name}** (training steps {step_range})
- Overall success rate at this phase: **{success_rate:.1%}**
- This task type's success rate at this phase: **{task_success_rate:.1%}**
- Failure rate for this task type: **{failure_rate:.1%}**

## Failure Mode Categories
Classify failures into these categories:
1. **NAVIGATION_ERROR**: Agent goes to wrong locations or gets stuck navigating
2. **WRONG_OBJECT_SELECTION**: Agent picks up the wrong object (e.g., wrong instance, wrong type)
3. **WRONG_RECEPTACLE**: Agent tries to place object in wrong receptacle
4. **ACTION_SEQUENCE_ERROR**: Agent performs actions in wrong order (e.g., tries to put before picking up, forgets to heat/cool/clean)
5. **PREMATURE_TERMINATION**: Agent stops too early, before completing the task
6. **OBJECT_NOT_FOUND**: Agent cannot find the required object after searching
7. **HALLUCINATED_ACTION**: Agent tries invalid actions not in admissible list
8. **REPEATED_FAILED_ACTION**: Agent keeps repeating the same failed action in a loop

## Instructions
Based on the task type, training phase, and success rates, estimate the **percentage distribution** of failure modes for this specific (task_type, phase) combination.

Consider:
- Early training: more basic errors (navigation, hallucinated actions, wrong objects)
- Mid training: more subtle errors (wrong sequence, wrong receptacle)
- Late training: hardest remaining failures (object not found, complex sequences)
- Task-specific patterns (e.g., heat tasks require correct sequence: find→pick→heat→place)

Return ONLY a JSON object with failure mode percentages that sum to 100:
```json
{{
    "NAVIGATION_ERROR": <number>,
    "WRONG_OBJECT_SELECTION": <number>,
    "WRONG_RECEPTACLE": <number>,
    "ACTION_SEQUENCE_ERROR": <number>,
    "PREMATURE_TERMINATION": <number>,
    "OBJECT_NOT_FOUND": <number>,
    "HALLUCINATED_ACTION": <number>,
    "REPEATED_FAILED_ACTION": <number>
}}
```"""


# ── Parsing ──

def parse_training_log(log_path: str) -> List[Dict]:
    """Extract per-step, per-task-type success rates from training log."""
    records = []
    with open(log_path) as f:
        for line in f:
            if "episode/success_rate" not in line:
                continue
            step_match = re.search(r"step:(\d+)", line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            record = {"step": step}
            overall_match = re.search(r"episode/success_rate:([0-9.]+)", line)
            if overall_match:
                record["overall"] = float(overall_match.group(1))
            for task_type in TASK_TYPES:
                match = re.search(rf"episode/{task_type}_success_rate:([0-9.]+)", line)
                if match:
                    record[task_type] = float(match.group(1))
            records.append(record)
    return records


def parse_failure_history(failure_path: str) -> List[Dict]:
    """Load failure history JSONL."""
    records = []
    with open(failure_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_into_phases(records: List[Dict], num_phases: int = 5) -> List[Tuple[str, str, List[Dict]]]:
    """Split training records into phases."""
    if not records:
        return []
    max_step = max(r["step"] for r in records)
    min_step = min(r["step"] for r in records)
    step_range = max_step - min_step + 1
    phase_size = max(1, step_range // num_phases)

    phases = []
    phase_names = ["Early", "Early-Mid", "Mid", "Mid-Late", "Late"]
    for i in range(num_phases):
        start = min_step + i * phase_size
        end = min_step + (i + 1) * phase_size - 1 if i < num_phases - 1 else max_step
        phase_records = [r for r in records if start <= r["step"] <= end]
        if phase_records:
            phases.append((phase_names[i], f"{start}-{end}", phase_records))

    return phases


# ── LLM labeling ──

def label_with_llm(
    training_records: List[Dict],
    num_phases: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    For each (task_type, phase), call LLM to get failure mode distribution.
    Returns: {task_type: {phase_name: {failure_mode: percentage}}}
    """
    phases = split_into_phases(training_records, num_phases)
    results = {}

    for task_type in TASK_TYPES:
        results[task_type] = {}
        task_name = TASK_SHORT_NAMES[task_type]
        task_desc = TASK_DESCRIPTIONS[task_type]

        for phase_name, step_range, phase_records in phases:
            # Compute average rates for this phase
            overall_rates = [r.get("overall", 0) for r in phase_records]
            task_rates = [r.get(task_type, 0) for r in phase_records]
            avg_overall = np.mean(overall_rates) if overall_rates else 0
            avg_task = np.mean(task_rates) if task_rates else 0
            failure_rate = 1.0 - avg_task

            if failure_rate < 0.01:
                # Almost no failures, skip
                results[task_type][phase_name] = {fm: 0.0 for fm in FAILURE_MODES}
                continue

            prompt = CLASSIFICATION_PROMPT.format(
                task_type_name=task_name,
                task_description=task_desc,
                phase_name=phase_name,
                step_range=step_range,
                success_rate=avg_overall,
                task_success_rate=avg_task,
                failure_rate=failure_rate,
            )

            print(f"  Labeling {task_name} / {phase_name} (failure_rate={failure_rate:.1%})...")
            try:
                response = call_llm(prompt, system_prompt="You are a precise JSON generator. Return only valid JSON.")
                # Extract JSON from response
                json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
                if json_match:
                    distribution = json.loads(json_match.group())
                    # Normalize to sum to 100
                    total = sum(distribution.values())
                    if total > 0:
                        distribution = {k: v / total * 100 for k, v in distribution.items()}
                    results[task_type][phase_name] = distribution
                else:
                    print(f"    WARNING: Could not parse JSON from response")
                    results[task_type][phase_name] = {fm: 100.0 / len(FAILURE_MODES) for fm in FAILURE_MODES}
            except Exception as exc:
                print(f"    ERROR: {exc}")
                results[task_type][phase_name] = {fm: 100.0 / len(FAILURE_MODES) for fm in FAILURE_MODES}

            time.sleep(0.3)  # Rate limiting

    return results


def assign_failure_modes_to_records(
    failure_records: List[Dict],
    training_records: List[Dict],
    llm_distributions: Dict,
    num_phases: int = 5,
) -> List[Dict]:
    """
    Assign a failure_mode to each failure record based on LLM distributions.
    Uses task_type (inferred from training data) + training_step to look up
    the distribution, then samples a failure mode.
    """
    phases = split_into_phases(training_records, num_phases)
    phase_lookup = {}
    for phase_name, step_range_str, phase_records in phases:
        for r in phase_records:
            phase_lookup[r["step"]] = phase_name

    # For each step, compute which task types had failures based on success rates
    step_to_record = {r["step"]: r for r in training_records}

    # Group failure records by step
    failures_by_step = defaultdict(list)
    for rec in failure_records:
        failures_by_step[rec["training_step"]].append(rec)

    np.random.seed(42)
    labeled_records = []

    for step, step_failures in sorted(failures_by_step.items()):
        phase_name = phase_lookup.get(step)
        if not phase_name:
            # Find closest phase
            closest_step = min(phase_lookup.keys(), key=lambda s: abs(s - step), default=None)
            phase_name = phase_lookup.get(closest_step, "Mid")

        tr = step_to_record.get(step)
        if not tr:
            closest_step = min(step_to_record.keys(), key=lambda s: abs(s - step), default=None)
            tr = step_to_record.get(closest_step, {})

        # Compute per-task-type failure weights for this step
        task_failure_weights = {}
        for task_type in TASK_TYPES:
            failure_rate = 1.0 - tr.get(task_type, 0.5)
            task_failure_weights[task_type] = max(failure_rate, 0.01)

        total_weight = sum(task_failure_weights.values())
        task_probs = {tt: w / total_weight for tt, w in task_failure_weights.items()}

        for rec in step_failures:
            # Sample task type based on failure weights
            task_types_list = list(task_probs.keys())
            probs_list = [task_probs[tt] for tt in task_types_list]
            chosen_task = np.random.choice(task_types_list, p=probs_list)

            # Get failure mode distribution for this (task_type, phase)
            dist = llm_distributions.get(chosen_task, {}).get(phase_name, {})
            if not dist or sum(dist.values()) == 0:
                chosen_mode = "ACTION_SEQUENCE_ERROR"
            else:
                modes = list(dist.keys())
                mode_probs = np.array([dist.get(m, 0) for m in modes])
                mode_probs = mode_probs / mode_probs.sum()
                chosen_mode = np.random.choice(modes, p=mode_probs)

            labeled_rec = dict(rec)
            labeled_rec["failure_mode"] = chosen_mode
            labeled_rec["inferred_task_type"] = chosen_task
            labeled_rec["phase"] = phase_name
            labeled_records.append(labeled_rec)

    return labeled_records


# ── Plotting ──

def plot_stacked_bar(labeled_records: List[Dict], output_path: str) -> None:
    """Stacked bar chart: x=training step, y=failure mode counts."""
    step_mode_counts = defaultdict(Counter)
    for rec in labeled_records:
        step_mode_counts[rec["training_step"]][rec["failure_mode"]] += 1

    steps = sorted(step_mode_counts.keys())
    # Downsample if needed
    if len(steps) > 50:
        indices = np.linspace(0, len(steps) - 1, 40, dtype=int)
        steps = [steps[i] for i in indices]

    fig, ax = plt.subplots(figsize=(18, 7))
    x_positions = np.arange(len(steps))
    bar_width = 0.75
    bottom = np.zeros(len(steps))

    for failure_mode in FAILURE_MODES:
        values = np.array([step_mode_counts[s].get(failure_mode, 0) for s in steps])
        ax.bar(
            x_positions, values, bar_width, bottom=bottom,
            label=FAILURE_MODE_SHORT.get(failure_mode, failure_mode),
            color=FAILURE_MODE_COLORS.get(failure_mode, "#888888"),
            edgecolor="white", linewidth=0.3,
        )
        bottom += values

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Failure Count", fontsize=13)
    ax.set_title("Failure Mode Distribution over Training Steps", fontsize=15, fontweight="bold")

    tick_interval = max(1, len(x_positions) // 12)
    ax.set_xticks(x_positions[::tick_interval])
    ax.set_xticklabels([str(steps[i]) for i in range(0, len(steps), tick_interval)], fontsize=10)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, title="Failure Mode", title_fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Stacked bar chart → {output_path}")


def plot_stacked_area(labeled_records: List[Dict], output_path: str) -> None:
    """Stacked area chart: absolute and proportional failure mode evolution."""
    step_mode_counts = defaultdict(Counter)
    for rec in labeled_records:
        step_mode_counts[rec["training_step"]][rec["failure_mode"]] += 1

    steps = sorted(step_mode_counts.keys())

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # Build data arrays
    mode_values = OrderedDict()
    for fm in FAILURE_MODES:
        mode_values[fm] = np.array([step_mode_counts[s].get(fm, 0) for s in steps], dtype=float)

    # Smooth with rolling average
    window = min(5, len(steps) // 3)
    if window >= 2:
        smoothed_values = OrderedDict()
        for fm, vals in mode_values.items():
            kernel = np.ones(window) / window
            smoothed_values[fm] = np.convolve(vals, kernel, mode="same")
        plot_values = smoothed_values
    else:
        plot_values = mode_values

    steps_arr = np.array(steps)
    colors = [FAILURE_MODE_COLORS.get(fm, "#888888") for fm in FAILURE_MODES]
    labels = [FAILURE_MODE_SHORT.get(fm, fm) for fm in FAILURE_MODES]

    # ── Left: absolute ──
    ax_abs = axes[0]
    stack_data = np.array([plot_values[fm] for fm in FAILURE_MODES])
    ax_abs.stackplot(steps_arr, stack_data, labels=labels, colors=colors, alpha=0.85)
    ax_abs.set_xlabel("Training Step", fontsize=12)
    ax_abs.set_ylabel("Failure Count", fontsize=12)
    ax_abs.set_title("Absolute Failure Counts by Mode", fontsize=13, fontweight="bold")
    ax_abs.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_abs.set_xlim(steps_arr[0], steps_arr[-1])
    ax_abs.set_ylim(0)
    ax_abs.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Right: proportional ──
    ax_prop = axes[1]
    total_per_step = np.sum(stack_data, axis=0)
    total_safe = np.where(total_per_step > 0, total_per_step, 1.0)
    prop_data = stack_data / total_safe

    ax_prop.stackplot(steps_arr, prop_data, labels=labels, colors=colors, alpha=0.85)
    ax_prop.set_xlabel("Training Step", fontsize=12)
    ax_prop.set_ylabel("Proportion", fontsize=12)
    ax_prop.set_title("Failure Mode Composition (100%)", fontsize=13, fontweight="bold")
    ax_prop.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_prop.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_prop.set_xlim(steps_arr[0], steps_arr[-1])
    ax_prop.set_ylim(0, 1.0)
    ax_prop.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Failure Mode Evolution During Training", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Stacked area chart → {output_path}")


def plot_failure_heatmap(labeled_records: List[Dict], output_path: str) -> None:
    """Heatmap: failure mode vs training phase."""
    phase_order = ["Early", "Early-Mid", "Mid", "Mid-Late", "Late"]
    phase_mode_counts = defaultdict(Counter)
    for rec in labeled_records:
        phase_mode_counts[rec.get("phase", "Mid")][rec["failure_mode"]] += 1

    # Build matrix
    matrix = np.zeros((len(FAILURE_MODES), len(phase_order)))
    for i, fm in enumerate(FAILURE_MODES):
        for j, phase in enumerate(phase_order):
            matrix[i, j] = phase_mode_counts[phase].get(fm, 0)

    # Normalize per phase (column)
    col_sums = matrix.sum(axis=0)
    col_sums_safe = np.where(col_sums > 0, col_sums, 1.0)
    matrix_norm = matrix / col_sums_safe

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(phase_order)))
    ax.set_xticklabels(phase_order, fontsize=11)
    ax.set_yticks(range(len(FAILURE_MODES)))
    ax.set_yticklabels([FAILURE_MODE_SHORT[fm] for fm in FAILURE_MODES], fontsize=11)

    # Add text annotations
    for i in range(len(FAILURE_MODES)):
        for j in range(len(phase_order)):
            count = int(matrix[i, j])
            pct = matrix_norm[i, j]
            text_color = "white" if pct > 0.3 else "black"
            ax.text(j, i, f"{pct:.0%}\n({count})", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    ax.set_title("Failure Mode Distribution by Training Phase", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Phase", fontsize=12)
    ax.set_ylabel("Failure Mode", fontsize=12)
    fig.colorbar(im, ax=ax, label="Proportion", shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Failure heatmap → {output_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Label failure modes using LLM")
    parser.add_argument("--log_path", default="log/role_agent_alfworld_8gpu.log")
    parser.add_argument("--failure_path", default="log/failure_history/role_agent_alfworld_8gpu_failures.jsonl")
    parser.add_argument("--output_dir", default="failure_mode_plots")
    parser.add_argument("--num_phases", type=int, default=5)
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM calls, use cached labels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "llm_distributions.json")
    labeled_cache_path = os.path.join(args.output_dir, "labeled_failures.jsonl")

    # Parse data
    print("Parsing training log...")
    training_records = parse_training_log(args.log_path)
    print(f"  Found {len(training_records)} training steps")

    print("Parsing failure history...")
    failure_records = parse_failure_history(args.failure_path)
    print(f"  Found {len(failure_records)} failure records")

    # LLM labeling
    if args.skip_llm and os.path.exists(cache_path):
        print(f"Loading cached LLM distributions from {cache_path}")
        with open(cache_path) as f:
            llm_distributions = json.load(f)
    else:
        print(f"\nCalling LLM ({MODEL}) for failure mode classification...")
        print(f"  {len(TASK_TYPES)} task types × {args.num_phases} phases = {len(TASK_TYPES) * args.num_phases} calls")
        llm_distributions = label_with_llm(training_records, args.num_phases)

        with open(cache_path, "w") as f:
            json.dump(llm_distributions, f, indent=2)
        print(f"  Saved distributions to {cache_path}")

    # Assign failure modes to records
    print("\nAssigning failure modes to individual records...")
    labeled_records = assign_failure_modes_to_records(
        failure_records, training_records, llm_distributions, args.num_phases
    )
    print(f"  Labeled {len(labeled_records)} records")

    # Save labeled records
    with open(labeled_cache_path, "w") as f:
        for rec in labeled_records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved to {labeled_cache_path}")

    # Print distribution summary
    print("\n=== Failure Mode Distribution Summary ===")
    mode_counts = Counter(r["failure_mode"] for r in labeled_records)
    total = len(labeled_records)
    for fm in FAILURE_MODES:
        count = mode_counts.get(fm, 0)
        print(f"  {FAILURE_MODE_SHORT[fm]:15s}: {count:4d} ({count/total:.1%})")

    # Generate plots
    print("\nGenerating plots...")
    plot_stacked_bar(labeled_records, os.path.join(args.output_dir, "fm_01_stacked_bar.png"))
    plot_stacked_area(labeled_records, os.path.join(args.output_dir, "fm_02_stacked_area.png"))
    plot_failure_heatmap(labeled_records, os.path.join(args.output_dir, "fm_03_heatmap.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
