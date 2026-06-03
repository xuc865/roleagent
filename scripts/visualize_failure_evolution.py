#!/usr/bin/env python3
"""
Failure Mode Evolution Lifecycle Visualizer for Role-Agent.

Reads a FailureBuffer .jsonl file (or generates demo data) and produces:
  1. Sankey diagram: failure mode transitions between consecutive epochs
  2. Heatmap: failure mode frequency per training epoch
  3. Lifecycle timeline: each failure mode's birth -> peak -> decay curve
  4. Evolution pathway summary: count of unique transition paths per mode
  5. Detailed content log: per-mode per-phase concrete failure contents

Usage:
    # From a real failure buffer:
    python scripts/visualize_failure_evolution.py \\
        --buffer_path /path/to/failure_buffer.jsonl \\
        --output_dir ./failure_evolution_plots

    # Demo mode (generates synthetic data):
    python scripts/visualize_failure_evolution.py \\
        --demo --output_dir ./failure_evolution_plots
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Failure mode taxonomy (matches GiGPO2's FAILURE_REFLECTION_PROMPT) ──

FAILURE_MODES = [
    "SEARCH_STRATEGY_ERROR",
    "REASONING_CHAIN_ERROR",
    "PREMATURE_TERMINATION",
    "HALLUCINATION",
    "ENTITY_CONFUSION",
    "TOOL_MISUSE",
    "CONTEXT_NEGLECT",
    "OTHER",
]

FAILURE_MODE_SHORT = {
    "SEARCH_STRATEGY_ERROR": "SearchErr",
    "REASONING_CHAIN_ERROR": "ReasonErr",
    "PREMATURE_TERMINATION": "PremTerm",
    "HALLUCINATION":         "Halluc",
    "ENTITY_CONFUSION":      "EntityConf",
    "TOOL_MISUSE":           "ToolMisuse",
    "CONTEXT_NEGLECT":       "CtxNeglect",
    "OTHER":                 "Other",
}

FAILURE_MODE_COLORS = {
    "SEARCH_STRATEGY_ERROR": "#e63946",
    "REASONING_CHAIN_ERROR": "#457b9d",
    "PREMATURE_TERMINATION": "#f4a261",
    "HALLUCINATION":         "#2a9d8f",
    "ENTITY_CONFUSION":      "#e76f51",
    "TOOL_MISUSE":           "#264653",
    "CONTEXT_NEGLECT":       "#a8dadc",
    "OTHER":                 "#888888",
}

_THINK_PATTERN = re.compile(r"", re.DOTALL)


# ── Parsing helpers ──

def extract_root_cause_type(failure_analysis: str) -> str:
    """Extract ROOT_CAUSE_TYPE from a structured reflection string."""
    if not failure_analysis:
        return "OTHER"
    match = re.search(r"ROOT_CAUSE_TYPE:\s*(\S+)", failure_analysis)
    if match:
        cause = match.group(1).strip().rstrip(",").rstrip(".")
        cause_upper = cause.upper()
        for fm in FAILURE_MODES:
            if fm in cause_upper or cause_upper in fm:
                return fm
        return "OTHER"
    return "OTHER"


def extract_reflection_fields(failure_analysis: str) -> Dict[str, str]:
    """Parse all structured fields from a reflection string."""
    fields = {
        "root_cause_type": "OTHER",
        "root_cause_detail": "",
        "critical_step": "",
        "transferable_lesson": "",
        "keywords": "",
    }
    if not failure_analysis:
        return fields
    field_patterns = {
        "root_cause_type":     r"ROOT_CAUSE_TYPE:\s*(.+?)(?:\n|$)",
        "root_cause_detail":   r"ROOT_CAUSE_DETAIL:\s*(.+?)(?:\n|$)",
        "critical_step":       r"CRITICAL_STEP:\s*(.+?)(?:\n|$)",
        "transferable_lesson": r"TRANSFERABLE_LESSON:\s*(.+?)(?:\n|$)",
        "keywords":            r"KEYWORDS:\s*(.+?)(?:\n|$)",
    }
    for key, pattern in field_patterns.items():
        match = re.search(pattern, failure_analysis, re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    fields["root_cause_type"] = extract_root_cause_type(failure_analysis)
    return fields


def load_failure_buffer(buffer_path: str) -> List[Dict[str, Any]]:
    """Load records from a FailureBuffer .jsonl file."""
    records = []
    with open(buffer_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def generate_demo_data(num_epochs: int = 30, failures_per_epoch: int = 20) -> List[Dict[str, Any]]:
    """Generate synthetic failure buffer data with realistic training dynamics."""
    np.random.seed(42)
    records = []

    phase_probs = {
        "early":  [0.30, 0.10, 0.08, 0.05, 0.05, 0.25, 0.12, 0.05],
        "mid":    [0.12, 0.25, 0.15, 0.12, 0.15, 0.08, 0.08, 0.05],
        "late":   [0.05, 0.15, 0.20, 0.25, 0.10, 0.03, 0.05, 0.17],
    }

    demo_details = {
        "SEARCH_STRATEGY_ERROR": {
            "detail": "Agent used overly broad search queries without refining based on retrieved results",
            "step": "Step 2: searched with generic terms instead of specific entity names",
            "lesson": "Always refine search queries based on initial results; use entity names from context",
            "kw": "search refinement, query specificity, iterative retrieval",
        },
        "REASONING_CHAIN_ERROR": {
            "detail": "Agent made a logical leap, skipping intermediate reasoning steps",
            "step": "Step 3: concluded without verifying intermediate premises",
            "lesson": "Verify each logical step before drawing final conclusions",
            "kw": "logical chain, intermediate verification, premise checking",
        },
        "PREMATURE_TERMINATION": {
            "detail": "Agent answered before gathering sufficient evidence from available sources",
            "step": "Step 1: provided answer after reading only the first search result",
            "lesson": "Continue gathering evidence until confidence exceeds threshold",
            "kw": "early stopping, insufficient evidence, confidence calibration",
        },
        "HALLUCINATION": {
            "detail": "Agent stated facts not supported by any retrieved document",
            "step": "Step 4: fabricated a date that was not in any retrieved passage",
            "lesson": "Only state facts that are directly supported by retrieved evidence",
            "kw": "fabrication, unsupported claims, grounding",
        },
        "ENTITY_CONFUSION": {
            "detail": "Agent confused two similarly named entities in the context",
            "step": "Step 3: mixed up person A with person B who share similar attributes",
            "lesson": "Explicitly track and disambiguate entities with similar names",
            "kw": "entity disambiguation, name confusion, attribute tracking",
        },
        "TOOL_MISUSE": {
            "detail": "Agent called the wrong tool for the task at hand",
            "step": "Step 2: used calculator tool when a search tool was needed",
            "lesson": "Match tool selection to the specific sub-task requirements",
            "kw": "tool selection, wrong API, parameter mismatch",
        },
        "CONTEXT_NEGLECT": {
            "detail": "Agent ignored relevant information already present in the context",
            "step": "Step 5: re-searched for information that was already retrieved in Step 2",
            "lesson": "Review existing context before initiating new searches",
            "kw": "context utilization, redundant search, information retention",
        },
        "OTHER": {
            "detail": "Miscellaneous failure not fitting standard categories",
            "step": "Step 3: unexpected behavior in response formatting",
            "lesson": "Follow structured output format consistently",
            "kw": "formatting, edge case, unexpected behavior",
        },
    }

    for epoch in range(num_epochs):
        progress = epoch / max(num_epochs - 1, 1)
        if progress < 0.33:
            alpha = progress / 0.33
            probs = np.array(phase_probs["early"]) * (1 - alpha) + np.array(phase_probs["mid"]) * alpha
        elif progress < 0.66:
            alpha = (progress - 0.33) / 0.33
            probs = np.array(phase_probs["mid"]) * (1 - alpha) + np.array(phase_probs["late"]) * alpha
        else:
            probs = np.array(phase_probs["late"])

        noise = np.random.dirichlet(np.ones(len(FAILURE_MODES)) * 2)
        probs = 0.8 * probs + 0.2 * noise
        probs /= probs.sum()

        decay_factor = max(0.3, 1.0 - 0.5 * progress)
        num_failures = max(3, int(failures_per_epoch * decay_factor + np.random.randint(-3, 4)))
        failure_indices = np.random.choice(len(FAILURE_MODES), size=num_failures, p=probs)

        for idx in failure_indices:
            fm = FAILURE_MODES[idx]
            reward = np.random.uniform(-1.0, 0.3)
            dd = demo_details[fm]
            analysis = (
                f"ROOT_CAUSE_TYPE: {fm}\n"
                f"ROOT_CAUSE_DETAIL: {dd['detail']}\n"
                f"CRITICAL_STEP: {dd['step']}\n"
                f"TRANSFERABLE_LESSON: {dd['lesson']}\n"
                f"KEYWORDS: {dd['kw']}"
            )
            records.append({
                "training_step": epoch,
                "timestamp": f"2026-01-{epoch+1:02d}T12:00:00",
                "traj_uid": f"traj_{epoch}_{len(records)}",
                "task_description": f"Demo task at epoch {epoch}: evaluate agent on multi-step reasoning",
                "episode_reward": round(reward, 4),
                "failure_analysis": analysis,
                "num_steps": int(np.random.randint(2, 8)),
                "steps": [
                    {"step_index": 0, "observation": "Initial task prompt", "action": "Agent response"},
                ],
            })

    return records


# ── Core analysis ──

def build_epoch_failure_matrix(records: List[Dict]) -> Tuple[List[int], Dict[str, List[int]]]:
    """Build a matrix: epochs x failure_modes -> count."""
    epoch_mode_counts = defaultdict(Counter)
    for rec in records:
        epoch = rec["training_step"]
        fm = extract_root_cause_type(rec.get("failure_analysis", ""))
        epoch_mode_counts[epoch][fm] += 1

    sorted_epochs = sorted(epoch_mode_counts.keys())
    matrix = {}
    for fm in FAILURE_MODES:
        matrix[fm] = [epoch_mode_counts[ep].get(fm, 0) for ep in sorted_epochs]
    return sorted_epochs, matrix


def compute_transitions(records: List[Dict]) -> Dict[Tuple[str, str], float]:
    """Compute population-level failure mode transitions between consecutive epochs."""
    epoch_mode_counts = defaultdict(Counter)
    for rec in records:
        epoch = rec["training_step"]
        fm = extract_root_cause_type(rec.get("failure_analysis", ""))
        epoch_mode_counts[epoch][fm] += 1

    sorted_epochs = sorted(epoch_mode_counts.keys())
    transitions = Counter()

    for i in range(len(sorted_epochs) - 1):
        ep_curr = sorted_epochs[i]
        ep_next = sorted_epochs[i + 1]
        total_curr = sum(epoch_mode_counts[ep_curr].values())
        total_next = sum(epoch_mode_counts[ep_next].values())
        if total_curr == 0 or total_next == 0:
            continue
        for fm_src, count_src in epoch_mode_counts[ep_curr].items():
            src_frac = count_src / total_curr
            for fm_dst, count_dst in epoch_mode_counts[ep_next].items():
                dst_frac = count_dst / total_next
                weight = src_frac * dst_frac * total_next
                if weight > 0.01:
                    transitions[(fm_src, fm_dst)] += weight

    return dict(transitions)


def compute_evolution_pathways(records: List[Dict]) -> Dict[str, Dict]:
    """For each failure mode, compute lifecycle and transition statistics."""
    sorted_epochs, matrix = build_epoch_failure_matrix(records)
    transitions = compute_transitions(records)

    pathways = {}
    for fm in FAILURE_MODES:
        counts = matrix[fm]
        nonzero_indices = [i for i, c in enumerate(counts) if c > 0]
        if not nonzero_indices:
            continue

        birth_epoch = sorted_epochs[nonzero_indices[0]]
        death_epoch = sorted_epochs[nonzero_indices[-1]]
        peak_idx = int(np.argmax(counts))
        peak_epoch = sorted_epochs[peak_idx]
        peak_count = counts[peak_idx]

        outgoing = {}
        incoming = {}
        for (src, dst), weight in transitions.items():
            if src == fm and dst != fm:
                outgoing[dst] = outgoing.get(dst, 0) + weight
            if dst == fm and src != fm:
                incoming[src] = incoming.get(src, 0) + weight

        connected_modes = set(outgoing.keys()) | set(incoming.keys())
        pathways[fm] = {
            "birth_epoch": birth_epoch,
            "death_epoch": death_epoch,
            "peak_epoch": peak_epoch,
            "peak_count": peak_count,
            "total_count": sum(counts),
            "lifespan": death_epoch - birth_epoch,
            "outgoing_transitions": outgoing,
            "incoming_transitions": incoming,
            "num_evolution_paths": len(connected_modes),
        }
    return pathways


# ── Plotting ──

def plot_heatmap(sorted_epochs, matrix, output_dir):
    """Plot 1: Heatmap of failure mode frequency per epoch."""
    fig, ax = plt.subplots(figsize=(max(14, len(sorted_epochs) * 0.5), 6))
    active_modes = [fm for fm in FAILURE_MODES if sum(matrix[fm]) > 0]
    data = np.array([matrix[fm] for fm in active_modes])
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(active_modes)))
    ax.set_yticklabels([FAILURE_MODE_SHORT[fm] for fm in active_modes], fontsize=10)
    tick_step = max(1, len(sorted_epochs) // 20)
    tick_indices = list(range(0, len(sorted_epochs), tick_step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(sorted_epochs[i]) for i in tick_indices], fontsize=8, rotation=45)
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Failure Mode", fontsize=12)
    ax.set_title("Failure Mode Frequency Heatmap Across Training", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "01_failure_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lifecycle_curves(sorted_epochs, matrix, output_dir):
    """Plot 2: Stacked area chart showing lifecycle of each failure mode."""
    fig, ax = plt.subplots(figsize=(14, 7))
    active_modes = [fm for fm in FAILURE_MODES if sum(matrix[fm]) > 0]
    data = np.array([matrix[fm] for fm in active_modes], dtype=float)
    totals = data.sum(axis=0)
    totals[totals == 0] = 1
    data_norm = data / totals
    colors = [FAILURE_MODE_COLORS[fm] for fm in active_modes]
    labels = [FAILURE_MODE_SHORT[fm] for fm in active_modes]
    ax.stackplot(sorted_epochs, data_norm, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Proportion of Failures", fontsize=12)
    ax.set_title("Failure Mode Lifecycle: Proportion Over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(sorted_epochs[0], sorted_epochs[-1])
    plt.tight_layout()
    path = os.path.join(output_dir, "02_failure_lifecycle_stacked.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_individual_curves(sorted_epochs, matrix, pathways, output_dir):
    """Plot 3: Individual lifecycle curves per failure mode."""
    active_modes = [fm for fm in FAILURE_MODES if fm in pathways]
    num_modes = len(active_modes)
    if num_modes == 0:
        return
    cols = min(4, num_modes)
    rows = (num_modes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for idx, fm in enumerate(active_modes):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        counts = matrix[fm]
        info = pathways[fm]
        color = FAILURE_MODE_COLORS[fm]
        ax.fill_between(sorted_epochs, counts, alpha=0.3, color=color)
        ax.plot(sorted_epochs, counts, color=color, linewidth=2)
        ax.axvline(info["birth_epoch"], color="green", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(info["peak_epoch"], color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(info["death_epoch"], color="gray", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_title(f"{FAILURE_MODE_SHORT[fm]}", fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        annotation = (
            f"paths: {info['num_evolution_paths']}\n"
            f"total: {info['total_count']:.0f}\n"
            f"peak@{info['peak_epoch']}: {info['peak_count']}"
        )
        ax.text(0.98, 0.95, annotation, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    for idx in range(num_modes, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)
    fig.suptitle("Individual Failure Mode Lifecycles (birth -- peak -- death)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "03_failure_individual_lifecycles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_transition_chord(transitions, pathways, output_dir):
    """Plot 4: Transition matrix as a heatmap."""
    active_modes = [fm for fm in FAILURE_MODES if fm in pathways]
    num_modes = len(active_modes)
    if num_modes == 0:
        return
    trans_matrix = np.zeros((num_modes, num_modes))
    for i, src in enumerate(active_modes):
        for j, dst in enumerate(active_modes):
            trans_matrix[i][j] = transitions.get((src, dst), 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trans_matrix, cmap="Blues", interpolation="nearest")
    short_names = [FAILURE_MODE_SHORT[fm] for fm in active_modes]
    ax.set_xticks(range(num_modes))
    ax.set_xticklabels(short_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(num_modes))
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("To (next epoch)", fontsize=11)
    ax.set_ylabel("From (current epoch)", fontsize=11)
    ax.set_title("Failure Mode Transition Matrix", fontsize=14, fontweight="bold")
    for i in range(num_modes):
        for j in range(num_modes):
            value = trans_matrix[i][j]
            if value > 0.5:
                text_color = "white" if value > trans_matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                        fontsize=8, color=text_color)
    plt.colorbar(im, ax=ax, label="Transition Weight", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "04_failure_transition_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pathway_summary(pathways, output_dir):
    """Plot 5: Bar chart summarizing evolution pathway counts per failure mode."""
    active_modes = sorted(pathways.keys(), key=lambda fm: pathways[fm]["num_evolution_paths"], reverse=True)
    if not active_modes:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    mode_names = [FAILURE_MODE_SHORT[fm] for fm in active_modes]
    path_counts = [pathways[fm]["num_evolution_paths"] for fm in active_modes]
    colors = [FAILURE_MODE_COLORS[fm] for fm in active_modes]
    bars = ax.barh(mode_names, path_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Evolution Pathways", fontsize=11)
    ax.set_title("Evolution Pathway Count per Failure Mode", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for bar, count in zip(bars, path_counts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, fontweight="bold")
    ax2 = axes[1]
    total_counts = [pathways[fm]["total_count"] for fm in active_modes]
    lifespans = [pathways[fm]["lifespan"] for fm in active_modes]
    x_pos = np.arange(len(active_modes))
    width = 0.35
    ax2.bar(x_pos - width / 2, total_counts, width, label="Total Count",
            color=[FAILURE_MODE_COLORS[fm] for fm in active_modes], alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x_pos + width / 2, lifespans, width, label="Lifespan (epochs)",
                 color=[FAILURE_MODE_COLORS[fm] for fm in active_modes], alpha=0.4,
                 edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([FAILURE_MODE_SHORT[fm] for fm in active_modes], fontsize=9, rotation=30, ha="right")
    ax2.set_ylabel("Total Count", fontsize=11)
    ax2_twin.set_ylabel("Lifespan (epochs)", fontsize=11)
    ax2.set_title("Failure Mode Scale & Persistence", fontsize=13, fontweight="bold")
    lines1 = mpatches.Patch(color="gray", alpha=0.8, label="Total Count")
    lines2 = mpatches.Patch(color="gray", alpha=0.4, label="Lifespan")
    ax2.legend(handles=[lines1, lines2], loc="upper right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "05_failure_pathway_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sankey_ascii(transitions, pathways, output_dir):
    """Plot 6: Text-based summary of all transition flows."""
    active_modes = sorted(pathways.keys(), key=lambda fm: pathways[fm]["total_count"], reverse=True)
    lines = []
    lines.append("=" * 70)
    lines.append("FAILURE MODE EVOLUTION: TRANSITION FLOWS")
    lines.append("=" * 70)
    for fm in active_modes:
        info = pathways[fm]
        lines.append(f"\n{'=' * 50}")
        lines.append(f"  {fm} ({FAILURE_MODE_SHORT[fm]})")
        lines.append(f"  Birth: epoch {info['birth_epoch']} | Peak: epoch {info['peak_epoch']} "
                      f"(count={info['peak_count']}) | Death: epoch {info['death_epoch']}")
        lines.append(f"  Total: {info['total_count']:.0f} | Lifespan: {info['lifespan']} epochs "
                      f"| Evolution paths: {info['num_evolution_paths']}")
        if info["outgoing_transitions"]:
            lines.append(f"  Evolves INTO ->")
            for dst, weight in sorted(info["outgoing_transitions"].items(), key=lambda x: -x[1]):
                bar = "#" * max(1, int(weight / 2))
                lines.append(f"    -> {FAILURE_MODE_SHORT[dst]:12s}  {bar}  ({weight:.1f})")
        if info["incoming_transitions"]:
            lines.append(f"  Evolves FROM <-")
            for src, weight in sorted(info["incoming_transitions"].items(), key=lambda x: -x[1]):
                bar = "#" * max(1, int(weight / 2))
                lines.append(f"    <- {FAILURE_MODE_SHORT[src]:12s}  {bar}  ({weight:.1f})")
    lines.append(f"\n{'=' * 70}")
    text = "\n".join(lines)
    path = os.path.join(output_dir, "06_transition_flows.txt")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"  Saved: {path}")
    print(text)


def generate_detailed_content_log(records, sorted_epochs, pathways, output_dir):
    """Generate detailed content log preserving specific contents per mode per phase."""
    num_ep = len(sorted_epochs)
    phase_boundaries = {
        "early": sorted_epochs[:max(1, num_ep // 3)],
        "mid":   sorted_epochs[num_ep // 3: 2 * num_ep // 3],
        "late":  sorted_epochs[2 * num_ep // 3:],
    }
    mode_phase = defaultdict(lambda: defaultdict(list))
    mode_epoch = defaultdict(lambda: defaultdict(list))

    for rec in records:
        epoch = rec["training_step"]
        analysis_text = rec.get("failure_analysis", "")
        fields = extract_reflection_fields(analysis_text)
        fm = fields["root_cause_type"]
        entry = {
            "epoch": epoch,
            "traj_uid": rec.get("traj_uid", ""),
            "task_description": rec.get("task_description", ""),
            "episode_reward": rec.get("episode_reward", 0.0),
            "num_steps": rec.get("num_steps", 0),
            "root_cause_type": fields["root_cause_type"],
            "root_cause_detail": fields["root_cause_detail"],
            "critical_step": fields["critical_step"],
            "transferable_lesson": fields["transferable_lesson"],
            "keywords": fields["keywords"],
            "raw_failure_analysis": analysis_text,
        }
        if rec.get("steps"):
            entry["trajectory_steps"] = rec["steps"]
        for pname, pepochs in phase_boundaries.items():
            if epoch in pepochs:
                mode_phase[fm][pname].append(entry)
                break
        mode_epoch[fm][epoch].append(entry)

    # JSON output
    json_report = {}
    for fm in FAILURE_MODES:
        if fm not in mode_phase:
            continue
        fm_data = {
            "short_name": FAILURE_MODE_SHORT[fm],
            "lifecycle": {k: v for k, v in pathways.get(fm, {}).items()
                          if k not in ("outgoing_transitions", "incoming_transitions")},
            "phases": {},
            "per_epoch": {},
        }
        for pname in ["early", "mid", "late"]:
            entries = mode_phase[fm].get(pname, [])
            fm_data["phases"][pname] = {"count": len(entries), "entries": entries}
        for ep in sorted(mode_epoch[fm].keys()):
            fm_data["per_epoch"][str(ep)] = mode_epoch[fm][ep]
        json_report[fm] = fm_data

    json_path = os.path.join(output_dir, "07_detailed_content_by_mode.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    # Markdown output
    md = []
    md.append("# Failure Mode Evolution - Detailed Content Log\n")
    md.append(f"Total records: {len(records)} | "
              f"Epochs: {sorted_epochs[0]}-{sorted_epochs[-1]} | "
              f"Active modes: {len(mode_phase)}\n")
    for fm in FAILURE_MODES:
        if fm not in mode_phase:
            continue
        info = pathways.get(fm, {})
        md.append(f"\n---\n")
        md.append(f"## {fm} ({FAILURE_MODE_SHORT[fm]})\n")
        if info:
            md.append(f"- **Birth**: epoch {info.get('birth_epoch', '?')} | "
                      f"**Peak**: epoch {info.get('peak_epoch', '?')} "
                      f"(count={info.get('peak_count', '?')}) | "
                      f"**Death**: epoch {info.get('death_epoch', '?')}")
            md.append(f"- **Total**: {info.get('total_count', 0):.0f} | "
                      f"**Lifespan**: {info.get('lifespan', 0)} epochs | "
                      f"**Evolution paths**: {info.get('num_evolution_paths', 0)}\n")
        for pname in ["early", "mid", "late"]:
            entries = mode_phase[fm].get(pname, [])
            if not entries:
                continue
            pe = phase_boundaries[pname]
            md.append(f"### {pname.upper()} phase (epochs {pe[0]}-{pe[-1]}) "
                      f"- {len(entries)} failures\n")
            for i, entry in enumerate(entries[:5], 1):
                md.append(f"**[{i}] Epoch {entry['epoch']}** | "
                          f"reward={entry['episode_reward']:.3f} | "
                          f"steps={entry['num_steps']}")
                md.append(f"- **Task**: {entry['task_description'][:200]}")
                if entry["root_cause_detail"]:
                    md.append(f"- **Root Cause Detail**: {entry['root_cause_detail']}")
                if entry["critical_step"]:
                    md.append(f"- **Critical Step**: {entry['critical_step']}")
                if entry["transferable_lesson"]:
                    md.append(f"- **Lesson**: {entry['transferable_lesson']}")
                if entry["keywords"]:
                    md.append(f"- **Keywords**: {entry['keywords']}")
                md.append("")
            if len(entries) > 5:
                md.append(f"*...and {len(entries) - 5} more entries (see JSON for full data)*\n")

    md_path = os.path.join(output_dir, "08_detailed_content_by_mode.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"  Saved: {md_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Visualize failure mode evolution lifecycle for Role-Agent training."
    )
    parser.add_argument("--buffer_path", type=str, default=None,
                        help="Path to FailureBuffer .jsonl file")
    parser.add_argument("--output_dir", type=str, default="./failure_evolution_plots",
                        help="Directory to save plots")
    parser.add_argument("--demo", action="store_true",
                        help="Generate and visualize demo data")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs for demo data")
    parser.add_argument("--failures_per_epoch", type=int, default=20,
                        help="Failures per epoch for demo data")
    args = parser.parse_args()

    if args.demo:
        print("Generating demo failure data...")
        records = generate_demo_data(args.num_epochs, args.failures_per_epoch)
    elif args.buffer_path:
        print(f"Loading failure buffer from: {args.buffer_path}")
        records = load_failure_buffer(args.buffer_path)
    else:
        print("ERROR: Provide --buffer_path or use --demo")
        sys.exit(1)

    if not records:
        print("No records found!")
        sys.exit(1)

    print(f"Loaded {len(records)} failure records.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Analysis
    sorted_epochs, matrix = build_epoch_failure_matrix(records)
    transitions = compute_transitions(records)
    pathways = compute_evolution_pathways(records)

    print(f"\nTraining epochs: {sorted_epochs[0]} -> {sorted_epochs[-1]} ({len(sorted_epochs)} epochs)")
    print(f"Active failure modes: {len(pathways)}")
    print(f"Unique transitions: {len(transitions)}")

    # Generate all plots
    print("\nGenerating plots...")
    plot_heatmap(sorted_epochs, matrix, args.output_dir)
    plot_lifecycle_curves(sorted_epochs, matrix, args.output_dir)
    plot_individual_curves(sorted_epochs, matrix, pathways, args.output_dir)
    plot_transition_chord(transitions, pathways, args.output_dir)
    plot_pathway_summary(pathways, args.output_dir)
    plot_sankey_ascii(transitions, pathways, args.output_dir)
    generate_detailed_content_log(records, sorted_epochs, pathways, args.output_dir)

    # Save JSON report
    report = {
        "total_records": len(records),
        "epochs": list(sorted_epochs),
        "pathways": {
            fm: {
                "short_name": FAILURE_MODE_SHORT[fm],
                "birth_epoch": info["birth_epoch"],
                "death_epoch": info["death_epoch"],
                "peak_epoch": info["peak_epoch"],
                "peak_count": info["peak_count"],
                "total_count": int(info["total_count"]),
                "lifespan": info["lifespan"],
                "num_evolution_paths": info["num_evolution_paths"],
                "outgoing": {k: round(v, 2) for k, v in info["outgoing_transitions"].items()},
                "incoming": {k: round(v, 2) for k, v in info["incoming_transitions"].items()},
            }
            for fm, info in pathways.items()
        },
        "transitions": {f"{src}->{dst}": round(w, 2) for (src, dst), w in
                        sorted(transitions.items(), key=lambda x: -x[1])},
    }
    report_path = os.path.join(args.output_dir, "evolution_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Saved: {report_path}")

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
