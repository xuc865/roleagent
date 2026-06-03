#!/usr/bin/env python3
"""
Task-Type Failure Mode Evolution Visualizer for ALFWorld training.

Parses the training log to extract per-task-type success rates at each
training step, then produces:
  1. Stacked bar chart: failure counts by task type over training steps
  2. Stacked area chart: failure proportion evolution over training steps

Usage:
    python scripts/visualize_task_failure_modes.py \
        --log_path log/role_agent_alfworld_8gpu.log \
        --output_dir failure_mode_plots
"""

import argparse
import json
import os
import re
from collections import OrderedDict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Task type display config ──

TASK_TYPES = [
    "pick_and_place",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
]

TASK_SHORT_NAMES = {
    "pick_and_place":                   "Pick & Place",
    "pick_cool_then_place_in_recep":    "Cool & Place",
    "pick_clean_then_place_in_recep":   "Clean & Place",
    "pick_heat_then_place_in_recep":    "Heat & Place",
    "look_at_obj_in_light":             "Look in Light",
    "pick_two_obj_and_place":           "Pick Two & Place",
}

TASK_COLORS = {
    "pick_and_place":                   "#e63946",
    "pick_cool_then_place_in_recep":    "#457b9d",
    "pick_clean_then_place_in_recep":   "#2a9d8f",
    "pick_heat_then_place_in_recep":    "#f4a261",
    "look_at_obj_in_light":             "#e76f51",
    "pick_two_obj_and_place":           "#264653",
}


# ── Parsing ──

def parse_training_log(log_path: str) -> List[Dict]:
    """Extract per-step, per-task-type success rates from the training log."""
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
                match = re.search(
                    rf"episode/{task_type}_success_rate:([0-9.]+)", line
                )
                if match:
                    record[task_type] = float(match.group(1))

            records.append(record)
    return records


# ── Plotting ──

def plot_stacked_bar(records: List[Dict], output_path: str) -> None:
    """
    Stacked bar chart: x = training step, y = failure rate (1 - success_rate)
    per task type.  Each bar is a step; segments are task-type failures.
    """
    steps = [r["step"] for r in records]
    failure_data = OrderedDict()
    for task_type in TASK_TYPES:
        failure_data[task_type] = [
            1.0 - r.get(task_type, 0.0) for r in records
        ]

    # Downsample if too many steps for readable bars
    total_steps = len(steps)
    if total_steps > 50:
        sample_indices = np.linspace(0, total_steps - 1, 40, dtype=int)
    else:
        sample_indices = np.arange(total_steps)

    sampled_steps = [steps[i] for i in sample_indices]
    sampled_failures = {
        tt: [vals[i] for i in sample_indices]
        for tt, vals in failure_data.items()
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    bar_width = 0.7
    x_positions = np.arange(len(sampled_steps))
    bottom = np.zeros(len(sampled_steps))

    for task_type in TASK_TYPES:
        values = np.array(sampled_failures[task_type])
        ax.bar(
            x_positions,
            values,
            bar_width,
            bottom=bottom,
            label=TASK_SHORT_NAMES[task_type],
            color=TASK_COLORS[task_type],
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += values

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Failure Rate (stacked)", fontsize=13)
    ax.set_title(
        "ALFWorld Task-Type Failure Modes over Training",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x_positions[::max(1, len(x_positions) // 10)])
    ax.set_xticklabels(
        [str(sampled_steps[i]) for i in range(0, len(sampled_steps),
         max(1, len(sampled_steps) // 10))],
        fontsize=10,
    )
    ax.set_ylim(0, max(bottom) * 1.05)
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        title="Task Type",
        title_fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Stacked bar chart → {output_path}")


def plot_stacked_area(records: List[Dict], output_path: str) -> None:
    """
    Stacked area chart: x = training step, y = failure rate proportion.
    Each band = one task type's failure rate, stacked to show composition.
    """
    steps = np.array([r["step"] for r in records])
    failure_rates = OrderedDict()
    for task_type in TASK_TYPES:
        failure_rates[task_type] = np.array([
            1.0 - r.get(task_type, 0.0) for r in records
        ])

    # Compute proportions (normalize to sum=1 per step)
    total_failure = np.zeros(len(steps))
    for vals in failure_rates.values():
        total_failure += vals
    # Avoid division by zero
    total_failure_safe = np.where(total_failure > 0, total_failure, 1.0)

    proportions = OrderedDict()
    for task_type, vals in failure_rates.items():
        proportions[task_type] = vals / total_failure_safe

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # ── Left: absolute stacked area ──
    ax_abs = axes[0]
    stack_values = np.array([failure_rates[tt] for tt in TASK_TYPES])
    colors = [TASK_COLORS[tt] for tt in TASK_TYPES]
    labels = [TASK_SHORT_NAMES[tt] for tt in TASK_TYPES]

    ax_abs.stackplot(steps, stack_values, labels=labels, colors=colors, alpha=0.85)
    ax_abs.set_xlabel("Training Step", fontsize=12)
    ax_abs.set_ylabel("Failure Rate (absolute)", fontsize=12)
    ax_abs.set_title("Absolute Failure Rate by Task Type", fontsize=13, fontweight="bold")
    ax_abs.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_abs.set_xlim(steps[0], steps[-1])
    ax_abs.set_ylim(0)
    ax_abs.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Right: proportional stacked area (100%) ──
    ax_prop = axes[1]
    prop_values = np.array([proportions[tt] for tt in TASK_TYPES])

    ax_prop.stackplot(steps, prop_values, labels=labels, colors=colors, alpha=0.85)
    ax_prop.set_xlabel("Training Step", fontsize=12)
    ax_prop.set_ylabel("Proportion of Failures", fontsize=12)
    ax_prop.set_title("Failure Composition by Task Type (100%)", fontsize=13, fontweight="bold")
    ax_prop.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_prop.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_prop.set_xlim(steps[0], steps[-1])
    ax_prop.set_ylim(0, 1.0)
    ax_prop.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "ALFWorld Failure Mode Evolution During Training",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Stacked area chart → {output_path}")


def plot_individual_failure_curves(records: List[Dict], output_path: str) -> None:
    """
    Line chart: each task type's failure rate as a separate line,
    showing individual decay trajectories.
    """
    steps = np.array([r["step"] for r in records])

    fig, ax = plt.subplots(figsize=(14, 7))

    for task_type in TASK_TYPES:
        failure_rate = np.array([1.0 - r.get(task_type, 0.0) for r in records])
        ax.plot(
            steps,
            failure_rate,
            label=TASK_SHORT_NAMES[task_type],
            color=TASK_COLORS[task_type],
            linewidth=2.2,
            alpha=0.9,
        )
        # Add smoothed trend
        if len(steps) > 5:
            window = min(7, len(steps) // 3)
            if window % 2 == 0:
                window += 1
            smoothed = np.convolve(
                failure_rate, np.ones(window) / window, mode="valid"
            )
            offset = (len(failure_rate) - len(smoothed)) // 2
            ax.plot(
                steps[offset : offset + len(smoothed)],
                smoothed,
                color=TASK_COLORS[task_type],
                linewidth=1.0,
                alpha=0.4,
                linestyle="--",
            )

    # Also plot overall failure rate
    overall_failure = np.array([1.0 - r.get("overall", 0.0) for r in records])
    ax.plot(
        steps,
        overall_failure,
        label="Overall",
        color="#333333",
        linewidth=3.0,
        alpha=0.7,
        linestyle="-.",
    )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Failure Rate", fontsize=13)
    ax.set_title(
        "Per-Task-Type Failure Rate Curves",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        title="Task Type",
        title_fontsize=11,
    )
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Individual failure curves → {output_path}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Visualize task-type failure modes during ALFWorld training."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="log/role_agent_alfworld_8gpu.log",
        help="Path to the training log file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="failure_mode_plots",
        help="Directory to save output plots.",
    )
    args = parser.parse_args()

    print(f"Parsing training log: {args.log_path}")
    records = parse_training_log(args.log_path)
    if not records:
        print("ERROR: No training records found in the log.")
        return

    print(f"Found {len(records)} training steps (step {records[0]['step']} - {records[-1]['step']})")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_stacked_bar(
        records,
        os.path.join(args.output_dir, "01_failure_mode_stacked_bar.png"),
    )
    plot_stacked_area(
        records,
        os.path.join(args.output_dir, "02_failure_mode_stacked_area.png"),
    )
    plot_individual_failure_curves(
        records,
        os.path.join(args.output_dir, "03_failure_mode_individual_curves.png"),
    )

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
