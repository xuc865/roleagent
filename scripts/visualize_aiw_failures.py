#!/usr/bin/env python3
"""
Visualize AIW failure history evolution from training .jsonl files.

Produces:
  1. Failure count per step (training progress)
  2. Task failure heatmap (step × task_idx)
  3. Weight evolution for top-N most-failed tasks
  4. Failure frequency distribution across tasks

Usage:
    python scripts/visualize_aiw_failures.py \
        --buffer_path log/failure_history/role_agent_alfworld_8gpu_failures.jsonl \
        --output_dir ./failure_evolution_plots
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(buffer_path: str):
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


def plot_failures_per_step(records, output_dir):
    """Plot 1: Number of failures per training step."""
    step_counts = Counter(r["training_step"] for r in records)
    steps = sorted(step_counts.keys())
    counts = [step_counts[s] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(steps, counts, color="#e63946", alpha=0.8, width=0.8)
    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Number of Failures", fontsize=13)
    ax.set_title("Failure Count per Training Step", fontsize=15, fontweight="bold")

    # Add trend line
    if len(steps) > 5:
        z = np.polyfit(steps, counts, 3)
        p = np.poly1d(z)
        smooth_x = np.linspace(min(steps), max(steps), 200)
        ax.plot(smooth_x, p(smooth_x), color="#457b9d", linewidth=2.5, label="Trend")
        ax.legend(fontsize=11)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "01_failures_per_step.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_heatmap(records, output_dir):
    """Plot 2: Heatmap of task_idx × training_step failures."""
    step_task = defaultdict(Counter)
    for r in records:
        step_task[r["training_step"]][r["task_idx"]] += 1

    all_steps = sorted(step_task.keys())
    all_tasks = sorted(set(r["task_idx"] for r in records))

    matrix = np.zeros((len(all_tasks), len(all_steps)), dtype=float)
    task_to_row = {t: i for i, t in enumerate(all_tasks)}
    step_to_col = {s: j for j, s in enumerate(all_steps)}

    for r in records:
        matrix[task_to_row[r["task_idx"]], step_to_col[r["training_step"]]] += 1

    fig, ax = plt.subplots(figsize=(16, max(8, len(all_tasks) * 0.18)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Label axes
    step_interval = max(1, len(all_steps) // 20)
    ax.set_xticks(range(0, len(all_steps), step_interval))
    ax.set_xticklabels([all_steps[i] for i in range(0, len(all_steps), step_interval)], fontsize=8)

    if len(all_tasks) <= 64:
        ax.set_yticks(range(len(all_tasks)))
        ax.set_yticklabels([f"task {t}" for t in all_tasks], fontsize=7)
    else:
        task_interval = max(1, len(all_tasks) // 30)
        ax.set_yticks(range(0, len(all_tasks), task_interval))
        ax.set_yticklabels([f"task {all_tasks[i]}" for i in range(0, len(all_tasks), task_interval)], fontsize=7)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Task Index", fontsize=13)
    ax.set_title("Task Failure Heatmap Across Training", fontsize=15, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Failure Count", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "02_task_failure_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_weight_evolution(records, output_dir, top_n=10):
    """Plot 3: Weight evolution for top-N most-failed tasks."""
    task_counts = Counter(r["task_idx"] for r in records)
    top_tasks = [t for t, _ in task_counts.most_common(top_n)]

    # Collect weight trajectory per task
    task_weight_traj = defaultdict(list)
    for r in records:
        if r["task_idx"] in top_tasks:
            task_weight_traj[r["task_idx"]].append(
                (r["training_step"], r["weight_after"])
            )

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab10", top_n)
    for i, task_idx in enumerate(top_tasks):
        points = task_weight_traj[task_idx]
        # Keep last weight per step for each task
        step_weight = {}
        for step, weight in points:
            step_weight[step] = weight
        sorted_steps = sorted(step_weight.keys())
        weights = [step_weight[s] for s in sorted_steps]
        ax.plot(sorted_steps, weights, marker=".", markersize=3,
                linewidth=1.5, color=cmap(i), label=f"task {task_idx} ({task_counts[task_idx]} fails)")

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("AIW Sampling Weight", fontsize=13)
    ax.set_title(f"Weight Evolution for Top-{top_n} Most-Failed Tasks", fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "03_weight_evolution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_task_failure_distribution(records, output_dir):
    """Plot 4: Total failure count per task (bar chart)."""
    task_counts = Counter(r["task_idx"] for r in records)
    tasks = sorted(task_counts.keys())
    counts = [task_counts[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e63946" if c > np.mean(counts) + np.std(counts) else
              "#f4a261" if c > np.mean(counts) else "#2a9d8f"
              for c in counts]
    ax.bar(range(len(tasks)), counts, color=colors, alpha=0.85, width=0.8)
    ax.set_xlabel("Task Index", fontsize=13)
    ax.set_ylabel("Total Failure Count", fontsize=13)
    ax.set_title("Failure Distribution Across Tasks", fontsize=15, fontweight="bold")

    if len(tasks) <= 64:
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, fontsize=7, rotation=45)
    else:
        ax.set_xticks(range(0, len(tasks), max(1, len(tasks) // 20)))

    # Add mean line
    mean_c = np.mean(counts)
    ax.axhline(mean_c, color="#264653", linestyle="--", linewidth=1.5, label=f"Mean = {mean_c:.1f}")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "04_task_failure_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_failure_reduction_curve(records, output_dir):
    """Plot 5: Failure rate reduction curve (moving average)."""
    step_counts = Counter(r["training_step"] for r in records)
    steps = sorted(step_counts.keys())
    counts = [step_counts[s] for s in steps]

    # Get total rollouts per step (from records, approximate batch_size)
    step_total = defaultdict(int)
    for r in records:
        step_total[r["training_step"]] += 1  # failures only
    # We know batch=64*8=512 rollouts or 16*8=128 rollouts per step
    # Use the first step's failure count as an upper bound estimate
    max_rollouts = max(counts) * 1.2 if counts else 128

    failure_rates = [c / max_rollouts * 100 for c in counts]

    # Moving average
    window = min(5, len(failure_rates) // 3 + 1)
    if window > 1 and len(failure_rates) >= window:
        moving_avg = np.convolve(failure_rates, np.ones(window) / window, mode="valid")
        moving_steps = steps[window - 1:]
    else:
        moving_avg = failure_rates
        moving_steps = steps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Left: raw failure count
    ax1.fill_between(steps, counts, alpha=0.3, color="#e63946")
    ax1.plot(steps, counts, color="#e63946", linewidth=1.5, marker=".", markersize=3)
    ax1.set_xlabel("Training Step", fontsize=13)
    ax1.set_ylabel("Failure Count", fontsize=13)
    ax1.set_title("Raw Failure Count", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Annotate improvement
    if len(counts) >= 2:
        reduction = (1 - counts[-1] / max(counts[0], 1)) * 100
        ax1.annotate(f"{reduction:.0f}% reduction",
                     xy=(steps[-1], counts[-1]),
                     xytext=(steps[-1] - len(steps) // 4, max(counts) * 0.7),
                     arrowprops=dict(arrowstyle="->", color="#264653"),
                     fontsize=12, color="#264653", fontweight="bold")

    # Right: moving average
    ax2.plot(moving_steps, moving_avg, color="#457b9d", linewidth=2.5)
    ax2.fill_between(moving_steps, moving_avg, alpha=0.2, color="#457b9d")
    ax2.set_xlabel("Training Step", fontsize=13)
    ax2.set_ylabel("Failure Rate (%, approx)", fontsize=13)
    ax2.set_title(f"Failure Rate (window={window} moving avg)", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_failure_reduction_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AIW failure history evolution")
    parser.add_argument("--buffer_path", required=True, help="Path to failure history .jsonl file")
    parser.add_argument("--output_dir", default="./failure_evolution_plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading records from {args.buffer_path}...")
    records = load_records(args.buffer_path)
    print(f"Loaded {len(records)} failure records")

    if not records:
        print("No records found. Exiting.")
        return

    steps = [r["training_step"] for r in records]
    print(f"Step range: {min(steps)} -> {max(steps)} ({len(set(steps))} unique steps)")

    print("\nGenerating plots...")
    plot_failures_per_step(records, args.output_dir)
    plot_task_heatmap(records, args.output_dir)
    plot_weight_evolution(records, args.output_dir)
    plot_task_failure_distribution(records, args.output_dir)
    plot_failure_reduction_curve(records, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
