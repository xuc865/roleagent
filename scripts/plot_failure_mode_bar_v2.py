"""
Revised failure mode bar chart for ALFWorld.
All 'search'-related names replaced with ALFWorld-appropriate terminology.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Embedded Data (cumulative counts per bin) ---
# 10 bins: training steps [0-14], [15-29], ..., [135-149]
# X-axis labels: 15, 30, 45, 60, 75, 90, 105, 120, 135, 150
BIN_COUNTS = {
    "repetitive_exploration":      [325, 561, 743, 898, 1021, 1103, 1174, 1224, 1257, 1288],
    "wrong_target_location":       [145, 297, 417, 499, 553, 577, 591, 605, 609, 610],
    "wrong_receptacle":            [138, 299, 409, 479, 537, 561, 573, 580, 585, 585],
    "premature_give_up":           [82, 147, 231, 286, 323, 346, 363, 369, 380, 395],
    "missing_precondition":        [112, 211, 262, 303, 331, 348, 353, 358, 359, 360],
    "repeated_failed_action":      [62, 124, 162, 194, 217, 225, 230, 234, 235, 236],
    "navigation_loop":             [72, 114, 132, 144, 148, 151, 151, 151, 152, 152],
    "entity_confusion":            [19, 57, 86, 106, 114, 127, 133, 139, 142, 144],
    "wrong_object_interaction":    [24, 60, 78, 90, 99, 106, 107, 107, 107, 110],
    "exhaustive_exploration_failure": [15, 19, 28, 32, 39, 41, 42, 44, 44, 44],
    "action_format_error":         [2, 6, 7, 7, 7, 7, 7, 7, 7, 7],
}

FAILURE_MODES = [
    "repetitive_exploration", "wrong_target_location", "wrong_receptacle",
    "premature_give_up", "missing_precondition", "repeated_failed_action",
    "navigation_loop", "entity_confusion", "wrong_object_interaction",
    "exhaustive_exploration_failure", "action_format_error",
]

MODE_LABELS = {
    "repetitive_exploration":         "Repetitive Exploration",
    "wrong_target_location":          "Wrong Target Location",
    "wrong_receptacle":               "Wrong Receptacle",
    "premature_give_up":              "Premature Give-up",
    "missing_precondition":           "Missing Precondition",
    "repeated_failed_action":         "Repeated Failed Action",
    "navigation_loop":                "Navigation Loop",
    "entity_confusion":               "Entity Confusion",
    "wrong_object_interaction":       "Wrong Object Interaction",
    "exhaustive_exploration_failure":  "Exhaustive Exploration Failure",
    "action_format_error":            "Action Format Error",
}

# Morandi color palette
MODE_COLORS = {
    "repetitive_exploration":         "#D4897A",
    "wrong_target_location":          "#E8B87A",
    "wrong_receptacle":               "#7BAFC4",
    "premature_give_up":              "#C27C6A",
    "missing_precondition":           "#9DB4A0",
    "repeated_failed_action":         "#C4A5CF",
    "navigation_loop":                "#8FBB8F",
    "entity_confusion":               "#E8C77A",
    "wrong_object_interaction":       "#B8B8D0",
    "exhaustive_exploration_failure":  "#D4A5A5",
    "action_format_error":            "#C0C0C0",
}


def main():
    x_labels = ["15", "30", "45", "60", "75", "90", "105", "120", "135", "150"]
    x = np.arange(len(x_labels))

    bin_total = [
        sum(BIN_COUNTS[mode][i] for mode in FAILURE_MODES)
        for i in range(len(x_labels))
    ]

    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    })

    fig, ax = plt.subplots(figsize=(4, 3))

    bottom = np.zeros(len(x_labels))
    for mode in FAILURE_MODES:
        heights = np.array(BIN_COUNTS[mode], dtype=float)
        ax.bar(
            x, heights, 0.7,
            bottom=bottom,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += heights

    ax.plot(
        x, bin_total,
        color="#2C2C2C", linewidth=1.5,
        marker="o", markersize=2.5,
        alpha=0.8, label="Total", zorder=5,
    )
    for i, v in enumerate(bin_total):
        ax.annotate(
            str(v), (x[i], v),
            textcoords="offset points", xytext=(0, 3),
            ha="center", fontsize=4.5, fontweight="bold", color="#2C2C2C",
        )

    ax.set_xlabel("Training Step", fontsize=6)
    ax.set_ylabel("Cumulative Failure Count", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=5.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--", color="#AAAAAA")
    ax.set_ylim(0, max(bin_total) * 1.1)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.45, 1.38), fontsize=5,
        framealpha=0.95, edgecolor="#CCCCCC", fancybox=False,
        ncol=3, columnspacing=0.5, handlelength=1.0, handletextpad=0.3,
        borderpad=0.3, labelspacing=0.3,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.62)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "failure_mode_plots")
    os.makedirs(output_dir, exist_ok=True)

    for ext in ["svg", "png", "pdf"]:
        output_path = os.path.join(output_dir, f"05_failure_mode_bar_v2.{ext}")
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"[OK] Saved: {os.path.abspath(output_path)}")

    plt.close(fig)


if __name__ == "__main__":
    main()
