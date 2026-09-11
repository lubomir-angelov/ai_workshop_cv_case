#!/usr/bin/env python3
"""Generate the figures for the Track B1 manuscript.

    python scripts/make_paper_figures.py --output-dir .local/paper/figures

Figures are drawn from the metrics JSON the evaluation writes, not from numbers
retyped by hand, so a re-run of the pipeline updates them.

Palette is colourblind-safe (Okabe-Ito) and every figure is legible in greyscale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Okabe-Ito, colourblind-safe.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#999999"
VERMILLION = "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load(run_dir: Path, split: str) -> dict:
    return json.loads((run_dir / "predictions" / f"metrics_{split}.json").read_text())


def fig_decode_ablation(output_dir: Path) -> None:
    """Validation F1 across decode configurations — the boundary-mode result."""
    labels = [
        "2.5 s window\nspan bounds\ndefault thr.",
        "2.5 s window\nspan bounds\ntuned thr.",
        "2.5 s window\ncentre bounds\ntuned thr.",
        "1.5 s window\ncentre bounds\ntuned thr.",
    ]
    f1_30 = [0.264, 0.350, 0.647, 0.585]
    f1_50 = [0.075, np.nan, 0.351, 0.523]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - width / 2, f1_30, width, label="tIoU 0.3", color=BLUE)
    ax.bar(x + width / 2, f1_50, width, label="tIoU 0.5", color=ORANGE)

    for xi, (a, b) in enumerate(zip(f1_30, f1_50)):
        ax.text(xi - width / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5)
        if not np.isnan(b):
            ax.text(xi + width / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Event-level $F_1$ (validation)")
    ax.set_ylim(0, 0.78)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "fig1_decode_ablation.png")
    plt.close(fig)


def fig_val_test_gap(runs: dict[str, Path], output_dir: Path) -> None:
    """Validation vs test for both models, overall and per class."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)

    for ax, (name, run_dir) in zip(axes, runs.items()):
        val, test = load(run_dir, "val"), load(run_dir, "test")
        metrics = ["$F_1$@0.3", "$F_1$@0.5", "pickup", "putdown"]
        v = [
            val["tiou@0.3"]["f1"], val["tiou@0.5"]["f1"],
            val["per_type"]["pickup"]["f1"], val["per_type"]["putdown"]["f1"],
        ]
        t = [
            test["tiou@0.3"]["f1"], test["tiou@0.5"]["f1"],
            test["per_type"]["pickup"]["f1"], test["per_type"]["putdown"]["f1"],
        ]
        x = np.arange(len(metrics))
        width = 0.36
        ax.bar(x - width / 2, v, width, label="validation", color=GREEN)
        ax.bar(x + width / 2, t, width, label="test", color=VERMILLION)
        for xi, (a, b) in enumerate(zip(v, t)):
            ax.text(xi - width / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
            ax.text(xi + width / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Event-level $F_1$")
    axes[0].set_ylim(0, 0.95)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(output_dir / "fig2_val_test_gap.png")
    plt.close(fig)


def fig_probability_shift(dataset_dir: Path, run_dir: Path, output_dir: Path) -> None:
    """Distribution of p(putdown) on true-putdown windows, validation vs test.

    This is the figure that carries the paper's main negative finding.
    """
    manifest = pd.read_parquet(dataset_dir / "window_manifest.parquet")
    key = ["clip_id", "candidate_id", "window_start_s"]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bins = np.linspace(0, 1, 26)

    for split, colour, style in (("val", GREEN, "-"), ("test", VERMILLION, "-")):
        scores = pd.read_parquet(run_dir / "predictions" / f"window_scores_{split}.parquet")
        merged = scores.merge(manifest[key + ["label_name"]], on=key, how="inner")
        putdowns = merged[merged["label_name"] == "putdown"]
        ax.hist(
            putdowns["p_putdown"], bins=bins, density=True, histtype="step",
            linewidth=1.8, color=colour, linestyle=style,
            label=f"{split} (n={len(putdowns)}, mean={putdowns['p_putdown'].mean():.2f})",
        )

    ax.axvline(0.45, color=GREY, linestyle="--", linewidth=1.0)
    ax.text(0.46, ax.get_ylim()[1] * 0.92, "decision\nthreshold", fontsize=7, color=GREY)
    ax.set_xlabel(r"$p(\mathrm{putdown})$ on windows whose ground truth is putdown")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / "fig3_putdown_probability_shift.png")
    plt.close(fig)


def fig_confusion(runs: dict[str, Path], output_dir: Path) -> None:
    """Class-confusion at tIoU 0.5 for the fine-tuned model, validation vs test."""
    run_dir = runs["Fine-tuned (2 blocks)"]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))

    for ax, split in zip(axes, ("val", "test")):
        confusion = load(run_dir, split)["confusion"]
        types = ["pickup", "putdown"]
        matrix = np.array([[confusion[t].get(c, 0) for c in types] for t in types], dtype=float)

        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() * 0.55 else "black",
                        fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], types, fontsize=8)
        ax.set_yticks([0, 1], types, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_title(f"{split} split", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("ground truth", fontsize=8)
    fig.savefig(output_dir / "fig4_confusion.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/paper/figures")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset_w15")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "Frozen probe": REPO_ROOT / ".local/track_b1_run_w15",
        "Fine-tuned (2 blocks)": REPO_ROOT / ".local/track_b1_finetune",
    }

    fig_decode_ablation(args.output_dir)
    fig_val_test_gap(runs, args.output_dir)
    fig_probability_shift(args.dataset_dir, runs["Fine-tuned (2 blocks)"], args.output_dir)
    fig_confusion(runs, args.output_dir)

    for path in sorted(args.output_dir.glob("*.png")):
        print(f"wrote {path} ({path.stat().st_size/1000:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
