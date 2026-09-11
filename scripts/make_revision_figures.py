#!/usr/bin/env python3
"""Manuscript figures for the revised submission, drawn from the results registry.

    python scripts/make_revision_figures.py

Replaces fig1 with a *controlled* decoder comparison (span vs centre on identical
window scores, identical thresholds) and adds the leave-one-day-out figure. Every bar
is a row of registry.csv or lodo_frozen.csv; nothing is typed in.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GREEN, VERMILLION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


def fig1_controlled_decoder(reg: pd.DataFrame, out: Path) -> None:
    """Span vs centre, same scores, same thresholds — both models, validation."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, model, title in zip(axes, ("frozen_probe", "finetuned_2blocks"),
                                ("Frozen probe", "Fine-tuned (2 blocks)")):
        sub = reg[(reg.model == model) & (reg.split == "val") & (reg.thresholds == "tuned")]
        span = sub[sub.boundary_mode == "window_span"].iloc[0]
        cen = sub[sub.boundary_mode == "window_centers"].iloc[0]
        x = np.arange(2); w = 0.36
        ax.bar(x - w / 2, [span["f1@0.3"], span["f1@0.5"]], w, label="window-span", color=GREY)
        ax.bar(x + w / 2, [cen["f1@0.3"], cen["f1@0.5"]], w, label="window-centre", color=BLUE)
        for xi, (a, b) in enumerate(zip([span["f1@0.3"], span["f1@0.5"]], [cen["f1@0.3"], cen["f1@0.5"]])):
            ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5)
            ax.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(x, ["tIoU 0.3", "tIoU 0.5"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Event $F_1$ (validation)"); axes[0].set_ylim(0, 0.95)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out / "fig1_decode_ablation.png"); plt.close(fig)


def fig5_lodo(lodo: pd.DataFrame, reg: pd.DataFrame, out: Path) -> None:
    """Per-day held-out F1 for the frozen probe, against the single-split numbers."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    days = [d[4:6] + "/" + d[6:8] for d in lodo.held_out_day.astype(str)]
    x = np.arange(len(days)); w = 0.26
    ax.bar(x - w, lodo["f1@0.3"], w, label="$F_1$ @ tIoU 0.3", color=BLUE)
    ax.bar(x, lodo["f1@0.5"], w, label="$F_1$ @ tIoU 0.5", color=ORANGE)
    ax.bar(x + w, lodo["putdown_f1@0.5"], w, label="putdown $F_1$ @ 0.5", color=VERMILLION)
    for xi, v in enumerate(lodo["f1@0.3"]):
        ax.text(xi - w, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5)
    m3 = lodo["f1@0.3"].mean(); s3 = lodo["f1@0.3"].std()
    ax.axhline(m3, color=BLUE, linestyle="--", linewidth=1); ax.axhspan(m3 - s3, m3 + s3, color=BLUE, alpha=0.08)
    single = reg[(reg.model == "frozen_probe") & (reg.split == "val") &
                 (reg.boundary_mode == "window_centers") & (reg.thresholds == "tuned")].iloc[0]["f1@0.3"]
    ax.axhline(single, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(len(days) - 0.5, single + 0.012, f"single-split validation {single:.2f}", ha="right", fontsize=7, color=GREY)
    ax.text(len(days) - 0.5, m3 - s3 - 0.05, f"LODO mean {m3:.2f} ± {s3:.2f}", ha="right", fontsize=7, color=BLUE)
    ax.set_xticks(x, [f"held out\n{d}" for d in days], fontsize=8)
    ax.set_ylabel("Event $F_1$ on held-out day"); ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    fig.savefig(out / "fig5_lodo.png"); plt.close(fig)


def fig6_putdown_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(sens.putdown_thr, sens.putdown_precision, "-o", ms=3, color=BLUE, label="precision")
    ax.plot(sens.putdown_thr, sens.putdown_recall, "-s", ms=3, color=ORANGE, label="recall")
    ax.plot(sens.putdown_thr, sens.putdown_f1, "-^", ms=3, color=VERMILLION, label="$F_1$")
    ax.axvline(0.45, color=GREY, linestyle="--", linewidth=1)
    ax.text(0.455, 0.92, "validation-\nselected", fontsize=7, color=GREY)
    ax.set_xlabel("putdown threshold (all other settings fixed)"); ax.set_ylabel("putdown, held-out day")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.6); ax.set_axisbelow(True)
    fig.savefig(out / "fig6_putdown_sensitivity.png"); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision-dir", type=Path, default=REPO_ROOT / ".local/paper/revision")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/paper/figures")
    a = p.parse_args()
    reg = pd.read_csv(a.revision_dir / "registry.csv")
    lodo = pd.read_csv(a.revision_dir / "lodo_frozen.csv")
    sens = pd.read_csv(a.revision_dir / "putdown_sensitivity.csv")
    fig1_controlled_decoder(reg, a.output_dir)
    fig5_lodo(lodo, reg, a.output_dir)
    fig6_putdown_sensitivity(sens, a.output_dir)
    for f in ("fig1_decode_ablation", "fig5_lodo", "fig6_putdown_sensitivity"):
        print("wrote", a.output_dir / f"{f}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
