#!/usr/bin/env python3
"""Consolidate Track B1 metrics from every run into one readable report.

    python scripts/report_track_b1_results.py

Reads the ``metrics_<split>.json`` files the inference script writes and prints a
side-by-side comparison, optionally writing it to Markdown with ``--output``.

Each run directory is expected to look like::

    <run_dir>/predictions/metrics_val.json
    <run_dir>/predictions/metrics_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RUNS = [
    ("frozen probe", REPO_ROOT / ".local/track_b1_run_w15"),
    ("fine-tuned (2 blocks)", REPO_ROOT / ".local/track_b1_finetune"),
]


def load(run_dir: Path, split: str) -> dict | None:
    path = run_dir / "predictions" / f"metrics_{split}.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([line, sep, *body])


def section(title: str, runs: list[tuple[str, dict | None]]) -> str:
    names = [name for name, metrics in runs if metrics is not None]
    loaded = [metrics for _, metrics in runs if metrics is not None]
    if not loaded:
        return f"### {title}\n\n(no metrics found)\n"

    def row(label: str, getter) -> list[str]:
        return [label] + [fmt(getter(m)) for m in loaded]

    rows = [
        row("F1 @ tIoU 0.3", lambda m: m.get("tiou@0.3", {}).get("f1")),
        row("precision @ 0.3", lambda m: m.get("tiou@0.3", {}).get("precision")),
        row("recall @ 0.3", lambda m: m.get("tiou@0.3", {}).get("recall")),
        row(
            "tp / fp / fn @ 0.3",
            lambda m: "{} / {} / {}".format(
                m.get("tiou@0.3", {}).get("tp"),
                m.get("tiou@0.3", {}).get("fp"),
                m.get("tiou@0.3", {}).get("fn"),
            ),
        ),
        row("F1 @ tIoU 0.5", lambda m: m.get("tiou@0.5", {}).get("f1")),
        row("precision @ 0.5", lambda m: m.get("tiou@0.5", {}).get("precision")),
        row("recall @ 0.5", lambda m: m.get("tiou@0.5", {}).get("recall")),
        row("pickup F1", lambda m: m.get("per_type", {}).get("pickup", {}).get("f1")),
        row(
            "pickup P / R",
            lambda m: "{} / {}".format(
                fmt(m.get("per_type", {}).get("pickup", {}).get("precision")),
                fmt(m.get("per_type", {}).get("pickup", {}).get("recall")),
            ),
        ),
        row("putdown F1", lambda m: m.get("per_type", {}).get("putdown", {}).get("f1")),
        row(
            "putdown P / R",
            lambda m: "{} / {}".format(
                fmt(m.get("per_type", {}).get("putdown", {}).get("precision")),
                fmt(m.get("per_type", {}).get("putdown", {}).get("recall")),
            ),
        ),
        row("mAP @ 0.3", lambda m: m.get("mAP", {}).get("mAP@0.3")),
        row("mAP avg", lambda m: m.get("mAP", {}).get("mAP_avg")),
        row("start MAE (s)", lambda m: m.get("start_mae_s")),
        row("end MAE (s)", lambda m: m.get("end_mae_s")),
        row("false positives / hour", lambda m: m.get("fp_per_hour")),
    ]
    out = [f"### {title}", "", table(rows, ["metric", *names]), ""]

    for name, metrics in runs:
        if metrics is None:
            continue
        confusion = metrics.get("confusion") or {}
        if confusion:
            out.append(
                f"Confusion @ tIoU {metrics.get('confusion_tiou')} — {name} "
                "(rows = ground truth, columns = prediction):"
            )
            types = sorted(confusion)
            out.append("")
            out.append(
                table(
                    [[t] + [str(confusion[t].get(c, 0)) for c in types] for t in types],
                    ["truth \\ pred", *types],
                )
            )
            out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Also write Markdown here")
    args = parser.parse_args()

    runs = [(name, path) for name, path in DEFAULT_RUNS if path.exists()]

    parts = ["# Track B1 results", ""]
    for split in ("val", "test"):
        parts.append(
            section(
                f"{split} split",
                [(name, load(path, split)) for name, path in runs],
            )
        )

    parts.append("Source files:")
    for name, path in runs:
        parts.append(f"- {name}: `{path}/predictions/metrics_{{val,test}}.json`")

    report = "\n".join(parts)
    print(report)
    if args.output:
        args.output.write_text(report)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
