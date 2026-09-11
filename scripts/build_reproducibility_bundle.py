#!/usr/bin/env python3
"""Assemble the Track B1 reproducibility bundle for external sharing.

    python scripts/build_reproducibility_bundle.py --output-dir .local/track_b1_bundle

Collects the CVAT annotation exports, the canonical dataset with its split
assignment, the trained model weights, every evaluation artefact, the frozen
configuration and the manuscript into one directory, with a checksum manifest.

Source videos are deliberately excluded. They total 17 GB and are commercial store
footage of members of the public; the bundle carries their S3 keys, sizes and
checksums instead, so a holder of bucket credentials can reconstruct the input set
without the footage being redistributed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_into(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def write_split_listing(dataset_dir: Path, out_path: Path) -> pd.DataFrame:
    """Human-readable statement of exactly which clip trained and which was held out."""
    clips = pd.read_parquet(dataset_dir / "clips.parquet")
    events = pd.read_parquet(dataset_dir / "events.parquet")

    counts = (
        events.groupby(["clip_id", "type"]).size().unstack(fill_value=0).reindex(clips["clip_id"])
        .fillna(0).astype(int)
    )
    table = clips[["clip_id", "recording_day", "split", "fps", "n_frames", "duration_s"]].copy()
    table = table.join(counts, on="clip_id")
    table = table.sort_values(["split", "clip_id"])

    lines = [
        "# Track B1 split assignment",
        "",
        "Splits are assigned by **recording day**, never by clip: clips recorded minutes",
        "apart share shoppers, lighting and shelf stock, so a clip-level split would put",
        "near-duplicates on both sides of the boundary.",
        "",
        "| split | days | clips | pickup | putdown |",
        "|---|---|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        part = table[table["split"] == split]
        lines.append(
            f"| {split} | {', '.join(sorted(part['recording_day'].unique()))} | {len(part)} "
            f"| {int(part.get('pickup', pd.Series(dtype=int)).sum())} "
            f"| {int(part.get('putdown', pd.Series(dtype=int)).sum())} |"
        )

    lines += ["", "## Per-clip assignment", "",
              "| clip_id | day | split | pickup | putdown | duration (s) |", "|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(
            f"| {row.clip_id} | {row.recording_day} | {row.split} "
            f"| {getattr(row, 'pickup', 0)} | {getattr(row, 'putdown', 0)} | {row.duration_s:.1f} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    return table


def write_source_manifest(clips: pd.DataFrame, video_dir: Path, out_path: Path) -> None:
    """S3 location and checksum for each source video, which the bundle does not carry."""
    rows = []
    for row in clips.itertuples():
        local = video_dir / f"{row.clip_id}.mp4"
        rows.append({
            "clip_id": row.clip_id,
            "s3_uri": f"s3://chillnbite-cameras/anon/{row.clip_id}.mp4",
            "split": row.split,
            "size_bytes": local.stat().st_size if local.exists() else "",
            "sha256": sha256(local) if local.exists() else "",
            "fps": row.fps,
            "n_frames": row.n_frames,
        })
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".local/track_b1_bundle")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / ".local/track_b1_dataset_w15")
    parser.add_argument("--export-dir", type=Path, default=REPO_ROOT / ".local/cvat_exports/2026-09-09")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / ".local/source_videos")
    parser.add_argument("--include-embeddings", action="store_true",
                        help="Include the 23 MB cached backbone embeddings")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. Annotations, exactly as exported from CVAT.
    copy_into(args.export_dir, out / "annotations" / "cvat_exports_2026-09-09")

    # 2. Canonical dataset, including the split assignment.
    for name in ("events", "ignore_intervals", "clips", "candidates", "window_manifest"):
        source = args.dataset_dir / f"{name}.parquet"
        copy_into(source, out / "data" / f"{name}.parquet")
        # CSV alongside parquet so the bundle is readable without pyarrow.
        pd.read_parquet(source).to_csv(out / "data" / f"{name}.csv", index=False)
    copy_into(args.dataset_dir / "actor_tracks", out / "data" / "actor_tracks")
    copy_into(args.dataset_dir / "build_metadata.json", out / "data" / "build_metadata.json")

    table = write_split_listing(args.dataset_dir, out / "data" / "SPLITS.md")
    write_source_manifest(table, args.video_dir, out / "data" / "source_videos_manifest.csv")

    # 3. Trained weights. checkpoint_epoch_5.pt is omitted: it is a later, worse epoch
    #    than best_model.pt and would double the bundle for nothing.
    copy_into(REPO_ROOT / ".local/track_b1_finetune/checkpoints/best_model.pt",
              out / "models" / "finetuned_last2blocks_best.pt")
    copy_into(REPO_ROOT / ".local/track_b1_run_w15/checkpoints/head_best.pt",
              out / "models" / "frozen_probe_head_best.pt")
    for source, target in [
        (".local/track_b1_finetune/run_config.json", "finetuned_run_config.json"),
        (".local/track_b1_finetune/training_results.json", "finetuned_training_results.json"),
        (".local/track_b1_run_w15/head_results.json", "frozen_probe_results.json"),
        (".local/track_b1_run_w15/head_training_history.csv", "frozen_probe_history.csv"),
    ]:
        if (REPO_ROOT / source).exists():
            copy_into(REPO_ROOT / source, out / "models" / target)

    # 4. Every evaluation artefact for both models.
    for label, run in [("finetuned", ".local/track_b1_finetune"),
                       ("frozen_probe", ".local/track_b1_run_w15")]:
        run_dir = REPO_ROOT / run
        for path in sorted((run_dir / "predictions").glob("*")):
            copy_into(path, out / "results" / label / path.name)
        for path in sorted(run_dir.glob("*threshold*")) + sorted(run_dir.glob("chosen_thresholds*")):
            copy_into(path, out / "results" / label / path.name)
    if (REPO_ROOT / ".local/track_b1_RESULTS.md").exists():
        copy_into(REPO_ROOT / ".local/track_b1_RESULTS.md", out / "results" / "RESULTS.md")

    # 5. Frozen configuration and the manuscript.
    copy_into(REPO_ROOT / "configs/track_b1.yaml", out / "configs" / "track_b1.yaml")
    copy_into(REPO_ROOT / "docs/TRACK_B1_CVAT.md", out / "docs" / "TRACK_B1_CVAT.md")
    for name in ("track_b1_paper.docx", "track_b1_paper.md"):
        source = REPO_ROOT / ".local/paper" / name
        if source.exists():
            copy_into(source, out / "paper" / name)
    if (REPO_ROOT / ".local/paper/figures").exists():
        copy_into(REPO_ROOT / ".local/paper/figures", out / "paper" / "figures")

    if args.include_embeddings:
        copy_into(REPO_ROOT / ".local/track_b1_embeddings_w15", out / "data" / "embeddings")

    # 6. Checksum manifest over everything shipped.
    manifest_path = out / "MANIFEST.csv"
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append({
                "path": str(path.relative_to(out)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["size_bytes"] for r in rows)
    print(f"bundle: {out}")
    print(f"  {len(rows)} files, {total/1e6:.0f} MB")
    for top in sorted({Path(r['path']).parts[0] for r in rows}):
        size = sum(r["size_bytes"] for r in rows if Path(r["path"]).parts[0] == top)
        count = sum(1 for r in rows if Path(r["path"]).parts[0] == top)
        print(f"  {top:14s} {count:4d} files  {size/1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
