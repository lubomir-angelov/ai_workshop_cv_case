"""Collect annotation- and deployment-input metrics of the same checkpoints into markdown tables."""

import json
import sys
from pathlib import Path

X = Path(".local/track_b1_deploy_eval_20260911")
E = Path(".local/track_b1_repro_cvat_20260911")
MODELS = ("frozen_head", "finetune_last2")
import pandas as pd

_clips = pd.read_parquet(X / "dataset" / "clips.parquet")
HOURS = (_clips.groupby("split")["duration_s"].sum() / 3600).to_dict()  # clips never overlap in time


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def runs(split):
    for m in MODELS:
        ann = {"val": "val_tuned_decode", "test": "test_frozen_decode"}[split]
        yield m, "annotation", "A (annotation-val decoder)", E / m / "eval" / ann, None
        yield m, "deployment", "A fixed (annotation-val decoder)", X / m / "A_fixed_decoder", "A"
        yield m, "deployment", "B deployment-val calibrated", X / m / "B_deploy_calibrated", "B"


def event_row(m, mode, dec, d, split):
    rec = load(d / f"metrics_{split}.json")
    if rec is None:
        return None
    by = rec.get("event_metrics_by_counting_policy") or load(
        d / f"rescored_by_counting_policy_{split}.json"
    )["event_metrics_by_counting_policy"]
    it, gr = by["per_item"], by["per_event_group"]
    fp_h3, fp_h5 = (it[f"tiou@{t}"]["fp"] / HOURS[split] for t in (0.3, 0.5))
    t3, t5 = it["tiou@0.3"], it["tiou@0.5"]
    dcd = rec["decode"]
    mae = (
        f"{it['start_mae_s']:.2f} / {it['end_mae_s']:.2f}" if it["start_mae_s"] is not None else "—"
    )
    return (
        f"| {m} | {mode} | {dec} ({dcd['pickup_threshold']}/{dcd['putdown_threshold']}/{dcd['smoothing_window']}) "
        f"| {rec['n_predictions']} | {t3['precision']:.3f} / {t3['recall']:.3f} / **{t3['f1']:.3f}** "
        f"| {t5['precision']:.3f} / {t5['recall']:.3f} / **{t5['f1']:.3f}** "
        f"| {it['per_type']['pickup']['f1']:.3f} / {it['per_type']['putdown']['f1']:.3f} "
        f"| {fp_h3:.1f} / {fp_h5:.1f} | {mae} | {gr['tiou@0.3']['f1']:.3f} / {gr['tiou@0.5']['f1']:.3f} |"
    )


def main():
    for split in ("val", "test"):
        print(f"\n### {split}: event level, per-item primary\n")
        print(
            "| checkpoint | input | decoder (pickup/putdown/smoothing) | predictions | P / R / F1 @0.3 | P / R / F1 @0.5 "
            "| pickup / putdown F1 @0.5 | FP/h @0.3 / @0.5 | start / end MAE s (midpoint-matched) | per-event-group F1 @0.3 / @0.5 |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m, mode, dec, d, _ in runs(split):
            row = event_row(m, mode, dec, d, split)
            if row:
                print(row)
        print(f"\nfootage hours ({split}): {HOURS.get(split)}")
    for split in ("val", "test"):
        for m in MODELS:
            for tag in ("A_fixed_decoder", "B_deploy_calibrated"):
                c = load(X / m / tag / f"coverage_{split}.json")
                if c:
                    print(f"\n#### coverage {split} {m} {tag}")
                    print(json.dumps({k: c[k] for k in ("ground_truth_counts", "n_suppressed_duplicates", "n_predictions", "false_positive_attribution", "n_candidates", "n_candidates_without_windows")}))
                    for pol in ("per_item", "per_event_group"):
                        p = c[pol]
                        print(pol, json.dumps({k: p[k] for k in ("n", "n_ignored_by_ignore_intervals", "association_status", "flags", "partition", "partition_reconciles")}))
                        print(pol, "matched", json.dumps(p["matched_by_partition"]))
                        print(pol, "crop", json.dumps(p["crop"]))
                    w = c["window_level"]
                    print("window", json.dumps(w))



def suppression_check(split):
    """Duplicate suppression: count, and per-item TP with vs without it (same decode)."""
    sys.path.insert(0, "src")
    from pickup_putdown.layer1.track_b1.inference import (
        InferenceConfig,
        decode_window_scores,
        evaluate_events,
        suppress_duplicate_events,
    )

    ds = X / "dataset"
    clips = pd.read_parquet(ds / "clips.parquet")
    split_clips = set(clips.loc[clips["split"] == split, "clip_id"])
    events = pd.read_parquet(ds / "events.parquet")
    truth = events[events["clip_id"].isin(split_clips)]
    ign = pd.read_parquet(ds / "ignore_intervals.parquet")
    ign = ign[ign["clip_id"].isin(split_clips)]
    dur = {c: float(d) for c, d in zip(clips["clip_id"], clips["duration_s"]) if c in split_clips}
    # same-type events of one CVAT actor closer than the merge gap: decode could fuse them
    at_risk = 0
    for _, g in truth.drop_duplicates("event_group_id").groupby(["clip_id", "actor_id", "type"]):
        g = g.sort_values("t_start")
        at_risk += int(((g["t_start"].to_numpy()[1:] - g["t_end"].to_numpy()[:-1]) < 0.75).sum())
    print(f"\n#### suppression / merge safeguards ({split}); same-type same-actor GT pairs with gap < 0.75 s: {at_risk}")
    for m in MODELS:
        for tag in ("A_fixed_decoder", "B_deploy_calibrated"):
            d = X / m / tag
            rec = load(d / f"metrics_{split}.json")
            if rec is None:
                continue
            scores = pd.read_parquet(d / f"window_scores_{split}.parquet")
            window = json.loads((ds / "build_metadata.json").read_text())["window"]
            cfg = InferenceConfig(
                window_duration_s=window["window_duration_s"],
                window_stride_s=window["window_stride_s"],
                **rec["decode"],
            )
            raw = decode_window_scores(scores, cfg)
            kept, n = suppress_duplicate_events(raw)
            saved = pd.read_csv(d / f"predictions_{split}.csv")
            same = len(kept) == len(saved) and set(kept["pred_id"]) == set(saved["pred_id"])
            out = []
            for name, preds in (("without", raw), ("with", kept)):
                met = evaluate_events(preds, truth, ign, dur, (0.3, 0.5))
                out.append(f"{name}: TP@0.3={met['tiou@0.3']['tp']} FP@0.3={met['tiou@0.3']['fp']} TP@0.5={met['tiou@0.5']['tp']} FP@0.5={met['tiou@0.5']['fp']}")
            print(f"{m} {tag}: suppressed={n} (recorded {rec['n_suppressed_duplicates']}), re-decode matches saved={same}; " + " | ".join(out))


def crop_vs_outcome(split):
    for m in MODELS:
        for tag in ("A_fixed_decoder", "B_deploy_calibrated"):
            f = X / m / tag / f"event_coverage_{split}.csv"
            if not f.exists():
                continue
            t = pd.read_csv(f)
            t = t[(t["coverage"] == "own_covered") & (~t["ignored"])]
            t["crop_bin"] = pd.cut(t["own_box_in_crop_mean"], [-0.01, 0.5, 0.9, 1.01], labels=["<0.5", "0.5-0.9", ">=0.9"])
            for thr in ("0.3", "0.5"):
                col = t[f"matched@{thr}"].astype(str) == "True"
                tab = t.assign(hit=col).groupby("crop_bin", observed=False)["hit"].agg(["sum", "count"])
                print(f"crop vs match {split} {m} {tag} tIoU {thr}: " + ", ".join(f"{k}: {int(r['sum'])}/{int(r['count'])}" for k, r in tab.iterrows()))


def annotation_crop_scale(split):
    """Same crop-scale measures on the annotation-mode inputs the checkpoints were trained on."""
    sys.path.insert(0, "src")
    import numpy as np

    from pickup_putdown.layer1.track_b1.actor_association import window_evidence
    from pickup_putdown.layer1.track_b1.dataset import (
        compute_actor_crop_box,
        crop_span,
        generate_inference_windows,
    )
    from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir

    ds = load_dataset_dir(E / "dataset")
    cfg = ds.window_config()
    clips = ds.table("clips").set_index("clip_id")
    split_clips = set(clips.index[clips["split"] == split])
    events = ds.table("events")
    events = events[events["clip_id"].isin(split_clips)].drop_duplicates("event_group_id")
    cands = ds.table("candidates")
    cands = cands[cands["clip_id"].isin(split_clips)]
    wins = pd.DataFrame([w.to_dict() for w in generate_inference_windows(cands, cfg)])
    share, area = [], []
    for _, ev in events.iterrows():
        track = pd.read_parquet(ds.tracks_dir / f"{ev['clip_id']}.parquet")
        clip = clips.loc[ev["clip_id"]]
        size = (int(clip["width"]), int(clip["height"]))
        w = wins[(wins["clip_id"] == ev["clip_id"]) & (wins["actor_id"] == ev["actor_id"])
                 & (wins["window_center_s"] >= ev["t_start"]) & (wins["window_center_s"] <= ev["t_end"])]
        s_ev, a_ev = [], []
        for _, row in w.iterrows():
            crop = compute_actor_crop_box(track[track["actor_id"] == row["actor_id"]], None, *crop_span(row, cfg), cfg.crop_margin, size)
            e = window_evidence(row["window_start_s"], row["window_end_s"], cfg.num_frames, float(clip["fps"]), crop, ev, track)
            if e["event_box_share_of_crop"] is not None:
                s_ev.append(e["event_box_share_of_crop"])
                a_ev.append((crop[2] - crop[0]) * (crop[3] - crop[1]) / (size[0] * size[1]))
        if s_ev:
            share.append(np.mean(s_ev))
            area.append(np.mean(a_ev))
    print(f"annotation-mode crop scale {split}: events={len(share)} box share of crop median={np.median(share):.3f} "
          f"(IQR {np.percentile(share, 25):.3f}-{np.percentile(share, 75):.3f}); crop share of frame median={np.median(area):.3f}")


if __name__ == "__main__":
    main()
    for split in ("val", "test"):
        suppression_check(split)
        crop_vs_outcome(split)
        annotation_crop_scale(split)
