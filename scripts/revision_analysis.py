#!/usr/bin/env python3
"""Revision analysis: a single results registry, plus the cheap reviewer-requested checks.

    python scripts/revision_analysis.py

Everything here is decoded from the *saved per-window scores*; no model is re-run.
That is what makes the numbers traceable: one config -> one row -> one set of TP/FP/FN.

Outputs (in --out-dir):
  registry.csv            one row per (model, split, decode config): every metric with its
                          exact tIoU, thresholds, smoothing, boundary mode, unique video time
  lodo_frozen.csv         leave-one-day-out over all five days, frozen probe; thresholds and
                          checkpoint selected on the inner days only
  putdown_sensitivity.csv post-hoc: test-day putdown P/R/F1 vs threshold, fixed everything
                          else. Diagnostic, NOT a reported result.
  event_durations.csv     distribution of ground-truth event durations and the fraction with
                          d < tau*w for the tIoU ceiling argument
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.evaluation.contracts import (  # noqa: E402
    EvaluationEvent, EvaluationIgnoreInterval, EvaluationPrediction,
)
from pickup_putdown.evaluation.metrics import aggregate_metrics  # noqa: E402
from pickup_putdown.layer1.track_b1.inference import (  # noqa: E402
    InferenceConfig, WindowPrediction, create_event_predictions,
    detect_score_peaks, merge_same_type_regions, smooth_predictions,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import ClassificationHead  # noqa: E402

DATASET = REPO_ROOT / ".local/track_b1_dataset_w15"
EMB = REPO_ROOT / ".local/track_b1_embeddings_w15"
RUNS = {
    "frozen_probe": REPO_ROOT / ".local/track_b1_run_w15",
    "finetuned_2blocks": REPO_ROOT / ".local/track_b1_finetune",
}
# Validation-selected operating points, as recorded in the runs' chosen_thresholds files.
CONFIGS = {
    "frozen_probe": dict(pickup=0.45, putdown=0.60, smoothing=3),
    "finetuned_2blocks": dict(pickup=0.40, putdown=0.45, smoothing=5),
}
WINDOW, STRIDE = 1.5, 0.25


# ---------------------------------------------------------------------------
# Decoding from saved window scores
# ---------------------------------------------------------------------------


def decode(scores: pd.DataFrame, cfg: InferenceConfig) -> pd.DataFrame:
    rows = []
    for (clip, cand, actor), g in scores.groupby(["clip_id", "candidate_id", "actor_id"], sort=False):
        g = g.sort_values("window_start_s")
        preds = [
            WindowPrediction(
                window_start_s=r.window_start_s, window_end_s=r.window_end_s,
                window_center_s=(r.window_start_s + r.window_end_s) / 2,
                probs=np.array([r.p_background, r.p_pickup, r.p_putdown]),
                predicted_class=int(np.argmax([r.p_background, r.p_pickup, r.p_putdown])),
                confidence=float(max(r.p_background, r.p_pickup, r.p_putdown)),
            )
            for r in g.itertuples()
        ]
        regions = detect_score_peaks(smooth_predictions(preds, cfg.smoothing_window), cfg)
        merged = merge_same_type_regions(regions, cfg.same_type_merge_gap_s, cfg.min_event_duration_s)
        rows += [e.to_dict() for e in create_event_predictions(merged, clip, cand, actor, cfg)]
    return pd.DataFrame(rows)


def truth_for(clip_ids: set[str], events: pd.DataFrame, ignores: pd.DataFrame, clips: pd.DataFrame):
    ev = [EvaluationEvent(event_id=r.event_id, clip_id=r.clip_id, type=r.type,
                          t_start=float(r.t_start), t_end=float(r.t_end))
          for r in events[events.clip_id.isin(clip_ids)].itertuples()]
    ig = [EvaluationIgnoreInterval(clip_id=r.clip_id, t_start=float(r.t_start), t_end=float(r.t_end))
          for r in ignores[ignores.clip_id.isin(clip_ids)].itertuples()]
    dur = {r.clip_id: float(r.duration_s) for r in clips[clips.clip_id.isin(clip_ids)].itertuples()}
    return ev, ig, dur


def score(pred_df: pd.DataFrame, ev, ig, dur) -> dict:
    pr = [EvaluationPrediction(pred_id=r.pred_id, clip_id=r.clip_id, type=r.type,
                               t_start=float(r.t_start), t_end=float(r.t_end),
                               score=float(r.score), model=r.model)
          for r in pred_df.itertuples()] if len(pred_df) else []
    return aggregate_metrics(events=ev, preds=pr, clip_durations=dur, ignores=ig,
                             tiou_thresholds=(0.3, 0.5))


def flat(metrics: dict, **meta) -> dict:
    row = dict(meta)
    for t in ("0.3", "0.5"):
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            row[f"{k}@{t}"] = metrics[f"tiou@{t}"][k]
    for cls in ("pickup", "putdown"):
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            row[f"{cls}_{k}@0.5"] = metrics["per_type"][cls][k]
    row["fp_per_hour@0.5"] = metrics["fp_per_hour"]
    row["start_mae_s"] = metrics["start_mae_s"]
    row["end_mae_s"] = metrics["end_mae_s"]
    row["n_matched_for_mae"] = metrics["tiou@0.5"]["tp"]
    row["mAP@0.3"] = metrics["mAP"]["mAP@0.3"]
    row["mAP@0.5"] = metrics["mAP"]["mAP@0.5"]
    row["mAP_avg"] = metrics["mAP"]["mAP_avg"]
    return row


def cfg_for(pickup, putdown, smoothing, boundary="window_centers") -> InferenceConfig:
    return InferenceConfig(window_duration_s=WINDOW, window_stride_s=STRIDE,
                           pickup_threshold=pickup, putdown_threshold=putdown,
                           smoothing_window=smoothing, boundary_mode=boundary)


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def build_registry(events, ignores, clips) -> pd.DataFrame:
    rows = []
    for model, run in RUNS.items():
        c = CONFIGS[model]
        for split in ("val", "test"):
            path = run / "predictions" / f"window_scores_{split}.parquet"
            if not path.exists():
                continue
            scores = pd.read_parquet(path)
            clip_ids = set(clips[clips.split == split].clip_id)
            ev, ig, dur = truth_for(clip_ids, events, ignores, clips)
            for boundary in ("window_span", "window_centers"):
                for tag, pk, pd_ in (("tuned", c["pickup"], c["putdown"]), ("default", 0.5, 0.5)):
                    cfg = cfg_for(pk, pd_, c["smoothing"], boundary)
                    m = score(decode(scores, cfg), ev, ig, dur)
                    rows.append(flat(
                        m, run_id=f"{model}|{split}|{boundary}|{tag}", model=model, split=split,
                        checkpoint=str(next((run / "checkpoints").glob("*best*.pt"), "")),
                        window_s=WINDOW, stride_s=STRIDE, boundary_mode=boundary,
                        thresholds=tag, pickup_thr=pk, putdown_thr=pd_, smoothing=c["smoothing"],
                        n_gt_events=len(ev), unique_video_hours=round(sum(dur.values()) / 3600, 3),
                    ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Leave-one-day-out, frozen probe
# ---------------------------------------------------------------------------


def train_head(x, y, w, x_val, y_val, epochs=300, patience=40, seed=42):
    torch.manual_seed(seed)
    head = ClassificationHead(x.shape[1], 3, dropout=0.2)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    counts = torch.bincount(y, minlength=3).float().clamp(min=1)
    crit = nn.CrossEntropyLoss(weight=counts.sum() / (3 * counts), reduction="none")
    best, best_state, since = -1.0, None, 0
    for _ in range(epochs):
        head.train()
        for i in torch.randperm(len(x)).split(64):
            opt.zero_grad()
            (crit(head(x[i]), y[i]) * w[i]).mean().backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            p = head(x_val).argmax(1).numpy()
        f1s = []
        for k in range(3):
            tp = ((p == k) & (y_val.numpy() == k)).sum(); fp = ((p == k) & (y_val.numpy() != k)).sum()
            fn = ((p != k) & (y_val.numpy() == k)).sum()
            pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        f1 = float(np.mean(f1s))
        if f1 > best + 1e-4:
            best, best_state, since = f1, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    head.load_state_dict(best_state)
    return head


def lodo(events, ignores, clips) -> pd.DataFrame:
    emb = np.load(EMB / "embeddings.npy")
    man = pd.read_parquet(EMB / "manifest.parquet")
    man = man.merge(clips[["clip_id", "recording_day"]], on="clip_id")
    x_all = torch.from_numpy(emb).float()
    y_all = torch.from_numpy(man.label.to_numpy()).long()
    w_all = torch.from_numpy(man.sample_weight.to_numpy()).float()
    days = sorted(man.recording_day.unique())
    grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    rows = []
    for held in days:
        inner = [d for d in days if d != held]
        # Inner split: the sparsest inner day is the inner validation day, mirroring the
        # main protocol so selection never touches the held-out day.
        counts = man[man.recording_day.isin(inner) & (man.label > 0)].groupby("recording_day").size()
        inner_val = counts.idxmin()
        tr = man.recording_day.isin([d for d in inner if d != inner_val]).to_numpy()
        va = (man.recording_day == inner_val).to_numpy()
        te = (man.recording_day == held).to_numpy()

        head = train_head(x_all[tr], y_all[tr], w_all[tr], x_all[va], y_all[va])
        head.eval()
        with torch.no_grad():
            probs_va = torch.softmax(head(x_all[va]), 1).numpy()
            probs_te = torch.softmax(head(x_all[te]), 1).numpy()

        def scores_df(mask, probs):
            sub = man[mask].reset_index(drop=True)
            return pd.DataFrame({
                "clip_id": sub.clip_id, "candidate_id": sub.candidate_id, "actor_id": sub.actor_id,
                "window_start_s": sub.window_start_s, "window_end_s": sub.window_end_s,
                "p_background": probs[:, 0], "p_pickup": probs[:, 1], "p_putdown": probs[:, 2],
            })

        s_va, s_te = scores_df(va, probs_va), scores_df(te, probs_te)
        ev_va, ig_va, dur_va = truth_for(set(s_va.clip_id), events, ignores, clips)
        ev_te, ig_te, dur_te = truth_for(set(s_te.clip_id), events, ignores, clips)

        # Threshold selection on the inner validation day only.
        best = None
        for pk in grid:
            for pd_ in grid:
                m = score(decode(s_va, cfg_for(pk, pd_, 3)), ev_va, ig_va, dur_va)
                obj = (m["tiou@0.3"]["f1"] + m["tiou@0.5"]["f1"]) / 2
                if best is None or obj > best[0]:
                    best = (obj, pk, pd_)
        _, pk, pd_ = best
        m = score(decode(s_te, cfg_for(pk, pd_, 3)), ev_te, ig_te, dur_te)
        rows.append(flat(m, held_out_day=held, inner_val_day=inner_val, pickup_thr=pk,
                         putdown_thr=pd_, n_gt_events=len(ev_te),
                         unique_video_hours=round(sum(dur_te.values()) / 3600, 3)))
        print(f"  LODO {held}: F1@0.3={m['tiou@0.3']['f1']:.3f} F1@0.5={m['tiou@0.5']['f1']:.3f} "
              f"putdown F1={m['per_type']['putdown']['f1']:.3f} (thr {pk}/{pd_})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Post-hoc putdown threshold sensitivity on test (diagnostic only)
# ---------------------------------------------------------------------------


def putdown_sensitivity(events, ignores, clips) -> pd.DataFrame:
    run, c = RUNS["finetuned_2blocks"], CONFIGS["finetuned_2blocks"]
    scores = pd.read_parquet(run / "predictions" / "window_scores_test.parquet")
    ev, ig, dur = truth_for(set(clips[clips.split == "test"].clip_id), events, ignores, clips)
    rows = []
    for thr in np.round(np.arange(0.05, 0.71, 0.05), 2):
        m = score(decode(scores, cfg_for(c["pickup"], float(thr), c["smoothing"])), ev, ig, dur)
        rows.append({"putdown_thr": thr, **{f"putdown_{k}": m["per_type"]["putdown"][k]
                                            for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                     "f1@0.3": m["tiou@0.3"]["f1"], "f1@0.5": m["tiou@0.5"]["f1"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Event-duration distribution for the ceiling argument
# ---------------------------------------------------------------------------


def durations(events) -> pd.DataFrame:
    d = (events.t_end - events.t_start).to_numpy()
    rows = [{"stat": "n", "value": len(d)}, {"stat": "median_s", "value": float(np.median(d))},
            {"stat": "p25_s", "value": float(np.percentile(d, 25))},
            {"stat": "p75_s", "value": float(np.percentile(d, 75))}]
    for w in (2.5, 1.5):
        for tau in (0.3, 0.5):
            rows.append({"stat": f"frac_d_lt_{tau}x{w}s", "value": float((d < tau * w).mean())})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / ".local/paper/revision")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(DATASET / "events.parquet")
    ignores = pd.read_parquet(DATASET / "ignore_intervals.parquet")
    clips = pd.read_parquet(DATASET / "clips.parquet")

    print("registry ...")
    reg = build_registry(events, ignores, clips)
    reg.to_csv(a.out_dir / "registry.csv", index=False)
    print("leave-one-day-out (frozen probe) ...")
    lo = lodo(events, ignores, clips)
    lo.to_csv(a.out_dir / "lodo_frozen.csv", index=False)
    print("putdown threshold sensitivity (post-hoc, test) ...")
    putdown_sensitivity(events, ignores, clips).to_csv(a.out_dir / "putdown_sensitivity.csv", index=False)
    durations(events).to_csv(a.out_dir / "event_durations.csv", index=False)

    print(f"\nwrote {a.out_dir}")
    cols = ["run_id", "f1@0.3", "f1@0.5", "pickup_f1@0.5", "putdown_f1@0.5", "fp_per_hour@0.5", "unique_video_hours"]
    print(reg[cols].to_string(index=False))
    print("\nLODO summary (frozen probe):")
    print(lo[["held_out_day", "f1@0.3", "f1@0.5", "pickup_f1@0.5", "putdown_f1@0.5"]].to_string(index=False))
    print(f"  mean F1@0.3 = {lo['f1@0.3'].mean():.3f} ± {lo['f1@0.3'].std():.3f}   "
          f"mean putdown F1 = {lo['putdown_f1@0.5'].mean():.3f} ± {lo['putdown_f1@0.5'].std():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
