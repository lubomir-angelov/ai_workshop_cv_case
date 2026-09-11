#!/usr/bin/env bash
# Track B1 reproduction (2026-09-11), annotation-conditioned. Code: commit 65a3f07 for the
# dataset/cache/embeddings/frozen head/warm fine-tune (plus the artifact-only checkpoint saving
# committed as f98cff3), 4956f77 for the cold-head run (adds lint fixes 77a1541, where zip() now
# raises on length mismatch, and formatting 4956f77). Training commands are the recorded
# $E/logs/*.cmd; evaluation commands are reconstructed from each output's metrics_<split>.json
# (checkpoint, split, decode), with every decode flag spelled out.
set -Eeuo pipefail
E=.local/track_b1_repro_cvat_20260911
export HF_HUB_OFFLINE=1

python scripts/build_track_b1_dataset.py --input-mode annotation --output-dir $E/dataset
python scripts/build_track_b1_cache.py --dataset-dir $E/dataset --workers 10
python scripts/precompute_track_b1_embeddings.py --dataset-dir $E/dataset --split train val --batch-size 8 --num-workers 4
python scripts/train_track_b1_head.py --dataset-dir $E/dataset --output-dir $E/frozen_head --epochs 300 --patience 40 --batch-size 64 --learning-rate 1e-3 --weight-decay 1e-2 --dropout 0.2 --seed 42 --device cpu
python $E/provenance/preflight_finetune.py   # copy: preflight_finetune.py
python scripts/train_track_b1.py --dataset-dir $E/dataset --output-dir $E/finetune_last2 --unfreeze-last-n-blocks 2 --backbone-lr 5e-5 --learning-rate 1e-3 --init-head-from $E/frozen_head/checkpoints/head_best.pt --epochs 20 --patience 5 --batch-size 8 --weight-decay 0.01 --dropout 0.1 --num-workers 4 --seed 42 --device cuda
python scripts/train_track_b1.py --dataset-dir $E/dataset --output-dir $E/finetune_last2_cold --unfreeze-last-n-blocks 2 --backbone-lr 5e-5 --learning-rate 1e-3 --epochs 20 --patience 5 --batch-size 8 --weight-decay 0.01 --dropout 0.1 --num-workers 4 --seed 42 --device cuda

# --- frozen_head ---
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split val --pickup-threshold 0.4 --putdown-threshold 0.45 --smoothing-window 5 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/frozen_head/eval/val_configured_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/frozen_head/eval/val_configured_decode --split val
mkdir -p $E/frozen_head/eval/val_sweep && cp $E/frozen_head/eval/val_configured_decode/window_scores_val.parquet $E/frozen_head/eval/val_sweep/
python scripts/tune_track_b1_thresholds.py --dataset-dir $E/dataset --predictions-dir $E/frozen_head/eval/val_sweep
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split val --pickup-threshold 0.5 --putdown-threshold 0.5 --smoothing-window 3 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/frozen_head/eval/val_historical_default_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/frozen_head/eval/val_historical_default_decode --split val
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split val --pickup-threshold 0.45 --putdown-threshold 0.6 --smoothing-window 3 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/frozen_head/eval/val_tuned_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/frozen_head/eval/val_tuned_decode --split val
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split test --pickup-threshold 0.45 --putdown-threshold 0.6 --smoothing-window 3 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/frozen_head/eval/test_frozen_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/frozen_head/eval/test_frozen_decode --split test
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split val --output-dir $E/frozen_head/diagnostics_val --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split test --output-dir $E/frozen_head/diagnostics_test --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json

# --- finetune_last2 ---
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2/checkpoints/best_model.pt --split val --pickup-threshold 0.4 --putdown-threshold 0.45 --smoothing-window 5 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2/eval/val_configured_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2/eval/val_configured_decode --split val
mkdir -p $E/finetune_last2/eval/val_sweep && cp $E/finetune_last2/eval/val_configured_decode/window_scores_val.parquet $E/finetune_last2/eval/val_sweep/
python scripts/tune_track_b1_thresholds.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2/eval/val_sweep
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2/checkpoints/best_model.pt --split val --pickup-threshold 0.5 --putdown-threshold 0.2 --smoothing-window 3 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2/eval/val_tuned_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2/eval/val_tuned_decode --split val
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2/checkpoints/best_model.pt --split test --pickup-threshold 0.5 --putdown-threshold 0.2 --smoothing-window 3 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2/eval/test_frozen_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2/eval/test_frozen_decode --split test
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2/checkpoints/best_model.pt --split val --output-dir $E/finetune_last2/diagnostics_val --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2/checkpoints/best_model.pt --split test --output-dir $E/finetune_last2/diagnostics_test --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json

# --- finetune_last2_cold ---
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2_cold/checkpoints/best_model.pt --split val --pickup-threshold 0.4 --putdown-threshold 0.45 --smoothing-window 5 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2_cold/eval/val_configured_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2_cold/eval/val_configured_decode --split val
mkdir -p $E/finetune_last2_cold/eval/val_sweep && cp $E/finetune_last2_cold/eval/val_configured_decode/window_scores_val.parquet $E/finetune_last2_cold/eval/val_sweep/
python scripts/tune_track_b1_thresholds.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2_cold/eval/val_sweep
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2_cold/checkpoints/best_model.pt --split val --pickup-threshold 0.4 --putdown-threshold 0.35 --smoothing-window 5 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2_cold/eval/val_tuned_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2_cold/eval/val_tuned_decode --split val
python scripts/infer_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2_cold/checkpoints/best_model.pt --split test --pickup-threshold 0.4 --putdown-threshold 0.35 --smoothing-window 5 --same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --output-dir $E/finetune_last2_cold/eval/test_frozen_decode
python scripts/rescore_track_b1_events.py --dataset-dir $E/dataset --predictions-dir $E/finetune_last2_cold/eval/test_frozen_decode --split test
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2_cold/checkpoints/best_model.pt --split val --output-dir $E/finetune_last2_cold/diagnostics_val --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/finetune_last2_cold/checkpoints/best_model.pt --split test --output-dir $E/finetune_last2_cold/diagnostics_test --workers 4   # pre-fix run: files named val_predictions.csv / metrics.json

# split-aware diagnostics re-run (frozen head, after the fix)
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split val --output-dir $E/frozen_head/diagnostics_splitaware --workers 4 --review-count 0
python scripts/diagnose_track_b1.py --dataset-dir $E/dataset --checkpoint $E/frozen_head/checkpoints/head_best.pt --split test --output-dir $E/frozen_head/diagnostics_splitaware --workers 4 --review-count 0
