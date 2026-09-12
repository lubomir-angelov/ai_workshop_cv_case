#!/usr/bin/env bash
# Deployment-input evaluation of the reproduced Track B1 checkpoints (see ../evaluation_protocol.json).
# Usage: run_eval.sh <phase>   phases: val_A | tune_B | val_B | test_A | test_B
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
X=.local/track_b1_deploy_eval_20260911
E=.local/track_b1_repro_cvat_20260911
export HF_HUB_OFFLINE=1
phase="${1:?phase required}"
declare -A CKPT=([frozen_head]="$E/frozen_head/checkpoints/head_best.pt" [finetune_last2]="$E/finetune_last2/checkpoints/best_model.pt")
declare -A FIXED=([frozen_head]="0.45 0.6 3" [finetune_last2]="0.5 0.2 3")
COMMON=(--same-type-merge-gap-s 0.75 --min-event-duration-s 0.3 --boundary-mode window_centers --num-workers 8 --batch-size 16)

infer() {  # model split pickup putdown smoothing outdir
  python scripts/infer_track_b1.py --dataset-dir "$X/dataset" --checkpoint "${CKPT[$1]}" --split "$2" \
    --pickup-threshold "$3" --putdown-threshold "$4" --smoothing-window "$5" "${COMMON[@]}" --output-dir "$6"
  python scripts/coverage_track_b1.py --dataset-dir "$X/dataset" --predictions-dir "$6" --split "$2"
}

chosen() {  # model -> "pickup putdown smoothing" from the frozen calibration selection
  python -c "import json,sys; d=json.load(open('$X/deploy_calibration_selection.json'))['models']['$1']['decode']; print(d['pickup_threshold'], d['putdown_threshold'], d['smoothing_window'])"
}

for m in frozen_head finetune_last2; do
  echo "=== $phase $m $(date -Iseconds)"
  case "$phase" in
    val_A|test_A)
      s="${phase%_A}"; read -r p d w <<<"${FIXED[$m]}"
      infer "$m" "$s" "$p" "$d" "$w" "$X/$m/A_fixed_decoder" ;;
    tune_B)
      mkdir -p "$X/$m/B_calibration_sweep"
      cp -n "$X/$m/A_fixed_decoder/window_scores_val.parquet" "$X/$m/B_calibration_sweep/"
      python scripts/tune_track_b1_thresholds.py --dataset-dir "$X/dataset" --predictions-dir "$X/$m/B_calibration_sweep" ;;
    val_B|test_B)
      s="${phase%_B}"; read -r p d w <<<"$(chosen "$m")"
      infer "$m" "$s" "$p" "$d" "$w" "$X/$m/B_deploy_calibrated" ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
done
echo "=== $phase done $(date -Iseconds)"
