# Results from Training on the Human-Annotated CVAT Dataset

Epoch 6 produced the best model. Training showed learning, but event recognition remains weak and validation performance is unstable.

## Observations

| Observation | Interpretation |
|---|---|
| Training macro F1 increased from **0.303 → 0.581** | The classification head is learning the sampled training examples. |
| Validation macro F1 peaked at **0.424**, then declined | Further training did not improve generalization. |
| Best-checkpoint pickup F1 was **0.261** and putdown F1 was **0.113** | Neither event class is reliable yet. |
| Training epoch duration dropped from **30.8 → 2.5 minutes** | Caching reduced epoch duration by approximately **12.5×**. |

## Interpreting Validation Accuracy

The best checkpoint achieved **80.7% validation accuracy**, but this metric is misleading in isolation. The validation set contains **1,284 background windows out of 1,391**. Predicting background for every window would achieve **92.3% accuracy**, but only approximately **0.320 macro F1**.

The trained model improves macro F1 to **0.424** by recognizing some events, despite having lower overall accuracy than the background-only baseline.

Training uses class-balanced sampling, while validation retains its natural class imbalance. Training and validation accuracy are therefore not directly comparable.

## Generalization Limitations

The training history is consistent with increasing overfitting after epoch 6, but it does not isolate the cause. Potential contributors include:

- Frozen encoder features.
- Ambiguous window labels.
- Crops that do not clearly show the relevant interaction.
- A small number of independent annotated events.

## Verified metrics and provenance (reproduced 2026-09-11)

Reproduced exactly on the integration branch with
`scripts/diagnose_track_b1.py --legacy-data-dir .local/track_b1_data` from
`.local/track_b1_output_human/checkpoints/best_model.pt` (best epoch 6):

| validation windows | macro F1 | background F1 | pickup F1 | putdown F1 |
|---|---|---|---|---|
| 1,391 | 0.4240668 | 0.8982630 | 0.2612613 | 0.1126761 |

Confusion matrix, rows = true, columns = predicted (background, pickup, putdown):
background 1086 / 103 / 95; pickup 39 / 29 / 14; putdown 9 / 8 / 8.

Setup: `TrainConfig` defaults (frozen VideoMAE-base, lr 1e-4), 2.5 s / 0.5 s windows,
16 frames, crop = pose person box + 15% margin (no shelf region was applied), clip-level
labels from `events_human.csv`, validation day 20260526 (`human_b1_2026_09_09` in
`configs/track_b1_splits.yaml`). Clip-level labels mean overlapping actors could receive
each other's events; with actor-resolved CVAT labels the same checkpoint scores 0.4177
on 1,372 windows of that day. See `docs/TRACK_B1_INTEGRATION.md`.
