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
