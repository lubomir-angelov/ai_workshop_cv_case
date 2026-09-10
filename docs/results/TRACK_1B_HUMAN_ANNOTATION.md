# Results from training on the human annotated dataset from CVAT

Epoch 6 is the best model. Training learned something, but event recognition remains weak and validation performance is unstable.

## Observation	Interpretation

Training F1 increases from 0.303 → 0.581	The head is learning the sampled training examples
Validation F1 peaks at 0.424, then falls	Further training did not improve generalization
Best pickup F1 0.261, putdown F1 0.113	Neither event class is reliable yet
Epoch duration drops 30.8 → 2.5 minutes	Caching reduced training time by approximately 12.5×

The 80.7% best validation accuracy is misleading by itself. Your validation set contains 1,284 background windows out of 1,391: predicting background everywhere would achieve 92.3% accuracy, but only about 0.320 macro F1. Your model improves macro F1 to 0.424 by recognizing some events.

Also, training uses class-balanced sampling while validation retains its natural class imbalance. Their accuracy figures therefore aren’t directly comparable.

The history is consistent with increasing overfitting after epoch 6, but it does not isolate the cause. Frozen features, ambiguous window labels, weak crops, and the small number of independent annotated events could all contribute.