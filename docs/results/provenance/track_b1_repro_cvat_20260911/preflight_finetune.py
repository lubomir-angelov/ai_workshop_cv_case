"""Fine-tune preflight: the checks the run must pass before the full pixel-path run.

Builds the model exactly as scripts/train_track_b1.py does, then checks the warm start
survives Gate B, the trainable set, optimizer groups, the pixel data path, and one real
batch's loss and gradients. Also scores the warm-started model on validation through
pixels, which must reproduce the embedding-trained head's validation macro F1.
"""

import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from pickup_putdown.layer1.track_b1.cache import CachedTrackB1Dataset  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset_dir import load_dataset_dir, open_window_dataset  # noqa: E402
from pickup_putdown.layer1.track_b1.train import (  # noqa: E402
    TrainConfig,
    _parameter_groups,
    run_tiny_overfit_test,
    validate,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import create_model  # noqa: E402
from train_track_b1 import class_weights_from  # noqa: E402

E = Path(".local/track_b1_repro_cvat_20260911")
HEAD = E / "frozen_head/checkpoints/head_best.pt"
report: dict = {}

torch.manual_seed(42)
dataset_dir = load_dataset_dir(E / "dataset")
manifest = dataset_dir.table("window_manifest")
open_split = lambda m: open_window_dataset(  # noqa: E731
    dataset_dir, m, Path(".local/source_videos"), cache_dir=Path(".local/track_b1_cache"),
    frame_cache_dir=Path(".local/track_b1_frame_cache"),
)
train_ds, val_ds = open_split(manifest[manifest.split == "train"]), open_split(manifest[manifest.split == "val"])
assert isinstance(train_ds, CachedTrackB1Dataset) and isinstance(val_ds, CachedTrackB1Dataset)
report["data_path"] = f"{type(train_ds).__name__} (pixels from per-candidate crop cache); no embeddings loaded"

model = create_model(model_name="MCG-NJU/videomae-base", num_classes=3, dropout=0.1,
                     freeze_backbone=True, unfreeze_last_n_blocks=2, device="cuda")
checkpoint = torch.load(HEAD, map_location="cuda", weights_only=False)
model.head.load_state_dict(checkpoint["head_state_dict"])
assert checkpoint["encoder_weights_sha256"] == model.encoder_weights_sha256
device = torch.device("cuda")

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
blocks = sorted({n.split(".")[3] for n in trainable if n.startswith("encoder.encoder.layer.")})
unexpected = [n for n in trainable if not (n.startswith("head.") or n.startswith("encoder.encoder.layer.10.")
                                           or n.startswith("encoder.encoder.layer.11."))]
assert blocks == ["10", "11"] and not unexpected, (blocks, unexpected)
report["trainable"] = {
    "blocks": blocks, "n_tensors": len(trainable),
    "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "n_total_params": sum(p.numel() for p in model.parameters()),
    "head_tensors": [n for n in trainable if n.startswith("head.")],
    "frozen_top_level_modules": sorted({n.split(".")[1] for n, p in model.named_parameters()
                                        if not p.requires_grad and n.startswith("encoder.")}),
}

config = TrainConfig(learning_rate=1e-3, backbone_lr=5e-5, weight_decay=0.01, num_epochs=20, patience=5,
                     dropout=0.1, unfreeze_last_n_blocks=2, checkpoint_dir=E / "scratch_ckpt")
groups = _parameter_groups(model, config)
head_ids = {id(p) for p in model.head.parameters()}
report["optimizer_groups"] = [
    {"lr": g["lr"], "n_tensors": len(g["params"]), "all_head": all(id(p) in head_ids for p in g["params"]),
     "no_head": not any(id(p) in head_ids for p in g["params"])} for g in groups]
assert [g["lr"] for g in groups] == [1e-3, 5e-5]
assert report["optimizer_groups"][0]["all_head"] and report["optimizer_groups"][1]["no_head"]

# Pixel-path validation of the warm start; must reproduce the head trainer's number.
workers = dict(num_workers=0)  # script has no __main__ guard, so no spawn workers
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, **workers)
weights = class_weights_from(train_ds.manifest).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
warm = validate(model, val_loader, criterion, device)
report["warm_start_val_pixel_path"] = {"f1_macro": float(warm.f1_macro), "f1_per_class": warm.f1_per_class,
                                       "head_trainer_val_f1_macro": checkpoint["val_f1_macro"]}

# Gate B exactly as train() runs it: on a deep copy. The warm start must be untouched.
before = {k: v.detach().clone() for k, v in model.state_dict().items()}
passed = run_tiny_overfit_test(copy.deepcopy(model), train_ds, device, config)
changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
assert passed and not changed, (passed, changed[:5])
report["gate_b"] = {"passed": passed, "tensors_changed_in_trained_model": len(changed)}

# One real training batch: finite loss, finite non-zero grads on every trainable tensor, none elsewhere.
model.train()
batch = next(iter(DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)))
logits = model(batch["pixel_values"].to(device))
per_sample = torch.nn.functional.cross_entropy(logits, batch["label"].to(device), reduction="none")
loss = (per_sample * batch["sample_weight"].to(device, torch.float32)).mean()  # train_one_epoch's loss
loss.backward()
grads = {n: p.grad for n, p in model.named_parameters()}
missing = [n for n in trainable if grads[n] is None or not torch.isfinite(grads[n]).all() or grads[n].abs().sum() == 0]
leaked = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
assert torch.isfinite(loss) and not missing and not leaked, (missing[:5], leaked[:5])
report["real_batch"] = {
    "pixel_values_shape": list(batch["pixel_values"].shape), "labels": batch["label"].tolist(),
    "loss": float(loss), "grad_norm_head": float(torch.norm(torch.stack([grads[n].norm() for n in trainable if n.startswith("head.")]))),
    "grad_norm_backbone": float(torch.norm(torch.stack([grads[n].norm() for n in trainable if not n.startswith("head.")]))),
    "trainable_tensors_with_finite_nonzero_grad": len(trainable) - len(missing), "frozen_tensors_with_grad": len(leaked),
}
report["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
(E / "provenance/preflight_finetune.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
