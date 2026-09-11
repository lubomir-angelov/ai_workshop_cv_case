"""Strict pretrained-encoder loading, checkpoints, and frozen vs fine-tuned paths."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import VideoMAEConfig, VideoMAEModel

from pickup_putdown.layer1.track_b1.train import TrainConfig, _parameter_groups, train
from pickup_putdown.layer1.track_b1.videomae_classifier import (
    VideoMAEClassifier,
    convert_encoder_state,
    load_checkpoint,
    save_checkpoint,
)

# The package re-exports the train() function under the module's name.
train_module = importlib.import_module("pickup_putdown.layer1.track_b1.train")


def _legacy_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """A tiny VideoMAE pretraining checkpoint in the MCG-NJU layout (q_bias/v_bias)."""
    config = VideoMAEConfig(
        image_size=32,
        patch_size=16,
        num_frames=2,
        tubelet_size=2,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        qkv_bias=True,
        use_mean_pooling=False,
    )
    torch.manual_seed(0)
    reference = VideoMAEModel(config).state_dict()
    legacy: dict[str, torch.Tensor] = {}
    for name, tensor in reference.items():
        value = torch.randn_like(tensor)  # distinct from any fresh initialisation
        if name.endswith("attention.attention.key.bias"):
            continue  # the legacy layout has no key bias
        name = name.replace("query.bias", "q_bias").replace("value.bias", "v_bias")
        legacy[f"videomae.{name}"] = value
    legacy["decoder.head.weight"] = torch.randn(4, 4)  # reconstruction decoder: discarded
    config.save_pretrained(tmp_path)
    save_file(legacy, str(tmp_path / "model.safetensors"))
    return tmp_path, legacy


def test_encoder_loads_pretrained_weights_strictly(tmp_path: Path):
    model_dir, legacy = _legacy_checkpoint(tmp_path)

    model = VideoMAEClassifier(model_name=str(model_dir), freeze_backbone=True)
    state = model.encoder.state_dict()

    base = "encoder.layer.1.attention.attention"
    assert torch.equal(state[f"{base}.query.bias"], legacy[f"videomae.{base}.q_bias"])
    assert torch.equal(state[f"{base}.value.bias"], legacy[f"videomae.{base}.v_bias"])
    assert torch.count_nonzero(state[f"{base}.key.bias"]) == 0
    for name, tensor in legacy.items():
        if name.startswith("videomae.") and not name.endswith(("q_bias", "v_bias")):
            assert torch.equal(state[name.removeprefix("videomae.")], tensor), name
    assert len(model.encoder_weights_sha256) == 64


def test_incomplete_pretrained_checkpoint_is_rejected(tmp_path: Path):
    model_dir, legacy = _legacy_checkpoint(tmp_path)
    del legacy["videomae.encoder.layer.0.output.dense.weight"]
    save_file(legacy, str(model_dir / "model.safetensors"))

    with pytest.raises(RuntimeError, match="Missing key"):
        VideoMAEClassifier(model_name=str(model_dir))


def test_legacy_bias_layout_is_left_alone_when_the_model_expects_it():
    checkpoint = {
        "videomae.encoder.layer.0.attention.attention.q_bias": torch.ones(2),
        "videomae.encoder.layer.0.attention.attention.v_bias": torch.ones(2),
    }
    model_keys = [
        "encoder.layer.0.attention.attention.q_bias",
        "encoder.layer.0.attention.attention.v_bias",
    ]

    assert set(convert_encoder_state(checkpoint, model_keys)) == set(model_keys)


def test_checkpoint_round_trip_is_strict_and_records_encoder_provenance(tmp_path: Path):
    model_dir, _ = _legacy_checkpoint(tmp_path / "pretrained")
    model = VideoMAEClassifier(model_name=str(model_dir), unfreeze_last_n_blocks=1)
    with torch.no_grad():
        model.head.classifier.weight.fill_(0.25)
    optimizer = torch.optim.AdamW(model.get_trainable_params())
    path = save_checkpoint(model, optimizer, 3, {"f1_macro": 0.5}, tmp_path / "best.pt")

    loaded = load_checkpoint(path, device="cpu")

    assert loaded["model_config"]["encoder_weights_sha256"] == model.encoder_weights_sha256
    assert torch.equal(loaded["model"].head.classifier.weight, model.head.classifier.weight)
    assert loaded["model"].unfreeze_last_n_blocks == 1


def test_frozen_and_fine_tuned_models_expose_the_right_parameters(tmp_path: Path):
    model_dir, _ = _legacy_checkpoint(tmp_path)
    frozen = VideoMAEClassifier(model_name=str(model_dir), freeze_backbone=True)
    tuned = VideoMAEClassifier(model_name=str(model_dir), unfreeze_last_n_blocks=1)

    assert all(
        name.startswith("head.") for name, p in frozen.named_parameters() if p.requires_grad
    )
    tuned_names = [name for name, p in tuned.named_parameters() if p.requires_grad]
    assert any(name.startswith("encoder.encoder.layer.1.") for name in tuned_names)
    assert not any(name.startswith("encoder.encoder.layer.0.") for name in tuned_names)

    groups = _parameter_groups(tuned, TrainConfig(learning_rate=1e-3, backbone_lr=5e-5))
    assert [group["lr"] for group in groups] == [1e-3, 5e-5]
    assert len(_parameter_groups(frozen, TrainConfig(backbone_lr=5e-5))) == 1


class _Windows(Dataset):
    def __init__(self, n: int = 6):
        torch.manual_seed(1)
        self.pixels = torch.randn(n, 2, 3, 32, 32)

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, index: int) -> dict:
        return {"pixel_values": self.pixels[index], "label": index % 3, "sample_weight": 1.0}


def test_gate_b_does_not_discard_a_warm_started_head(tmp_path: Path, monkeypatch):
    model_dir, _ = _legacy_checkpoint(tmp_path / "pretrained")
    model = VideoMAEClassifier(model_name=str(model_dir), unfreeze_last_n_blocks=1)
    with torch.no_grad():
        model.head.classifier.weight.fill_(0.5)  # the warm start

    def gate_b_that_trains(model, dataset, device, config):
        with torch.no_grad():
            model.head.classifier.weight.fill_(99.0)
        return True

    monkeypatch.setattr(train_module, "run_tiny_overfit_test", gate_b_that_trains)
    monkeypatch.setattr(train_module, "create_model", lambda **_: pytest.fail("model re-created"))
    loader = torch.utils.data.DataLoader(_Windows(), batch_size=3)
    config = TrainConfig(
        learning_rate=0.0,
        backbone_lr=0.0,
        num_epochs=1,
        warmup_epochs=0,
        checkpoint_dir=tmp_path / "ckpt",
        model_name=str(model_dir),
    )

    train(model, loader, loader, config, torch.device("cpu"))

    saved = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=False)
    head = saved["model_state_dict"]["head.classifier.weight"]
    # The cosine schedule's eta_min=1e-7 moves weights negligibly; a re-created head
    # (N(0, 0.02)) or Gate B's copy (99) would be far away.
    assert torch.allclose(head, torch.full_like(head, 0.5), atol=1e-3)
