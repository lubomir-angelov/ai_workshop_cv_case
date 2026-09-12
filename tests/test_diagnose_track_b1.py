"""scripts/diagnose_track_b1.py: stored checkpoint F1 is compared only on its own val data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_track_b1 import comparison_blocker  # noqa: E402

STORED = {"f1_macro": 0.61, "dataset_dir": "/data/ds", "input_mode": "annotation"}


def test_same_validation_data_is_compared():
    assert comparison_blocker("val", STORED, "/data/ds", "annotation", legacy=False) is None


def test_test_split_is_never_compared_with_validation_f1():
    assert "validation" in comparison_blocker("test", STORED, "/data/ds", "annotation", False)


def test_other_dataset_or_input_mode_is_not_compared():
    assert comparison_blocker("val", STORED, "/data/deploy", "deployment", False) is not None
    assert comparison_blocker("val", STORED, "/data/ds", "deployment", False) is not None


def test_unknown_provenance_is_not_compared_except_legacy_layout():
    unknown = {"f1_macro": 0.42}
    assert comparison_blocker("val", unknown, "/data/ds", "annotation", False) is not None
    assert comparison_blocker("val", unknown, "/legacy", "legacy_pose_clip_level", True) is None
