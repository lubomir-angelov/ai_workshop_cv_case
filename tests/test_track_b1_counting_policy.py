"""GT counting policies for Track B1 event scoring: per item (primary) vs per event group."""

from __future__ import annotations

import pandas as pd
import pytest

from pickup_putdown.layer1.track_b1.inference import (
    evaluate_events_by_policy,
    ground_truth_counts,
    ground_truth_for_policy,
)

CLIP = "clip_a"


def _event(event_id, group, t0, t1, kind="pickup", actor="trk000", items=1, index=0, clip=CLIP):
    return {
        "event_id": event_id,
        "clip_id": clip,
        "type": kind,
        "t_start": t0,
        "t_end": t1,
        "actor_id": actor,
        "item_count": items,
        "item_index": index,
        "event_group_id": group,
        "confidence": "high",
        "hard_case": False,
    }


def _three_items(group="g3", t0=10.0, t1=11.0, clip=CLIP):
    return [_event(f"{group}_{i}", group, t0, t1, items=3, index=i, clip=clip) for i in range(3)]


def _pred(pred_id, t0, t1, kind="pickup", clip=CLIP):
    return {
        "pred_id": pred_id,
        "clip_id": clip,
        "type": kind,
        "t_start": t0,
        "t_end": t1,
        "score": 0.9,
        "model": "test",
    }


def _score(truth: pd.DataFrame, preds: list[dict]) -> dict:
    ignores = pd.DataFrame(columns=["clip_id", "t_start", "t_end"])
    return evaluate_events_by_policy(
        pd.DataFrame(preds), truth, ignores, {CLIP: 60.0, "clip_b": 60.0}, (0.5,)
    )


def test_multi_item_event_expands_per_item_and_collapses_per_group():
    truth = pd.DataFrame(_three_items())
    assert len(ground_truth_for_policy(truth, "per_item")) == 3
    collapsed = ground_truth_for_policy(truth, "per_event_group")
    assert len(collapsed) == 1 and collapsed.iloc[0]["item_index"] == 0
    assert ground_truth_counts(truth) == {
        "item_rows": 3,
        "event_groups": 1,
        "distinct_intervals": 1,
    }


def test_one_prediction_matches_one_item_row_only():
    """No prediction is duplicated to fill item rows: 1 TP + 2 FN per item, 1 TP per group."""
    metrics = _score(pd.DataFrame(_three_items()), [_pred("p1", 10.0, 11.0)])
    assert (metrics["per_item"]["tiou@0.5"]["tp"], metrics["per_item"]["tiou@0.5"]["fn"]) == (1, 2)
    assert metrics["per_item"]["n_ground_truth_rows"] == 3
    grouped = metrics["per_event_group"]["tiou@0.5"]
    assert (grouped["tp"], grouped["fp"], grouped["fn"]) == (1, 0, 0)


def test_two_predictions_on_one_group_leave_a_false_positive_per_group():
    preds = [_pred("p1", 10.0, 11.0), _pred("p2", 10.1, 11.0)]
    metrics = _score(pd.DataFrame(_three_items()), preds)
    assert metrics["per_item"]["tiou@0.5"]["tp"] == 2
    grouped = metrics["per_event_group"]["tiou@0.5"]
    assert (grouped["tp"], grouped["fp"]) == (1, 1)


def test_independent_simultaneous_events_are_not_merged_by_timestamp():
    """Different actors, same interval, separate groups: two events under both policies."""
    truth = pd.DataFrame(
        [
            _event("a", "grp_a", 10.0, 11.0, actor="trk000"),
            _event("b", "grp_b", 10.0, 11.0, actor="trk001"),
            _event("c", "grp_c", 11.0, 12.0, kind="putdown", actor="trk000"),
        ]
    )
    collapsed = ground_truth_for_policy(truth, "per_event_group")
    assert sorted(collapsed["event_id"]) == ["a", "b", "c"]
    assert ground_truth_counts(truth)["distinct_intervals"] == 2
    metrics = _score(truth, [_pred("p1", 10.0, 11.0)])
    for policy in ("per_item", "per_event_group"):
        assert metrics[policy]["tiou@0.5"]["fn"] == 2


def test_group_ids_are_scoped_by_clip():
    truth = pd.DataFrame(
        _three_items(group="g", clip=CLIP) + _three_items(group="g", clip="clip_b")
    )
    assert len(ground_truth_for_policy(truth, "per_event_group")) == 2


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_group_id_is_rejected(missing):
    truth = pd.DataFrame(_three_items())
    truth.loc[1, "event_group_id"] = missing
    with pytest.raises(ValueError, match="lack event_group_id"):
        ground_truth_for_policy(truth, "per_event_group")
    assert len(ground_truth_for_policy(truth, "per_item")) == 3


def test_missing_group_column_is_rejected():
    truth = pd.DataFrame(_three_items()).drop(columns="event_group_id")
    with pytest.raises(ValueError, match="event_group_id column"):
        ground_truth_for_policy(truth, "per_event_group")


@pytest.mark.parametrize(
    ("column", "value"),
    [("type", "putdown"), ("t_end", 12.0), ("actor_id", "trk009"), ("item_count", 2)],
)
def test_ambiguous_group_is_rejected(column, value):
    truth = pd.DataFrame(_three_items())
    truth.loc[2, column] = value
    with pytest.raises(ValueError, match="event group"):
        ground_truth_for_policy(truth, "per_event_group")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown counting policy"):
        ground_truth_for_policy(pd.DataFrame(_three_items()), "per_interval")
