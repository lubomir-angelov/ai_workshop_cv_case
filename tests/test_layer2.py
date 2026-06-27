"""Comprehensive tests for Layer 2 modules.

Covers:
- Deterministic window generation (short/partial windows, clip-boundary clamping)
- Timestamp metadata
- Strict schema validation (reject unsupported types, invalid intervals, extra fields)
- Retry after invalid JSON/schema validation
- Failure after second invalid output
- Raw response preservation
- Timestamp conversion
- Pickup/putdown remaining separate
- Duplicate merging, repeated-event preservation, two-item preservation
- Task 8 evaluator compatibility
- No dependency on Layer 1 files
- No live endpoint requirement (mocked)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pickup_putdown.evaluation import (
    Criterion,
    EvaluationEvent,
)
from pickup_putdown.layer2 import (
    evaluation,
    prompts,
    schemas,
)
from pickup_putdown.layer2.merge_predictions import merge_predictions as merge_fn
from pickup_putdown.layer2.qwen_client import QwenClientConfig, call_qwen
from pickup_putdown.layer2.schemas import (
    Layer2Event,
    Layer2Prediction,
    Layer2WindowResponse,
)
from pickup_putdown.layer2.window_generator import (
    WindowConfig,
    generate_all_windows,
    generate_windows,
)
from pickup_putdown.vlm import VlmError, VlmResponse

# ===================================================================
# Window generation tests
# ===================================================================


class TestWindowGeneration:
    """Deterministic window generation tests."""

    def test_basic_windows(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=20.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=5.0),
        )
        assert len(windows) == 4
        assert windows[0].window_start_s == 0.0
        assert windows[0].window_end_s == 10.0
        assert windows[1].window_start_s == 5.0
        assert windows[1].window_end_s == 15.0
        assert windows[2].window_start_s == 10.0
        assert windows[2].window_end_s == 20.0
        assert windows[3].window_start_s == 15.0
        assert windows[3].window_end_s == 20.0

    def test_deterministic_ids(self):
        w1 = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=10.0,
            config=WindowConfig(window_duration_s=5.0, step_duration_s=5.0),
        )
        w2 = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=10.0,
            config=WindowConfig(window_duration_s=5.0, step_duration_s=5.0),
        )
        assert w1[0].window_id == w2[0].window_id

    def test_short_span(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=1.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=5.0),
        )
        assert len(windows) == 0

    def test_final_partial_window(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=12.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=5.0),
        )
        # Windows at [0,10], [5,10], [10,12] or similar
        assert len(windows) >= 1
        assert windows[-1].window_end_s == 12.0

    def test_clip_boundary_clamping(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=30.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=5.0),
            clip_duration_s=20.0,
        )
        assert all(w.window_end_s <= 20.0 for w in windows)

    def test_stable_source_timestamps(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=5.0,
            active_end_s=25.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=5.0),
        )
        assert all(w.source_timestamp_s == 5.0 for w in windows)

    def test_all_windows_deterministic_order(self):
        spans = [
            {"clip_id": "b", "active_span_id": "s1", "t_start": 0.0, "t_end": 10.0},
            {"clip_id": "a", "active_span_id": "s1", "t_start": 0.0, "t_end": 10.0},
        ]
        windows = generate_all_windows(
            spans, config=WindowConfig(window_duration_s=5.0, step_duration_s=5.0)
        )
        clip_ids = [w.clip_id for w in windows]
        assert clip_ids == sorted(clip_ids)

    def test_invalid_span(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=10.0,
            active_end_s=5.0,
        )
        assert len(windows) == 0

    def test_equal_start_end(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=5.0,
            active_end_s=5.0,
        )
        assert len(windows) == 0

    def test_window_metadata(self):
        windows = generate_windows(
            clip_id="clip_1",
            active_span_id="span_1",
            active_start_s=0.0,
            active_end_s=10.0,
            config=WindowConfig(window_duration_s=10.0, step_duration_s=10.0),
        )
        assert len(windows) == 1
        w = windows[0]
        assert w.clip_id == "clip_1"
        assert w.active_span_id == "span_1"
        assert w.duration_s == 10.0
        assert w.overlap_s == 0.0
        assert w.n_samples == 10


# ===================================================================
# Schema validation tests
# ===================================================================


class TestSchemas:
    """Strict Pydantic schema validation."""

    def test_valid_event(self):
        ev = Layer2Event(
            event_type="pickup",
            relative_start_s=1.0,
            relative_end_s=3.0,
            item_count=2,
            visibility="visible",
            confidence=0.85,
        )
        assert ev.event_type == "pickup"
        assert ev.item_count == 2

    def test_invalid_event_type(self):
        with pytest.raises(ValueError, match="event_type"):
            Layer2Event(
                event_type="invalid_type",
                relative_start_s=1.0,
                relative_end_s=3.0,
            )

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="relative_end_s"):
            Layer2Event(
                event_type="pickup",
                relative_start_s=3.0,
                relative_end_s=1.0,
            )

    def test_out_of_window_timestamps(self):
        with pytest.raises(ValueError, match="non-negative"):
            Layer2Event(
                event_type="pickup",
                relative_start_s=-1.0,
                relative_end_s=1.0,
            )

    def test_invalid_count(self):
        with pytest.raises(ValueError):
            Layer2Event(
                event_type="pickup",
                relative_start_s=0.0,
                relative_end_s=1.0,
                item_count=0,
            )

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            Layer2Event(
                event_type="pickup",
                relative_start_s=0.0,
                relative_end_s=1.0,
                extra_field="should_fail",
            )

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            Layer2Event(
                event_type="pickup",
                relative_start_s=0.0,
                relative_end_s=1.0,
                confidence=1.5,
            )

    def test_confidence_zero(self):
        ev = Layer2Event(
            event_type="pickup",
            relative_start_s=0.0,
            relative_end_s=1.0,
            confidence=0.0,
        )
        assert ev.confidence == 0.0

    def test_window_response_empty_events(self):
        resp = Layer2WindowResponse(
            window_id="w1",
            clip_id="c1",
            window_start_s=0.0,
            window_end_s=10.0,
        )
        assert resp.events == []

    def test_window_response_invalid_window_bounds(self):
        with pytest.raises(ValueError, match="window_end_s"):
            Layer2WindowResponse(
                window_id="w1",
                clip_id="c1",
                window_start_s=10.0,
                window_end_s=5.0,
            )

    def test_layer2_prediction_to_canonical(self):
        pred = Layer2Prediction(
            clip_id="c1",
            pred_id="l2_abc",
            event_type="pickup",
            t_start_s=1.0,
            t_end_s=3.0,
            confidence=0.9,
        )
        rows = pred.to_canonical()
        assert len(rows) == 1
        assert rows[0]["clip_id"] == "c1"
        assert rows[0]["type"] == "pickup"
        assert rows[0]["t_start"] == 1.0
        assert rows[0]["t_end"] == 3.0
        assert rows[0]["score"] == 0.9

    def test_layer2_prediction_to_canonical_two_items(self):
        pred = Layer2Prediction(
            clip_id="c1",
            pred_id="l2_abc",
            event_type="pickup",
            t_start_s=1.0,
            t_end_s=3.0,
            item_count=2,
            confidence=0.9,
            event_group_id="l2_abc",
        )
        rows = pred.to_canonical()
        assert len(rows) == 2
        assert rows[0]["pred_id"] == "l2_abc_item0"
        assert rows[1]["pred_id"] == "l2_abc_item1"
        assert rows[0]["event_group_id"] == "l2_abc"
        assert rows[1]["event_group_id"] == "l2_abc"


# ===================================================================
# Qwen client tests (mocked, no live endpoint)
# ===================================================================


def _make_mock_vlm_response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": content, "reasoning_content": ""},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }


class TestQwenClient:
    """VLM client wrapper tests with mocked HTTP."""

    def test_success_returns_validated_events(self):
        valid_json = json.dumps(
            {
                "events": [
                    {
                        "event_type": "pickup",
                        "relative_start_s": 1.0,
                        "relative_end_s": 3.0,
                        "item_count": 1,
                        "visibility": "visible",
                        "confidence": 0.9,
                    }
                ],
                "reasoning": "saw it",
            }
        )
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content=valid_json, finish_reason="stop"),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )

        assert result.validated_response is not None
        assert len(result.predictions) == 1
        assert result.predictions[0].event_type == "pickup"

    def test_retry_after_invalid_json(self):
        """One retry after invalid JSON."""
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return VlmResponse(content="not json", finish_reason="stop")
            return VlmResponse(
                content=json.dumps(
                    {
                        "events": [],
                        "reasoning": "ok",
                    }
                ),
                finish_reason="stop",
            )

        with patch("pickup_putdown.layer2.qwen_client.call_vlm", side_effect=_side_effect):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=2, retry_delay_s=0.0),
            )

        assert call_count == 2
        assert result.validated_response is not None
        assert len(result.attempts) == 2

    def test_failure_after_second_invalid(self):
        """No predictions after exhausting retries on invalid JSON."""
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return VlmResponse(content="bad json", finish_reason="stop")

        with patch("pickup_putdown.layer2.qwen_client.call_vlm", side_effect=_side_effect):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=2, retry_delay_s=0.0),
            )

        assert call_count == 2
        assert result.validated_response is None
        assert result.error is not None
        assert len(result.predictions) == 0

    def test_retry_on_validation_error(self):
        """Retry on invalid event_type, not just bad JSON."""
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return VlmResponse(
                    content=json.dumps(
                        {
                            "events": [{"event_type": "bad_type"}],
                        }
                    ),
                    finish_reason="stop",
                )
            return VlmResponse(
                content=json.dumps(
                    {
                        "events": [
                            {
                                "event_type": "pickup",
                                "relative_start_s": 1.0,
                                "relative_end_s": 3.0,
                            }
                        ],
                        "reasoning": "ok",
                    }
                ),
                finish_reason="stop",
            )

        with patch("pickup_putdown.layer2.qwen_client.call_vlm", side_effect=_side_effect):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=2, retry_delay_s=0.0),
            )

        assert call_count == 2
        assert result.validated_response is not None
        assert len(result.predictions) == 1
        assert result.predictions[0].event_type == "pickup"

    def test_raw_response_preserved(self):
        valid_json = json.dumps({"events": [], "reasoning": "test"})
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content=valid_json, finish_reason="stop"),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )

        assert len(result.attempts) == 1
        assert result.attempts[0].raw_response == valid_json
        assert result.attempts[0].is_success is True

    def test_validation_errors_recorded(self):
        invalid_json = json.dumps({"events": [{"event_type": "bad_type"}]})
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content=invalid_json, finish_reason="stop"),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )

        assert result.validated_response is None
        assert result.error is not None
        assert len(result.attempts) == 1
        assert result.attempts[0].validation_errors

    def test_request_metadata_recorded(self):
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content=json.dumps({"events": []}), finish_reason="stop"),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )

        assert len(result.attempts) == 1
        assert result.attempts[0].request.model == "llamacpp/Qwen3.6-35B-A3B-UD-Q4_K_XL"
        assert result.attempts[0].attempt_number == 1

    def test_failed_response_not_in_predictions(self):
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content="", error=VlmError(message="timeout")),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )

        assert result.predictions == []
        assert result.error == "timeout"

    def test_configurable_model_id(self):
        with patch(
            "pickup_putdown.layer2.qwen_client.call_vlm",
            return_value=VlmResponse(content=json.dumps({"events": []}), finish_reason="stop"),
        ):
            result = call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(model_id="custom-model", max_attempts=1),
            )

        assert result.attempts[0].request.model == "custom-model"

    def test_no_live_endpoint_required(self):
        """All code paths use mocked call_vlm — no network calls."""
        with patch("pickup_putdown.layer2.qwen_client.call_vlm") as mock_call:
            mock_call.return_value = VlmResponse(
                content=json.dumps({"events": []}), finish_reason="stop"
            )
            call_qwen(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                frame_count=10,
                fps=5.0,
                qwen_config=QwenClientConfig(max_attempts=1),
            )
            mock_call.assert_called_once()


# ===================================================================
# Merge predictions tests
# ===================================================================


class TestMergePredictions:
    """Duplicate merging: deterministic, event-type aware."""

    def _make_response(
        self,
        window_id,
        clip_id,
        start,
        end,
        event_type="pickup",
        item_count=1,
        confidence=0.9,
        rel_start=0.5,
        rel_end=1.5,
    ):
        return Layer2WindowResponse(
            window_id=window_id,
            clip_id=clip_id,
            window_start_s=start,
            window_end_s=end,
            events=[
                schemas.Layer2Event(
                    event_type=event_type,
                    relative_start_s=rel_start,
                    relative_end_s=rel_end,
                    item_count=item_count,
                    confidence=confidence,
                )
            ],
        )

    def test_merge_duplicates_same_type(self):
        """Overlapping windows with same event type should merge."""
        # w1 [0,10] reports event at relative [2,4] → absolute [2,4]
        # w2 [5,15] reports event at relative [0,2] → absolute [5,7]
        # These overlap at absolute [5,4] — actually [5,4] is invalid. Use w2 rel [2,4] → abs [7,9].
        # Better: w1 [0,10] rel [2,4] → abs [2,4]; w2 [3,13] rel [0,2] → abs [3,5]
        # abs [2,4] and [3,5] overlap → merge
        responses = [
            self._make_response(
                "w1", "c1", 0.0, 10.0, "pickup", 1, 0.9, rel_start=2.0, rel_end=4.0
            ),
            self._make_response(
                "w2", "c1", 3.0, 13.0, "pickup", 1, 0.8, rel_start=0.0, rel_end=2.0
            ),
        ]
        # abs: [2,4] and [3,5] → overlap → merge into one
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].event_type == "pickup"
        assert len(merged[0].contributing_window_ids) == 2

    def test_pickup_putdown_remain_separate(self):
        """Never merge pickup with putdown."""
        responses = [
            self._make_response("w1", "c1", 0.0, 10.0, "pickup", 1, 0.9),
            self._make_response("w2", "c1", 0.0, 10.0, "putdown", 1, 0.8),
        ]
        merged = merge_fn(responses)
        types = {p.event_type for p in merged}
        assert "pickup" in types
        assert "putdown" in types
        assert len(merged) == 2

    def test_repeated_events_preserved(self):
        """Repeated pickup/putdown actions should not be over-merged."""
        responses = [
            self._make_response("w1", "c1", 0.0, 10.0, "pickup", 1, 0.9),
            self._make_response("w2", "c1", 15.0, 25.0, "pickup", 1, 0.8),
        ]
        merged = merge_fn(responses)
        # Two separate pickups with no overlap
        assert len(merged) == 2

    def test_two_item_preservation(self):
        """item_count=2 should be preserved."""
        responses = [
            self._make_response("w1", "c1", 0.0, 10.0, "pickup", item_count=2, confidence=0.9),
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].item_count == 2

    def test_two_item_has_group_id(self):
        """Two-item events get event_group_id set."""
        responses = [
            self._make_response("w1", "c1", 0.0, 10.0, "pickup", item_count=2, confidence=0.9),
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].item_count == 2
        assert merged[0].event_group_id == merged[0].pred_id

    def test_single_item_no_group_id(self):
        """Single-item events have empty event_group_id."""
        responses = [
            self._make_response("w1", "c1", 0.0, 10.0, "pickup", item_count=1, confidence=0.9),
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].event_group_id == ""

    def test_contributing_window_ids_retained(self):
        # All three windows report the same absolute event [2,4]
        # w_a [0,10] rel [2,4] → abs [2,4]
        # w_b [0,10] rel [2,4] → abs [2,4] (same window range, different ID)
        # w_c [0,10] rel [2,4] → abs [2,4]
        responses = [
            self._make_response(
                "w_a", "c1", 0.0, 10.0, "pickup", 1, 0.9, rel_start=2.0, rel_end=4.0
            ),
            self._make_response(
                "w_b", "c1", 0.0, 10.0, "pickup", 1, 0.8, rel_start=2.0, rel_end=4.0
            ),
            self._make_response(
                "w_c", "c1", 0.0, 10.0, "pickup", 1, 0.7, rel_start=2.0, rel_end=4.0
            ),
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert len(merged[0].contributing_window_ids) == 3

    def test_no_events_returns_empty(self):
        responses = [
            Layer2WindowResponse(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
            )
        ]
        merged = merge_fn(responses)
        assert merged == []

    def test_max_confidence_preserved(self):
        # w1 [0,10] rel [2,4] → abs [2,4]; w2 [3,13] rel [0,2] → abs [3,5] — overlap
        responses = [
            self._make_response(
                "w1", "c1", 0.0, 10.0, "pickup", 1, 0.6, rel_start=2.0, rel_end=4.0
            ),
            self._make_response(
                "w2", "c1", 3.0, 13.0, "pickup", 1, 0.95, rel_start=0.0, rel_end=2.0
            ),
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].confidence == 0.95


# ===================================================================
# Evaluation tests
# ===================================================================


class TestEvaluation:
    """Task 8 evaluator compatibility and timestamp conversion."""

    def test_predictions_to_canonical(self):
        preds = [
            Layer2Prediction(
                clip_id="c1",
                pred_id="l2_abc",
                event_type="pickup",
                t_start_s=1.0,
                t_end_s=3.0,
                confidence=0.85,
            )
        ]
        canonical = evaluation.predictions_to_canonical(preds)
        assert len(canonical) == 1
        assert canonical[0].clip_id == "c1"
        assert canonical[0].type == "pickup"
        assert canonical[0].t_start == 1.0
        assert canonical[0].t_end == 3.0
        assert canonical[0].score == 0.85

    def test_evaluate_with_matches(self):
        preds = [
            Layer2Prediction(
                clip_id="c1",
                pred_id="l2_1",
                event_type="pickup",
                t_start_s=1.0,
                t_end_s=3.0,
                confidence=0.9,
            )
        ]
        gt = [EvaluationEvent(clip_id="c1", type="pickup", t_start=1.0, t_end=3.0)]
        metrics = evaluation.evaluate_layer2(preds, gt, criterion=Criterion("tiou", 0.5))
        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_predictions_to_canonical_two_items(self):
        """Two-item events expand to separate canonical rows."""
        preds = [
            Layer2Prediction(
                clip_id="c1",
                pred_id="l2_abc",
                event_type="pickup",
                t_start_s=1.0,
                t_end_s=3.0,
                item_count=2,
                confidence=0.9,
                event_group_id="l2_abc",
            )
        ]
        canonical = evaluation.predictions_to_canonical(preds)
        assert len(canonical) == 2
        assert canonical[0].pred_id == "l2_abc_item0"
        assert canonical[1].pred_id == "l2_abc_item1"

    def test_evaluate_no_predictions(self):
        gt = [EvaluationEvent(clip_id="c1", type="pickup", t_start=1.0, t_end=3.0)]
        metrics = evaluation.evaluate_layer2([], gt)
        assert metrics["tp"] == 0
        assert metrics["note"] == "no predictions"

    def test_evaluate_no_ground_truth(self):
        preds = [
            Layer2Prediction(
                clip_id="c1",
                pred_id="l2_1",
                event_type="pickup",
                t_start_s=1.0,
                t_end_s=3.0,
                confidence=0.9,
            )
        ]
        metrics = evaluation.evaluate_layer2(preds, [])
        assert metrics["fp"] == 1
        assert metrics["note"] == "no ground truth"

    def test_evaluate_from_responses(self):
        responses = [
            Layer2WindowResponse(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                events=[
                    schemas.Layer2Event(
                        event_type="pickup",
                        relative_start_s=1.0,
                        relative_end_s=3.0,
                        confidence=0.9,
                    )
                ],
            )
        ]
        gt = [EvaluationEvent(clip_id="c1", type="pickup", t_start=1.0, t_end=3.0)]
        metrics = evaluation.evaluate_from_responses(responses, gt)
        assert metrics["tp"] == 1

    def test_timestamp_conversion_absolute(self):
        """Relative timestamps correctly converted to absolute."""
        responses = [
            Layer2WindowResponse(
                window_id="w1",
                clip_id="c1",
                window_start_s=5.0,
                window_end_s=15.0,
                events=[
                    schemas.Layer2Event(
                        event_type="putdown",
                        relative_start_s=2.0,
                        relative_end_s=4.0,
                        confidence=0.8,
                    )
                ],
            )
        ]
        merged = merge_fn(responses)
        assert len(merged) == 1
        assert merged[0].t_start_s == 7.0  # 5.0 + 2.0
        assert merged[0].t_end_s == 9.0  # 5.0 + 4.0

    def test_provenance_preserved(self):
        responses = [
            Layer2WindowResponse(
                window_id="w1",
                clip_id="c1",
                window_start_s=0.0,
                window_end_s=10.0,
                events=[
                    schemas.Layer2Event(
                        event_type="pickup",
                        relative_start_s=1.0,
                        relative_end_s=2.0,
                        confidence=0.9,
                    )
                ],
            )
        ]
        merged = merge_fn(responses)
        assert merged[0].contributing_window_ids == ["w1"]


# ===================================================================
# Prompt tests
# ===================================================================


class TestPrompts:
    """Prompt coverage and versioning."""

    def test_prompt_version_exists(self):
        assert prompts.PROMPT_VERSION == "2026-06-27"

    def test_system_prompt_covers_all_cases(self):
        sys = prompts.SYSTEM_PROMPT
        for keyword in [
            "pickup",
            "putdown",
            "restocking",
            "occlusion",
            "multiple people",
            "immediate return",
            "multiple events",
            "two-item",
            "ambiguous",
        ]:
            assert keyword.lower() in sys.lower(), f"Missing: {keyword}"

    def test_no_layer1_info_in_prompt(self):
        sys = prompts.SYSTEM_PROMPT
        for term in ["layer1", "track_a", "track_b", "candidate", "pose"]:
            assert term.lower() not in sys.lower(), f"Leaked Layer 1 info: {term}"

    def test_build_prompt_returns_pair(self):
        system, user = prompts.build_prompt(0.0, 10.0, 10, 5.0)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0

    def test_user_prompt_includes_window_info(self):
        _, user = prompts.build_prompt(5.0, 15.0, 10, 5.0)
        assert "10.0s" in user
        assert "5.0s" in user


# ===================================================================
# No Layer 1 dependency tests
# ===================================================================


class TestNoLayer1Dependency:
    """Verify Layer 2 modules don't import Layer 1."""

    def test_window_generator_no_layer1_import(self):
        import importlib

        mod = importlib.import_module("pickup_putdown.layer2.window_generator")
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        assert "layer1" not in content
        assert "track_a" not in content
        assert "track_b" not in content

    def test_schemas_no_layer1_import(self):
        import importlib

        mod = importlib.import_module("pickup_putdown.layer2.schemas")
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        assert "layer1" not in content
        assert "track_a" not in content

    def test_qwen_client_no_layer1_import(self):
        import importlib

        mod = importlib.import_module("pickup_putdown.layer2.qwen_client")
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        assert "layer1" not in content
        assert "track_a" not in content

    def test_merge_no_layer1_import(self):
        import importlib

        mod = importlib.import_module("pickup_putdown.layer2.merge_predictions")
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        assert "layer1" not in content
        assert "track_a" not in content

    def test_evaluation_no_layer1_import(self):
        import importlib

        mod = importlib.import_module("pickup_putdown.layer2.evaluation")
        src = mod.__file__
        with open(src) as f:
            content = f.read()
        assert "layer1" not in content
        assert "track_a" not in content


# ===================================================================
# Renderer tests (structure, not actual video files)
# ===================================================================


class TestRendererStructure:
    """Renderer module structure tests without requiring video files."""

    def test_frame_info_fields(self):
        from pickup_putdown.layer2 import renderer

        assert hasattr(renderer, "FrameInfo")
        assert hasattr(renderer, "WindowRender")
        assert hasattr(renderer, "render_window")
        assert hasattr(renderer, "render_all_windows")
        assert hasattr(renderer, "frames_to_base64")

    def test_frames_to_base64(self):
        import base64

        from pickup_putdown.layer2.renderer import frames_to_base64

        raw = b"\xff\xd8\xff\xe0fake jpeg data"
        encoded = frames_to_base64([raw])
        assert len(encoded) == 1
        assert base64.b64decode(encoded[0]) == raw

    def test_render_window_with_video(self):
        """Render frames from a real video file and verify overlay text."""
        import cv2
        import numpy as np

        from pickup_putdown.layer2.renderer import render_window

        # Create a 1-second test video at 10fps
        tmp_path = "/tmp/test_overlay_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (100, 100))
        for _ in range(10):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        from pickup_putdown.layer2.window_generator import Window

        w = Window(
            window_id="w1",
            clip_id="c1",
            active_span_id="s1",
            window_start_s=0.0,
            window_end_s=1.0,
            duration_s=1.0,
            overlap_s=0.5,
            source_timestamp_s=0.0,
        )
        result = render_window(w, tmp_path, 10.0, 1.0, n_frames=3)
        assert result is not None
        assert len(result.frame_infos) == 3

        # Decode the rendered frame and check overlay is present
        frame = cv2.imdecode(np.frombuffer(result.frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None

        # Sample text region where overlay should be (top-left corner)
        region = frame[0:25, 0:150]
        # Green text on dark background — sample center of region for green pixels
        green_count = np.sum(
            (region[:, :, 1] > 100) & (region[:, :, 0] < 50) & (region[:, :, 2] < 50)
        )
        assert green_count > 0, "No green overlay text found in rendered frame"

        import os

        os.remove(tmp_path)
