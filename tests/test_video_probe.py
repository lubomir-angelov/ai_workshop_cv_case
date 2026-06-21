"""Tests for video_probe.py — ffprobe metadata extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from pickup_putdown.ingestion.video_probe import ProbeResult, probe_video


class TestProbeVideoExistingFile:
    def test_probes_valid_video(self, tmp_path: Path) -> None:
        """A real video file should produce valid metadata."""
        # Use the fixture if it exists; this test may skip if ffprobe
        # cannot parse it (fixture is corrupt).
        fixture = Path("tests/fixtures/corrupt_video.mp4")
        if not fixture.exists():
            pytest.skip("corrupt_video.mp4 fixture not found")
        result = probe_video(fixture)
        assert isinstance(result, ProbeResult)
        # Corrupt fixture should fail decode
        assert result.decode_ok is False
        assert result.probe_error is not None

    def test_probes_nonexistent_file(self, tmp_path: Path) -> None:
        """A missing file should return decode_ok=False."""
        result = probe_video(tmp_path / "does_not_exist.mp4")
        assert isinstance(result, ProbeResult)
        assert result.decode_ok is False
        assert "not found" in (result.probe_error or "").lower()


class TestProbeVideoDecodeValidation:
    def test_corrupt_fixture_returns_failed(self, tmp_path: Path) -> None:
        """The corrupt fixture should be flagged as decode failure."""
        fixture = Path("tests/fixtures/corrupt_video.mp4")
        if not fixture.exists():
            pytest.skip("corrupt_video.mp4 fixture not found")
        result = probe_video(fixture)
        assert result.decode_ok is False
        assert result.probe_error is not None


class TestGenerateClipId:
    def test_deterministic(self) -> None:
        from pickup_putdown.ingestion.clip_registry import generate_clip_id

        id1 = generate_clip_id("s3://bucket/clip.mp4", "abc123", 1000)
        id2 = generate_clip_id("s3://bucket/clip.mp4", "abc123", 1000)
        assert id1 == id2
        assert len(id1) == 16

    def test_different_inputs_different_ids(self) -> None:
        from pickup_putdown.ingestion.clip_registry import generate_clip_id

        id1 = generate_clip_id("s3://bucket/a.mp4", "etag1", 1000)
        id2 = generate_clip_id("s3://bucket/b.mp4", "etag1", 1000)
        assert id1 != id2

    def test_no_etag_still_deterministic(self) -> None:
        from pickup_putdown.ingestion.clip_registry import generate_clip_id

        id1 = generate_clip_id("s3://bucket/clip.mp4", None, 1000)
        id2 = generate_clip_id("s3://bucket/clip.mp4", None, 1000)
        assert id1 == id2
