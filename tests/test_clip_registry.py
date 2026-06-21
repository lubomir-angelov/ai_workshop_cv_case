"""Tests for clip_registry.py — parquet/CSV export."""

from __future__ import annotations

from pathlib import Path

import pytest

from pickup_putdown.common.schemas import Clip
from pickup_putdown.ingestion.clip_registry import ClipRegistry, generate_clip_id


@pytest.fixture
def sample_clip() -> Clip:
    return Clip(
        clip_id="clip_001",
        s3_key="s3://bucket/clip_001.mp4",
        duration_s=30.0,
        fps=30.0,
        width=1920,
        height=1080,
        n_person_tracks=2,
        usable=True,
        active_start_s=5.0,
        active_end_s=25.0,
        split="train",
        session_id="session_01",
        notes=None,
        etag="abc123",
        object_size_bytes=1048576,
        video_codec="h264",
        audio_codec="aac",
        decode_ok=True,
        probe_error=None,
        duplicate_of=None,
        probe_fps=30.0,
    )


class TestClipRegistry:
    def test_add_clip(self, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        registry.add_clip(sample_clip)
        assert registry.count() == 1
        assert registry.get_clip("clip_001") is sample_clip

    def test_add_batch(self, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        clip2 = Clip(
            clip_id="clip_002",
            s3_key="s3://bucket/clip_002.mp4",
            duration_s=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        registry.add_batch([sample_clip, clip2])
        assert registry.count() == 2

    def test_get_clip_not_found(self) -> None:
        registry = ClipRegistry()
        assert registry.get_clip("nonexistent") is None

    def test_all_clips_order(self, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        clip2 = Clip(
            clip_id="clip_002",
            s3_key="s3://bucket/clip_002.mp4",
            duration_s=60.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        registry.add_clip(sample_clip)
        registry.add_clip(clip2)
        clips = registry.all_clips()
        assert len(clips) == 2
        assert clips[0].clip_id == "clip_001"
        assert clips[1].clip_id == "clip_002"

    def test_export_csv(self, tmp_path: Path, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        registry.add_clip(sample_clip)
        csv_path = tmp_path / "clips.csv"
        registry.export_csv(csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "clip_id" in content
        assert "clip_001" in content
        assert "s3://bucket/clip_001.mp4" in content

    def test_export_csv_empty(self, tmp_path: Path) -> None:
        """Exporting an empty registry logs a warning but does not crash."""
        registry = ClipRegistry()
        csv_path = tmp_path / "clips.csv"
        registry.export_csv(csv_path)
        # Empty export does not create a file (logged warning instead)
        assert not csv_path.exists()

    def test_export_parquet(self, tmp_path: Path, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        registry.add_clip(sample_clip)
        parquet_path = tmp_path / "clips.parquet"
        registry.export_parquet(parquet_path)
        assert parquet_path.exists()

    def test_export_parquet_empty(self, tmp_path: Path) -> None:
        """Exporting an empty registry logs a warning but does not crash."""
        registry = ClipRegistry()
        parquet_path = tmp_path / "clips.parquet"
        registry.export_parquet(parquet_path)
        # Empty export does not create a file (logged warning instead)
        assert not parquet_path.exists()

    def test_summary(self, sample_clip: Clip) -> None:
        registry = ClipRegistry()
        registry.add_clip(sample_clip)
        summary = registry.summary(cache_usage={"used_mb": 1.5, "count": 1})
        assert summary["indexed_count"] == 1
        assert summary["failures"] == 0
        assert summary["duplicate_candidates"] == 0
        assert summary["total_source_bytes"] == 1048576
        assert summary["cache_usage"]["used_mb"] == 1.5

    def test_summary_with_failures(self) -> None:
        clip = Clip(
            clip_id="clip_fail",
            s3_key="s3://bucket/bad.mp4",
            duration_s=0.0,
            fps=0.0,
            width=0,
            height=0,
            decode_ok=False,
            probe_error="no video stream",
        )
        registry = ClipRegistry()
        registry.add_clip(clip)
        summary = registry.summary()
        assert summary["failures"] == 1

    def test_summary_with_duplicates(self) -> None:
        dup_clip = Clip(
            clip_id="clip_dup",
            s3_key="s3://bucket/dup.mp4",
            duration_s=30.0,
            fps=30.0,
            width=1920,
            height=1080,
            duplicate_of="clip_001",
        )
        registry = ClipRegistry()
        registry.add_clip(dup_clip)
        summary = registry.summary()
        assert summary["duplicate_candidates"] == 1


class TestGenerateClipId:
    def test_deterministic(self) -> None:
        id1 = generate_clip_id("s3://bucket/clip.mp4", "etag1", 1000)
        id2 = generate_clip_id("s3://bucket/clip.mp4", "etag1", 1000)
        assert id1 == id2
        assert len(id1) == 16

    def test_different_key_different_id(self) -> None:
        id1 = generate_clip_id("s3://bucket/a.mp4", "etag", 1000)
        id2 = generate_clip_id("s3://bucket/b.mp4", "etag", 1000)
        assert id1 != id2

    def test_different_size_different_id(self) -> None:
        id1 = generate_clip_id("s3://bucket/clip.mp4", "etag", 1000)
        id2 = generate_clip_id("s3://bucket/clip.mp4", "etag", 2000)
        assert id1 != id2

    def test_none_etag_handled(self) -> None:
        id1 = generate_clip_id("s3://bucket/clip.mp4", None, 1000)
        id2 = generate_clip_id("s3://bucket/clip.mp4", None, 1000)
        assert id1 == id2
