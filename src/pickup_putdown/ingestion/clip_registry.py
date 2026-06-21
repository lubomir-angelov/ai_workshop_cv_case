"""Clip registry: collect indexed clips and export to parquet/CSV."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from pickup_putdown.common.schemas import Clip

logger = logging.getLogger(__name__)

# Canonical column order for clips.csv (case-required columns first)
_CANONICAL_COLUMNS = [
    "clip_id",
    "s3_key",
    "duration_s",
    "fps",
    "width",
    "height",
    "n_person_tracks",
    "usable",
    "active_start_s",
    "active_end_s",
    "split",
    "session_id",
    "notes",
    "etag",
    "object_size_bytes",
    "video_codec",
    "audio_codec",
    "decode_ok",
    "probe_error",
    "duplicate_of",
    "probe_fps",
]


def generate_clip_id(s3_key: str, etag: str | None, object_size: int) -> str:
    """Generate a stable clip_id from immutable source attributes.

    Uses SHA-256 of (s3_key + etag + size) so the same source
    always produces the same clip_id regardless of local cache paths.

    Parameters
    ----------
    s3_key : str
        The S3 object key.
    etag : str or None
        The S3 ETag (checksum) of the object.
    object_size : int
        The object size in bytes.

    Returns
    -------
    str
        A 16-character hex clip identifier.
    """
    raw = f"{s3_key}|{etag or ''}|{object_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ClipRegistry:
    """In-memory registry of indexed clips with parquet/CSV export.

    Parameters
    ----------
    storage : dict, optional
        Initial storage dict mapping clip_id -> Clip. Used for
        deserialization.
    """

    def __init__(self, storage: dict[str, Clip] | None = None) -> None:
        self._clips: dict[str, Clip] = storage if storage is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_clip(self, clip: Clip) -> None:
        """Add or replace a single clip in the registry.

        Parameters
        ----------
        clip : Clip
            The clip to register.
        """
        self._clips[clip.clip_id] = clip

    def add_batch(self, clips: list[Clip]) -> None:
        """Add multiple clips to the registry.

        Parameters
        ----------
        clips : list[Clip]
            Clips to register.
        """
        for clip in clips:
            self.add_clip(clip)

    def get_clip(self, clip_id: str) -> Clip | None:
        """Retrieve a clip by ID, or None if not found."""
        return self._clips.get(clip_id)

    def all_clips(self) -> list[Clip]:
        """Return all registered clips in insertion order."""
        return list(self._clips.values())

    def count(self) -> int:
        """Return the number of registered clips."""
        return len(self._clips)

    def export_parquet(self, path: str | Path) -> None:
        """Export all clips to a Parquet file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        if not self._clips:
            logger.warning("No clips to export to %s", path)
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        records = [clip.model_dump() for clip in self._clips.values()]
        table = pa.Table.from_pylist(records)
        pq.write_table(table, str(path))  # type: ignore[no-untyped-call]
        logger.info("Exported %d clips to %s", len(records), path)

    def _empty_parquet_schema(self) -> Any:
        """Return an empty PyArrow schema matching the Clip model."""
        import pyarrow as pa

        clip = Clip(
            clip_id="",
            s3_key="",
            duration_s=0.0,
            fps=0.0,
            width=0,
            height=0,
        )
        return pa.Table.from_pylist([clip.model_dump()]).schema

    def export_csv(self, path: str | Path) -> None:
        """Export all clips to a canonical CSV file.

        Columns are ordered to match the case-required schema.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        path = Path(path)
        if not self._clips:
            logger.warning("No clips to export to %s", path)
            return

        import csv

        records = [clip.model_dump() for clip in self._clips.values()]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CANONICAL_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        logger.info("Exported %d clips to %s", len(records), path)

    def summary(self, cache_usage: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a machine-readable summary of the registry state.

        Parameters
        ----------
        cache_usage : dict, optional
            Cache usage dict from DownloadCache.usage().

        Returns
        -------
        dict
            Keys: indexed_count, failures, duplicate_candidates,
                  total_source_bytes, cache_usage
        """
        failures = sum(1 for c in self._clips.values() if not c.decode_ok)
        duplicates = sum(1 for c in self._clips.values() if c.duplicate_of is not None)
        total_bytes = sum(c.object_size_bytes for c in self._clips.values())

        result: dict[str, Any] = {
            "indexed_count": len(self._clips),
            "failures": failures,
            "duplicate_candidates": duplicates,
            "total_source_bytes": total_bytes,
        }
        if cache_usage is not None:
            result["cache_usage"] = cache_usage
        return result
