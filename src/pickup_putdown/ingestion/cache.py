"""Bounded on-demand download cache with eviction and concurrency safety."""

from __future__ import annotations

import fcntl
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DownloadCache:
    """Manages a bounded local cache of downloaded video files.

    Provides idempotent retrieval (repeated get returns same path) and
    concurrency safety via file locks to prevent duplicate downloads.
    Evicts oldest entries (FIFO) when size or count limits are exceeded.

    Parameters
    ----------
    base_dir : str or Path
        Root directory for cached files.
    max_size_mb : int
        Maximum total cache size in megabytes.
    max_count : int
        Maximum number of files in the cache.
    """

    def __init__(
        self,
        base_dir: str | Path,
        max_size_mb: int = 5120,
        max_count: int = 50,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_mb = max_size_mb
        self._max_count = max_count
        self._lock_dir = self._base_dir / ".locks"
        self._lock_dir.mkdir(exist_ok=True)
        self._manifest_path = self._base_dir / ".manifest.json"
        self._manifest: dict[str, dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, s3_key: str) -> Path:
        """Return the cached path for *s3_key*, downloading if missing.

        If the file is already cached, returns its path immediately.
        If another process is downloading it, waits for the lock and
        returns the result.

        Parameters
        ----------
        s3_key : str
            The S3 object key to retrieve.

        Returns
        -------
        Path
            Local path to the cached file.

        Raises
        ------
        RuntimeError
            If the download fails.
        """
        cache_path = self._resolve_key(s3_key)

        if cache_path.exists() and cache_path.is_file():
            self._touch(s3_key)
            return cache_path

        # Acquire a per-key lock to prevent concurrent duplicate downloads
        lock_path = self._lock_dir / f"{self._safe_key(s3_key)}.lock"
        self._acquire_lock(lock_path)

        try:
            # Double-check after acquiring lock
            if cache_path.exists() and cache_path.is_file():
                self._touch(s3_key)
                return cache_path

            logger.info("Cache miss for %s — downloading to %s", s3_key, cache_path)
            self._download(s3_key, cache_path)
            self._manifest[s3_key] = {
                "path": str(cache_path),
                "size_bytes": cache_path.stat().st_size,
                "added_at": _now_iso(),
                "accessed_at": _now_iso(),
            }
            self._save_manifest()
            self._evict()
        finally:
            self._release_lock(lock_path)

        return cache_path

    def put(self, s3_key: str, local_path: str | Path) -> None:
        """Register a locally downloaded file in the cache.

        Useful when files were downloaded by other means (e.g. manual
        download or external tool).

        Parameters
        ----------
        s3_key : str
            The S3 object key.
        local_path : str or Path
            Path to the already-downloaded file.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found for cache registration: {local_path}")

        cache_path = self._resolve_key(s3_key)
        shutil.copy2(str(local_path), str(cache_path))
        self._manifest[s3_key] = {
            "path": str(cache_path),
            "size_bytes": cache_path.stat().st_size,
            "added_at": _now_iso(),
            "accessed_at": _now_iso(),
        }
        self._save_manifest()

    def count(self) -> int:
        """Return the number of entries in the cache manifest."""
        return len(self._manifest)

    def usage(self) -> dict[str, Any]:
        """Return current cache usage statistics.

        Returns
        -------
        dict
            Keys: used_mb, count, max_mb, max_count
        """
        total_bytes = sum(
            entry.get("size_bytes", 0)
            for entry in self._manifest.values()
            if Path(entry.get("path", "")).exists()
        )
        return {
            "used_mb": round(total_bytes / (1024 * 1024), 2),
            "count": len(self._manifest),
            "max_mb": self._max_size_mb,
            "max_count": self._max_count,
        }

    def clear(self) -> None:
        """Remove all cached files and the manifest."""
        for entry in self._manifest.values():
            path = Path(entry.get("path", ""))
            if path.exists():
                path.unlink()
        self._manifest.clear()
        self._save_manifest()
        # Clean up lock files
        if self._lock_dir.exists():
            for lock_file in self._lock_dir.iterdir():
                if lock_file.suffix == ".lock":
                    lock_file.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, s3_key: str) -> Path:
        """Map an S3 key to a deterministic local cache path."""
        # Replace / with _ for directory structure, keep extension
        safe = self._safe_key(s3_key)
        return self._base_dir / safe

    @staticmethod
    def _safe_key(s3_key: str) -> str:
        """Create a filesystem-safe key from an S3 key."""
        return s3_key.replace("/", "_").replace(":", "_").replace("\\", "_")

    def _download(self, s3_key: str, cache_path: Path) -> None:
        """Download an object from S3 to *cache_path*.

        Subclass or monkey-patch this method to inject a real S3 client.
        Default implementation raises — callers must set a download
        callback via :meth:`set_download_fn`.
        """
        if self._download_fn is None:
            raise RuntimeError(
                "No download function configured. "
                "Set cache.set_download_fn() before calling get()."
            )
        self._download_fn(s3_key, cache_path)

    _download_fn: Any = None  # noqa: UP032

    def set_download_fn(self, fn: Any) -> None:
        """Set the function used to download objects from S3.

        The function signature should be: fn(s3_key: str, local_path: Path) -> None

        Parameters
        ----------
        fn : callable
            Download callback.
        """
        self._download_fn = fn

    def _evict(self) -> None:
        """Evict oldest entries if cache exceeds size or count limits."""
        while True:
            if not self._manifest:
                break

            used_mb = sum(entry.get("size_bytes", 0) for entry in self._manifest.values()) / (
                1024 * 1024
            )

            if used_mb <= self._max_size_mb and len(self._manifest) <= self._max_count:
                break

            # Evict the oldest entry (first inserted, FIFO)
            oldest_key = next(iter(self._manifest))
            entry = self._manifest.pop(oldest_key)
            path = Path(entry.get("path", ""))
            if path.exists():
                path.unlink()
                logger.debug("Evicted %s from cache", oldest_key)

        self._save_manifest()

    def _touch(self, s3_key: str) -> None:
        """Update the accessed_at timestamp for a cached entry."""
        if s3_key in self._manifest:
            self._manifest[s3_key]["accessed_at"] = _now_iso()

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Load the cache manifest from disk."""
        if self._manifest_path.exists():
            try:
                import json

                with open(self._manifest_path) as fh:
                    return json.load(fh)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt manifest at %s, starting fresh", self._manifest_path)
        return {}

    def _save_manifest(self) -> None:
        """Persist the cache manifest to disk."""
        import json

        with open(self._manifest_path, "w") as fh:
            json.dump(self._manifest, fh, indent=2)

    def _acquire_lock(self, lock_path: Path) -> None:
        """Acquire an advisory file lock."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(lock_path, "w")  # noqa: SIM115, UP032
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)

    def _release_lock(self, lock_path: Path) -> None:
        """Release an advisory file lock."""
        if hasattr(self, "_lock_file"):
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()


def _now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()  # noqa: UP017
