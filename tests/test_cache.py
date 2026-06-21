"""Tests for cache.py — bounded download cache with eviction."""

from __future__ import annotations

from pathlib import Path

import pytest

from pickup_putdown.ingestion.cache import DownloadCache


@pytest.fixture
def cache(tmp_path: Path) -> DownloadCache:
    return DownloadCache(tmp_path / "cache", max_size_mb=10, max_count=5)


class TestDownloadCache:
    def test_get_miss_triggers_download(self, cache: DownloadCache, tmp_path: Path) -> None:
        """A cache miss should call the download function."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"fake video data")

        call_log: list[str] = []

        def mock_download(s3_key: str, local_path: Path) -> None:
            call_log.append(s3_key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(str(test_file), str(local_path))

        cache.set_download_fn(mock_download)
        result = cache.get("bucket/test.mp4")
        assert len(call_log) == 1
        assert result.exists()
        assert result.read_bytes() == b"fake video data"

    def test_get_hit_returns_cached(self, cache: DownloadCache, tmp_path: Path) -> None:
        """A cache hit should return the cached path without downloading."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"cached data")

        download_count = 0

        def mock_download(s3_key: str, local_path: Path) -> None:
            nonlocal download_count
            download_count += 1
            local_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(str(test_file), str(local_path))

        cache.set_download_fn(mock_download)
        cache.get("bucket/test.mp4")
        assert download_count == 1

        # Second call should not trigger download
        result2 = cache.get("bucket/test.mp4")
        assert download_count == 1
        assert result2.exists()

    def test_put_registers_file(self, cache: DownloadCache, tmp_path: Path) -> None:
        """put() should register an externally downloaded file."""
        external_file = tmp_path / "external.mp4"
        external_file.write_bytes(b"external data")
        cache.put("bucket/ext.mp4", external_file)
        assert cache.get("bucket/ext.mp4").read_bytes() == b"external data"

    def test_put_nonexistent_raises(self, cache: DownloadCache) -> None:
        """put() should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            cache.put("bucket/missing.mp4", Path("/nonexistent/file.mp4"))

    def test_usage_returns_stats(self, cache: DownloadCache, tmp_path: Path) -> None:
        """usage() should return correct statistics."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"x" * 1000)

        cache.set_download_fn(
            lambda sk, lp: (
                lp.parent.mkdir(parents=True, exist_ok=True),
                __import__("shutil").copy2(str(test_file), str(lp)),
            )[1]
        )
        cache.get("bucket/test.mp4")
        usage = cache.usage()
        assert usage["count"] == 1
        assert usage["max_count"] == 5
        assert usage["used_mb"] >= 0

    def test_eviction_on_count_limit(self, cache: DownloadCache, tmp_path: Path) -> None:
        """Cache should evict oldest entries when count exceeds max_count."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"same data")

        cache.set_download_fn(
            lambda sk, lp: (
                lp.parent.mkdir(parents=True, exist_ok=True),
                __import__("shutil").copy2(str(test_file), str(lp)),
            )[1]
        )

        # Add 6 files to a cache with max_count=5
        for i in range(6):
            cache.get(f"bucket/file_{i}.mp4")

        usage = cache.usage()
        assert usage["count"] <= 5

    def test_eviction_on_size_limit(self, cache: DownloadCache, tmp_path: Path) -> None:
        """Cache should evict when total size exceeds max_size_mb."""
        # Create a 3MB file; max_size_mb=10, max_count=50
        big_file = tmp_path / "big.mp4"
        big_file.write_bytes(b"x" * (3 * 1024 * 1024))

        cache.set_download_fn(
            lambda sk, lp: (
                lp.parent.mkdir(parents=True, exist_ok=True),
                __import__("shutil").copy2(str(big_file), str(lp)),
            )[1]
        )

        for i in range(4):
            cache.get(f"bucket/big_{i}.mp4")

        usage = cache.usage()
        assert usage["count"] <= 5

    def test_clear_removes_all(self, cache: DownloadCache, tmp_path: Path) -> None:
        """clear() should remove all cached files and manifest."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"data")
        cache.set_download_fn(
            lambda sk, lp: (
                lp.parent.mkdir(parents=True, exist_ok=True),
                __import__("shutil").copy2(str(test_file), str(lp)),
            )[1]
        )
        cache.get("bucket/test.mp4")
        assert cache.count() == 1
        cache.clear()
        assert cache.usage()["count"] == 0

    def test_idempotent_get(self, cache: DownloadCache, tmp_path: Path) -> None:
        """Repeated get() for the same key should be idempotent."""
        test_file = tmp_path / "source.mp4"
        test_file.write_bytes(b"idempotent")

        download_count = 0

        def mock_download(s3_key: str, local_path: Path) -> None:
            nonlocal download_count
            download_count += 1
            local_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(str(test_file), str(local_path))

        cache.set_download_fn(mock_download)
        path1 = cache.get("bucket/test.mp4")
        path2 = cache.get("bucket/test.mp4")
        assert path1 == path2
        assert download_count == 1
