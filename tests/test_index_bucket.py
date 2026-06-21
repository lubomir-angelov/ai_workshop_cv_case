"""Tests for index_bucket.py — S3 bucket listing."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from pickup_putdown.ingestion.index_bucket import _parse_bucket_uri, list_objects


class TestParseBucketUri:
    def test_parses_standard_uri(self) -> None:
        bucket, prefix = _parse_bucket_uri("s3://my-bucket/videos")
        assert bucket == "my-bucket"
        assert prefix == "videos"

    def test_parses_uri_with_nested_prefix(self) -> None:
        bucket, prefix = _parse_bucket_uri("s3://bucket/path/to/clips")
        assert bucket == "bucket"
        assert prefix == "path/to/clips"

    def test_invalid_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid bucket URI"):
            _parse_bucket_uri("http://bucket/key")

    def test_empty_path(self) -> None:
        bucket, prefix = _parse_bucket_uri("s3://bucket/")
        assert bucket == "bucket"
        assert prefix == ""


class TestListObjects:
    def test_raises_on_s3_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_objects should propagate S3 errors."""
        import boto3

        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("network error")

        boto3.client = lambda *a, **kw: mock_client  # type: ignore[assignment]

        try:
            with pytest.raises(Exception, match="network error"):
                list_objects("s3://bucket/key")
        finally:
            # Restore is not needed — boto3.client is called fresh each time
            pass

    def test_calls_boto3_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_objects should call boto3 list_objects_v2."""
        now = datetime.now(UTC)
        page_data = {
            "Contents": [
                {
                    "Key": "clip1.mp4",
                    "Size": 1000,
                    "ETag": '"abc"',
                    "LastModified": now,
                },
                {
                    "Key": "clip2.mp4",
                    "Size": 2000,
                    "ETag": '"def"',
                    "LastModified": now,
                },
            ]
        }

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [page_data]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        import boto3

        original_client = boto3.client
        boto3.client = lambda *a, **kw: mock_client  # type: ignore[assignment]

        try:
            objects = list_objects("s3://test-bucket/videos")
            assert len(objects) == 2
            assert objects[0].key == "clip1.mp4"
            assert objects[0].size == 1000
            assert objects[0].etag == "abc"
            assert objects[1].key == "clip2.mp4"
        finally:
            boto3.client = original_client  # type: ignore[assignment]

    def test_skips_directory_markers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keys ending with / should be skipped."""
        now = datetime.now(UTC)
        page_data = {
            "Contents": [
                {"Key": "folder/", "Size": 0, "ETag": '"-"', "LastModified": now},
                {"Key": "video.mp4", "Size": 500, "ETag": '"xyz"', "LastModified": now},
            ]
        }

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [page_data]

        mock_client = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        import boto3

        original_client = boto3.client
        boto3.client = lambda *a, **kw: mock_client  # type: ignore[assignment]

        try:
            objects = list_objects("s3://test-bucket/videos")
            assert len(objects) == 1
            assert objects[0].key == "video.mp4"
        finally:
            boto3.client = original_client  # type: ignore[assignment]
