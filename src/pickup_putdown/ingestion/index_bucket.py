"""List objects in an S3/S3-compatible bucket without downloading them."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_BUCKET_RE = re.compile(r"^s3://([^/]+)/(.*)$")


@dataclass
class BucketObject:
    """Metadata for a single object in an S3 bucket."""

    key: str
    size: int
    etag: str | None = None
    last_modified: datetime | None = None


def _parse_bucket_uri(bucket_uri: str) -> tuple[str, str]:
    """Parse an s3:// URI into (bucket, prefix).

    Raises ValueError if the URI format is invalid.
    """
    match = _BUCKET_RE.match(bucket_uri)
    if not match:
        raise ValueError(f"Invalid bucket URI '{bucket_uri}'. Expected format: s3://bucket/prefix")
    return match.group(1), match.group(2)


def _build_client_kwargs(
    endpoint_url: str | None,
    region: str | None,
    anonymous: bool,
) -> dict[str, Any]:
    """Build boto3 client keyword arguments from config."""
    kwargs: dict[str, Any] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region:
        kwargs["region_name"] = region
    if anonymous:
        kwargs["aws_access_key_id"] = ""
        kwargs["aws_secret_access_key"] = ""
        kwargs["aws_session_token"] = ""
    return kwargs


def list_objects(
    bucket_uri: str,
    *,
    endpoint_url: str | None = None,
    region: str | None = None,
    anonymous: bool = False,
    prefix: str | None = None,
) -> list[BucketObject]:
    """List objects in an S3 bucket without downloading any content.

    Only lists object metadata (key, size, etag, last_modified).
    Does not download video files.

    Parameters
    ----------
    bucket_uri : str
        S3 URI in the form s3://bucket/prefix.
    endpoint_url : str, optional
        S3-compatible endpoint URL.
    region : str, optional
        AWS region name.
    anonymous : bool
        If True, access without credentials.
    prefix : str, optional
        Additional prefix filter beyond what is in bucket_uri.

    Returns
    -------
    list[BucketObject]
    """
    bucket, base_prefix = _parse_bucket_uri(bucket_uri)
    full_prefix = f"{base_prefix}/{prefix}" if prefix else base_prefix

    client_kwargs = _build_client_kwargs(endpoint_url, region, anonymous)

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 listing. Install it with: pip install boto3"
        ) from exc

    client = boto3.client("s3", **client_kwargs)  # noqa: S301

    objects: list[BucketObject] = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=bucket,
            Prefix=full_prefix,
            PaginationConfig={"PageSize": 1000},
        )

        for page in page_iterator:
            contents = page.get("Contents", [])
            for obj in contents:
                # Skip directory markers (keys ending with /)
                if obj["Key"].endswith("/"):
                    continue
                etag = obj.get("ETag", "").strip('"')
                objects.append(
                    BucketObject(
                        key=obj["Key"],
                        size=obj["Size"],
                        etag=etag if etag else None,
                        last_modified=obj.get("LastModified"),
                    )
                )

    except Exception as exc:
        logger.error("Failed to list objects in bucket '%s': %s", bucket, exc)
        raise

    logger.info(
        "Listed %d objects in s3://%s/%s",
        len(objects),
        bucket,
        full_prefix,
    )
    return objects
