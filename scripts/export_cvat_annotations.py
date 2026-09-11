#!/usr/bin/env python3
"""Export completed CVAT jobs to S3 in native "CVAT for video 1.1" format.

Read-only with respect to CVAT: annotations are exported, nothing in CVAT is
modified — no status, stage, or assignee changes.

Selection rule: a task is exported when its single job has state ``completed``.
Tasks whose job state is ``new`` are excluded. Completed jobs with zero events
ARE exported and recorded in the manifest — they are verified negatives, which
are as informative for training as positives.

Archive naming preserves the source video:
    <source_video_stem>__task_<task_id>__job_<job_id>.zip

Usage:
    python scripts/export_cvat_annotations.py --dry-run
    python scripts/export_cvat_annotations.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parent.parent

CVAT_HOST = "https://app.cvat.ai"
CVAT_ORG = "CVCASE"
EXPORT_FORMAT = "CVAT for video 1.1"

S3_BUCKET = "chillnbite-cameras"
S3_PREFIX_TEMPLATE = "anon/annotations/cvat/{export_date}/raw"

MANIFEST_NAME = "export_manifest.csv"
MANIFEST_COLUMNS = [
    "task_id",
    "job_id",
    "source_video_filename",
    "job_status",
    "event_count",
    "s3_uri",
    "export_timestamp",
]

COMPLETED = "completed"
EXCLUDED = "new"


def cvat_client() -> Any:
    """Authenticated high-level CVAT client scoped to the organization."""
    from cvat_sdk import Client, Config

    token_file = Path.home() / ".cvat_token"
    token = os.environ.get("CVAT_TOKEN") or (
        token_file.read_text().strip() if token_file.exists() else None
    )
    if not token:
        raise SystemExit("No CVAT token: set $CVAT_TOKEN or write ~/.cvat_token")

    client = Client(url=CVAT_HOST, config=Config(verify_ssl=True))
    client.api_client.default_headers["Authorization"] = f"Token {token}"
    client.api_client.default_headers["X-Organization"] = CVAT_ORG
    client.organization_slug = CVAT_ORG
    return client


def count_events(archive: Path) -> tuple[int, str | None]:
    """Count annotated events in a CVAT-for-video archive.

    An event is one temporal interval: a <track> (a shape tracked between
    keyframes) or a standalone <tag>. Returns (count, error).
    """
    try:
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n.endswith("annotations.xml")]
            if not names:
                return 0, f"no annotations.xml in {archive.name}"
            root = ElementTree.fromstring(zf.read(names[0]))
    except (zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return 0, f"unreadable archive {archive.name}: {exc}"

    return len(root.findall(".//track")) + len(root.findall(".//tag")), None


def collect_jobs(client: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split tasks into (to_export, excluded) by their job state."""
    api = client.api_client
    tasks, _ = api.tasks_api.list(page_size=500)
    jobs, _ = api.jobs_api.list(page_size=1000)

    jobs_by_task: dict[int, list[Any]] = {}
    for job in jobs.results:
        jobs_by_task.setdefault(job.task_id, []).append(job)

    to_export: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for task in sorted(tasks.results, key=lambda t: t.name):
        task_jobs = jobs_by_task.get(task.id, [])
        if len(task_jobs) != 1:
            # The naming scheme assumes one job per task; surface any exception
            # rather than silently exporting only the first job.
            excluded.append(
                {
                    "task_id": task.id,
                    "name": task.name,
                    "job_id": None,
                    "state": f"unexpected job count: {len(task_jobs)}",
                }
            )
            continue

        job = task_jobs[0]
        record = {
            "task_id": task.id,
            "name": task.name,
            "job_id": job.id,
            "state": str(job.state),
        }
        (to_export if record["state"] == COMPLETED else excluded).append(record)

    return to_export, excluded


def export_task(client: Any, record: dict[str, Any], out_dir: Path) -> tuple[Path | None, str | None]:
    """Export one task's annotations. Returns (archive_path, error)."""
    archive = out_dir / f"{record['name']}__task_{record['task_id']}__job_{record['job_id']}.zip"
    if archive.exists():
        return archive, None

    try:
        task = client.tasks.retrieve(record["task_id"])
        task.export_dataset(
            format_name=EXPORT_FORMAT,
            filename=archive,
            include_images=False,
        )
    except Exception as exc:  # noqa: BLE001 — report per-task, keep going
        return None, f"{type(exc).__name__}: {exc}"

    if not archive.exists():
        return None, "export reported success but no archive was written"
    return archive, None


def s3_client() -> Any:
    import boto3

    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
    return session.client("s3")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-date",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="Export snapshot date; names the local out-dir and the S3 prefix (default: today, UTC)",
    )
    parser.add_argument("--out-dir", default=None, help="Default: .local/cvat_exports/<export-date>")
    parser.add_argument("--dry-run", action="store_true", help="Select and report, do not export or upload")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    s3_prefix = S3_PREFIX_TEMPLATE.format(export_date=args.export_date)
    out_dir = Path(args.out_dir or REPO_ROOT / ".local" / "cvat_exports" / args.export_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cvat_client()
    to_export, excluded = collect_jobs(client)
    if args.limit:
        to_export = to_export[: args.limit]

    print(f"Completed jobs to export : {len(to_export)}")
    print(f"Excluded (state != completed): {len(excluded)}")
    if args.dry_run:
        for record in to_export:
            print(f"  export  {record['name']}  task={record['task_id']} job={record['job_id']}")
        for record in excluded:
            print(f"  skip    {record['name']}  ({record['state']})")
        return 0

    s3 = s3_client()
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for index, record in enumerate(to_export, start=1):
        label = f"[{index}/{len(to_export)}] {record['name']}"
        archive, error = export_task(client, record, out_dir)
        if error or archive is None:
            print(f"  FAIL   {label}: {error}")
            failures.append((record["name"], f"export: {error}"))
            continue

        event_count, count_error = count_events(archive)
        if count_error:
            print(f"  FAIL   {label}: {count_error}")
            failures.append((record["name"], count_error))
            continue

        s3_key = f"{s3_prefix}/{archive.name}"
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        try:
            s3.upload_file(str(archive), S3_BUCKET, s3_key)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL   {label}: upload: {exc}")
            failures.append((record["name"], f"upload: {exc}"))
            continue

        rows.append(
            {
                "task_id": record["task_id"],
                "job_id": record["job_id"],
                "source_video_filename": f"{record['name']}.mp4",
                "job_status": record["state"],
                "event_count": event_count,
                "s3_uri": s3_uri,
                "export_timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        print(f"  ok     {label}  events={event_count}")

    # Manifest — includes zero-event completed jobs as verified negatives.
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    manifest_key = f"{s3_prefix}/{MANIFEST_NAME}"
    s3.upload_file(str(manifest_path), S3_BUCKET, manifest_key)
    print(f"\nManifest -> s3://{S3_BUCKET}/{manifest_key}  ({len(rows)} rows)")

    # Verification: every uploaded object must exist with a matching size.
    print("\nVerifying uploaded objects...")
    verified, mismatched = 0, []
    for row in rows:
        key = row["s3_uri"].split(f"s3://{S3_BUCKET}/", 1)[1]
        local = out_dir / Path(key).name
        try:
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
        except Exception as exc:  # noqa: BLE001
            mismatched.append((key, f"head_object failed: {exc}"))
            continue
        if head["ContentLength"] != local.stat().st_size:
            mismatched.append((key, f"size {head['ContentLength']} != local {local.stat().st_size}"))
            continue
        verified += 1

    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=manifest_key)
        manifest_ok = head["ContentLength"] == manifest_path.stat().st_size
    except Exception as exc:  # noqa: BLE001
        manifest_ok = False
        mismatched.append((manifest_key, str(exc)))

    zero_event = [r for r in rows if r["event_count"] == 0]
    total_events = sum(r["event_count"] for r in rows)

    print("\n" + "=" * 66)
    print(f"Exported and uploaded : {len(rows)}")
    print(f"Verified in S3        : {verified}/{len(rows)}  (manifest ok: {manifest_ok})")
    print(f"Total events          : {total_events}")
    print(f"Zero-event completed  : {len(zero_event)}  (verified negatives)")
    for row in zero_event:
        print(f"    {row['source_video_filename']}  task={row['task_id']} job={row['job_id']}")
    print(f"Excluded (not completed): {len(excluded)}")
    for record in excluded:
        print(f"    {record['name']}  state={record['state']}")
    print(f"Failures              : {len(failures) + len(mismatched)}")
    for name, reason in failures:
        print(f"    {name}: {reason}")
    for key, reason in mismatched:
        print(f"    {key}: {reason}")
    print("=" * 66)

    return 1 if (failures or mismatched or not manifest_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
