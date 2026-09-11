#!/usr/bin/env python3
"""Create the CVAT project, labels, and per-source-video tasks.

Implements docs/CVAT_ANNOTATION_SETUP.md sections 2 and 3: one CVAT task per
source video, with the candidate windows for that video pasted into the task
description as annotator reference.

Authentication (first match wins):
    --token / $CVAT_TOKEN / ~/.cvat_token   API token
    $CVAT_USERNAME + $CVAT_PASSWORD         basic credentials

Video data reaches CVAT one of two ways:
    --cloud-storage-id N    CVAT pulls from an S3 cloud storage configured in
                            the CVAT UI (CVAT holds the AWS credentials).
    --presigned             We hand CVAT time-limited presigned URLs, so no
                            AWS credentials are stored in CVAT.

Examples:
    python scripts/cvat_setup.py create-project
    python scripts/cvat_setup.py create-tasks --presigned --limit 1
    python scripts/cvat_setup.py create-tasks --cloud-storage-id 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_HOST = "https://app.cvat.ai"
DEFAULT_ORG = "CVCASE"
DEFAULT_PROJECT = "pickup_putdown_source_annotation"
PROJECT_DESCRIPTION = (
    "Temporal annotation on source videos. "
    "Candidate windows in task description are reference only."
)

DEFAULT_LABELS = REPO_ROOT / "annotation" / "cvat_labels.json"
DEFAULT_METADATA_DIR = REPO_ROOT / ".local" / "candidate_staging" / "metadata"

S3_BUCKET = "chillnbite-cameras"
S3_REGION = "eu-central-1"
# Source videos live flat under this prefix as <source_video_id>.mp4
S3_SOURCE_PREFIX = "anon"

# Presigned URL lifetime. CVAT downloads the video when the task is created,
# so this only needs to outlive task creation.
PRESIGN_EXPIRY_S = 12 * 3600


# ---------------------------------------------------------------------------
# Candidate windows
# ---------------------------------------------------------------------------


def load_sources_with_candidates(metadata_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Map source_video_id -> candidate windows, skipping sources with none.

    Reads the Task 6.1 metadata layout: one JSON per source video with the
    windows under ``.candidates[]``.
    """
    if not metadata_dir.exists():
        raise SystemExit(
            f"Metadata directory not found: {metadata_dir}\n"
            "Sync it first:  aws s3 sync "
            f"s3://{S3_BUCKET}/anon/candidates/metadata/ {metadata_dir}/"
        )

    sources: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(metadata_dir.rglob("*.json")):
        data = json.loads(path.read_text())
        source_id = data.get("source_video_id") or path.stem
        candidates = data.get("candidates") or []
        if not candidates:
            continue
        sources[str(source_id)] = sorted(
            candidates, key=lambda c: float(c.get("source_start_s", 0.0))
        )
    return sources


def format_task_description(source_id: str, candidates: list[dict[str, Any]]) -> str:
    """Build the annotator-facing candidate window reference for one task."""
    lines = [
        f"Source video: {source_id}",
        f"Candidate windows: {len(candidates)} (REFERENCE ONLY — not constraints)",
        "",
        "Review the full active span. Confirm or correct events inside these",
        "windows, and add any events the candidate generator missed.",
        "",
    ]
    for cand in candidates:
        start = float(cand.get("source_start_s", 0.0))
        end = float(cand.get("source_end_s", 0.0))
        extras = " ".join(
            str(cand.get(key, "?"))
            for key in ("actor_id", "hand_side", "region_id")
            if cand.get(key)
        )
        suffix = f"  {extras}" if extras else ""
        lines.append(f"  [{start:7.1f}–{end:7.1f}s] {cand.get('candidate_id', '?')}{suffix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auth / client
# ---------------------------------------------------------------------------


def resolve_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit.strip()
    env = os.environ.get("CVAT_TOKEN")
    if env:
        return env.strip()
    token_file = Path.home() / ".cvat_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def make_api_client(host: str, org: str, token: str | None) -> Any:
    """Build an authenticated low-level CVAT ApiClient."""
    from cvat_sdk.api_client import ApiClient, Configuration

    configuration = Configuration(host=host)
    client = ApiClient(configuration)

    if not token:
        username = os.environ.get("CVAT_USERNAME")
        password = os.environ.get("CVAT_PASSWORD")
        if not (username and password):
            raise SystemExit(
                "No CVAT credentials. Provide one of:\n"
                "  --token <api token>\n"
                "  CVAT_TOKEN=<api token>\n"
                "  ~/.cvat_token containing the token\n"
                "  CVAT_USERNAME + CVAT_PASSWORD"
            )
        # Exchange credentials for a session key, then use it like a token.
        login, _ = client.auth_api.create_login({"email": username, "password": password})
        token = login.key

    # Set the header directly: passing the "Token " prefix through api_key
    # double-prefixes it and CVAT rejects the result.
    client.default_headers["Authorization"] = f"Token {token}"
    if org:
        client.default_headers["X-Organization"] = org
    return client


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def find_project(api: Any, name: str) -> dict[str, Any] | None:
    projects, _ = api.projects_api.list(search=name)
    for project in projects.results:
        if project.name == name:
            return project
    return None


def cmd_create_project(args: argparse.Namespace) -> int:
    labels = json.loads(Path(args.labels).read_text())

    with make_api_client(args.host, args.org, resolve_token(args.token)) as api:
        existing = find_project(api, args.project)
        if existing is not None:
            print(f"Project already exists: {args.project} (id={existing.id})")
            print("Labels are left untouched. Delete the project to recreate it.")
            return 0

        project, _ = api.projects_api.create(
            {
                "name": args.project,
                "labels": labels,
            }
        )
        print(f"Created project {project.name} (id={project.id}) in org {args.org}")
        for label in labels:
            attrs = ", ".join(a["name"] for a in label["attributes"])
            print(f"  {label['name']:<8} [{label['color']}]  {attrs}")
    return 0


def presign(source_id: str) -> str:
    import boto3

    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
    s3 = session.client("s3", region_name=S3_REGION)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": f"{S3_SOURCE_PREFIX}/{source_id}.mp4"},
        ExpiresIn=PRESIGN_EXPIRY_S,
    )


def cmd_create_tasks(args: argparse.Namespace) -> int:
    sources = load_sources_with_candidates(Path(args.metadata_dir))
    if not sources:
        raise SystemExit("No sources with candidates found.")

    source_ids = sorted(sources)
    if args.limit:
        source_ids = source_ids[: args.limit]

    print(f"{len(sources)} source(s) with candidates; creating {len(source_ids)} task(s).")

    with make_api_client(args.host, args.org, resolve_token(args.token)) as api:
        project = find_project(api, args.project)
        if project is None:
            raise SystemExit(
                f"Project {args.project!r} not found. Run create-project first."
            )

        existing_tasks, _ = api.tasks_api.list(project_id=project.id, page_size=1000)
        existing_names = {task.name for task in existing_tasks.results}

        created = 0
        for source_id in source_ids:
            if source_id in existing_names:
                print(f"  skip (exists)  {source_id}")
                continue

            if args.dry_run:
                print(f"  would create   {source_id}  ({len(sources[source_id])} windows)")
                continue

            task, _ = api.tasks_api.create(
                {
                    "name": source_id,
                    "project_id": project.id,
                    # Candidate windows as annotator reference (doc §1, §3).
                    "bug_tracker": "",
                    "subset": "",
                }
            )

            data: dict[str, Any] = {"image_quality": args.image_quality}
            if args.cloud_storage_id:
                data["cloud_storage_id"] = args.cloud_storage_id
                data["server_files"] = [f"{S3_SOURCE_PREFIX}/{source_id}.mp4"]
                data["use_cache"] = True
            else:
                data["remote_files"] = [presign(source_id)]

            api.tasks_api.create_data(task.id, data_request=data, _content_type="application/json")
            print(f"  created        {source_id}  (task id={task.id})")
            created += 1

        print(f"\nCreated {created} task(s).")
        if not args.dry_run and created:
            print(
                "CVAT is decoding the videos asynchronously — check task status in the UI.\n"
                "Task descriptions with candidate windows: run `set-descriptions`."
            )
    return 0


def cmd_set_descriptions(args: argparse.Namespace) -> int:
    """Attach the candidate window summary to each task as an annotation guide.

    CVAT tasks have no free-text description field, so the windows go into the
    task's annotation guide — which is what annotators see in the UI.
    """
    sources = load_sources_with_candidates(Path(args.metadata_dir))

    with make_api_client(args.host, args.org, resolve_token(args.token)) as api:
        project = find_project(api, args.project)
        if project is None:
            raise SystemExit(f"Project {args.project!r} not found.")

        tasks, _ = api.tasks_api.list(project_id=project.id, page_size=1000)
        updated = 0
        for task in tasks.results:
            candidates = sources.get(task.name)
            if not candidates:
                continue
            markdown = "```\n" + format_task_description(task.name, candidates) + "\n```"
            if args.dry_run:
                print(f"--- {task.name} (id={task.id}) ---\n{markdown}\n")
                continue

            guide_id = getattr(task, "guide_id", None)
            if guide_id:
                api.guides_api.partial_update(
                    guide_id, patched_annotation_guide_write_request={"markdown": markdown}
                )
            else:
                api.guides_api.create(
                    annotation_guide_write_request={"task_id": task.id, "markdown": markdown}
                )
            print(f"  guide  {task.name}  ({len(candidates)} windows)")
            updated += 1
        print(f"\n{updated} annotation guide(s) written.")
    return 0


def cmd_windows(args: argparse.Namespace) -> int:
    """Print candidate windows per source (doc §1), no CVAT access needed."""
    sources = load_sources_with_candidates(Path(args.metadata_dir))
    for source_id in sorted(sources):
        candidates = sources[source_id]
        print(f"\n=== {source_id} ({len(candidates)} candidates) ===")
        print(format_task_description(source_id, candidates))
    print(f"\n{len(sources)} source(s) with candidates.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--org", default=DEFAULT_ORG)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--token", default=None, help="CVAT API token")
    parser.add_argument("--metadata-dir", default=str(DEFAULT_METADATA_DIR))

    sub = parser.add_subparsers(dest="command", required=True)

    p_project = sub.add_parser("create-project", help="Create project with labels")
    p_project.add_argument("--labels", default=str(DEFAULT_LABELS))
    p_project.set_defaults(func=cmd_create_project)

    p_tasks = sub.add_parser("create-tasks", help="Create one task per source video")
    p_tasks.add_argument("--cloud-storage-id", type=int, default=None)
    p_tasks.add_argument("--presigned", action="store_true")
    p_tasks.add_argument("--image-quality", type=int, default=70)
    p_tasks.add_argument("--limit", type=int, default=None)
    p_tasks.add_argument("--dry-run", action="store_true")
    p_tasks.set_defaults(func=cmd_create_tasks)

    p_desc = sub.add_parser("set-descriptions", help="Write candidate windows into tasks")
    p_desc.add_argument("--dry-run", action="store_true")
    p_desc.set_defaults(func=cmd_set_descriptions)

    p_windows = sub.add_parser("windows", help="Print candidate windows per source")
    p_windows.set_defaults(func=cmd_windows)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
