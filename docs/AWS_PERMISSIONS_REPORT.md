# AWS Permissions & Upload Readiness Report

Generated: 2026-06-24

## Credentials

| Field | Value |
|---|---|
| IAM user | `cameras-reader-anon` |
| Account | `158309450155` |
| ARN | `arn:aws:iam::158309450155:user/cameras-reader-anon` |
| Region | `eu-central-1` |
| Target bucket | `s3://chillnbite-cameras/anon/` |

## Permission test results

| Permission | Status | Detail |
|---|---|---|
| `s3:ListBucket` | Working | Can list `anon/` prefix and source videos |
| `s3:GetObject` | Working | Can read objects from bucket |
| `s3:HeadObject` | Working | Implicit via listing |
| `s3:PutObject` | **Denied** | `AccessDenied` — no identity-based policy allows this action on `arn:aws:s3:::chillnbite-cameras/anon/candidates/*` |
| `s3:HeadBucket` | 403 Forbidden | Expected for cross-account or restricted users; does not block ListBucket/GetObject |

## Candidate staging status

Ledger: `.local/candidate_staging/local_processing.csv`

| State | Count | Notes |
|---|---|---|
| `generated=true, uploaded=false` | **75** | Ready for upload |
| `downloaded=true, generated=false` | **13** | Need generation before upload |
| Already uploaded | 0 | — |

### Candidate directories

- Total dirs under `.local/candidate_staging/candidates/`: **75**
- With candidate videos (`.mp4`): **42** (1,615 total candidate videos)
- Empty (only `.json` metadata, `candidate_count: 0`): **33**

## Upload target structure

The `candidates-upload` command uploads to:

```
s3://chillnbite-cameras/anon/candidates/videos/{video_id}/{candidate}.mp4
s3://chillnbite-cameras/anon/candidates/metadata/{video_id}/{video_id}.json
```

No `candidates/` prefix currently exists in S3 — will be created on first upload.

## Blocking item

**`s3:PutObject` permission is required.** Ask the account admin to attach an inline or managed policy to `cameras-reader-anon` granting at minimum:

```json
{
  "Effect": "Allow",
  "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::chillnbite-cameras/anon/candidates/*"
}
```

## Next steps (once write access granted)

```bash
# Quick test with 1 entry
pickup-putdown candidates-upload -t 1

# Upload all 75 generated candidates
pickup-putdown candidates-upload

# Generate the remaining 13, then upload those too
pickup-putdown candidates-generate -t 13
pickup-putdown candidates-upload -t 13
```

## Notes

- The 33 empty candidate dirs have `candidate_count: 0` in their metadata. Uploading them sends only the `.json` with no videos. Regeneration may yield results if the pipeline has been updated since original run.
- The `candidates-upload` command uses the ledger at `.local/candidate_staging/local_processing.csv` to select entries with `generated=true, uploaded=false`.
- Use `--upload-ledger` flag to write upload state to a separate ledger and avoid race conditions with a running generation pipeline.
