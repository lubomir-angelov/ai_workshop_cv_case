# Data Privacy

Project footage contains people. Use it only for pickup/putdown event detection.
Person identification, face recognition, identity inference, and source-video
redistribution are prohibited.

## Source-data privacy status

Source videos available to this project are pre-blurred and stored in the
approved AWS bucket. The project does not receive or require original
unblurred footage.

Pre-blurring reduces privacy risk but is not proof that every identifying
detail is obscured. Treat source clips and all derived media as private even
when faces appear blurred. Do not attempt to reverse, weaken, or reconstruct
the applied blurring.

The organization or process that performed source blurring is not recorded in
this repository. Add that provenance only when confirmed by the data owner.

## Handling rules

- Read source footage only from the approved AWS location or private cache.
- Use clip-local tracking IDs; they are not identities.
- Store credentials in environment variables or ignored local files.
- Never log credentials, tokens, private keys, or signed URLs.
- Keep raw videos, previews, real annotations, crops, features, and model
  checkpoints outside Git.
- Publish only synthetic illustrations or manually approved derivatives of the
  pre-blurred source.
- Manually review selected media before publication; source pre-blurring may
  miss faces or other identifying details.
- Restrict artifact access to project members who need it.

## Artifact policy

| Artifact | Owner | Approved location | Git policy | Retention |
|---|---|---|---|---|
| Pre-blurred source videos | Data owner | Approved AWS bucket or private cache | Prohibited | Source-data agreement |
| Credentials | Operator | Environment or `.local/env/` | Prohibited | Remove when access ends |
| Safe manifests | Dataset owner | Versioned storage | Review required | Project audit period |
| Real annotation exports | Annotation owner | Private artifact storage | Prohibited | Dataset audit period |
| Candidate/event previews | Run owner | Private artifact storage | Prohibited | Delete after review or per policy |
| Crops and features | Model owner | Private artifact storage | Prohibited | While needed for reproduction |
| Weights and checkpoints | Model owner | Private model storage | Prohibited | Keep selected reproducible versions |
| Metadata and metrics | Run owner | Versioned results storage | Allowed after review | Project audit period |
| Approved report media | Report owner | Documentation assets | Allowed after manual review | While report is maintained |

If no retention period is defined by the data agreement, project owner must
define one before collecting derived media.

## Secrets

Safe example configurations contain placeholders only. Runtime secrets may use:

```text
PICKUP_PUTDOWN_STORAGE_BUCKET_URI
PICKUP_PUTDOWN_STORAGE_REGION
PICKUP_PUTDOWN_STORAGE_ENDPOINT_URL
PICKUP_PUTDOWN_STORAGE_ANONYMOUS
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

Do not place secret values in YAML, shell history, screenshots, reports, or
issues.

## Publication workflow

```text
AWS pre-blurred source
    -> private local cache
    -> selected derivative
    -> manual privacy review
    -> approved publication artifact
```

1. Select the minimum image or clip required for the report.
2. Create the derivative in private working storage.
3. Review the complete derivative frame by frame.
4. Confirm that faces remain obscured.
5. Check reflections, screens, badges, readable documents, and other
   identifying details.
6. Apply additional masking or use a synthetic illustration when review fails.
7. Record source `clip_id`, reviewer, review date, and approval status without
   personal information.
8. Publish only the approved derivative.

Example review record:

```json
{
  "clip_id": "clip_example_001",
  "artifact": "privacy_safe/example_event_preview.mp4",
  "reviewer": "reviewer_alias",
  "review_date": "2026-06-25",
  "source_preblurred": true,
  "approval_status": "approved",
  "notes": "No visible faces or identifying details after manual review."
}
```

## Face-blurring utility decision

Task 17 permits a face-blurring utility or a documented report workflow. This
repository uses the documented workflow because the AWS source is already
pre-blurred. A second automatic blurring tool is not required now and could
create false confidence. Add one later only if manual review finds recurring
missed faces or publication policy requires independent reprocessing.

## Synthetic annotation fixtures

Repository contains these generated fixtures:

```text
resources/annotations/clip_001.h264.mp4
resources/annotations/clip_002.h264.mp4
```

They are programmatically generated, contain geometric shapes only, and are
marked `SYNTHETIC FIXTURE - NO CUSTOMER MEDIA` in every frame. They preserve
annotation-demo paths without storing customer-derived footage. Existing Git
history may still contain superseded customer-derived versions; history
rewriting requires separate repository-owner approval.

## Incident response

If sensitive data is committed or shared:

1. Stop further distribution.
2. Notify repository and data owners.
3. Revoke or rotate exposed credentials.
4. Remove the artifact using the repository host's approved process.
5. Record exposure and corrective actions.

Do not rewrite repository history without explicit owner approval.
