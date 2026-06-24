# Upload Report

Generated: 2026-06-24

## Upload run 1 — single-entry test

```bash
pickup-putdown candidates-upload -t 1
```

Result: success. Uploaded 42 candidate videos + 1 metadata JSON for `D2_S20260520135131_E20260520135549_anon` to `s3://chillnbite-cameras/anon/candidates/videos/...`. Ledger marked entry as `uploaded=true`.

## Upload run 2 — full upload attempt

```bash
pickup-putdown candidates-upload
```

Result: `No candidates ready for upload.` — exited immediately with no uploads.

## Bug: `target_count=0` returns empty selection

**Root cause:** `LocalProcessingLedger.select_ready_for_upload()` at `src/pickup_putdown/remote/local_ledger.py:155` does `candidates[:target_count]`. The CLI default for `--target-count` is `0` (intended to mean "all"), but Python's `[:0]` returns an empty list.

Affected methods:
- `select_not_downloaded` (line 125)
- `select_ready_for_generation` (line 140)
- `select_ready_for_upload` (line 155)

**Workaround:** Pass an explicit count larger than total entries:
```bash
pickup-putdown candidates-upload -t 100
```

**Fix needed:** Treat `target_count == 0` as "no limit":
```python
return candidates if target_count == 0 else candidates[:target_count]
```

## Remaining entries

- Uploaded: 1 source (42 videos)
- Remaining `generated=true, uploaded=false`: 74 sources
- Still `generated=false`: 13 sources (need generation first)

## Next

1. Run `pickup-putdown candidates-upload -t 100` to upload remaining 74 sources.
2. Fix `target_count=0` bug in `local_ledger.py` before future runs.
