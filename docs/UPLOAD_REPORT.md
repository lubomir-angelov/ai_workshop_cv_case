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

**Surviving SSH disconnects:** The process terminates on SIGHUP when the session drops. Use `tmux` (recommended) or `nohup`.

**Using tmux:**
```bash
# Create a named session
tmux new -s upload

# Must run from the project root (ledger path is relative)
cd /home/naim/repos/ai_workshop_cv_case
pickup-putdown candidates-upload -t 100

# Detach: Ctrl+B then D
# Reattach later:
tmux attach -t upload
```

**Using nohup:**
```bash
# Must run from the project root — the ledger is at .local/candidate_staging/local_processing.csv
cd /home/naim/repos/ai_workshop_cv_case
nohup pickup-putdown candidates-upload -t 100 > upload.log 2>&1 &
```

**Ledger safety:** The ledger saves atomically per entry. If the process is killed mid-upload, the last entry is rolled back — no partial state. Re-running resumes from the next unuploaded entry.

**Fix needed:** Treat `target_count == 0` as "no limit":
```python
return candidates if target_count == 0 else candidates[:target_count]
```

## Upload history

| Run | Command | Result |
|---|---|---|
| 1 | `candidates-upload -t 1` | 1 source uploaded (42 videos) |
| 2 | `candidates-upload` | 0 — `target_count=0` bug |
| 3 | `candidates-upload -t 100` | 11 more uploaded, then killed by SSH disconnect |
| 4 | `nohup candidates-upload -t 100` | Failed — ran from `~` instead of project root; relative ledger path resolved to `~/.local/` which doesn't exist |

## Current status (2026-06-24 ~12:15)

- Uploaded: **12** sources
- Remaining `generated=true, uploaded=false`: **63** sources
- Still `generated=false`: **13** sources (need generation first)

## Next

1. Start tmux session and run `pickup-putdown candidates-upload -t 100` from project root.
2. Fix `target_count=0` bug in `local_ledger.py` before future runs.
3. Generate the remaining 13 sources, then upload those.
