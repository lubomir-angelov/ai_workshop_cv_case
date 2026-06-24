# VLM Annotation Pipeline

Run the annotate-vlm command to send candidate contact sheets to a local llama.cpp VLM and produce structured event annotations.

## Prerequisites

- llama.cpp server running with a **vision-capable** model and mmproj loaded
- Default endpoint: `http://localhost:8000`
- Verify vision support:
  ```bash
  curl -s http://localhost:8000/v1/models
  ```

## Quick run

```bash
pickup-putdown annotate-vlm .local/candidate_staging/candidates \
  --output-dir .local/vlm_annotations \
  --force \
  --vlm-base-url http://localhost:8000 \
  --vlm-model Qwen3.6-27B-UD-Q4_K_XL \
  -v
```

## Run all night with tmux + logging

```bash
# Start detached session
tmux new-session -d -s vlm-annotate

# Run pipeline, tee output to timestamped log
tmux send-keys -t vlm-annotate \
  'pickup-putdown annotate-vlm .local/candidate_staging/candidates \
    --output-dir .local/vlm_annotations \
    --force \
    --vlm-base-url http://localhost:8000 \
    --vlm-model Qwen3.6-27B-UD-Q4_K_XL \
    --vlm-timeout 180 \
    -v 2>&1 | tee .local/vlm_annotations/vlm_run_$(date +%Y%m%d_%H%M%S).log' \
  Enter

# Attach to watch
tmux attach-session -t vlm-annotate
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t vlm-annotate`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir, -o` | `.local/vlm_annotations` | Output directory |
| `--review-fps` | `5.0` | Frame extraction rate |
| `--max-frame-width` | `640` | Review frame max width |
| `--force` | — | Reprocess already-annotated candidates |
| `--limit N` | all | Process at most N candidates |
| `--vlm-base-url` | `http://localhost:8080` | llama.cpp server URL |
| `--vlm-model` | auto | Model name for VLM |
| `--vlm-temperature` | `0.0` | Sampling temperature |
| `--vlm-max-tokens` | `2048` | Max response tokens |
| `--vlm-timeout` | `120` | Per-request timeout (seconds) |
| `--no-vlm` | — | Skip VLM, produce frames only |
| `--verbose, -v` | — | Enable debug logging |

## Outputs

Written to `--output-dir`:

- `raw/<candidate_id>.json` — raw probe info, frame count, metadata
- `normalized/<candidate_id>.json` — normalized events and ignore intervals
- `review_frames/<candidate_id>/` — extracted frames and contact sheet
- `events.csv` — canonical event rows across all candidates
- `processing.csv` — per-candidate processing ledger
- `summary.json` — aggregate pipeline summary
- `vlm_run_<timestamp>.log` — combined stdout/stderr (when using tee)

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Connection refused` | llama.cpp not running | Start server, verify port mapping |
| `image input is not supported` | Text-only model | Load vision model with `--mmproj` |
| `Failed to load image or audio file` | mmproj missing | Add `--mmproj /path/to/mmproj.gguf` to server |
| `HTTP 500` | Vision path misconfigured | Check server logs for mmproj errors |
