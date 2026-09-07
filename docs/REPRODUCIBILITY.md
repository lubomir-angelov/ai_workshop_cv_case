# Reproducibility

This guide covers functionality implemented through Tasks 1–6 and the Task 8
evaluator. Add commands for later tasks only after they are implemented and
verified.

The reference case repository is read-only. Keep implementation code in this
solution repository and keep private data, generated media, features,
checkpoints, and run artifacts in approved private storage.

## Environment

Requirements:

- Python version supported by `pyproject.toml`
- FFmpeg and `ffprobe`
- Docker and Docker Compose for Label Studio
- Optional GPU for YOLO inference

Install the project and inspect available commands:

```powershell
make install-dev
pickup-putdown --help
make help
```

Run commands from repository root. Keep credentials, source videos, model
weights, caches, previews, annotations, and generated outputs outside Git.
Small synthetic annotation fixtures under `resources/annotations/` are the
only video exception; each frame is visibly marked as synthetic.

## Storage and ingestion

Create private storage configuration:

```powershell
make env-setup
```

This writes secrets to the ignored path:

```text
.local/env/storage.env
```

Do not print, share, or commit this file. Run ingestion:

```powershell
make ingest
```

Runtime storage values come from `configs/storage.yaml` and environment
variables. `configs/storage.example.yaml` is the safe sharing template.
Source videos in the approved AWS location are pre-blurred but remain private.
Follow `docs/DATA_PRIVACY.md` for derived media and publication review.

## Layer 0 reproduction

### WSL environment used for acceptance

The verified Task 17 acceptance run used:

```bash
source /home/constantine/venvs/cv_case/bin/activate
cd /mnt/c/Users/const/Desktop/project/ai_workshop_cv_case
python --version
```

Expected Python version is 3.12 or newer. Verify required tools:

```bash
python -c "import cv2, pyarrow, ultralytics, yaml; print('Dependencies OK')"
command -v make
command -v ffprobe
```

Run Tasks 3–5 for one private pre-blurred video:

```bash
make tasks-3-5 \
  VIDEO=/mnt/c/path/to/private/preblurred-video.mp4 \
  RENDER_PREVIEWS=0
```

Outputs are written below `.local/task_runs/<timestamp>/`:

```text
task_3/tracks_person.parquet
task_3/active_spans.parquet
task_3/clips.parquet
task_4/shelf_validation.json
task_5/tracks_pose.parquet
task_5/candidates.parquet
task_5/propose_run_metadata.json
```

Set a stable run ID when the run is acceptance evidence:

```bash
make tasks-3-5 \
  RUN_ID=20260625_task17_acceptance \
  VIDEO=/mnt/c/path/to/private/preblurred-video.mp4 \
  RENDER_PREVIEWS=0
```

Reproduction inputs include source checksum, resolved configurations, model
checksums, Git commit, and random seed. Stage A/B candidates are never
ground-truth events or canonical predictions.

The verified acceptance run completed on CPU because the installed NVIDIA
driver was older than the active PyTorch build requirement. GPU availability
therefore affects runtime but not required output structure.

## Annotation

Start and verify Label Studio:

```powershell
make annotation-up
make annotation-config-validate
make annotation-test
make annotation-acceptance
```

Follow `docs/ANNOTATION_WORKFLOW.md` and `docs/LABELING_GUIDELINES.md`.
Generated tasks, exports, Label Studio state, media, and real annotator data
remain outside Git.

## Evaluator

Task 8 currently provides a Python API and tests, not an evaluation CLI command:

```powershell
python -m pytest tests/test_evaluation.py
```

All future models must use this shared evaluator. Select thresholds using
validation data only.

## Run metadata

Every reproducible run must record:

```text
run_id
timestamp
git_commit
dataset_version
split_version
config
resolved_config
seed
model_identifier
checkpoint_hash
```

Use SHA-256 for source, model, and checkpoint hashes. Record threshold values
and the validation run used to choose them. Never silently overwrite an
incompatible run or immutable dataset version.

Recommended names:

```text
dataset: manifest_v<integer>
split: split_v<integer>
run: <YYYYMMDD>_<stage>_<model>_v<integer>
model: <layer>_<approach>_v<integer>
```

See `samples/example_run_metadata.json` for a fictional, machine-readable
Task 3–5 metadata example. It demonstrates structure only and is not evidence
of a real run.

## Verification

Run relevant checks before committing:

```powershell
ruff check .
ruff format --check .
python -m pytest
python -m compileall src
git diff --check
git status --short
git diff
```

Do not report a check as passing unless it completed successfully.

## Pending final documentation

Add exact commands and evidence when their owning tasks land:

- immutable dataset validation and split freeze;
- Track A and Track B training and inference;
- standalone Layer 2 inference;
- evaluation report generation;
- batch inference;
- final checkpoints, thresholds, hardware, quantization, runtime, and metrics.

Track progress in `docs/TASK_17_CHECKLIST.md`.
