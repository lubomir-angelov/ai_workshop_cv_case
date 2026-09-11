SHELL := /bin/bash

PYTHON ?= python
PICKUP_PUTDOWN ?= $(PYTHON) -m pickup_putdown.cli

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

STORAGE_ENV ?= .local/env/storage.env
STORAGE_CONFIG ?= configs/storage.yaml

TRIAGE_CONFIG ?= configs/triage.yaml
TRACKER_CONFIG ?= configs/bytetrack_triage.yaml

PROPOSALS_CONFIG ?= configs/proposals.yaml
SHELF_CONFIG ?= configs/shelves.yaml
CAMERA_ID ?= store_camera_01

PERSON_MODEL ?= models/person_detector.pt
POSE_MODEL ?= models/pose_detector.pt

# One timestamp is resolved once per make invocation. It can be overridden:
#   make tasks-3-5 RUN_ID=20260622_120000 VIDEO=/path/video.mp4
RUN_ID := $(shell date +%Y%m%d_%H%M%S)

RUN_ROOT ?= .local/task_runs/$(RUN_ID)

TASK3_OUTPUT ?= $(RUN_ROOT)/task_3
TASK4_OUTPUT ?= $(RUN_ROOT)/task_4
TASK5_OUTPUT ?= $(RUN_ROOT)/task_5

# Backward-compatible Task 3 output name.
TRIAGE_OUTPUT ?= $(TASK3_OUTPUT)

PERSON_TRACKS ?= $(TRIAGE_OUTPUT)/tracks_person.parquet
ACTIVE_SPANS ?= $(TRIAGE_OUTPUT)/active_spans.parquet
CLIPS ?= $(TRIAGE_OUTPUT)/clips.parquet

# Render the 15-style candidate review videos by default.
# Disable with:
#   make tasks-3-5 VIDEO=/path/video.mp4 RENDER_PREVIEWS=0
RENDER_PREVIEWS ?= 1

# Task 3 triage QA previews are for human review only; no downstream task
# consumes them, so batch runs skip them by default. Enable with:
#   make task-3 VIDEO=/path/video.mp4 TRIAGE_RENDER_PREVIEWS=1
TRIAGE_RENDER_PREVIEWS ?= 0

# Explicit VIDEO=/path is recommended. This fallback selects the first local MP4.
DEFAULT_VIDEO := $(shell find .local \
	-type f \
	-iname '*.mp4' \
	-not -path '*/task_runs/*' \
	-not -path '*/triage_acceptance/output_*/*' \
	-not -path '*/triage_previews/*' \
	-not -path '*/candidate_previews/*' \
	2>/dev/null | sort | head -n 1)

TRIAGE_INPUT ?= $(DEFAULT_VIDEO)
VIDEO ?= $(TRIAGE_INPUT)

.PHONY: help install install-dev lint format format-check typecheck test compile clean \
	env-setup env-ingest ingest \
	show-run models task-3 task-4 task-5 tasks-3-5 \
	task_3 task_4 task_5 \
 candidates-remote candidates-download candidates-upload candidates-generate candidates-process-local \
 track-a-dataset train-track-a infer-track-a evaluate-track-a \
 cvat-export track-b1-dataset track-b1-cache track-b1-embeddings \
 train-track-b1-head train-track-b1 infer-track-b1 track-b1-inspect \
 track-b1-sources tune-track-b1-thresholds track-b1-results track-b1-all

# ---------------------------------------------------------------------------
# General development targets
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Example:"
	@echo "  make tasks-3-5 VIDEO=/path/to/video.mp4"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	$(PYTHON) -m pip install -e .

install-dev: install ## Install package with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run Ruff linting
	ruff check .

format: ## Format Python files with Ruff
	ruff format .

format-check: ## Check formatting without modifying files
	ruff format --check .

typecheck: ## Run mypy
	mypy src/pickup_putdown

test: ## Run the complete pytest suite
	$(PYTHON) -m pytest

compile: ## Compile all Python files
	$(PYTHON) -m compileall src

clean: ## Remove build artifacts and Python caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Storage and ingestion
# ---------------------------------------------------------------------------

env-setup: ## Prompt for storage settings and save a protected environment file
	@mkdir -p "$(dir $(STORAGE_ENV))"
	@read -rp "AWS Access Key ID: " ACCESS_KEY && \
	read -rsp "AWS Secret Access Key: " SECRET_KEY && echo && \
	read -rp "S3 Bucket URI (s3://bucket/prefix): " BUCKET_URI && \
	read -rp "S3 Region (e.g. us-east-1): " REGION && \
	read -rp "Endpoint URL (leave empty for AWS default): " ENDPOINT && \
	read -rp "Anonymous access (y/N): " ANON && \
	umask 077 && \
	{ \
		printf 'export AWS_ACCESS_KEY_ID=%q\n' "$$ACCESS_KEY"; \
		printf 'export AWS_SECRET_ACCESS_KEY=%q\n' "$$SECRET_KEY"; \
		printf 'export PICKUP_PUTDOWN_STORAGE_BUCKET_URI=%q\n' "$$BUCKET_URI"; \
		printf 'export PICKUP_PUTDOWN_STORAGE_REGION=%q\n' "$$REGION"; \
		if [ -n "$$ENDPOINT" ]; then \
			printf 'export PICKUP_PUTDOWN_STORAGE_ENDPOINT_URL=%q\n' "$$ENDPOINT"; \
		fi; \
		if [[ "$$ANON" =~ ^[yY]$$ ]]; then \
			printf 'export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=true\n'; \
		else \
			printf 'export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=false\n'; \
		fi; \
	} > "$(STORAGE_ENV)" && \
	chmod 600 "$(STORAGE_ENV)" && \
	echo "Storage environment saved to $(STORAGE_ENV)"

ingest: ## Run ingestion using the saved storage environment
	@test -f "$(STORAGE_ENV)" || \
		(echo "Missing $(STORAGE_ENV). Run 'make env-setup' first." && exit 1)
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	$(PICKUP_PUTDOWN) ingest --config "$(STORAGE_CONFIG)"

env-ingest: env-setup ## Configure storage and immediately run ingestion
	@$(MAKE) ingest

# ---------------------------------------------------------------------------
# Model installation
# ---------------------------------------------------------------------------

$(PERSON_MODEL):
	@mkdir -p "$(dir $(PERSON_MODEL))"
	@echo "Downloading YOLO11n person detector..."
	@$(PYTHON) -c 'from pathlib import Path; import shutil; from ultralytics import YOLO; dst=Path("$(PERSON_MODEL)"); model=YOLO("yolo11n.pt"); src=Path(model.ckpt_path); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print(f"Installed {src} -> {dst}")'

$(POSE_MODEL):
	@mkdir -p "$(dir $(POSE_MODEL))"
	@echo "Downloading YOLO11n pose detector..."
	@$(PYTHON) -c 'from pathlib import Path; import shutil; from ultralytics import YOLO; dst=Path("$(POSE_MODEL)"); model=YOLO("yolo11n-pose.pt"); src=Path(model.ckpt_path); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); print(f"Installed {src} -> {dst}")'

models: $(PERSON_MODEL) $(POSE_MODEL) ## Download required YOLO models

# ---------------------------------------------------------------------------
# Pipeline information
# ---------------------------------------------------------------------------

show-run: ## Show resolved input, configuration, and output paths
	@echo "Run ID:             $(RUN_ID)"
	@echo "Video:              $(VIDEO)"
	@echo "Run root:           $(RUN_ROOT)"
	@echo "Task 3 output:      $(TRIAGE_OUTPUT)"
	@echo "Task 4 output:      $(TASK4_OUTPUT)"
	@echo "Task 5 output:      $(TASK5_OUTPUT)"
	@echo "Person tracks:      $(PERSON_TRACKS)"
	@echo "Active spans:       $(ACTIVE_SPANS)"
	@echo "Shelf config:       $(SHELF_CONFIG)"
	@echo "Camera ID:          $(CAMERA_ID)"
	@echo "Render previews:    $(RENDER_PREVIEWS)"

# ---------------------------------------------------------------------------
# Task 3: person triage and active spans
# ---------------------------------------------------------------------------

task-3: $(PERSON_MODEL) ## Run Task 3 person triage for VIDEO
	@test -n "$(VIDEO)" || \
		(echo "No input video selected." && \
		 echo "Use: make task-3 VIDEO=/path/to/video.mp4" && \
		 exit 1)
	@test -f "$(VIDEO)" || \
		(echo "Input video does not exist: $(VIDEO)" && exit 1)
	@test -f "$(TRIAGE_CONFIG)" || \
		(echo "Missing triage configuration: $(TRIAGE_CONFIG)" && exit 1)
	@test -f "$(TRACKER_CONFIG)" || \
		(echo "Missing tracker configuration: $(TRACKER_CONFIG)" && exit 1)
	@mkdir -p "$(TRIAGE_OUTPUT)"
	@echo "Task 3 input:  $(VIDEO)"
	@echo "Task 3 output: $(TRIAGE_OUTPUT)"
	@set -o pipefail; \
	$(PICKUP_PUTDOWN) triage \
		"$(VIDEO)" \
		--config "$(TRIAGE_CONFIG)" \
		--tracker-config "$(TRACKER_CONFIG)" \
		--output-dir "$(TRIAGE_OUTPUT)" \
		$(if $(filter 1 true yes,$(TRIAGE_RENDER_PREVIEWS)),--render-previews,) \
		--verbose \
		2>&1 | tee "$(TRIAGE_OUTPUT)/task_3.log"
	@test -f "$(PERSON_TRACKS)" || \
		(echo "Task 3 did not produce $(PERSON_TRACKS)" && exit 1)
	@test -f "$(ACTIVE_SPANS)" || \
		(echo "Task 3 did not produce $(ACTIVE_SPANS)" && exit 1)
	@test -f "$(CLIPS)" || \
		(echo "Task 3 did not produce $(CLIPS)" && exit 1)
	@echo "Task 3 completed successfully."

# ---------------------------------------------------------------------------
# Task 4: shelf-region configuration validation
# ---------------------------------------------------------------------------

task-4: ## Validate Task 4 shelf regions for CAMERA_ID
	@test -f "$(SHELF_CONFIG)" || \
		(echo "Missing shelf configuration: $(SHELF_CONFIG)" && exit 1)
	@mkdir -p "$(TASK4_OUTPUT)"
	@SHELF_CONFIG="$(SHELF_CONFIG)" \
	CAMERA_ID="$(CAMERA_ID)" \
	TASK4_OUTPUT="$(TASK4_OUTPUT)" \
	$(PYTHON) -c 'import json, os; from pathlib import Path; from pickup_putdown.perception.shelf_regions import load_shelf_config, get_regions_for_camera, get_expanded_regions; config=load_shelf_config(Path(os.environ["SHELF_CONFIG"])); camera_id=os.environ["CAMERA_ID"]; camera=get_regions_for_camera(config, camera_id); expanded=get_expanded_regions(camera); regions=[{"region_id": region.region_id, "points": region.points, "expanded_points": expanded[region.region_id]} for region in camera.regions]; assert regions, f"No shelf regions configured for {camera_id}"; output=Path(os.environ["TASK4_OUTPUT"]); output.mkdir(parents=True, exist_ok=True); payload={"camera_id": camera_id, "source_width": camera.source_width, "source_height": camera.source_height, "n_regions": len(regions), "regions": regions}; path=output / "shelf_validation.json"; path.write_text(json.dumps(payload, indent=2)); print(f"Validated {len(regions)} region(s) for {camera_id}"); print(f"Validation output: {path}")' \
		2>&1 | tee "$(TASK4_OUTPUT)/task_4.log"
	@test -f "$(TASK4_OUTPUT)/shelf_validation.json" || \
		(echo "Task 4 validation output was not created." && exit 1)
	@echo "Task 4 completed successfully."

# ---------------------------------------------------------------------------
# Task 5: pose-based proposal generation
# ---------------------------------------------------------------------------

task-5: $(POSE_MODEL) ## Run Task 5 pose inference and candidate generation
	@test -n "$(VIDEO)" || \
		(echo "No input video selected." && \
		 echo "Use: make task-5 VIDEO=/path/to/video.mp4" && \
		 exit 1)
	@test -f "$(VIDEO)" || \
		(echo "Input video does not exist: $(VIDEO)" && exit 1)
	@test -f "$(PROPOSALS_CONFIG)" || \
		(echo "Missing proposals configuration: $(PROPOSALS_CONFIG)" && exit 1)
	@test -f "$(SHELF_CONFIG)" || \
		(echo "Missing shelf configuration: $(SHELF_CONFIG)" && exit 1)
	@test -f "$(PERSON_TRACKS)" || \
		(echo "Missing Task 3 person tracks: $(PERSON_TRACKS)" && \
		 echo "Run Task 3 first using the same RUN_ID." && exit 1)
	@test -f "$(ACTIVE_SPANS)" || \
		(echo "Missing Task 3 active spans: $(ACTIVE_SPANS)" && \
		 echo "Run Task 3 first using the same RUN_ID." && exit 1)
	@mkdir -p "$(TASK5_OUTPUT)"
	@echo "Task 5 input:         $(VIDEO)"
	@echo "Task 5 person tracks: $(PERSON_TRACKS)"
	@echo "Task 5 active spans:  $(ACTIVE_SPANS)"
	@echo "Task 5 output:        $(TASK5_OUTPUT)"
	@set -o pipefail; \
	$(PICKUP_PUTDOWN) propose \
		"$(VIDEO)" \
		--config "$(PROPOSALS_CONFIG)" \
		--shelves-config "$(SHELF_CONFIG)" \
		--camera-id "$(CAMERA_ID)" \
		--person-tracks "$(PERSON_TRACKS)" \
		--active-spans "$(ACTIVE_SPANS)" \
		--output-dir "$(TASK5_OUTPUT)" \
		$(if $(filter 1 true yes,$(RENDER_PREVIEWS)),--render-previews,) \
		--verbose \
		2>&1 | tee "$(TASK5_OUTPUT)/task_5.log"
	@test -f "$(TASK5_OUTPUT)/tracks_pose.parquet" || \
		(echo "Task 5 did not produce tracks_pose.parquet" && exit 1)
	@test -f "$(TASK5_OUTPUT)/candidates.parquet" || \
		(echo "Task 5 did not produce candidates.parquet" && exit 1)
	@test -f "$(TASK5_OUTPUT)/propose_run_metadata.json" || \
		(echo "Task 5 did not produce run metadata." && exit 1)
	@echo "Task 5 completed successfully."

# ---------------------------------------------------------------------------
# Consecutive Task 3 -> Task 4 -> Task 5 pipeline
# ---------------------------------------------------------------------------

tasks-3-5: ## Run Tasks 3, 4, and 5 consecutively for VIDEO
	@test -n "$(VIDEO)" || \
		(echo "No input video selected." && \
		 echo "Use: make tasks-3-5 VIDEO=/path/to/video.mp4" && \
		 exit 1)
	@test -f "$(VIDEO)" || \
		(echo "Input video does not exist: $(VIDEO)" && exit 1)
	@echo "Starting Tasks 3-5"
	@echo "Run ID:   $(RUN_ID)"
	@echo "Video:    $(VIDEO)"
	@echo "Run root: $(RUN_ROOT)"
	@$(MAKE) task-3 \
		RUN_ID="$(RUN_ID)" \
		RUN_ROOT="$(RUN_ROOT)" \
		VIDEO="$(VIDEO)"
	@$(MAKE) task-4 \
		RUN_ID="$(RUN_ID)" \
		RUN_ROOT="$(RUN_ROOT)" \
		VIDEO="$(VIDEO)"
	@$(MAKE) task-5 \
		RUN_ID="$(RUN_ID)" \
		RUN_ROOT="$(RUN_ROOT)" \
		VIDEO="$(VIDEO)"
	@echo ""
	@echo "Tasks 3-5 completed successfully."
	@echo "Run output: $(RUN_ROOT)"

# Underscore aliases matching the task document names.
task_3: task-3 ## Alias for task-3

task_4: task-4 ## Alias for task-4

task_5: task-5 ## Alias for task-5

# ---------------------------------------------------------------------------
# Remote candidate generation (Task 6.1 Hard)
# ---------------------------------------------------------------------------

CANDIDATE_STORAGE_CONFIG ?= configs/storage.s3.yaml
CANDIDATE_PIPELINE_CONFIG ?= configs/candidates.yaml
CANDIDATE_TARGET_COUNT ?= 5
CANDIDATE_WORKERS ?= 4
CANDIDATE_TRANSFER_WORKERS ?= 4
CANDIDATE_GPU_WORKERS ?= 8
CANDIDATE_ENCODE_WORKERS ?= 12
CANDIDATE_WORK_DIR ?= .local/remote_candidates
CANDIDATE_KEEP_LOCAL_FILES ?=
CANDIDATE_FAIL_FAST ?=
CANDIDATE_OVERWRITE ?=
CANDIDATE_DRY_RUN ?=

CANDIDATE_DEFER_UPLOAD ?=
CANDIDATE_LOCAL_SOURCE_DIR ?= .local/source_videos
CANDIDATE_LOCAL_OUTPUT_DIR ?= .local/candidate_staging
CANDIDATE_MIN_FREE_DISK_GB ?= 0
CANDIDATE_REFRESH_CHANGED ?=

candidates-download: ## Download source videos from S3 to local cache in batches
	@echo "=== Source Video Download ==="
	@echo "Storage config:    $(CANDIDATE_STORAGE_CONFIG)"
	@echo "Target count:      $(CANDIDATE_TARGET_COUNT)"
	@echo "Transfer workers:  $(CANDIDATE_TRANSFER_WORKERS)"
	@echo "Source dir:        $(CANDIDATE_LOCAL_SOURCE_DIR)"
	@echo "Output dir:        $(CANDIDATE_LOCAL_OUTPUT_DIR)"
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	$(PICKUP_PUTDOWN) candidates-download \
		--storage-config "$(CANDIDATE_STORAGE_CONFIG)" \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		--transfer-workers $(CANDIDATE_TRANSFER_WORKERS) \
		--local-source-dir "$(CANDIDATE_LOCAL_SOURCE_DIR)" \
		--local-output-dir "$(CANDIDATE_LOCAL_OUTPUT_DIR)" \
		--minimum-free-disk-gb $(CANDIDATE_MIN_FREE_DISK_GB) \
		$(if $(CANDIDATE_REFRESH_CHANGED),--refresh-changed,) \
		-v

candidates-remote: ## Generate annotation candidates from remote S3 source videos
	@echo "=== Remote Candidate Generation ==="
	@echo "Storage config:    $(CANDIDATE_STORAGE_CONFIG)"
	@echo "Pipeline config:   $(CANDIDATE_PIPELINE_CONFIG)"
	@echo "Target count:      $(CANDIDATE_TARGET_COUNT)"
	@echo "Workers:           $(CANDIDATE_WORKERS)"
	@echo "Transfer workers:  $(CANDIDATE_TRANSFER_WORKERS)"
	@echo "GPU workers:       $(CANDIDATE_GPU_WORKERS)"
	@echo "Encode workers:    $(CANDIDATE_ENCODE_WORKERS)"
	@echo "Work dir:          $(CANDIDATE_WORK_DIR)"
	@echo "Source dir:        $(CANDIDATE_LOCAL_SOURCE_DIR)"
	@echo "Output dir:        $(CANDIDATE_LOCAL_OUTPUT_DIR)"
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	$(PICKUP_PUTDOWN) candidates-remote \
		--storage-config "$(CANDIDATE_STORAGE_CONFIG)" \
		--pipeline-config "$(CANDIDATE_PIPELINE_CONFIG)" \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		--workers $(CANDIDATE_WORKERS) \
		--transfer-workers $(CANDIDATE_TRANSFER_WORKERS) \
		--gpu-workers $(CANDIDATE_GPU_WORKERS) \
		--encode-workers $(CANDIDATE_ENCODE_WORKERS) \
		--work-dir "$(CANDIDATE_WORK_DIR)" \
		--local-source-dir "$(CANDIDATE_LOCAL_SOURCE_DIR)" \
		--local-output-dir "$(CANDIDATE_LOCAL_OUTPUT_DIR)" \
		$(if $(CANDIDATE_KEEP_LOCAL_FILES),--keep-local-files,) \
		$(if $(CANDIDATE_FAIL_FAST),--fail-fast,) \
		$(if $(CANDIDATE_DEFER_UPLOAD),--defer-upload,) \
		$(if $(CANDIDATE_OVERWRITE),--overwrite,) \
		$(if $(CANDIDATE_DRY_RUN),--dry-run,) \
		-v

candidates-upload: ## Upload locally staged candidates to S3
	@echo "=== Candidate Upload ==="
	@echo "Storage config:    $(CANDIDATE_STORAGE_CONFIG)"
	@echo "Output dir:        $(CANDIDATE_LOCAL_OUTPUT_DIR)"
	@echo "Target count:      $(CANDIDATE_TARGET_COUNT)"
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	$(PICKUP_PUTDOWN) candidates-upload \
		--storage-config "$(CANDIDATE_STORAGE_CONFIG)" \
		--local-output-dir "$(CANDIDATE_LOCAL_OUTPUT_DIR)" \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		-v

candidates-generate: ## Process downloaded sources locally without uploading to S3
	@echo "=== Local Candidate Generation (deferred upload) ==="
	@echo "Storage config:    $(CANDIDATE_STORAGE_CONFIG)"
	@echo "Pipeline config:   $(CANDIDATE_PIPELINE_CONFIG)"
	@echo "Target count:      $(CANDIDATE_TARGET_COUNT)"
	@echo "Workers:           $(CANDIDATE_WORKERS)"
	@echo "GPU workers:       $(CANDIDATE_GPU_WORKERS)"
	@echo "Encode workers:    $(CANDIDATE_ENCODE_WORKERS)"
	@echo "Work dir:          $(CANDIDATE_WORK_DIR)"
	@echo "Source dir:        $(CANDIDATE_LOCAL_SOURCE_DIR)"
	@echo "Output dir:        $(CANDIDATE_LOCAL_OUTPUT_DIR)"
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	$(PICKUP_PUTDOWN) candidates-remote \
		--storage-config "$(CANDIDATE_STORAGE_CONFIG)" \
		--pipeline-config "$(CANDIDATE_PIPELINE_CONFIG)" \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		--workers $(CANDIDATE_WORKERS) \
		--transfer-workers $(CANDIDATE_TRANSFER_WORKERS) \
		--gpu-workers $(CANDIDATE_GPU_WORKERS) \
		--encode-workers $(CANDIDATE_ENCODE_WORKERS) \
		--work-dir "$(CANDIDATE_WORK_DIR)" \
		--local-source-dir "$(CANDIDATE_LOCAL_SOURCE_DIR)" \
		--local-output-dir "$(CANDIDATE_LOCAL_OUTPUT_DIR)" \
		$(if $(CANDIDATE_KEEP_LOCAL_FILES),--keep-local-files,) \
		$(if $(CANDIDATE_FAIL_FAST),--fail-fast,) \
		--defer-upload \
		$(if $(CANDIDATE_OVERWRITE),--overwrite,) \
		$(if $(CANDIDATE_DRY_RUN),--dry-run,) \
		-v

candidates-process-local: ## Process downloaded sources locally (GPU parallel, CPU parallel)
	@echo "=== Local Candidate Processing ==="
	@echo "Pipeline config:   $(CANDIDATE_PIPELINE_CONFIG)"
	@echo "Target count:      $(CANDIDATE_TARGET_COUNT)"
	@echo "GPU workers:       $(CANDIDATE_GPU_WORKERS)"
	@echo "Encode workers:    $(CANDIDATE_ENCODE_WORKERS)"
	@echo "Work dir:          $(CANDIDATE_WORK_DIR)"
	@echo "Source dir:        $(CANDIDATE_LOCAL_SOURCE_DIR)"
	@echo "Output dir:        $(CANDIDATE_LOCAL_OUTPUT_DIR)"
	@$(PICKUP_PUTDOWN) candidates-process-local \
		--pipeline-config "$(CANDIDATE_PIPELINE_CONFIG)" \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		--gpu-workers $(CANDIDATE_GPU_WORKERS) \
		--encode-workers $(CANDIDATE_ENCODE_WORKERS) \
		--work-dir "$(CANDIDATE_WORK_DIR)" \
		--local-source-dir "$(CANDIDATE_LOCAL_SOURCE_DIR)" \
		--local-output-dir "$(CANDIDATE_LOCAL_OUTPUT_DIR)" \
		$(if $(CANDIDATE_KEEP_LOCAL_FILES),--keep-local-files,) \
		$(if $(CANDIDATE_OVERWRITE),--overwrite,) \
		-v

# ---------------------------------------------------------------------------
# Track A: reviewed feature dataset
# ---------------------------------------------------------------------------

TRACK_A_EVENTS_CSV ?= .local/task_7_vlm/events.csv
TRACK_A_CLIPS_CSV ?= .local/task_7_vlm/clips.csv
TRACK_A_REVIEW_MANIFEST ?= .local/task_7_review/review_manifest.csv
TRACK_A_CANDIDATE_DIR ?= .local/candidate_staging
TRACK_A_SOURCE_VIDEO_DIR ?= .local/source_videos
TRACK_A_OUTPUT_DIR ?= .local/track_a_features
TRACK_A_SPLIT_SEED ?= 42
TRACK_A_CONFIG ?= configs/proposals.yaml
TRACK_A_SHELVES_CONFIG ?= configs/shelves.yaml
TRACK_A_CAMERA_ID ?= store_camera_01

track-a-dataset: ## Build the reviewed Track A feature dataset
	@echo "=== Building Reviewed Track A Dataset ==="
	@$(PICKUP_PUTDOWN) build-track-a-dataset \
		--events-csv "$(TRACK_A_EVENTS_CSV)" \
		--clips-csv "$(TRACK_A_CLIPS_CSV)" \
		--review-manifest "$(TRACK_A_REVIEW_MANIFEST)" \
		--candidate-metadata-dir "$(TRACK_A_CANDIDATE_DIR)" \
		--source-video-dir "$(TRACK_A_SOURCE_VIDEO_DIR)" \
		--output-dir "$(TRACK_A_OUTPUT_DIR)" \
		--split-seed $(TRACK_A_SPLIT_SEED) \
		--config "$(TRACK_A_CONFIG)" \
		--shelves-config "$(TRACK_A_SHELVES_CONFIG)" \
		--camera-id "$(TRACK_A_CAMERA_ID)" \
		-v

# ---------------------------------------------------------------------------
# Track A: classifier training
# ---------------------------------------------------------------------------

TRACK_A_ARTIFACT_DIR ?= .local/track_a_artifacts
TRACK_A_CONFIG ?= configs/track_a.yaml
TRACK_A_FEATURE_MANIFEST ?= .local/track_a_features/feature_dataset.parquet
TRACK_A_CANDIDATE_METADATA ?= .local/candidate_staging/metadata
TRACK_A_CACHE_DIR ?= .local/track_a_features
TRACK_A_INFER_OUTPUT_DIR ?= .local/track_a_output
TRACK_A_CLIP_ID ?=
TRACK_A_CANDIDATE_ID ?=
TRACK_A_DEBUG_TRACES ?=
TRACK_A_FORCE ?=

# Track A evaluation (Phase 6)
TRACK_A_EVAL_SPLIT ?= val
TRACK_A_EVAL_LIMIT ?=
TRACK_A_EVAL_OUTPUT ?= .local/track_a_evaluation

evaluate-track-a: ## Run Track A evaluation: inference + Task-8 metrics + reports
	@echo "=== Track A Evaluation ==="
	@echo "Split:        $(TRACK_A_EVAL_SPLIT)"
	@echo "Output:       $(TRACK_A_EVAL_OUTPUT)"
	@TRACK_A_EVAL_EXTRA_ARGS=""; \
	if [ -n "$(TRACK_A_EVAL_LIMIT)" ]; then TRACK_A_EVAL_EXTRA_ARGS="$$TRACK_A_EVAL_EXTRA_ARGS --limit-clips $(TRACK_A_EVAL_LIMIT)"; fi; \
	if [ -n "$(TRACK_A_CLIP_ID)" ]; then TRACK_A_EVAL_EXTRA_ARGS="$$TRACK_A_EVAL_EXTRA_ARGS --clip-id $(TRACK_A_CLIP_ID)"; fi; \
	set -o pipefail; \
	$(PICKUP_PUTDOWN) evaluate-track-a \
		--config "$(TRACK_A_CONFIG)" \
		--splits "$(TRACK_A_OUTPUT_DIR)/splits.json" \
		--feature-manifest "$(TRACK_A_FEATURE_MANIFEST)" \
		--events "$(TRACK_A_EVENTS_CSV)" \
		--clips "$(TRACK_A_CLIPS_CSV)" \
		--artifact-dir "$(TRACK_A_ARTIFACT_DIR)" \
		--candidate-metadata "$(TRACK_A_CANDIDATE_METADATA)" \
		--source-video-dir "$(TRACK_A_SOURCE_VIDEO_DIR)" \
		--shelves-config "$(TRACK_A_SHELVES_CONFIG)" \
		--camera-id "$(TRACK_A_CAMERA_ID)" \
		--output-dir "$(TRACK_A_EVAL_OUTPUT)" \
		--split "$(TRACK_A_EVAL_SPLIT)" \
		$$TRACK_A_EVAL_EXTRA_ARGS \
		-v
	@echo "=== Training Track A Classifiers ==="
	@$(PICKUP_PUTDOWN) train-track-a \
		--config "$(TRACK_A_CONFIG)" \
		--feature-manifest "$(TRACK_A_FEATURE_MANIFEST)" \
		--output-dir "$(TRACK_A_ARTIFACT_DIR)" \
		-v

# ---------------------------------------------------------------------------
# Track A: inference (Phase 5)
# ---------------------------------------------------------------------------

infer-track-a: ## Run Track A inference pipeline on candidates
	@echo "=== Track A Inference ==="
	@TRACK_A_EXTRA_ARGS=""; \
	if [ -n "$(TRACK_A_CLIP_ID)" ]; then TRACK_A_EXTRA_ARGS="$$TRACK_A_EXTRA_ARGS --clip-id $(TRACK_A_CLIP_ID)"; fi; \
	if [ -n "$(TRACK_A_CANDIDATE_ID)" ]; then TRACK_A_EXTRA_ARGS="$$TRACK_A_EXTRA_ARGS --candidate-id $(TRACK_A_CANDIDATE_ID)"; fi; \
	if [ -n "$(TRACK_A_DEBUG_TRACES)" ]; then TRACK_A_EXTRA_ARGS="$$TRACK_A_EXTRA_ARGS --debug-traces"; fi; \
	if [ -n "$(TRACK_A_FORCE)" ]; then TRACK_A_EXTRA_ARGS="$$TRACK_A_EXTRA_ARGS --force"; fi; \
	set -o pipefail; \
	$(PICKUP_PUTDOWN) infer-track-a \
		--config "$(TRACK_A_CONFIG)" \
		--candidate-metadata "$(TRACK_A_CANDIDATE_METADATA)" \
		--source-video-dir "$(TRACK_A_SOURCE_VIDEO_DIR)" \
		--shelves-config "$(TRACK_A_SHELVES_CONFIG)" \
		--camera-id "$(TRACK_A_CAMERA_ID)" \
		--artifact-dir "$(TRACK_A_ARTIFACT_DIR)" \
--cache-dir "$(TRACK_A_CACHE_DIR)" \
	--output-dir "$(TRACK_A_INFER_OUTPUT_DIR)" \
	$$TRACK_A_EXTRA_ARGS \
	-v


# ---------------------------------------------------------------------------
# Track B1: VideoMAE window classifier (task_12)
#
# Ground truth is CVAT source-video annotation. Actor crop boxes come from the
# annotation tracks, so no pose-pipeline output is needed for these clips.
#
# Full path from a fresh checkout:
#   make cvat-export          # completed CVAT jobs -> S3 + .local/cvat_exports
#   make track-b1-sources     # 18 GB of 4K source video from S3
#   make track-b1-dataset     # canonical events/clips/tracks + window manifest
#   make track-b1-inspect     # Gate A: eyeball the loader before training
#   make track-b1-cache       # per-candidate crop cache (decode once, not per epoch)
#   make track-b1-embeddings  # frozen-backbone features (valid while frozen)
#   make train-track-b1-head  # Gate B + head training
#   make infer-track-b1       # sliding-window decode + Task 8 metrics
# ---------------------------------------------------------------------------

# Native arm64 interpreter: the x86 conda env runs under Rosetta, where torch is
# capped at 2.2.2 and Conv3D is unsupported on MPS.
TRACK_B1_PYTHON ?= .venv-arm64/bin/python

TRACK_B1_EXPORT_DATE ?= 2026-09-09
TRACK_B1_EXPORT_DIR ?= .local/cvat_exports/$(TRACK_B1_EXPORT_DATE)
TRACK_B1_VIDEO_DIR ?= .local/source_videos
TRACK_B1_DATASET_DIR ?= .local/track_b1_dataset
TRACK_B1_CACHE_DIR ?= .local/track_b1_cache
TRACK_B1_EMBEDDINGS_DIR ?= .local/track_b1_embeddings
TRACK_B1_RUN_DIR ?= .local/track_b1_run
TRACK_B1_MODEL ?= MCG-NJU/videomae-base
TRACK_B1_WORKERS ?= 6
TRACK_B1_SPLIT ?= val

cvat-export: ## Export completed CVAT jobs to S3 (read-only with respect to CVAT)
	@echo "=== Exporting CVAT annotations ($(TRACK_B1_EXPORT_DATE)) ==="
	@$(PYTHON) scripts/export_cvat_annotations.py --export-date "$(TRACK_B1_EXPORT_DATE)"

track-b1-sources: ## Download the annotated source videos named in the export manifest
	@echo "=== Downloading annotated source videos ==="
	@./scripts/download_annotated_sources.sh \
		"$(TRACK_B1_EXPORT_DIR)/export_manifest.csv" "$(TRACK_B1_VIDEO_DIR)"

track-b1-dataset: ## Import CVAT annotations into canonical tables + window manifest
	@echo "=== Building Track B1 dataset ==="
	@$(TRACK_B1_PYTHON) scripts/build_track_b1_dataset.py \
		--export-dir "$(TRACK_B1_EXPORT_DIR)" \
		--video-dir "$(TRACK_B1_VIDEO_DIR)" \
		--output-dir "$(TRACK_B1_DATASET_DIR)"

track-b1-inspect: ## Gate A: render sampled-frame grids from the loader
	@echo "=== Track B1 Gate A: visual loader inspection ==="
	@$(TRACK_B1_PYTHON) scripts/inspect_track_b1_windows.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--video-dir "$(TRACK_B1_VIDEO_DIR)"

track-b1-cache: ## Build the per-candidate crop cache
	@echo "=== Building Track B1 crop cache ==="
	@$(TRACK_B1_PYTHON) scripts/build_track_b1_cache.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--video-dir "$(TRACK_B1_VIDEO_DIR)" \
		--cache-dir "$(TRACK_B1_CACHE_DIR)" \
		--workers $(TRACK_B1_WORKERS)

track-b1-embeddings: ## Precompute frozen-backbone embeddings for every window
	@echo "=== Precomputing Track B1 embeddings ==="
	@$(TRACK_B1_PYTHON) scripts/precompute_track_b1_embeddings.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--cache-dir "$(TRACK_B1_CACHE_DIR)" \
		--output-dir "$(TRACK_B1_EMBEDDINGS_DIR)" \
		--model-name "$(TRACK_B1_MODEL)"

train-track-b1-head: ## Gate B + train the head on cached embeddings (fast path)
	@echo "=== Training Track B1 head ==="
	@$(TRACK_B1_PYTHON) scripts/train_track_b1_head.py \
		--embeddings-dir "$(TRACK_B1_EMBEDDINGS_DIR)" \
		--output-dir "$(TRACK_B1_RUN_DIR)"

train-track-b1: ## Train through the full pixel path (needed to unfreeze backbone blocks)
	@echo "=== Training Track B1 (full pixel path) ==="
	@$(TRACK_B1_PYTHON) scripts/train_track_b1.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--cache-dir "$(TRACK_B1_CACHE_DIR)" \
		--output-dir "$(TRACK_B1_RUN_DIR)" \
		--model-name "$(TRACK_B1_MODEL)"

infer-track-b1: ## Sliding-window inference + shared Task 8 evaluation
	@echo "=== Track B1 inference on $(TRACK_B1_SPLIT) ==="
	@$(TRACK_B1_PYTHON) scripts/infer_track_b1.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--cache-dir "$(TRACK_B1_CACHE_DIR)" \
		--checkpoint "$(TRACK_B1_RUN_DIR)/checkpoints/head_best.pt" \
		--output-dir "$(TRACK_B1_RUN_DIR)/predictions" \
		--model-name "$(TRACK_B1_MODEL)" \
		--split "$(TRACK_B1_SPLIT)"

tune-track-b1-thresholds: ## Tune decode thresholds on validation window scores only
	@echo "=== Tuning Track B1 thresholds (validation only) ==="
	@$(TRACK_B1_PYTHON) scripts/tune_track_b1_thresholds.py \
		--dataset-dir "$(TRACK_B1_DATASET_DIR)" \
		--run-dir "$(TRACK_B1_RUN_DIR)"

track-b1-results: ## Print the consolidated Track B1 metrics comparison
	@$(TRACK_B1_PYTHON) scripts/report_track_b1_results.py --output .local/track_b1_RESULTS.md

track-b1-all: track-b1-dataset track-b1-cache track-b1-embeddings train-track-b1-head infer-track-b1 ## Full Track B1 pipeline from an existing export
