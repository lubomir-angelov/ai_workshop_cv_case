"""Annotation workflow: schemas, import/export, and Label Studio integration."""

from __future__ import annotations

from pickup_putdown.annotation.import_export import (
    build_candidate_tasks,
    build_label_studio_tasks,
    convert_candidates_to_predictions,
    export_candidate_annotations,
    export_events_csv,
    export_ignore_intervals_parquet,
    validate_candidate_metadata,
    validate_export,
)
from pickup_putdown.annotation.schemas import (
    AnnotationEvent,
    AnnotationRegion,
    CandidateMetadata,
    CandidateValidationError,
    CanonicalEvent,
    ConversionResult,
    IgnoreIntervalExport,
    LabelStudioPrediction,
    LabelStudioTask,
    ReviewMetadata,
    ValidationErrors,
)

__all__ = [
    "AnnotationEvent",
    "AnnotationRegion",
    "CandidateMetadata",
    "CandidateValidationError",
    "CanonicalEvent",
    "ConversionResult",
    "IgnoreIntervalExport",
    "LabelStudioTask",
    "LabelStudioPrediction",
    "ReviewMetadata",
    "ValidationErrors",
    "build_candidate_tasks",
    "build_label_studio_tasks",
    "convert_candidates_to_predictions",
    "export_candidate_annotations",
    "export_events_csv",
    "export_ignore_intervals_parquet",
    "validate_candidate_metadata",
    "validate_export",
]
