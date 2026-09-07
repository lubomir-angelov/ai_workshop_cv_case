# Annotation (CVAT)

Human annotation runs in **CVAT online** on full source videos.

- Setup and workflow: [`docs/CVAT_ANNOTATION_SETUP.md`](../docs/CVAT_ANNOTATION_SETUP.md)
- Labeling rules: [`docs/LABELING_GUIDELINES.md`](../docs/LABELING_GUIDELINES.md)
- Export conversion: `python scripts/convert_cvat_source_export.py --cvat-export annotation/cvat_export_source.xml`

Generated outputs land in `annotation/exports/` (`events.csv`,
`ignore_intervals.parquet`, `event_provenance.parquet`).
