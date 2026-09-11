"""Window-level diagnostics for a Track B1 checkpoint, plus review clips.

    # a dataset built by build_track_b1_dataset.py (either input mode)
    python scripts/diagnose_track_b1.py --dataset-dir .local/track_b1_dataset_deploy \
        --checkpoint .local/track_b1_run/checkpoints/head_best.pt --output-dir .local/diag

    # the historical frozen-head baseline, rebuilt exactly as it was trained
    python scripts/diagnose_track_b1.py --legacy-data-dir .local/track_b1_data \
        --checkpoint .local/track_b1_output_human/checkpoints/best_model.pt \
        --evidence-dataset-dir .local/track_b1_dataset_deploy --output-dir .local/diag_baseline

Writes per-window predictions (val_predictions.csv), the confusion matrix (rows =
true, columns = predicted), per-class metrics, and a review manifest with preview
videos of the actual model input. When CVAT boxes are available each event window
also gets evidence columns that separate a transfer cropped out of view
(``event_box_in_crop``) from one that fell between sampled frames
(``sampled_frames_in_event``), and ``crop_is_full_frame`` flags a missing actor box.

Does not train or modify annotations. Only load checkpoints you created/trust:
torch.load uses pickle because saved metrics can contain NumPy scalars.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pickup_putdown.evaluation.intervals import tiou  # noqa: E402
from pickup_putdown.layer1.track_b1.actor_association import window_evidence  # noqa: E402
from pickup_putdown.layer1.track_b1.dataset import (  # noqa: E402
    TrackB1Dataset,
    WindowConfig,
    build_window_manifest,
    compute_actor_crop_box,
    crop_span,
    effective_shelf_region,
    load_shelf_regions,
)
from pickup_putdown.layer1.track_b1.dataset_dir import (  # noqa: E402
    load_dataset_dir,
    open_window_dataset,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import (  # noqa: E402
    ClassificationHead,
    VideoMAEClassifier,
)

NAMES = ['background', 'pickup', 'putdown']
LOG = logging.getLogger('diagnose_track_b1')


def select_review(df, limit=12):
    """Round-robin error groups; favor confident errors and distinct clips.

    This is a diagnostic sample, not a representative performance estimate.
    """
    groups = {
        'missed_pickup': df[(df.true_id == 1) & (df.pred_id == 0)],
        'missed_putdown': df[(df.true_id == 2) & (df.pred_id == 0)],
        'false_event': df[(df.true_id == 0) & (df.pred_id != 0)],
        'event_confusion': df[(df.true_id != 0) & (df.pred_id != 0)
                              & (df.true_id != df.pred_id)],
    }
    groups = {name: list(rows.sort_values('confidence', ascending=False).index)
              for name, rows in groups.items()}
    selected, used_clips = {}, set()
    for distinct_clips in (True, False):
        while len(selected) < limit:
            added = False
            for name, indices in groups.items():
                for idx in indices:
                    if idx in selected:
                        continue
                    if distinct_clips and df.loc[idx, 'clip_id'] in used_clips:
                        continue
                    selected[idx] = name
                    used_clips.add(df.loc[idx, 'clip_id'])
                    added = True
                    break
                if len(selected) >= limit:
                    break
            if not added:
                break
    result = df.loc[list(selected)].copy()
    result['error_group'] = list(selected.values())
    return result


def write_preview(pixel_values, path, duration):
    """Export the actual sampled model input, after inverse normalization."""
    frames = pixel_values.detach().cpu().permute(0, 2, 3, 1).numpy()
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    frames = np.rint(np.clip(frames * std + mean, 0, 1) * 255).astype(np.uint8)
    count, height, width, _ = frames.shape
    subprocess.run([
        'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(count / max(duration, .1)),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path),
    ], input=frames.tobytes(), check=True)


def legacy_dataset(args):
    """The pose-route baseline layout, with the manifest rebuilt as train.main did.

    Clip-level labels from events_human.csv, WindowConfig defaults unless the same
    flat YAML passed to training is given. Used to reproduce historical metrics.
    """
    cfg = (yaml.safe_load(args.config_path.read_text()) or {}) if args.config_path else {}
    window_keys = ('window_duration_s', 'window_stride_s', 'num_frames', 'image_size',
                   'crop_margin', 'crop_scope', 'resize_interpolation', 'include_shelf_region')
    window = WindowConfig(**{k: cfg[k] for k in window_keys if k in cfg})
    data = args.legacy_data_dir
    clips = pd.read_csv(data / 'clips.csv')
    ignore_path = data / 'ignore_intervals.parquet'
    ignores = (pd.read_parquet(ignore_path) if ignore_path.exists() else
               pd.DataFrame(columns=['clip_id', 't_start', 't_end', 'reason']))
    if not ignore_path.exists():
        LOG.warning('No ignore_intervals.parquet found; verify this matches training')
    manifest = build_window_manifest(
        candidates_df=pd.read_parquet(data / 'candidates.parquet'),
        events_df=pd.read_csv(data / 'events_human.csv'), ignore_intervals_df=ignores,
        clips_df=clips, config=window, split=args.split)
    frame_cache = (Path(cfg.get('frame_cache_dir', args.frame_cache_dir))
                   if cfg.get('cache_frames', True) else None)
    dataset = TrackB1Dataset(
        manifest, args.video_dir, data / 'pose_tracks',
        load_shelf_regions(args.shelves), window,
        clips_df=clips, cache_dir=frame_cache)
    return dataset, window, 'legacy_pose_clip_level'


def load_model(checkpoint_path: Path, model_name: str):
    """Full-model or head-only checkpoint, loaded strictly; returns (model, record)."""
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'head_state_dict' in saved:
        model = VideoMAEClassifier(model_name=model_name, num_classes=3, freeze_backbone=True)
        model.head = ClassificationHead(saved['hidden_dim'], 3, dropout=saved.get('dropout', 0.1))
        model.head.load_state_dict(saved['head_state_dict'], strict=True)
        return model, {'epoch': saved.get('epoch'), 'f1_macro': saved.get('val_f1_macro')}
    mc = saved['model_config']
    model = VideoMAEClassifier(
        model_name=mc['model_name'], num_classes=mc['num_classes'],
        freeze_backbone=mc['freeze_backbone'],
        unfreeze_last_n_blocks=mc['unfreeze_last_n_blocks'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model, {'epoch': saved.get('epoch'), 'f1_macro': saved.get('metrics', {}).get('f1_macro')}


def cvat_event_lookup(evidence_dir, legacy_events=None):
    """event_id -> CVAT event row (with ``cvat_actor_id``) for windows' event ids.

    Legacy baseline events come from the older converter and carry other ids; each is
    mapped to the same-clip, same-type CVAT event with the highest temporal IoU (>= 0.5).
    """
    events = (evidence_dir.table('events_pose_identity') if evidence_dir.input_mode == 'deployment'
              else evidence_dir.table('events').assign(cvat_actor_id=lambda e: e['actor_id']))
    lookup = {row.event_id: row for _, row in events.iterrows()}
    if legacy_events is None:
        return lookup
    mapped = {}
    for _, legacy in legacy_events.iterrows():
        same = events[(events.clip_id == legacy.clip_id) & (events.type == legacy.type)]
        scored = [(tiou(legacy, cvat), cvat.event_id) for _, cvat in same.iterrows()]
        best = max(scored, default=(0.0, None))
        if best[0] >= 0.5:
            mapped[legacy.event_id] = lookup[best[1]]
    return mapped


def add_evidence(result, window, tracks_dir, shelf_regions, evidence_dir, lookup):
    """Crop/sampling evidence per event window from CVAT boxes (never model inputs)."""
    clips = evidence_dir.table('clips').set_index('clip_id')
    cvat_dir = evidence_dir.path / ('cvat_tracks' if evidence_dir.input_mode == 'deployment'
                                    else 'actor_tracks')
    crop_tracks, cvat_tracks = {}, {}
    columns = {k: [] for k in ('matched_cvat_event_id', 'event_box_in_crop',
                               'sampled_frames_in_event', 'crop_is_full_frame')}
    for _, row in result.iterrows():
        clip = clips.loc[row.clip_id]
        if row.clip_id not in crop_tracks:
            crop_tracks[row.clip_id] = pd.read_parquet(tracks_dir / f'{row.clip_id}.parquet')
            cvat_tracks[row.clip_id] = pd.read_parquet(cvat_dir / f'{row.clip_id}.parquet')
        actor = crop_tracks[row.clip_id]
        actor = actor[actor.actor_id == row.actor_id]
        shelf = effective_shelf_region(shelf_regions, row.region_id, window)
        frame_size = (int(clip.width), int(clip.height))
        crop = compute_actor_crop_box(actor, shelf, *crop_span(row, window), window.crop_margin,
                                      frame_size)
        event = lookup.get(row.event_id) if isinstance(row.event_id, str) else None
        if event is not None:
            event = event.copy()
            event['actor_id'] = event['cvat_actor_id']
        evidence = window_evidence(row.window_start_s, row.window_end_s, window.num_frames,
                                   float(clip.fps), crop, event, cvat_tracks[row.clip_id])
        columns['matched_cvat_event_id'].append(None if event is None else event['event_id'])
        columns['event_box_in_crop'].append(evidence['event_box_in_crop'])
        columns['sampled_frames_in_event'].append(evidence['sampled_frames_in_event'])
        columns['crop_is_full_frame'].append(tuple(crop) == (0, 0, *frame_size))
    for name, values in columns.items():
        result[name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-dir', type=Path, help='built dataset (either input mode)')
    source.add_argument('--legacy-data-dir', type=Path,
                        help='pose-route baseline layout (candidates.parquet, events_human.csv, clips.csv)')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-name', default='MCG-NJU/videomae-base',
                        help='encoder for head-only checkpoints')
    parser.add_argument('--split', default='val')
    parser.add_argument('--video-dir', type=Path, default=Path('.local/source_videos'))
    parser.add_argument('--cache-dir', type=Path, default=Path('.local/track_b1_cache'))
    parser.add_argument('--frame-cache-dir', type=Path, default=Path('.local/track_b1_frame_cache'))
    parser.add_argument('--shelves', type=Path, default=Path('configs/shelves.yaml'))
    parser.add_argument('--config-path', type=Path,
                        help='legacy: the flat YAML passed to training; omit if training used defaults.')
    parser.add_argument('--evidence-dataset-dir', type=Path,
                        help='dataset dir supplying CVAT boxes for evidence columns '
                             '(default: --dataset-dir itself)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--review-count', type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.workers < 0 or args.batch_size < 1 or args.review_count < 0:
        parser.error('workers/review-count must be nonnegative and batch-size positive')
    if args.review_count and shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required for previews; use --review-count 0 for metrics only')

    legacy_events = None
    if args.legacy_data_dir:
        dataset, window, input_mode = legacy_dataset(args)
        tracks_dir, shelf_regions = args.legacy_data_dir / 'pose_tracks', dataset.shelf_regions
        legacy_events = pd.read_csv(args.legacy_data_dir / 'events_human.csv')
    else:
        built = load_dataset_dir(args.dataset_dir)
        manifest = built.table('window_manifest')
        manifest = manifest[manifest['split'] == args.split].reset_index(drop=True)
        dataset = open_window_dataset(built, manifest, args.video_dir, cache_dir=args.cache_dir,
                                      frame_cache_dir=args.frame_cache_dir)
        window, input_mode = built.window_config(), built.input_mode
        tracks_dir, shelf_regions = built.tracks_dir, built.shelf_regions()
    if len(dataset) == 0:
        raise ValueError(f'{args.split} manifest is empty')

    model, expected = load_model(args.checkpoint, args.model_name)
    model.to(args.device).eval()
    options = {}
    if args.workers:
        options = {'multiprocessing_context': 'spawn', 'prefetch_factor': 1}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                        **options)
    outputs = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            logits = model(batch['pixel_values'].to(args.device, non_blocking=True))
            outputs.append(logits.softmax(-1).cpu().numpy())
            if step == 1 or step % 20 == 0 or step == len(loader):
                LOG.info('Batch %d/%d', step, len(loader))
    probabilities = np.concatenate(outputs)
    result = dataset.manifest.copy()
    result.insert(0, 'dataset_index', np.arange(len(result)))
    result['true_id'] = result['label'].astype(int)
    result['pred_id'] = probabilities.argmax(axis=1)
    result['true_label'] = result.true_id.map(dict(enumerate(NAMES)))
    result['pred_label'] = result.pred_id.map(dict(enumerate(NAMES)))
    result['confidence'] = probabilities.max(axis=1)
    for i, name in enumerate(NAMES):
        result[f'p_{name}'] = probabilities[:, i]

    evidence_path = args.evidence_dataset_dir or args.dataset_dir
    if evidence_path is not None:
        evidence_dir = load_dataset_dir(evidence_path)
        result = add_evidence(result, window, tracks_dir, shelf_regions, evidence_dir,
                              cvat_event_lookup(evidence_dir, legacy_events))

    report = classification_report(result.true_id, result.pred_id, labels=[0, 1, 2],
                                   target_names=NAMES, output_dict=True, zero_division=0)
    report['input_mode'] = input_mode
    report['checkpoint'] = str(args.checkpoint)
    report['checkpoint_epoch'] = expected.get('epoch')
    report['window_count'] = len(result)
    report['accuracy'] = float((result.true_id == result.pred_id).mean())
    report['always_background_accuracy'] = float((result.true_id == 0).mean())
    report['macro_f1_difference_from_checkpoint'] = (
        float(report['macro avg']['f1-score'] - expected['f1_macro'])
        if expected.get('f1_macro') is not None else None)
    matrix = confusion_matrix(result.true_id, result.pred_id, labels=[0, 1, 2])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'val_predictions.csv', index=False)
    pd.DataFrame(matrix, index=pd.Index(NAMES, name='true_label'),
                 columns=[f'pred_{x}' for x in NAMES]).to_csv(out / 'confusion_matrix.csv')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    selected = select_review(result, args.review_count)
    review_dir = out / 'review_clips'
    review_dir.mkdir(exist_ok=True)
    selected['preview_path'] = ''
    selected['review_status'] = 'pending'
    selected['review_notes'] = ''
    for idx, row in selected.iterrows():
        path = review_dir / f"{int(row.dataset_index):05d}_{row.true_label}_to_{row.pred_label}.mp4"
        item = dataset[int(row.dataset_index)]
        write_preview(item['pixel_values'], path, row.window_end_s - row.window_start_s)
        selected.loc[idx, 'preview_path'] = str(path)
    selected.to_csv(out / 'review_manifest.csv', index=False)
    LOG.info('Done [%s]: %s | accuracy=%.4f macro F1=%.4f | review clips=%d', input_mode,
             out, report['accuracy'], report['macro avg']['f1-score'], len(selected))
    delta = report['macro_f1_difference_from_checkpoint']
    if delta is not None and abs(delta) > 1e-4:
        LOG.warning('Metrics differ from checkpoint by %.6f F1; check matching YAML/data/split', delta)


if __name__ == '__main__':
    main()
