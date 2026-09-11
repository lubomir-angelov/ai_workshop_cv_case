"""Evaluate a locally trained Track B1 checkpoint and export review clips.

Run from the repository root with its Python environment. Uses the optimized
TrackB1Dataset (cache_dir support). Does not train or modify annotations.
Only load checkpoints you created/trust: torch.load uses pickle here because
this project's saved metrics can contain NumPy scalars.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from pickup_putdown.layer1.track_b1.dataset import (
    WindowConfig, TrackB1Dataset, build_window_manifest,
)
from pickup_putdown.layer1.track_b1.videomae_classifier import VideoMAEClassifier

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('.local/track_b1_output_human/checkpoints/best_model.pt'))
    parser.add_argument('--data-dir', type=Path, default=Path('.local/track_b1_data'))
    parser.add_argument('--video-dir', type=Path, default=Path('.local/source_videos'))
    parser.add_argument('--shelves', type=Path, default=Path('configs/shelves.yaml'))
    parser.add_argument('--config-path', type=Path,
                        help='Use the same YAML passed to training; omit if training used defaults.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('.local/track_b1_output_human/diagnostics'))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--review-count', type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.workers < 0 or args.batch_size < 1 or args.review_count < 0:
        parser.error('workers/review-count must be nonnegative and batch-size positive')
    if args.review_count and shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required for previews; use --review-count 0 for metrics only')
    cfg = yaml.safe_load(args.config_path.read_text()) or {} if args.config_path else {}
    # Match train.main exactly: other WindowConfig fields retain their defaults.
    window = WindowConfig(window_duration_s=cfg.get('window_duration_s', 2.5),
                          window_stride_s=cfg.get('window_stride_s', .5),
                          num_frames=cfg.get('num_frames', 16))
    data = args.data_dir
    clips = pd.read_csv(data / 'clips.csv')
    ignore_path = data / 'ignore_intervals.parquet'
    ignores = (pd.read_parquet(ignore_path) if ignore_path.exists() else
               pd.DataFrame(columns=['clip_id', 't_start', 't_end', 'reason']))
    if not ignore_path.exists():
        LOG.warning('No ignore_intervals.parquet found; verify this matches training')
    manifest = build_window_manifest(
        candidates_df=pd.read_parquet(data / 'candidates.parquet'),
        events_df=pd.read_csv(data / 'events_human.csv'), ignore_intervals_df=ignores,
        clips_df=clips, config=window, split='val')
    if manifest.empty:
        raise ValueError('Validation manifest is empty')
    shelves = yaml.safe_load(args.shelves.read_text())
    dataset = TrackB1Dataset(
        manifest, args.video_dir, data / 'pose_tracks',
        {r['region_id']: r for r in shelves.get('regions', [])}, window,
        clips_df=clips,
        cache_dir=Path(cfg.get('frame_cache_dir', '.local/track_b1_frame_cache'))
                  if cfg.get('cache_frames', True) else None)
    # CPU load avoids allocating optimizer tensors in GPU memory.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    mc = saved['model_config']
    model = VideoMAEClassifier(
        model_name=mc['model_name'], num_classes=mc['num_classes'],
        freeze_backbone=mc['freeze_backbone'],
        unfreeze_last_n_blocks=mc['unfreeze_last_n_blocks'])
    model.load_state_dict(saved['model_state_dict'], strict=True)
    expected_metrics = saved.get('metrics', {})
    checkpoint_epoch = saved.get('epoch')
    del saved
    model.to(args.device).eval()
    options = {}
    if args.workers:
        options = dict(multiprocessing_context='spawn', prefetch_factor=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'),
                        **options)
    outputs = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            logits = model(batch['pixel_values'].to(args.device, non_blocking=True))
            outputs.append(logits.softmax(-1).cpu().numpy())
            if step == 1 or step % 20 == 0 or step == len(loader):
                LOG.info('Validation batch %d/%d', step, len(loader))
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
    report = classification_report(result.true_id, result.pred_id, labels=[0, 1, 2],
                                   target_names=NAMES, output_dict=True, zero_division=0)
    report['checkpoint_epoch'] = checkpoint_epoch
    report['window_count'] = len(result)
    report['accuracy'] = float((result.true_id == result.pred_id).mean())
    report['macro_f1_difference_from_checkpoint'] = (
        float(report['macro avg']['f1-score'] - expected_metrics['f1_macro'])
        if 'f1_macro' in expected_metrics else None)
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
    LOG.info('Done: %s | accuracy=%.4f macro F1=%.4f | review clips=%d',
             out, report['accuracy'], report['macro avg']['f1-score'], len(selected))
    delta = report['macro_f1_difference_from_checkpoint']
    if delta is not None and abs(delta) > 1e-4:
        LOG.warning('Metrics differ from checkpoint by %.6f F1; check matching YAML/data/split', delta)


if __name__ == '__main__':
    main()
