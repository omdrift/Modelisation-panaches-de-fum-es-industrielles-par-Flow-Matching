"""Content leakage checks on immutable prepared clips, with held-out isolation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scientific.data import WindowDataset, validate_indices
from scientific.protocol import file_hash, read_jsonl, validate_prepared_data


class DatasetLeakageError(ValueError):
    """An invalid split, with the evidence retained for an audit report."""

    def __init__(self, report):
        self.report = report
        super().__init__('Exact raw-frame duplicate content occurs across dataset splits')


def _raw_fingerprint(raw):
    identity = f'{raw.dtype.str}:{raw.shape}'.encode()
    return hashlib.sha256(identity + raw.tobytes(order='C')).hexdigest()


def _thumbnail(raw):
    # Four temporal bins and a 4x4 RGB grid retain broad video appearance.
    # This is a screening representation, not a perceptual equivalence claim.
    return (raw.reshape(4, 9, 4, 16, 4, 16, 3).mean((1, 3, 5)) / 255).astype(np.float32).reshape(-1)


def _pair(left, right, **extra):
    return dict(left_split=left['split'], left_window_id=left['window_id'],
                right_split=right['split'], right_window_id=right['window_id'], **extra)


def _check_metadata(root, manifest):
    seen_ids, seen_sources, records = {}, {}, {}
    for split in ('train', 'val', 'test_dates', 'test_views'):
        if split not in manifest['manifests']:
            continue
        path = root / f'{split}.jsonl'
        if file_hash(path) != manifest['manifests'][split]:
            raise ValueError(f'Manifest changed after preparation: {split}')
        records[split] = read_jsonl(path)
        for record in records[split]:
            if record['split'] != split:
                raise ValueError('Window declares a different split from its manifest')
            window_id = record['window_id']
            if window_id in seen_ids:
                raise ValueError(f'Duplicate window identifier in split manifests: {window_id}')
            seen_ids[window_id] = split
            source = record.get('video_id')
            if source in seen_sources and seen_sources[source] != split:
                raise ValueError(f'Source video occurs in multiple splits: {source}')
            if source is not None:
                seen_sources[source] = split
            if record.get('frame_count') != 36:
                raise ValueError('Prepared window must preserve all thirty-six source frames')
            validate_indices(record['frame_indices'], record['timestamps'])
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Prepared window path escapes the dataset directory')
    # Test dates may overlap between the two test partitions, never with train/val.
    dates = {split: {item['date'] for item in items} for split, items in records.items()}
    if dates.get('train', set()) & dates.get('val', set()):
        raise ValueError('Train and validation dates overlap')
    test_dates = dates.get('test_dates', set()) | dates.get('test_views', set())
    if test_dates & (dates.get('train', set()) | dates.get('val', set())):
        raise ValueError('Held-out test dates overlap with training or validation')
    training_views = {(r['camera_id'], r['view_id']) for split in ('train', 'val') for r in records.get(split, [])}
    unseen_views = {(r['camera_id'], r['view_id']) for r in records.get('test_views', [])}
    if training_views & unseen_views:
        raise ValueError('Reserved test views occur in training')
    return records


def audit_prepared_data(data, cfg, *, include_test=False, final_selection_frozen=False,
                        near_mse_threshold=0.0025, max_near_pairs=1000):
    """Verify train/val content before search; inspect test pixels only after freeze.

    Exact duplicate RGB sequences across inspected splits reject the study.
    Thumbnail similarities are reported for investigation; partitions and source
    clips are never changed. Zero-valued smoke extractions are not compared:
    content fingerprints use the preserved, unmasked ``raw_frames`` instead.
    """
    if include_test and not final_selection_frozen:
        raise ValueError('Test pixel audit requires frozen final selection')
    if not np.isfinite(near_mse_threshold) or near_mse_threshold < 0 or type(max_near_pairs) is not int or max_near_pairs < 0:
        raise ValueError('Near-duplicate screening parameters must be finite and nonnegative')
    root = Path(data)
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    validate_prepared_data(manifest, cfg)
    metadata = _check_metadata(root, manifest)
    splits = ['train', 'val'] + (['test_dates', 'test_views'] if include_test else [])
    report = dict(dataset_manifest_sha256=file_hash(root / 'dataset_manifest.json'),
                  audited_pixel_splits=splits, counts={}, raw_fingerprints={}, exact_duplicate_pairs=[],
                  within_split_duplicate_pairs=[],
                  near_duplicate_pairs=[], near_duplicate_pair_count=0, near_pairs_truncated=False,
                  near_mse_threshold=float(near_mse_threshold),
                  near_representation='4 temporal bins x 4x4 RGB mean; MSE in [0,1]',
                  timing_validation='stored indices and timestamps; original videos not required or revalidated')
    signatures = defaultdict(list)
    features = {}
    for split in splits:
        if split not in metadata:
            raise ValueError(f'Required prepared split is missing: {split}')
        dataset = WindowDataset(root, split)
        report['counts'][split] = len(dataset)
        split_features = []
        report['raw_fingerprints'][split] = {}
        for index, record in enumerate(dataset.records):
            # Reuse frame, mask and continuity validation before hashing raw RGB.
            dataset[index]
            with np.load(root / record['path'], allow_pickle=False) as package:
                raw = package['raw_frames']
                signature = _raw_fingerprint(raw)
                split_features.append(_thumbnail(raw))
            report['raw_fingerprints'][split][record['window_id']] = signature
            for previous in signatures[signature]:
                if previous['split'] != split:
                    report['exact_duplicate_pairs'].append(_pair(previous, record, raw_sha256=signature))
                else:
                    report['within_split_duplicate_pairs'].append(_pair(previous, record, raw_sha256=signature))
            signatures[signature].append(record)
        features[split] = np.asarray(split_features)
    for position, left_split in enumerate(splits):
        left = features[left_split]
        for right_split in splits[position + 1:]:
            right = features[right_split]
            # Keep temporary pairwise arrays bounded even for larger snapshots.
            right_norm = np.einsum('ij,ij->i', right, right)
            for start in range(0, len(left), 128):
                block = left[start:start + 128]
                distances = (np.einsum('ij,ij->i', block, block)[:, None] + right_norm[None, :]
                             - 2 * block @ right.T) / left.shape[1]
                for left_index, right_index in np.argwhere(distances <= near_mse_threshold):
                    left_record = metadata[left_split][start + left_index]
                    right_record = metadata[right_split][right_index]
                    if report['raw_fingerprints'][left_split][left_record['window_id']] == report['raw_fingerprints'][right_split][right_record['window_id']]:
                        continue
                    report['near_duplicate_pair_count'] += 1
                    if len(report['near_duplicate_pairs']) < max_near_pairs:
                        report['near_duplicate_pairs'].append(_pair(left_record, right_record,
                            thumbnail_mse=max(0.0, float(distances[left_index, right_index]))))
    report['near_pairs_truncated'] = report['near_duplicate_pair_count'] > len(report['near_duplicate_pairs'])
    if report['exact_duplicate_pairs']:
        raise DatasetLeakageError(report)
    return report
