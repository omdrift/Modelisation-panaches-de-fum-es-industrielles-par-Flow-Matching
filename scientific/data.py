"""Causal smoke extraction and prepared window loading; never drop interior frames."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from scientific.protocol import file_hash, partition, stratified_selection, write_json, write_jsonl, read_jsonl, digest, data_spec
from scientific.segmentation import ClipSegExtractor


def extract_smoke(frames: np.ndarray, settings: dict, extractor=None) -> tuple[np.ndarray, np.ndarray]:
    """RGB [T,H,W,3] in [0,1]; extraction uses each source image independently."""
    if extractor is None:
        extractor = ClipSegExtractor(settings)
    return extractor(frames)


def validate_indices(indices, timestamps) -> None:
    if len(indices) != 36 or not np.all(np.diff(indices) == 1):
        raise ValueError('A window must contain thirty-six consecutive source frame indices')
    dt = np.diff(timestamps)
    if not np.all(np.isfinite(timestamps)) or not np.all(dt > 0) or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-5):
        raise ValueError('Source acquisition intervals must be positive and constant')


def prepare(metadata: list[dict], video_root: Path, output: Path, cfg: dict, metadata_path: Path | None = None,
            device='auto', cache_dir=None, local_files_only=False, batch_frames=None) -> dict:
    import cv2
    from tqdm import tqdm

    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a prepared dataset: {output}')
    if metadata_path is not None and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Metadata input differs from its declared source file')
    # Load and verify one model before creating any dataset artifacts.
    extractor = ClipSegExtractor(cfg['preprocessing'], device=device, cache_dir=cache_dir, local_files_only=local_files_only, batch_frames=batch_frames)
    extraction_identity = extractor.provenance
    output.mkdir(parents=True)
    (output / 'clips').mkdir()
    records, events, tasks = [], [], []
    # Account for all metadata, but estimate time only over actual source videos.
    for item in sorted(metadata, key=lambda r: r['id']):
        split = partition(item)
        event = {'video_id': item['id'], 'split': split, 'clips': 0}
        if split.startswith('excluded'):
            event['status'] = split
            events.append(event)
            continue
        path = video_root / f"{item['id']}_{item['file_name']}.mp4"
        if not path.is_file():
            event['status'] = 'missing_video'
            events.append(event)
            continue
        tasks.append((item, split, path, event))
    print(f'CLIPSeg: device={extractor.device}, batch_frames={extractor.batch_frames}, source_clips={len(tasks)}', flush=True)
    for item, split, path, event in tqdm(tasks, desc='Preparing 36-frame clips', unit='clip'):
        capture = cv2.VideoCapture(str(path))
        expected_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        event.update(container_fps=capture.get(cv2.CAP_PROP_FPS), source_width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), source_height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        decoded = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        event['decoded_frames'] = len(decoded)
        if len(decoded) != expected_count or len(decoded) != cfg['task']['clip_frames']:
            event['status'] = 'decode_count_mismatch' if len(decoded) != expected_count else 'unexpected_clip_length'
            events.append(event)
            continue
        parts = item['file_name'].split('-')
        start, end = int(parts[-2]), int(parts[-1])
        if end <= start or int(item['start_time']) != start:
            raise ValueError(f"Inconsistent source timestamps for video {item['id']}")
        delta = (end - start) / (len(decoded) - 1)
        event.update(source_sha256=file_hash(path), source_interval_seconds=delta, unused_tail_frames=0)
        indices = list(range(len(decoded)))
        times = [start + i * delta for i in indices]
        validate_indices(indices, times)
        window_id = f"{item['id']}_0000"
        source = np.asarray(decoded, dtype=np.uint8)
        raw = np.stack([cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA) for frame in source])
        relative = f'clips/{window_id}.npz'
        try:
            smoke, alpha = extract_smoke(source.astype(np.float32) / 255, cfg['preprocessing'], extractor=extractor)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Preserve all source images even when foreground extraction fails.
            np.savez_compressed(output / relative, raw_frames=raw)
            event.update(status='segmentation_failed', raw_path=relative, reason=str(exc))
            events.append(event)
            continue
        # Segment at native resolution, then reduce RGB×mask and mask separately.
        smoke = np.stack([cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA) for frame in smoke])
        alpha = np.stack([cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA) for mask in alpha])
        # INTER_AREA may exceed 1 by one float32 ULP even for constant masks.
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        np.savez_compressed(output / relative, raw_frames=raw, frames=np.round(np.clip(smoke, 0, 1) * 255).astype(np.uint8), alpha=alpha)
        records.append(dict(window_id=window_id, video_id=item['id'], split=split, date='-'.join(parts[2:5]), camera_id=item['camera_id'], view_id=item['view_id'], frame_count=len(decoded), frame_indices=indices, timestamps=times, timestamp_basis='linear_source_endpoints_not_playback_fps', path=relative, sha256=file_hash(output / relative), source_sha256=event['source_sha256'], extraction_sha256=digest(extraction_identity), source_resolution=list(source.shape[1:3])))
        event.update(status='prepared', clips=1)
        events.append(event)
    selection = {}
    for split in ('train', 'val', 'test_dates', 'test_views'):
        subset = [r for r in records if r['split'] == split]
        count = len(subset) if split == 'train' else cfg['selection']['validation_windows' if split == 'val' else 'test_windows_per_split']
        selection[split] = stratified_selection(subset, count, cfg['selection_seed'])
        write_jsonl(output / f'{split}.jsonl', selection[split])
    write_jsonl(output / 'all_windows.jsonl', records)
    write_jsonl(output / 'preparation_events.jsonl', events)
    summary = dict(protocol=cfg, protocol_sha256=digest(cfg), data_spec_sha256=digest(data_spec(cfg)), extraction=extraction_identity, metadata_content_sha256=digest(metadata), metadata_file_sha256=file_hash(metadata_path) if metadata_path is not None else None, statuses=dict(Counter(e['status'] for e in events)), eligible_metadata=sum(not partition(i).startswith('excluded_label') for i in metadata), selected={s: len(v) for s, v in selection.items()}, manifests={s: file_hash(output / f'{s}.jsonl') for s in selection}, acquisition_timing='Derived from metadata endpoints and decoded frame count; verify against source before interpreting seconds.')
    write_json(output / 'dataset_manifest.json', summary)
    return summary


class WindowDataset(Dataset):
    """One complete 36-frame clip per sample; class name retained for callers."""
    def __init__(self, root, split: str, verify: bool = True):
        self.root = Path(root)
        self.clip_frames = 36
        self.records = read_jsonl(self.root / f'{split}.jsonl')
        if not self.records:
            raise ValueError(f'No prepared windows for {split}')
        manifest = json.loads((self.root / 'dataset_manifest.json').read_text())
        if manifest['protocol'].get('version') != '1.1' or manifest['protocol']['task'].get('clip_frames') != self.clip_frames:
            raise ValueError('Reprepare the dataset as complete 36-frame clips (protocol v1.1)')
        if file_hash(self.root / f'{split}.jsonl') != manifest['manifests'][split]:
            raise ValueError(f'Manifest changed after preparation: {split}')
        for record in self.records:
            if 'extraction_sha256' in record and record['extraction_sha256'] != digest(manifest.get('extraction')):
                raise ValueError('Window extraction provenance differs from the dataset manifest')
            validate_indices(record['frame_indices'], record['timestamps'])
            if verify and file_hash(self.root / record['path']) != record['sha256']:
                raise ValueError(f"Window checksum mismatch: {record['window_id']}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(self.root / record['path'], allow_pickle=False) as pack:
            if pack['raw_frames'].shape != (36, 64, 64, 3) or pack['raw_frames'].dtype != np.uint8:
                raise ValueError(f"The complete source RGB clip is missing or malformed: {record['window_id']}")
            if pack['frames'].dtype != np.uint8:
                raise ValueError(f"Processed RGB frames must be uint8: {record['window_id']}")
            frames = torch.from_numpy(pack['frames'].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1
            alpha = torch.from_numpy(pack['alpha'].copy()).unsqueeze(1)
        if frames.shape != (36, 3, 64, 64) or alpha.shape != (36, 1, 64, 64):
            raise ValueError(f"Malformed window: {record['window_id']}")
        if not torch.isfinite(alpha).all() or torch.any(alpha < 0) or torch.any(alpha > 1):
            raise ValueError(f"Invalid alpha values: {record['window_id']}")
        return {'frames': frames, 'alpha': alpha, 'window_id': record['window_id'], 'date': record['date']}
