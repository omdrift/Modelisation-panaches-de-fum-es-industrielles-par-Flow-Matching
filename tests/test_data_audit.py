"""Training audits can inspect split metadata without opening held-out pixels."""
import json

import numpy as np
import pytest

from scientific.data_audit import DatasetLeakageError, audit_prepared_data
from scientific.protocol import data_spec, digest, file_hash, load_protocol, write_json, write_jsonl


def prepared(root, values):
    cfg = load_protocol()
    (root / 'clips').mkdir()
    manifests = {}
    for index, (split, value) in enumerate(values.items()):
        relative = f'clips/{split}.npz'
        rgb = np.full((36, 64, 64, 3), value, np.uint8)
        np.savez_compressed(root / relative, raw_frames=rgb, frames=rgb, alpha=np.ones((36, 64, 64), np.float32))
        record = dict(window_id=split, video_id=index, split=split, date=f'2019-01-{index + 1:02}',
                      camera_id=0, view_id=1 if split == 'test_views' else 0, frame_count=36,
                      frame_indices=list(range(36)), timestamps=list(range(36)), path=relative,
                      sha256=file_hash(root / relative))
        write_jsonl(root / f'{split}.jsonl', [record])
        manifests[split] = file_hash(root / f'{split}.jsonl')
    write_json(root / 'dataset_manifest.json', dict(protocol=cfg, data_spec_sha256=digest(data_spec(cfg)), manifests=manifests))
    return cfg


def test_training_audit_never_reads_test_npz_and_reports_similarity(tmp_path):
    cfg = prepared(tmp_path, dict(train=100, val=101, test_dates=100, test_views=100))
    (tmp_path / 'clips/test_dates.npz').unlink()
    (tmp_path / 'clips/test_views.npz').unlink()
    report = audit_prepared_data(tmp_path, cfg)
    assert report['audited_pixel_splits'] == ['train', 'val']
    assert report['counts'] == dict(train=1, val=1)
    assert report['near_duplicate_pairs'][0]['left_split'] == 'train'
    assert report['near_duplicate_pairs'][0]['right_split'] == 'val'
    assert report['near_duplicate_pairs'][0]['thumbnail_mse'] < .001


def test_exact_cross_split_raw_duplicate_rejects_with_report(tmp_path):
    cfg = prepared(tmp_path, dict(train=100, val=100))
    with pytest.raises(DatasetLeakageError, match='duplicate') as error:
        audit_prepared_data(tmp_path, cfg)
    assert error.value.report['exact_duplicate_pairs'][0]['left_window_id'] == 'train'


def test_test_pixel_audit_requires_frozen_final_selection(tmp_path):
    cfg = prepared(tmp_path, dict(train=30, val=100, test_dates=160, test_views=220))
    with pytest.raises(ValueError, match='frozen'):
        audit_prepared_data(tmp_path, cfg, include_test=True)
    report = audit_prepared_data(tmp_path, cfg, include_test=True, final_selection_frozen=True)
    assert report['audited_pixel_splits'] == ['train', 'val', 'test_dates', 'test_views']


def test_temporal_discontinuity_and_changed_array_fail_before_training(tmp_path):
    cfg = prepared(tmp_path, dict(train=30, val=100))
    record = json.loads((tmp_path / 'val.jsonl').read_text())
    record['frame_indices'][3] = 8
    write_jsonl(tmp_path / 'val.jsonl', [record])
    manifest = json.loads((tmp_path / 'dataset_manifest.json').read_text())
    manifest['manifests']['val'] = file_hash(tmp_path / 'val.jsonl')
    write_json(tmp_path / 'dataset_manifest.json', manifest)
    with pytest.raises(ValueError, match='consecutive'):
        audit_prepared_data(tmp_path, cfg)
