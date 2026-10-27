"""Automatic data checks must remain effective after removing human approval."""
import numpy as np
import pytest

from scientific.cli import check_data
from scientific.protocol import data_spec, digest, file_hash, load_protocol, write_json, write_jsonl


@pytest.mark.parametrize('defect,message', [
    ('nan', 'Invalid alpha'),
    ('range', 'Invalid alpha'),
    ('shape', 'Malformed window'),
    ('rgb_dtype', 'RGB frames must be uint8'),
])
def test_automatic_check_reads_and_rejects_invalid_arrays(tmp_path, defect, message):
    cfg = load_protocol()
    manifests = {}
    for index, split in enumerate(('train', 'val', 'test_dates', 'test_views')):
        alpha = np.zeros((36, 64, 64), dtype=np.float32)
        frames = np.zeros((36, 64, 64, 3), dtype=np.uint8)
        if index == 0:
            if defect == 'nan':
                alpha[0, 0, 0] = np.nan
            elif defect == 'range':
                alpha[0, 0, 0] = 2
            elif defect == 'shape':
                alpha = alpha[:-1]
            else:
                frames = frames.astype(np.float32)
        path = tmp_path / f'{split}.npz'
        np.savez_compressed(path, raw_frames=np.zeros((36, 64, 64, 3), dtype=np.uint8), frames=frames, alpha=alpha)
        record = dict(window_id=str(index), video_id=index, frame_indices=list(range(36)), timestamps=list(range(36)), path=path.name, sha256=file_hash(path), date='2019-01-01')
        write_jsonl(tmp_path / f'{split}.jsonl', [record])
        manifests[split] = file_hash(tmp_path / f'{split}.jsonl')
    write_json(tmp_path / 'dataset_manifest.json', dict(protocol=cfg, data_spec_sha256=digest(data_spec(cfg)), manifests=manifests))
    with pytest.raises(ValueError, match=message):
        check_data(tmp_path, cfg)
