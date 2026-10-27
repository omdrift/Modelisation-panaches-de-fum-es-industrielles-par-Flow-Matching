"""Prepared campaigns must remain runnable after their source MP4s are removed."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from scientific import cli, pipeline
from scientific.protocol import data_spec, digest, file_hash, load_protocol, write_json, write_jsonl


@pytest.fixture
def prepared(tmp_path):
    cfg = load_protocol()
    data = tmp_path / 'prepared'
    (data / 'clips').mkdir(parents=True)
    hashes = {}
    for index, split in enumerate(('train', 'val', 'test_dates', 'test_views')):
        clip = data / 'clips' / f'{split}.npz'
        frames = np.zeros((36, 64, 64, 3), dtype=np.uint8)
        np.savez_compressed(clip, raw_frames=frames, frames=frames, alpha=np.zeros((36, 64, 64), dtype=np.float32))
        write_jsonl(data / f'{split}.jsonl', [dict(video_id=index, window_id=split, date='2019-01-01',
            path=f'clips/{split}.npz', sha256=file_hash(clip), frame_indices=list(range(36)), timestamps=list(range(36)))])
        hashes[split] = file_hash(data / f'{split}.jsonl')
    write_json(data / 'dataset_manifest.json', dict(protocol=cfg, data_spec_sha256=digest(data_spec(cfg)), manifests=hashes))
    args = SimpleNamespace(videos=str(tmp_path / 'deleted_videos'), data=str(data), campaign=str(tmp_path / 'campaign'),
        methods=['flow_uniform'], seeds=[42], batch_frames=1, cache_dir=None, local_files_only=False,
        resume=False, dry_run=False, wandb=False, wandb_log_every=25, wandb_media_every=250)
    return cfg, args


class ReachedPilot(Exception):
    """Stop a simulated campaign before any training or GPU work."""


def test_prepared_data_needs_no_videos_and_is_checked_before_pilot(prepared):
    cfg, args = prepared
    calls = []

    def executor(command, log, heartbeat):
        stage = command[6]
        calls.append(stage)
        if stage == 'doctor':
            return
        if stage == 'check-data':
            assert cli.check_data(args.data, cfg)['clips_checked'] == 4
            return
        assert stage == 'pilot'  # Never prepare, download or train.
        raise ReachedPilot

    with pytest.raises(ReachedPilot):
        pipeline.run_pipeline(cfg, args, executor=executor)
    state_path = Path(args.campaign) / 'pipeline.json'
    state = json.loads(state_path.read_text())
    assert state['stages']['prepare']['status'] == 'complete'
    assert state['stages']['check_data']['status'] == 'complete'
    assert calls == ['doctor', 'check-data', 'pilot']
    args.resume = True
    original_videos = state['identity']['videos']
    args.videos = 'smoke_videos'  # Default used when --videos is omitted on resume.
    with pytest.raises(ReachedPilot):
        pipeline.run_pipeline(cfg, args, executor=executor)
    assert calls == ['doctor', 'check-data', 'pilot'] * 2
    assert json.loads(state_path.read_text())['identity']['videos'] == original_videos
    args.seeds = [17]
    with pytest.raises(ValueError, match='identical sources'):
        pipeline.run_pipeline(cfg, args, executor=executor)
    assert calls == ['doctor', 'check-data', 'pilot'] * 2


@pytest.mark.parametrize('tampering', ['missing', 'modified'])
def test_invalid_npz_blocks_pilot_even_with_a_complete_manifest(prepared, tampering):
    cfg, args = prepared
    clip = Path(args.data) / 'clips/train.npz'
    if tampering == 'missing':
        clip.unlink()
    else:
        clip.write_bytes(b'changed NPZ')
    calls = []

    def executor(command, log, heartbeat):
        stage = command[6]
        calls.append(stage)
        if stage == 'check-data':
            cli.check_data(args.data, cfg)
        else:
            assert stage == 'doctor'

    with pytest.raises(FileNotFoundError if tampering == 'missing' else ValueError):
        pipeline.run_pipeline(cfg, args, executor=executor)
    assert calls == ['doctor', 'check-data']


@pytest.mark.parametrize('mismatch', [False, True])
def test_invalid_or_incomplete_preparation_requires_sources_without_creating_campaign(prepared, mismatch):
    cfg, args = prepared
    manifest = Path(args.data) / 'dataset_manifest.json'
    if mismatch:
        cfg['preprocessing']['guided_radius'] += 1
    else:
        manifest.unlink()
    calls = []
    with pytest.raises(ValueError if mismatch else FileNotFoundError):
        pipeline.run_pipeline(cfg, args, executor=lambda *a: calls.append(a))
    assert not calls
    assert not Path(args.campaign).exists()


def test_pipeline_still_prepares_when_only_source_videos_are_available(prepared):
    cfg, args = prepared
    args.data = str(Path(args.data).parent / 'new_dataset')
    Path(args.videos).mkdir()
    calls = []

    def executor(command, log, heartbeat):
        calls.append(command[6])
        if command[6] == 'prepare':
            assert command[command.index('--videos') + 1] == args.videos
            raise ReachedPilot
        assert command[6] == 'doctor'

    with pytest.raises(ReachedPilot):
        pipeline.run_pipeline(cfg, args, executor=executor)
    assert calls == ['doctor', 'prepare']
    args.resume = True
    args.videos = str(Path(args.videos).with_name('other_videos'))
    Path(args.videos).mkdir()
    with pytest.raises(ValueError, match='identical sources'):
        pipeline.run_pipeline(cfg, args, executor=executor)
    assert calls == ['doctor', 'prepare']


def test_cli_accepts_prepared_data_without_video_argument(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'run_pipeline', lambda cfg, args: calls.append(args))
    assert cli.main(['pipeline', '--data', str(tmp_path / 'data'), '--campaign', str(tmp_path / 'campaign')]) == 0
    assert calls[0].videos == 'smoke_videos'


def test_dry_run_explains_dataset_reuse_without_creating_campaign(prepared, capsys):
    cfg, args = prepared
    args.dry_run = True
    pipeline.run_pipeline(cfg, args, executor=lambda *a: pytest.fail('dry-run must not execute'))
    assert 'reuse prepared dataset' in capsys.readouterr().out
    assert not Path(args.campaign).exists()
