"""Monitoring must stay optional, resumable and independent of training RNG."""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scientific.monitoring import WandbMonitor, WandbOptions, image_panel
from scientific.protocol import load_protocol
from training.checkpoints import capture_rng_state, restore_rng_state


def fake_sdk(monkeypatch):
    calls = []
    class FakeRun:
        entity = 'test-user'
        url = 'https://example.invalid/test-run'
        def define_metric(self, *args, **kwargs):
            calls.append(('define', args, kwargs))
        def log(self, values):
            random.random(); np.random.random(); torch.rand(1)
            calls.append(('log', values))
        def finish(self, **kwargs):
            calls.append(('finish', kwargs))
    def init(**kwargs):
        random.random(); np.random.random(); torch.rand(1)
        calls.append(('init', kwargs))
        return FakeRun()
    sdk = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs,
                          Image=lambda pixels, caption: (pixels, caption), __version__='test')
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    return calls


def test_optional_sdk_rng_and_online_resume(tmp_path, monkeypatch):
    calls = fake_sdk(monkeypatch)
    cfg = load_protocol()
    state = capture_rng_state()
    expected = random.random(), np.random.random(), torch.rand(2)
    restore_rng_state(state)
    with WandbMonitor(tmp_path, cfg, 'vqgan', 42, WandbOptions()) as monitor:
        monitor.log(661, {'train/generator': 0.2})
    actual = random.random(), np.random.random(), torch.rand(2)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    identity = json.loads((tmp_path / 'wandb.json').read_text())
    with WandbMonitor(tmp_path, cfg, 'vqgan', 42, WandbOptions(), resume=True) as monitor:
        # An earlier optimizer checkpoint is legal on an independent chart axis.
        monitor.log(650, {'train/generator': 0.3})
    init_calls = [c[1] for c in calls if c[0] == 'init']
    assert init_calls[0]['id'] == init_calls[1]['id'] == identity['id']
    assert init_calls[1]['resume'] == 'allow'
    assert init_calls[1]['entity'] == 'test-user'
    assert init_calls[0]['settings']['disable_code']
    assert [c[1]['global_step'] for c in calls if c[0] == 'log'] == [661, 650]
    assert ('finish', {'exit_code': 0}) in calls


def test_offline_sessions_are_distinct_and_do_not_request_remote_resume(tmp_path, monkeypatch):
    calls = fake_sdk(monkeypatch)
    for resume in (False, True):
        with WandbMonitor(tmp_path, load_protocol(), 'vqgan', 42, WandbOptions(mode='offline'), resume=resume):
            pass
    init_calls = [c[1] for c in calls if c[0] == 'init']
    assert all(c['mode'] == 'offline' and c['resume'] is None for c in init_calls)
    assert init_calls[0]['dir'] != init_calls[1]['dir']
    assert init_calls[0]['id'] == init_calls[1]['id']


def test_requested_missing_sdk_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'wandb', None)
    with pytest.raises(RuntimeError, match='install wandb'):
        with WandbMonitor(tmp_path, load_protocol(), 'vqgan', 42, WandbOptions()):
            pass


def test_monitor_finishes_failed_run(tmp_path, monkeypatch):
    calls = fake_sdk(monkeypatch)
    with pytest.raises(RuntimeError, match='training failed'):
        with WandbMonitor(tmp_path, load_protocol(), 'vqgan', 42, WandbOptions()):
            raise RuntimeError('training failed')
    assert ('finish', {'exit_code': 1}) in calls


def test_preview_rgb_has_explicit_uint8_range():
    panel = image_panel(torch.full((2, 3, 4, 4), -1.0), torch.ones(2, 3, 4, 4))
    assert panel.shape == (8, 8, 3) and panel.dtype == np.uint8
    assert panel[:, :4].max() == 0 and panel[:, 4:].min() == 255


def test_resume_rejects_changed_provenance():
    from scientific.runner import check_resume_provenance
    origin = {'source_sha256': 'source', 'protocol_sha256': 'protocol', 'dataset_sha256': 'data', 'seed': 42, 'command': ['train']}
    assert check_resume_provenance(origin, origin | {'command': ['train', '--resume', 'latest']}) is None
    for change in ({'source_sha256': 'other'}, {'protocol_sha256': 'changed'}, {'dataset_sha256': 'other'}, {'seed': 17}):
        with pytest.raises(ValueError, match='identical training source'):
            check_resume_provenance(origin, origin | change)
