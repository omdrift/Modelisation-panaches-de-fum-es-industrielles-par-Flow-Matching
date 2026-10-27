"""CLI dispatch and measurement must work with an existing campaign directory."""
import fcntl
import json

import pytest
import torch

from scientific import cli, runner
from scientific.protocol import load_protocol


def test_cli_starts_autoencoder_and_prevents_concurrent_writes(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cli, 'check_data', lambda *args: {'validation': 'automatic'})
    def train(cfg, root, run, device, resume=None):
        calls.append((cfg['training']['max_updates'], root, run, device.type, resume))
    monkeypatch.setattr(runner, 'train_autoencoder', train)
    args = ['--context-frames', '3', '--horizon', '7', 'autoencoder', '--campaign', str(campaign), '--data', str(tmp_path / 'data')]
    assert cli.main(args) == 0
    assert calls == [(100000, str(tmp_path / 'data'), campaign / 'autoencoder', 'cuda', None)]
    assert {p.name for p in campaign.iterdir()} == {'campaign.lock'}

    with (campaign / 'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already running'):
            cli.main(args)
    assert len(calls) == 1


def test_profile_keeps_configured_training_length(tmp_path, monkeypatch):
    cfg = load_protocol()
    cfg['evaluation'].update(context_frames=3, horizon=7)
    cfg['training']['max_updates'] = 37
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
    monkeypatch.setattr(runner, 'model_for', lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(runner, 'build_lpips', lambda *args: torch.nn.Identity())
    monkeypatch.setattr(runner, 'parameters', lambda model, method: model.parameters())
    monkeypatch.setattr(runner, 'loss_for', lambda model, *args: (model.weight.square(), None))
    monkeypatch.setattr(runner, 'predict', lambda *args: (None, 1))
    monkeypatch.setattr(runner, 'provenance', lambda *args: {})
    result = runner.profile(cfg, tmp_path / 'pilot.json', torch.device('cpu'))
    assert result['common_updates'] == 37
    assert all(v['micro_batch'] == 4 for v in result['methods'].values())


def test_cli_wandb_flags_and_latest_checkpoint_are_forwarded(tmp_path, monkeypatch):
    import scientific.monitoring as monitoring
    directory = tmp_path / 'autoencoder'
    (directory / 'checkpoints').mkdir(parents=True)
    (directory / 'run.json').write_text('{}')
    for step in (25, 100):
        (directory / f'checkpoints/step_{step}.pth').write_text('checkpoint placeholder')
    monitors, calls = [], []
    class StubMonitor:
        def __init__(self, directory, cfg, method, seed, options, resume):
            self.options, self.resume = options, resume
            monitors.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def train(cfg, root, run, device, resume=None, monitor=None):
        calls.append((resume, monitor))
    monkeypatch.setattr(monitoring, 'WandbMonitor', StubMonitor)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cli, 'check_data', lambda *args: {})
    monkeypatch.setattr(runner, 'train_autoencoder', train)
    assert cli.main(['--context-frames', '5', '--horizon', '15', 'autoencoder', '--campaign', str(tmp_path), '--data', 'prepared',
        '--resume', 'latest', '--wandb', '--wandb-project', 'smoke-test', '--wandb-mode', 'offline', '--wandb-log-every', '10']) == 0
    assert calls == [(str(directory / 'checkpoints/step_100.pth'), monitors[0])]
    assert monitors[0].options.mode == 'offline' and monitors[0].options.project == 'smoke-test'
    assert monitors[0].options.log_every == 10 and monitors[0].resume


def test_cli_prepare_passes_segmentation_device_and_offline_cache(tmp_path, monkeypatch):
    from scientific import data
    calls = []
    monkeypatch.setattr(data, 'prepare', lambda *args, **kwargs: calls.append(kwargs))
    assert cli.main(['prepare', '--videos', 'videos', '--output', str(tmp_path),
                     '--device', 'cpu', '--cache-dir', 'scientific_models/huggingface', '--local-files-only', '--batch-frames', '8']) == 0
    assert calls[0]['device'] == 'cpu' and calls[0]['local_files_only']
    assert calls[0]['cache_dir'] == 'scientific_models/huggingface'
    assert calls[0]['batch_frames'] == 8
