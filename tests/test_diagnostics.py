"""Diagnostics must explain model quality without changing training state."""
import json
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from lutils.dict_wrapper import DictWrapper
from scientific.diagnostics import (
    codebook_statistics, evaluation_mode, export_snapshot, forecast_snapshot,
    image_errors, reconstruction_snapshot, trend_diagnostics,
)
from scientific.protocol import load_protocol, write_json
from training.checkpoints import capture_rng_state


class Panel:
    def __init__(self, split='val', size=4):
        self.records = [dict(window_id=f'{split}_{i}', date=f'2019-01-{22+i%2}', camera_id=0, view_id=i % 2) for i in range(size)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(frames=torch.full((36, 3, 8, 8), (index + 1) / 10), alpha=torch.ones(36, 1, 8, 8))


class TinyAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.8))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, images):
        codes = F.one_hot(torch.zeros(len(images), 2, 2, dtype=torch.long), 4).float()
        return DictWrapper(reconstructed_images=self.dropout(images) * self.weight, quantized_latents_ids=codes)


class Perceptual(torch.nn.Module):
    def forward(self, left, right):
        return (left - right).square().mean((1, 2, 3), keepdim=True)


def test_codebook_perplexity_detects_concentration_without_claiming_overfit():
    assert codebook_statistics([8, 0, 0, 0]) == dict(perplexity=1.0, active_codes=1, used_fraction=0.25, assignments=8)
    assert codebook_statistics([2, 2, 2, 2])['perplexity'] == pytest.approx(4)
    assert codebook_statistics([0, 0, 0, 0])['perplexity'] is None
    with pytest.raises(ValueError):
        codebook_statistics([1, -1])


def test_smoke_metrics_keep_absent_regions_missing():
    prediction, target = torch.ones(1, 3, 8, 8), -torch.ones(1, 3, 8, 8)
    rows = image_errors(prediction, target, torch.zeros(1, 1, 8, 8), torch.tensor([0.2]))
    assert rows[0]['mse'] == 1 and rows[0]['background_mse'] == 1
    assert rows[0]['smoke_mse'] is None
    with pytest.raises(FloatingPointError):
        image_errors(prediction * float('nan'), target, torch.ones(1, 1, 8, 8), torch.tensor([0.2]))


def test_reconstruction_diagnostics_preserve_rng_weights_modes_and_primary_score(tmp_path):
    cfg = load_protocol()
    model, dataset, metric = TinyAE().train(), Panel(), Perceptual()
    model.dropout.eval()  # Preserve mixed modes, not just the root training flag.
    state = capture_rng_state()
    expected = random.random(), np.random.random(), torch.rand(2)
    from training.checkpoints import restore_rng_state
    restore_rng_state(state)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    snapshot = reconstruction_snapshot(model, dataset, metric, cfg, 'cpu')
    actual = random.random(), np.random.random(), torch.rand(2)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert model.training and not model.dropout.training
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    reference = []
    with evaluation_mode(model):
        for index in range(len(dataset)):
            frame = dataset[index]['frames'][index:index + 1]
            reference.append(float(metric(model(frame).reconstructed_images.clamp(-1, 1), frame).mean()))
    assert snapshot.score == float(np.mean(reference))
    assert snapshot.summary()['codebook/active_codes'] == 1
    assert snapshot.summary()['smoke_codebook/used_fraction'] == 0.25
    train_snapshot = reconstruction_snapshot(model, Panel('train', 7), metric, cfg, 'cpu', train_panel=True)
    values = export_snapshot(tmp_path, 100, 'vqgan', train_snapshot, snapshot, snapshot.score, 100)
    report = json.loads((tmp_path / 'diagnostics/step_100.json').read_text())
    assert values['generalization/lpips_gap'] == pytest.approx(snapshot.score - train_snapshot.score)
    assert {r['window_id'] for r in report['train_panel']}.isdisjoint({r['window_id'] for r in report['validation_panel']})
    assert report['baseline'] == 'black image'


def test_forecast_diagnostics_pass_only_context_and_use_fixed_draws():
    cfg = load_protocol()
    cfg['evaluation'].update(context_frames=5, horizon=3)
    model = torch.nn.Linear(1, 1).train()
    observations = []

    def predict(context, generator):
        assert context.shape == (1, 5, 3, 8, 8)
        observations.append(context.clone())
        return context[:, -1:].expand(1, 3, 3, 8, 8) + torch.rand((1, 3, 3, 8, 8), generator=generator) * 0.01

    first = forecast_snapshot(model, Panel(), Perceptual(), cfg, 'cpu', predict)
    second = forecast_snapshot(model, Panel(), Perceptual(), cfg, 'cpu', predict)
    assert first.rows == second.rows and model.training
    assert len(observations) == 8
    reference = []
    model.eval()
    with torch.no_grad():
        for index in range(len(Panel())):
            frames = Panel()[index]['frames']
            generator = torch.Generator(device='cpu').manual_seed(42000 + index)
            prediction = predict(frames[None, :5], generator)[0].clamp(-1, 1)
            reference.append(float(Perceptual()(prediction, frames[5:8]).mean()))
    assert first.score == float(np.mean(reference))


def test_possible_overfit_requires_a_full_advisory_trend(tmp_path):
    assert not trend_diagnostics(tmp_path, 250, 0.1, 0.2, 0.2)['generalization/trend_available']
    for index in range(4):
        write_json(tmp_path / 'diagnostics' / f'step_{(index+1)*250}.json', dict(step=(index+1)*250,
            metrics={'train_eval/lpips': 0.1 - index*0.01, 'validation/lpips': 0.2 + index*0.01}))
    values = trend_diagnostics(tmp_path, 1250, 0.05, 0.25, 0.2)
    assert values['generalization/possible_overfit']
    assert values['generalization/train_lpips_slope_per_1000'] < 0
    assert values['generalization/validation_lpips_slope_per_1000'] > 0


@pytest.mark.parametrize('replace_checkpoint', [False, True])
@pytest.mark.parametrize('upgraded', [False, True])
def test_autoencoder_inspection_never_loads_test_splits(tmp_path, monkeypatch, replace_checkpoint, upgraded):
    from scientific import runner
    from scientific import protocol
    cfg = load_protocol()
    previous = __import__('copy').deepcopy(cfg)
    previous['training'].pop('early_stopping')
    old_sources, new_sources = {'scientific/runner.py': 'before'}, {'scientific/runner.py': 'after'}
    if upgraded:
        protocol.write_json(tmp_path / 'maintenance.json', dict(kind='validation_early_stopping',
            before_protocol=previous, after_protocol=cfg, protocol_sha256=protocol.digest(cfg),
            before_sources=old_sources, after_sources=new_sources))
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    origin = dict(method='vqgan', seed=42, protocol_sha256=protocol.digest(previous if upgraded else cfg),
                  dataset_sha256=protocol.file_hash(data / 'dataset_manifest.json'), source_sha256=protocol.digest(old_sources))
    checkpoint = tmp_path / 'best.pth'
    torch.save(dict(step=7, model=TinyAE().state_dict(), extra={'provenance': origin}), checkpoint)
    splits = []

    def dataset(root, split):
        splits.append(split)
        assert split in ('train', 'val')
        return Panel(split)

    monkeypatch.setattr(protocol, 'validate_prepared_data', lambda *a: None)
    monkeypatch.setattr(runner, 'WindowDataset', dataset)
    monkeypatch.setattr(runner, 'provenance', lambda *a: {'source_sha256': protocol.digest(new_sources if upgraded else old_sources)})
    monkeypatch.setattr(runner, 'build_vqvae', lambda *a: TinyAE())
    monkeypatch.setattr(runner, 'build_lpips', lambda *a: Perceptual())
    output = tmp_path / 'inspection'
    if replace_checkpoint:
        original_snapshot = runner.reconstruction_snapshot

        def snapshot(*args, **kwargs):
            replacement = tmp_path / 'replacement.pth'
            torch.save(dict(step=8, model=TinyAE().state_dict(), extra={'provenance': origin}), replacement)
            replacement.replace(checkpoint)
            return original_snapshot(*args, **kwargs)

        monkeypatch.setattr(runner, 'reconstruction_snapshot', snapshot)
        with pytest.raises(RuntimeError, match='Checkpoint changed during inspection'):
            runner.inspect_autoencoder(cfg, data, tmp_path, checkpoint, torch.device('cpu'), output)
        assert not (output / 'inspection.json').exists()
        return
    report = runner.inspect_autoencoder(cfg, data, tmp_path, checkpoint, torch.device('cpu'), output)
    assert splits == ['train', 'val']
    assert report['step'] == 7 and (output / 'inspection.json').exists()
    assert (output / 'validation_reconstructions.png').exists()
    assert (output / 'diagnostics/summary.png').exists()
