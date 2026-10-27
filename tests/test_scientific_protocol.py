import json
import random

import numpy as np
import pytest
import torch

from model.conditioning import ConditioningSampler
from scientific.baselines import SimVP, persistence, optical_flow
from scientific.data import extract_smoke, validate_indices
from scientific.metrics import frame_metrics, paired_bootstrap
from scientific.protocol import load_protocol, partition, eligible, stratified_selection
from training.checkpoints import save_checkpoint, load_checkpoint


def configured_protocol(context=3, horizon=7):
    """Explicit test scenario, never a default imposed on the scientific campaign."""
    cfg = load_protocol()
    cfg['evaluation'].update(context_frames=context, horizon=horizon)
    return cfg


def test_metadata_counts_and_disjoint_partitions():
    records = json.load(open('prepare_data/metadata_02242020.json'))
    from collections import Counter
    assert len(records) == 12567
    assert sum(map(eligible, records)) == 5025
    counts = Counter(map(partition, records))
    assert [counts[s] for s in ('train', 'val', 'test_dates', 'test_views')] == [1673, 441, 706, 328]
    train = [r for r in records if partition(r) == 'train']
    held = [r for r in records if partition(r) == 'test_views']
    assert not {(r['camera_id'], r['view_id']) for r in train} & {(r['camera_id'], r['view_id']) for r in held}
    assert not {r['file_name'][4:14] for r in train} & {r['file_name'][4:14] for r in held}


def test_causal_extraction_is_independent_of_all_future_frames(synthetic_segmenter):
    cfg = load_protocol()['preprocessing']
    extractor = synthetic_segmenter(cfg)
    frames = np.random.default_rng(42).random((36, 16, 16, 3), dtype=np.float32)
    a, alpha_a = extract_smoke(frames, cfg, extractor=extractor)
    frames[2:] = 0
    b, alpha_b = extract_smoke(frames, cfg, extractor=extractor)
    np.testing.assert_array_equal(a[:2], b[:2])
    np.testing.assert_array_equal(alpha_a[:2], alpha_b[:2])
    frames = np.random.default_rng(42).random((36, 16, 16, 3), dtype=np.float32)
    frames[3:] = 1
    c, _ = extract_smoke(frames, cfg, extractor=extractor)
    np.testing.assert_array_equal(a[2], c[2])


def test_temporal_gaps_are_rejected():
    validate_indices(range(36), np.arange(36) * 5.0)
    with pytest.raises(ValueError, match='consecutive'):
        validate_indices(list(range(35)) + [37], np.arange(36) * 5.0)
    with pytest.raises(ValueError, match='intervals'):
        validate_indices(range(36), [0, 1, 1] + list(range(3, 36)))


def test_target_sampling_covers_the_complete_clip_like_river():
    sampled = ConditioningSampler().sample(4096, 36, torch.device('cpu'))
    assert set(sampled.target.tolist()) == set(range(2, 36))
    assert torch.all(sampled.reference == sampled.target - 1)
    assert torch.all(sampled.condition < sampled.reference)


def test_selection_independent_of_input_order_and_has_strata():
    rows = [dict(window_id=str(i), date=str(i % 3), camera_id=0, view_id=i % 2) for i in range(30)]
    selected = stratified_selection(rows, 12, 42)
    assert selected == stratified_selection(rows[::-1], 12, 42)
    assert len({(r['date'], r['view_id']) for r in selected}) == 6


def test_checkpoint_restores_all_rngs_and_rejects_partial_weights(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    path = save_checkpoint(model, optimizer, scheduler, tmp_path, 7)
    expected = random.random(), np.random.random(), torch.rand(3)
    assert load_checkpoint(model, optimizer, scheduler, path, torch.device('cpu')) == 7
    actual = random.random(), np.random.random(), torch.rand(3)
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    torch.save({'model': {}}, tmp_path / 'bad.pth')
    with pytest.raises(RuntimeError):
        load_checkpoint(model, optimizer, scheduler, tmp_path / 'bad.pth', torch.device('cpu'))


def test_baselines_use_only_context_and_simvp_backpropagates():
    context = torch.zeros(1, 4, 3, 64, 64) - 1
    assert torch.equal(persistence(context, 12), optical_flow(context, 12))
    model = SimVP(spatial_channels=8, temporal_channels=16, temporal_depth=2, context_frames=4)
    prediction = model(context, horizon=12)
    assert prediction.shape == (1, 12, 3, 64, 64)
    prediction.mean().backward()
    assert model.output.weight.grad is not None
    assert model.last_nfe == 3


def test_metrics_handle_empty_masks_and_use_unit_pixel_range():
    target = torch.zeros(2, 3, 64, 64)
    prediction = target.clone()
    prediction[1] = 1
    fake_perceptual = lambda a, b: ((a - b) ** 2).mean((1, 2, 3), keepdim=True)
    rows = frame_metrics(prediction, target, torch.zeros(2, 1, 64, 64), fake_perceptual)
    assert rows[0]['mse'] == 0 and rows[0]['psnr'] == 100
    assert rows[1]['mse'] == 0.25
    assert rows[1]['smoke_mse'] is None
    assert rows[1]['background_mse'] == 0.25


def test_paired_bootstrap_does_not_treat_frames_as_independent():
    rows = [dict(method=m, split='test_dates', date=str(day), window_id=str(day), seed=42, horizon=h, lpips=value) for m, value in [('a', 1), ('b', 2)] for day in range(3) for h in range(1, 13)]
    result = paired_bootstrap(rows, 'a', 'b', replicates=30)
    assert result['difference'] == -1 and result['ci95'] == [-1, -1] and result['days'] == 3
    with pytest.raises(ValueError, match='identical'):
        paired_bootstrap(rows[:-1], 'a', 'b')


def test_model_generation_generator_controls_all_random_draws():
    import torch.nn as nn
    from lutils.dict_wrapper import DictWrapper
    from model.model import Model
    from model.autoencoder_adapter import AutoencoderAdapter

    class Backbone:
        def decode_from_latents(self, value):
            return value

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()

    class Vector(nn.Module):
        def forward(self, input_latents, conditioning_latents, **kwargs):
            return conditioning_latents * 0.1

    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = DictWrapper({'autoencoder': {'encoder': {'out_channels': 2}}})
    model.sigma = 0.001
    model.ae = AE()
    model.autoencoder = AutoencoderAdapter(model.ae)
    model.vector_field_regressor = Vector()
    context = torch.randn(1, 4, 2, 4, 4)
    def generate(seed):
        return model.generate_frames(context, num_frames=3, steps=3, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(generate(42), generate(42))
    assert not torch.equal(generate(42), generate(43))
    assert model.last_nfe == 24
    with pytest.raises(ValueError):
        model.generate_frames(context, num_frames=3, steps=1)


def test_alpha_loss_uses_actual_mask_and_is_normalized():
    from scientific.runner import loss_for
    from lutils.dict_wrapper import DictWrapper
    class FakeModel:
        def __call__(self, frames, alpha_masks):
            return DictWrapper(reconstructed_vectors=torch.tensor([[[[1., 3.]]]]), target_vectors=torch.zeros(1, 1, 1, 2), smoke_mask=alpha_masks, latent_mask_fraction=torch.tensor(1.))
    alpha = torch.tensor([[[[1., 0.]]]])
    cfg = load_protocol()
    uniform, _ = loss_for(FakeModel(), 'flow_uniform', None, alpha, cfg)
    weighted, _ = loss_for(FakeModel(), 'flow_alpha', None, alpha, cfg)
    assert uniform == 5
    assert weighted.item() == pytest.approx(7 / 3)


def test_prepare_load_and_evaluate_pipeline_on_synthetic_mp4(tmp_path, monkeypatch, synthetic_segmenter):
    import cv2
    from scientific.data import prepare, WindowDataset
    from scientific.runner import evaluate
    from scientific.cli import check_data, summarize
    from scientific.protocol import write_json

    videos = tmp_path / 'videos'
    videos.mkdir()
    metadata = []
    cases = [(10, '2018-05-11', 0), (20, '2019-01-22', 0), (30, '2019-02-03', 0), (40, '2019-02-03', 1)]
    for identifier, date, view in cases:
        name = f'0-{view}-{date}-0-0-64-64-180-180-0-1000-1175'
        record = dict(id=identifier, file_name=name, camera_id=0, view_id=view, start_time=1000, label_state=23, label_state_admin=-1)
        metadata.append(record)
        writer = cv2.VideoWriter(str(videos / f'{identifier}_{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, (64, 64))
        assert writer.isOpened()
        for index in range(36):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            begin, end = {10: (0, 36), 20: (0, 8), 30: (12, 24), 40: (28, 36)}[identifier]
            if begin <= index < end:
                cv2.circle(frame, (12 + index, 32), 8, (180, 180, 180), -1)
            writer.write(frame)
        writer.release()
    cfg = configured_protocol()
    prepared = tmp_path / 'prepared'
    result = prepare(metadata, videos, prepared, cfg)
    assert len(synthetic_segmenter.instances) == 1
    assert len(synthetic_segmenter.instances[0].shapes) == 4 * 36
    assert result['selected'] == {s: 1 for s in ('train', 'val', 'test_dates', 'test_views')}
    dataset = WindowDataset(prepared, 'test_dates')
    assert dataset[0]['frames'].shape == (36, 3, 64, 64)
    assert dataset.records[0]['frame_indices'] == list(range(36))
    with np.load(prepared / dataset.records[0]['path']) as pack:
        assert pack['raw_frames'].shape == (36, 64, 64, 3)
        assert pack['raw_frames'][0].max() == 0
        assert pack['raw_frames'][12, 32, 24, 0] > 100
        assert pack['raw_frames'][35].max() == 0
    late = WindowDataset(prepared, 'test_views')
    with np.load(prepared / late.records[0]['path']) as pack:
        assert pack['raw_frames'][0].max() == 0
        assert pack['raw_frames'][35, 32, 47, 0] > 100
    checked = check_data(prepared, cfg)
    assert checked['validation'] == 'automatic' and checked['clips_checked'] == 4
    class TestMetric(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean((1, 2, 3), keepdim=True)
    output = tmp_path / 'evaluation'
    rows = evaluate(cfg, prepared, 'persistence', 42, None, None, 'test_dates', output, torch.device('cpu'), metric=TestMetric())
    assert len(rows) == 7 and {r['horizon'] for r in rows} == set(range(1, 8))
    manifest = json.loads((output / 'evaluation.json').read_text())
    assert manifest['status'] == 'complete' and manifest['windows_completed'] == 1
    alternate = configured_protocol(context=5, horizon=9)
    alternate_rows = evaluate(alternate, prepared, 'persistence', 42, None, None, 'test_dates', tmp_path / 'alternate_task', torch.device('cpu'), metric=TestMetric())
    assert len(alternate_rows) == 9
    # Synthetic injected scores must never be presented as actual LPIPS.
    with pytest.raises(ValueError, match='Injected'):
        summarize([output], tmp_path / 'summary', cfg)
    manifest['lpips'] = 'official alex v0.1'
    write_json(output / 'evaluation.json', manifest)
    metric_rows = (output / 'metrics.csv').read_text().splitlines()
    (output / 'metrics.csv').write_text('\n'.join(metric_rows[:-1]) + '\n')
    with pytest.raises(ValueError, match='grid'):
        summarize([output], tmp_path / 'summary', cfg)
    # Files are immutable inputs: tampering is detected even with valid shapes.
    first = dataset.records[0]
    (prepared / first['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        WindowDataset(prepared, 'test_dates')
    import scientific.data as data_module
    monkeypatch.setattr(data_module, 'extract_smoke', lambda *a, **kw: (_ for _ in ()).throw(ValueError('bad trimap')))
    failed = tmp_path / 'failed_segmentation'
    failure = prepare(metadata, videos, failed, cfg)
    assert failure['statuses']['segmentation_failed'] == 4
    with np.load(failed / 'clips/40_0000.npz') as pack:
        assert pack['raw_frames'].shape == (36, 64, 64, 3)
        assert pack['raw_frames'][35, 32, 47, 0] > 100


@pytest.mark.parametrize('failure', ['validation', 'best_copy'])
def test_predictor_resume_completes_pending_final_validation(tmp_path, monkeypatch, failure):
    from scientific import runner
    from scientific.protocol import digest, write_json
    cfg = configured_protocol()
    cfg['training']['max_updates'] = 1
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'dataset_manifest.json').write_text('{}')
    pilot = tmp_path / 'pilot.json'
    write_json(pilot, {'provenance': {'protocol_sha256': digest(cfg), 'source_sha256': 'source', 'gpu': None, 'torch': 'test', 'cuda': None}, 'methods': {'simvp': {'micro_batch': 4, 'update_seconds': 0.001, 'inference_seconds': 0.001}}, 'common_updates': 1})

    class TinyDataset:
        def __init__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {'frames': torch.ones(36, 3, 4, 4) * 0.2, 'alpha': torch.ones(36, 1, 4, 4)}
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.last_nfe = 1
        def forward(self, context, horizon=12):
            return context[:, -1:] * self.weight + torch.zeros(len(context), horizon, 3, 4, 4)
    class TinyMetric(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean((1, 2, 3), keepdim=True)
    monkeypatch.setattr(runner, 'WindowDataset', TinyDataset)
    monkeypatch.setattr(runner, 'model_for', lambda *a, **kw: TinyModel())
    monkeypatch.setattr(runner, 'build_lpips', lambda device: TinyMetric())
    monkeypatch.setattr(runner, 'provenance', lambda cfg, root=None: dict(protocol_sha256=digest(cfg), source_sha256='source', gpu=None, torch='test', cuda=None))
    original_validation = runner.validation_score
    original_copy = runner.copy_checkpoint
    from training.checkpoints import CheckpointWriteError
    if failure == 'validation':
        monkeypatch.setattr(runner, 'validation_score', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('validation interrupted')))
    else:
        monkeypatch.setattr(runner, 'copy_checkpoint', lambda *a: (_ for _ in ()).throw(CheckpointWriteError('best copy interrupted')))
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError if failure == 'validation' else CheckpointWriteError, match='interrupted'):
        runner.train_predictor(cfg, root, 'simvp', 17, None, pilot, run, torch.device('cpu'))
    assert json.loads((run / 'run.json').read_text())['status'] == 'incomplete'
    checkpoint = run / 'checkpoints/step_1.pth'
    state = torch.load(checkpoint, weights_only=True)
    assert state['step'] == 1
    assert state['extra']['last_validated_step'] == (0 if failure == 'validation' else 1)
    assert bool(state['extra'].get('best_checkpoint_pending')) == (failure == 'best_copy')
    monkeypatch.setattr(runner, 'validation_score', original_validation)
    monkeypatch.setattr(runner, 'copy_checkpoint', original_copy)
    runner.train_predictor(cfg, root, 'simvp', 17, None, pilot, run, torch.device('cpu'), resume=checkpoint)
    assert json.loads((run / 'run.json').read_text())['status'] == 'complete'
    selected = torch.load(run / 'best.pth', weights_only=True)
    assert selected['extra']['last_validated_step'] == 1
    assert selected['extra']['completion']['completed_updates'] == 1


def test_unvalidated_autoencoder_is_rejected(tmp_path):
    from scientific.runner import validate_artifact
    cfg = load_protocol()
    (tmp_path / 'dataset_manifest.json').write_text('{}')
    checkpoint = tmp_path / 'unvalidated.pth'
    torch.save({'model': {}}, checkpoint)
    with pytest.raises(ValueError, match='scientific checkpoint'):
        validate_artifact(checkpoint, cfg, tmp_path, 'vqgan')


def test_autoencoder_honors_configured_architecture():
    from model.vqgan.encoder import build_encoder
    from model.vqgan.decoder import build_decoder
    from model.vqgan.vector_quantizer import build_vector_quantizer
    encoder_config = {'in_channels': 3, 'out_channels': 8, 'mid_channels': 64}
    decoder_config = {'in_channels': 8, 'out_channels': 3, 'mid_channels': 128}
    strict_encoder = build_encoder(encoder_config)
    strict_decoder = build_decoder(decoder_config)
    assert strict_encoder.conv_in.out_channels == 64
    assert strict_decoder.conv_in.out_channels == 128
    with torch.inference_mode():
        assert strict_decoder(torch.randn(1, 8, 8, 8)).shape == (1, 3, 64, 64)
    quantizer = build_vector_quantizer({'num_embeddings': 16, 'embedding_dimension': 8, 'commitment_cost': 0.1})
    assert quantizer.beta == 0.1


def test_completion_proof_is_bound_to_selected_step(tmp_path, monkeypatch):
    from scientific import runner
    from scientific.protocol import digest, file_hash
    cfg = load_protocol()
    cfg['training']['max_updates'] = 10
    (tmp_path / 'dataset_manifest.json').write_text('{}')
    monkeypatch.setattr(runner, 'provenance', lambda cfg, root=None: {'source_sha256': 'source'})
    state = {'step': 5, 'model': {}, 'extra': {'provenance': {'method': 'simvp', 'protocol_sha256': digest(cfg), 'dataset_sha256': file_hash(tmp_path / 'dataset_manifest.json'), 'source_sha256': 'source'}, 'best_validation_lpips': 0.2, 'selected_artifact': True, 'selection_step': 4, 'completion': {'status': 'complete', 'completed_updates': 10, 'planned_updates': 10}}}
    path = tmp_path / 'selected.pth'
    torch.save(state, path)
    with pytest.raises(ValueError, match='bound'):
        runner.validate_artifact(path, cfg, tmp_path, 'simvp')
    state['step'] = 4
    torch.save(state, path)
    assert runner.validate_artifact(path, cfg, tmp_path, 'simvp')['step'] == 4


def test_full_summary_rejects_partial_grid_and_reports_paired_results(tmp_path):
    import csv
    from scientific.cli import summarize
    from scientific.protocol import digest, write_json, METHODS
    cfg = configured_protocol()
    folders = []
    for method_index, method in enumerate(METHODS):
        for seed in cfg['seeds']:
            for split in ('test_dates', 'test_views'):
                folder = tmp_path / f'{method}_{seed}_{split}'
                folder.mkdir()
                count = 4 if method.startswith('flow') else 1
                write_json(folder / 'evaluation.json', {'status': 'complete', 'lpips': 'official alex v0.1', 'protocol_sha256': digest(cfg), 'dataset_sha256': 'dataset', 'metric_weights_sha256': 'weights', 'source_sha256': 'source', 'training_complete': True, 'method': method, 'seed': seed, 'split': split, 'expected_window_ids': ['a', 'b'], 'windows_completed': 2, 'expected_windows': 2})
                rows = [dict(method=method, seed=seed, split=split, window_id=window, date=str(day), horizon=h, realization=r, lpips=method_index * 0.1, mse=0.1, psnr=10, ssim=0.9, smoke_mse=None, background_mse=0.1, smoke_fraction=0, diversity_mse=0) for day, window in enumerate(('a', 'b')) for h in range(1, 8) for r in range(count)]
                with (folder / 'metrics.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with (folder / 'runtime.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['window_id', 'realization', 'seconds', 'nfe', 'peak_memory_bytes'])
                    writer.writeheader()
                    writer.writerows(dict(window_id=w, realization=r, seconds=1, nfe=1, peak_memory_bytes=1) for w in ('a', 'b') for r in range(count))
                folders.append(folder)
    # These fixture scores test aggregation, not model quality.
    result = summarize(folders, tmp_path / 'summary', cfg)
    assert result['complete_campaign']
    assert len(result['comparisons']) == 4
    assert (tmp_path / 'summary/test_dates_horizons.png').exists()
    summary = json.loads((tmp_path / 'summary/summary.json').read_text())
    assert all(v['smoke_mse'] is None for v in summary['summaries'])
    # A complete file can still contain invalid values or a different test panel.
    first = folders[0]
    original_csv = (first / 'metrics.csv').read_text()
    (first / 'metrics.csv').write_text(original_csv.replace(',0.0,0.1,', ',nan,0.1,', 1))
    with pytest.raises(ValueError, match='non-finite metric'):
        summarize(folders, tmp_path / 'invalid_metrics', cfg)
    (first / 'metrics.csv').write_text(original_csv)
    changed = json.loads((folders[2] / 'evaluation.json').read_text())
    changed['expected_window_ids'] = ['a', 'other']
    write_json(folders[2] / 'evaluation.json', changed)
    with pytest.raises(ValueError, match='same frozen test panel'):
        summarize(folders, tmp_path / 'invalid_panel', cfg)


def test_scientific_vq_commitment_coefficient_weights_encoder_gradient():
    from model.vqgan.vector_quantizer import build_vector_quantizer
    quantizer = build_vector_quantizer({'num_embeddings': 16, 'embedding_dimension': 8, 'commitment_cost': 0.1})
    with torch.no_grad():
        quantizer.embedding.weight.zero_()
    latent = torch.ones(1, 8, 1, 1, requires_grad=True)
    loss, _, _ = quantizer(latent)
    loss.backward()
    assert torch.allclose(latent.grad, torch.full_like(latent, 0.025))
    assert torch.allclose(quantizer.embedding.weight.grad[0], torch.full((8,), -0.25))


def test_complete_clips_do_not_impose_an_inference_horizon():
    from scientific.protocol import forecast_task, data_spec, digest
    cfg = load_protocol()
    assert cfg['task']['clip_frames'] == 36
    assert cfg['model']['min_target_index'] == 2
    assert forecast_task(cfg, required=False) == (5, 15)
    assert forecast_task(cfg) == (5, 15)
    configured = configured_protocol(context=5, horizon=25)
    assert forecast_task(configured) == (5, 25)
    assert digest(data_spec(cfg)) == digest(data_spec(configured))
    assert digest(cfg) != digest(configured)


def test_prepared_data_identity_is_strict():
    import copy
    from scientific.protocol import data_spec, digest, validate_prepared_data
    cfg = load_protocol()
    manifest = {'protocol': cfg, 'data_spec_sha256': digest(data_spec(cfg))}
    validate_prepared_data(manifest, cfg)
    changed = copy.deepcopy(cfg)
    changed['preprocessing']['guided_radius'] = 5
    with pytest.raises(ValueError, match='does not match'):
        validate_prepared_data(manifest, changed)
    manifest['data_spec_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='inconsistent'):
        validate_prepared_data(manifest, cfg)


@pytest.mark.parametrize('context,horizon', [(2, 35), (1, 12), (36, 1), (4, 0)])
def test_forecast_task_rejects_unsupported_or_unavailable_targets(context, horizon):
    from scientific.protocol import forecast_task
    with pytest.raises(ValueError, match='Require'):
        forecast_task(configured_protocol(context, horizon))


@pytest.mark.parametrize('context,horizon', [(2, 5), (3, 7), (5, 25)])
def test_simvp_supports_explicit_contexts_and_horizons(context, horizon):
    model = SimVP(spatial_channels=16, temporal_channels=16, temporal_depth=2, context_frames=context)
    with torch.inference_mode():
        result = model(torch.zeros(1, context, 3, 64, 64), horizon)
    assert result.shape == (1, horizon, 3, 64, 64)
    assert model.last_nfe == (horizon + context - 1) // context
