"""Plateau decisions must survive interruptions and produce usable artifacts."""
import json

import pytest
import torch
import yaml

from scientific.protocol import digest, file_hash, load_protocol, write_json


def configured_policy(**overrides):
    cfg = load_protocol()
    cfg['training']['early_stopping'] = dict(enabled=True, patience=2, min_delta=0.0001, min_updates=0) | overrides
    return cfg


def test_small_gains_accumulate_against_the_last_significant_improvement():
    from scientific.early_stopping import EarlyStopping
    stopper = EarlyStopping(configured_policy(patience=3))
    for step, score in enumerate([0.3, 0.29996, 0.29992, 0.29985], 1):
        stopper.observe(score, step)
    assert not stopper.stopped
    assert stopper.state_dict()['bad_checks'] == 0
    assert stopper.state_dict()['best_score'] == 0.29985
    stopper.observe(0.29985, 5)
    stopper.observe(0.31, 6)
    stopper.observe(0.32, 7)
    assert stopper.stopped


def test_minimum_updates_delays_patience_counting_and_ties_do_not_improve():
    from scientific.early_stopping import EarlyStopping
    stopper = EarlyStopping(configured_policy(min_updates=5, min_delta=0.0))
    for step in range(1, 5):
        stopper.observe(0.2, step)
        assert stopper.state_dict()['bad_checks'] == 0
    stopper.observe(0.2, 5)
    assert not stopper.stopped
    stopper.observe(0.2, 6)
    assert stopper.stopped


def test_missing_policy_preserves_frozen_protocol_behavior():
    from scientific.early_stopping import EarlyStopping
    cfg = load_protocol()
    cfg['training'].pop('early_stopping', None)
    stopper = EarlyStopping(cfg)
    for step in range(1, 20):
        stopper.observe(0.2, step)
    assert not stopper.stopped
    assert not stopper.state_dict()['settings']['enabled']


def test_state_round_trip_and_rejection_of_changed_policy():
    from scientific.early_stopping import EarlyStopping
    cfg = configured_policy()
    stopper = EarlyStopping(cfg)
    stopper.observe(0.2, 1)
    stopper.observe(0.3, 2)
    state = json.loads(json.dumps(stopper.state_dict()))
    resumed = EarlyStopping(cfg, state)
    resumed.observe(0.3, 3)
    assert resumed.stopped
    assert not stopper.stopped
    with pytest.raises(ValueError, match='policy|settings'):
        EarlyStopping(configured_policy(patience=3), state)


@pytest.mark.parametrize('override', [
    {'enabled': 1}, {'patience': 0}, {'patience': True}, {'patience': 1.5},
    {'min_delta': -1}, {'min_delta': float('nan')}, {'min_delta': float('inf')},
    {'min_delta': True}, {'min_updates': -1}, {'min_updates': 0.5},
    {'min_updates': True}, {'typo': 1},
])
def test_invalid_policy_is_rejected(override):
    from scientific.early_stopping import EarlyStopping
    with pytest.raises(ValueError):
        EarlyStopping(configured_policy(**override))


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_score_cannot_consume_patience(score):
    from scientific.early_stopping import EarlyStopping
    stopper = EarlyStopping(configured_policy())
    before = stopper.state_dict()
    with pytest.raises(ValueError, match='finite'):
        stopper.observe(score, 1)
    assert stopper.state_dict() == before


@pytest.fixture
def tiny_training(tmp_path, monkeypatch):
    from scientific import runner
    from lutils.dict_wrapper import DictWrapper
    import model.vqgan.discriminator as discriminator_module

    cfg = configured_policy()
    cfg['training'].update(max_updates=12, validation_every=2, discriminator_start=0, warmup_updates=1)
    cfg['training']['autoencoder']['validation_every'] = 2
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dataset_manifest.json').write_text('{}')
    ae = tmp_path / 'common.pth'
    ae.write_bytes(b'fixed common autoencoder fixture')
    pilot = tmp_path / 'pilot.json'

    def provenance(cfg, root=None):
        return dict(protocol_sha256=digest(cfg), source_sha256='source', gpu=None, torch='test', cuda=None,
                    dataset_sha256=file_hash(data / 'dataset_manifest.json') if root else None)

    write_json(pilot, dict(provenance=provenance(cfg), common_updates=12,
                          methods={m: {'micro_batch': 4} for m in ('simvp', 'flow_uniform', 'flow_alpha')}))

    class TinyDataset:
        records = [dict(window_id='fixture', date='2019-02-03', camera_id=0, view_id=0)]
        def __init__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return dict(frames=torch.full((36, 3, 4, 4), 0.2), alpha=torch.ones(36, 1, 4, 4))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vector_field_regressor = torch.nn.Linear(1, 1, bias=False)
        def forward(self, images):
            weight = self.vector_field_regressor.weight.reshape(())
            return DictWrapper(reconstructed_images=images * weight, vq_loss=weight.square())

    class TinyMetric(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).square().mean((1, 2, 3), keepdim=True)

    class Snapshot:
        score = 0.2
        def summary(self):
            return {'lpips': self.score}

    controls = dict(pending_validation=False, validation_calls=0, score=0.2)
    original_batch = runner.batch_at

    def batch(*args, **kwargs):
        assert not controls['pending_validation'], 'Resume skipped an interrupted validation before an optimizer update'
        return original_batch(*args, **kwargs)

    def validation(*args, **kwargs):
        if kwargs.get('train_panel'):
            return Snapshot()
        controls['pending_validation'] = False
        controls['validation_calls'] += 1
        result = Snapshot()
        result.score = controls.get('scores', [controls['score']] * 20)[controls['validation_calls'] - 1]
        if controls.get('failure') == 'validation' and controls['validation_calls'] == 2:
            raise RuntimeError('validation interrupted')
        return result

    def loss(model, method, frames, alpha, cfg):
        return model.vector_field_regressor.weight.square().mean(), None

    monkeypatch.setattr(runner, 'WindowDataset', TinyDataset)
    monkeypatch.setattr(runner, 'batch_at', batch)
    monkeypatch.setattr(runner, 'build_vqvae', lambda *a: TinyModel())
    monkeypatch.setattr(runner, 'model_for', lambda *a, **kw: TinyModel())
    monkeypatch.setattr(runner, 'build_lpips', lambda *a: TinyMetric())
    monkeypatch.setattr(discriminator_module, 'build_discriminator', lambda **kw: torch.nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(runner, 'provenance', provenance)
    monkeypatch.setattr(runner, 'loss_for', loss)
    monkeypatch.setattr(runner, 'validation_score', validation)
    monkeypatch.setattr(runner, 'reconstruction_snapshot', validation)
    monkeypatch.setattr(runner, 'forecast_snapshot', lambda *a, **kw: Snapshot())
    monkeypatch.setattr(runner, 'export_snapshot', lambda run, step, method, train, val, *a: {'validation/lpips': val.score})
    monkeypatch.setattr(runner, 'plot_history', lambda *a: None)
    original_validate = runner.validate_artifact
    monkeypatch.setattr(runner, 'validate_artifact', lambda *a: None)
    run = tmp_path / 'run'

    def train(method, resume=None):
        if method == 'vqgan':
            return runner.train_autoencoder(cfg, data, run, torch.device('cpu'), resume=resume)
        return runner.train_predictor(cfg, data, method, 17, ae, pilot, run, torch.device('cpu'), resume=resume)

    return cfg, run, controls, train, lambda method: original_validate(run / 'best.pth', cfg, data, method)


@pytest.mark.parametrize('method', ['vqgan', 'simvp', 'flow_uniform', 'flow_alpha'])
@pytest.mark.parametrize('failure', [None, 'validation', 'logging', 'finalization', 'missing_metadata'])
def test_all_trainers_stop_resume_and_finalize_before_the_cap(tiny_training, monkeypatch, method, failure):
    from scientific import runner
    from scientific.pipeline import Stage, validate_stage
    from training.checkpoints import latest_checkpoint
    cfg, run, controls, train, validate = tiny_training
    controls['failure'] = 'validation' if failure == 'missing_metadata' else failure
    original_log = runner.record_metrics
    original_finalize = runner.finalize_selection

    def logging(directory, step, values, *args):
        if failure == 'logging' and step == 4 and 'validation/lpips' in values:
            raise RuntimeError('logging interrupted')
        return original_log(directory, step, values, *args)

    monkeypatch.setattr(runner, 'record_metrics', logging)
    if failure == 'finalization':
        monkeypatch.setattr(runner, 'finalize_selection', lambda *a: (_ for _ in ()).throw(RuntimeError('finalization interrupted')))
    if failure:
        with pytest.raises(RuntimeError, match='interrupted'):
            train(method)
        saved = torch.load(latest_checkpoint(run), weights_only=True)
        if failure in ('validation', 'logging', 'missing_metadata'):
            assert saved['step'] == 4 and saved['extra']['last_validated_step'] == 2
            controls['pending_validation'] = True
        if failure == 'missing_metadata':
            (run / 'run.json').unlink()
        controls['failure'] = None
        monkeypatch.setattr(runner, 'record_metrics', original_log)
        monkeypatch.setattr(runner, 'finalize_selection', original_finalize)
        train(method, latest_checkpoint(run))
    else:
        train(method)
    metadata = json.loads((run / 'run.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['completed_updates'] == 6
    assert metadata['termination_reason'] == 'early_stopping'
    selected = validate(method)
    assert selected['step'] == 2  # Plateau keeps the original minimum.
    assert selected['extra']['completion']['completed_updates'] == 6
    assert selected['extra']['completion']['early_stopping']['stopped']
    stage_name = 'vqgan' if method == 'vqgan' else f'train_{method}_17'
    validate_stage(Stage(stage_name, (), (run / 'run.json', run / 'best.pth')), cfg)
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert any(r.get('early_stopping/stopped') for r in metrics)
    before = file_hash(run / 'best.pth')
    train(method, latest_checkpoint(run))
    assert file_hash(run / 'best.pth') == before
    assert json.loads((run / 'run.json').read_text())['completed_updates'] == 6


def test_early_stop_proof_cannot_replace_incomplete_training(tmp_path, monkeypatch):
    from scientific import runner
    cfg = configured_policy()
    cfg['training']['max_updates'] = 12
    (tmp_path / 'dataset_manifest.json').write_text('{}')
    monkeypatch.setattr(runner, 'provenance', lambda *a: {'source_sha256': 'source'})
    origin = dict(method='simvp', protocol_sha256=digest(cfg), source_sha256='source',
                  dataset_sha256=file_hash(tmp_path / 'dataset_manifest.json'))
    state = dict(step=2, model={}, extra=dict(provenance=origin, selected_artifact=True,
        selection_step=2, best_validation_lpips=0.2,
        completion=dict(status='complete', termination_reason='early_stopping', completed_updates=6, planned_updates=12)))
    path = tmp_path / 'best.pth'
    torch.save(state, path)
    with pytest.raises(ValueError, match='Incomplete|early.stop|proof'):
        runner.validate_artifact(path, cfg, tmp_path, 'simvp')


@pytest.mark.parametrize('enabled', [False, True])
def test_completion_cannot_lower_the_frozen_cap_or_claim_an_unproven_plateau(enabled):
    from scientific.early_stopping import completion_valid
    cfg = configured_policy(enabled=enabled)
    cfg['training']['max_updates'] = 12
    assert not completion_valid(cfg, dict(status='complete', completed_updates=6, planned_updates=6,
                                         termination_reason='early_stopping'))
    assert not completion_valid(cfg, dict(status='complete', completed_updates=12, planned_updates=12,
                                         termination_reason='early_stopping'))


@pytest.mark.parametrize('method', ['vqgan', 'simvp', 'flow_uniform', 'flow_alpha'])
def test_raw_best_can_improve_without_resetting_patience(tiny_training, method):
    cfg, run, controls, train, validate = tiny_training
    controls['scores'] = [0.3, 0.25, 0.24995, 0.28]
    train(method)
    assert json.loads((run / 'run.json').read_text())['completed_updates'] == 8
    selected = validate(method)
    assert selected['step'] == 6
    assert selected['extra']['best_validation_lpips'] == 0.24995


@pytest.mark.parametrize('method', ['vqgan', 'simvp', 'flow_uniform', 'flow_alpha'])
def test_stopping_waits_for_warmup_or_adversarial_activation(tiny_training, method):
    cfg, run, _, train, validate = tiny_training
    cfg['training']['discriminator_start' if method == 'vqgan' else 'warmup_updates'] = 6 if method == 'vqgan' else 8
    pilot_path = run.parent / 'pilot.json'
    pilot = json.loads(pilot_path.read_text())
    pilot['provenance']['protocol_sha256'] = digest(cfg)
    write_json(pilot_path, pilot)
    train(method)
    assert json.loads((run / 'run.json').read_text())['completed_updates'] == 10
    assert validate(method)['extra']['completion']['termination_reason'] == 'early_stopping'


@pytest.mark.parametrize('method', ['vqgan', 'simvp', 'flow_uniform', 'flow_alpha'])
@pytest.mark.parametrize('already_complete', [False, True])
def test_legacy_opt_in_continues_from_saved_optimizer_step(tiny_training, tmp_path, monkeypatch, method, already_complete):
    from scientific import runner
    from scientific.protocol import _source_manifest
    from tools.enable_early_stopping import enable
    from training.checkpoints import latest_checkpoint
    cfg, run, controls, train, validate = tiny_training
    cfg['training'].pop('early_stopping')
    campaign = run.parent
    (campaign / 'protocol.yaml').write_text(yaml.safe_dump(cfg))
    before, after = tmp_path / 'before', tmp_path / 'after'
    for root, content in ((before, 'before'), (after, 'after')):
        (root / 'scientific').mkdir(parents=True)
        (root / 'scientific/runner.py').write_text(content)
    source = {'sha': digest(_source_manifest(before))}
    original_provenance = runner.provenance
    monkeypatch.setattr(runner, 'provenance', lambda cfg, root=None:
                        original_provenance(cfg, root) | {'source_sha256': source['sha']})
    pilot_path = campaign / 'pilot.json'
    pilot = json.loads(pilot_path.read_text())
    pilot['provenance'].update(protocol_sha256=digest(cfg), source_sha256=source['sha'])
    write_json(pilot_path, pilot)
    original_log = runner.record_metrics

    def interrupted(directory, step, values, *args):
        if step == 4 and 'validation/lpips' in values:
            raise RuntimeError('saved legacy checkpoint')
        return original_log(directory, step, values, *args)

    if already_complete:
        train(method)
    else:
        monkeypatch.setattr(runner, 'record_metrics', interrupted)
        with pytest.raises(RuntimeError, match='saved legacy checkpoint'):
            train(method)
    checkpoint = latest_checkpoint(run)
    assert torch.load(checkpoint, weights_only=True)['step'] == (12 if already_complete else 4)
    original_hash = file_hash(checkpoint)
    selected_hash = file_hash(run / 'best.pth')
    metadata_hash = file_hash(run / 'run.json')
    enable(campaign, before, source_root=after, patience=2, min_updates=0)
    assert file_hash(checkpoint) == original_hash
    updated = load_protocol(campaign / 'protocol.yaml')
    cfg.clear()
    cfg.update(updated)
    source['sha'] = digest(_source_manifest(after))
    monkeypatch.setattr(runner, 'record_metrics', original_log)
    controls['pending_validation'] = not already_complete
    monkeypatch.setattr(runner, 'validate_artifact', lambda path, *a: validate(method) if path == run / 'best.pth' else None)
    train(method, checkpoint)
    assert json.loads((run / 'run.json').read_text())['completed_updates'] == (12 if already_complete else 6)
    selected = validate(method)
    assert selected['extra']['completion']['termination_reason'] == ('max_updates' if already_complete else 'early_stopping')
    if already_complete:
        assert file_hash(run / 'best.pth') == selected_hash
        assert file_hash(run / 'run.json') == metadata_hash
    if method != 'vqgan':
        monkeypatch.setattr(runner, 'validate_artifact', lambda *a: selected)
        monkeypatch.setattr(runner, 'predict', lambda model, method, context, cfg, generator=None:
                            (context[:, -1:].repeat(1, cfg['evaluation']['horizon'], 1, 1, 1), 1))
        monkeypatch.setattr(runner, 'frame_metrics', lambda prediction, *a:
                            [dict(horizon=i + 1, lpips=0.2) for i in range(len(prediction))])
        rows = runner.evaluate(cfg, campaign / 'data', method, 17, run / 'best.pth', campaign / 'common.pth',
                               'test_dates', campaign / 'evaluated', torch.device('cpu'), metric=torch.nn.Identity())
        assert len(rows) == cfg['evaluation']['horizon'] * (4 if method.startswith('flow') else 1)
