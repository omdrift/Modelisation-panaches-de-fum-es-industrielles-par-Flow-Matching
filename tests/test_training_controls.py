"""Study controls preserve the legacy path and reproducible recovery."""
import copy
import json
import random

import numpy as np
import pytest
import torch
import yaml

from scientific.hyperparameters import autoencoder_settings, schedule_factor
from scientific.protocol import load_protocol, write_json
from scientific.training_control import (autoencoder_protocol, fine_tune,
    objective_snapshot, optimizer_step, scaler_for, scheduler_for, study_pruned)
from training.checkpoints import capture_rng_state, restore_rng_state, load_checkpoint, save_checkpoint


def test_cosine_floor_and_legacy_warmup_are_explicit():
    cosine = dict(schedule='cosine', schedule_min_factor=0.1)
    assert schedule_factor(0, 100, cosine) == 1
    assert schedule_factor(100, 100, cosine) == pytest.approx(0.1)
    assert schedule_factor(999, 100, cosine) == pytest.approx(0.1)
    assert schedule_factor(0, 100, cosine, warmup=10) == pytest.approx(0.1)
    assert schedule_factor(10, 100, cosine, warmup=10) == 1
    assert schedule_factor(100, 100, cosine, warmup=10) == pytest.approx(0.1)
    assert [schedule_factor(step, 100, {}, warmup=10) for step in (0, 4, 9, 40)] == [0.1, 0.5, 1.0, 1.0]
    cfg = load_protocol()
    cfg['training']['autoencoder'].update(cosine)
    assert autoencoder_settings(cfg)['schedule'] == 'cosine'


@pytest.mark.parametrize('minimum', [0, -1, True, float('nan'), 2])
def test_invalid_schedule_floor_rejected(minimum):
    cfg = load_protocol()
    cfg['training']['autoencoder']['schedule_min_factor'] = minimum
    with pytest.raises(ValueError, match='schedule_min_factor'):
        autoencoder_settings(cfg)


def test_scheduler_optimizer_and_scaler_resume_at_exact_same_update(tmp_path):
    cfg = load_protocol()
    cfg['training'].update(max_updates=10, amp=True)
    settings = dict(schedule='cosine', schedule_min_factor=0.1)
    torch.manual_seed(17)
    initial = torch.nn.Linear(2, 1)
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = scheduler_for(optimizer, cfg, settings)
    scaler = scaler_for(cfg, torch.device('cpu'))

    def update(model, optimizer, scheduler, scaler):
        optimizer.zero_grad()
        scaler.scale(model(torch.ones(2, 2)).square().mean()).backward()
        optimizer_step(optimizer, model.parameters(), scaler, 1)
        scaler.update()
        scheduler.step()

    update(model, optimizer, scheduler, scaler)
    checkpoint = save_checkpoint(model, optimizer, scheduler, tmp_path, 1, dict(scaler=scaler.state_dict()))
    update(model, optimizer, scheduler, scaler)
    restored = copy.deepcopy(initial)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_scheduler = scheduler_for(restored_optimizer, cfg, settings)
    restored_scaler = scaler_for(cfg, torch.device('cpu'))
    assert load_checkpoint(restored, restored_optimizer, restored_scheduler, checkpoint, torch.device('cpu')) == 1
    restored_scaler.load_state_dict(torch.load(checkpoint, weights_only=True)['extra']['scaler'])
    update(restored, restored_optimizer, restored_scheduler, restored_scaler)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    assert scaler.state_dict() == restored_scaler.state_dict()


def test_nonfinite_gradients_fail_before_optimizer_step():
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('nan'))
    scaler = scaler_for(load_protocol(), torch.device('cpu'))
    with pytest.raises(FloatingPointError, match='Non-finite'):
        optimizer_step(optimizer, model.parameters(), scaler, 1)
    for key in before:
        torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)


def test_fine_tune_requires_explicit_resets_and_strict_geometry(tmp_path):
    parent = torch.nn.Linear(2, 1)
    path = tmp_path / 'parent.pth'
    torch.save(dict(model=parent.state_dict()), path)
    cfg = load_protocol()
    cfg['training']['init_from'] = dict(path=str(path), reset=['optimizer', 'scheduler', 'rng', 'early_stopping'])
    child = torch.nn.Linear(2, 1)
    receipt = fine_tune(child, cfg, None)
    assert receipt['parent_sha256'] and receipt['parent_checkpoint'] == str(path)
    for left, right in zip(parent.parameters(), child.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='size mismatch'):
        fine_tune(torch.nn.Linear(3, 1), cfg, None)
    cfg['training']['init_from']['reset'].remove('rng')
    with pytest.raises(ValueError, match='explicit reset'):
        fine_tune(child, cfg, None)


def test_frozen_autoencoder_protocol_may_differ_in_optimizer_but_not_geometry(tmp_path):
    cfg = load_protocol()
    parent = copy.deepcopy(cfg)
    parent['training']['autoencoder_seed'] = 17
    path = tmp_path / 'autoencoder.yaml'
    path.write_text(yaml.safe_dump(parent))
    cfg['training']['autoencoder_protocol'] = str(path)
    assert autoencoder_protocol(cfg) == parent
    cfg['model']['autoencoder']['vector_quantizer']['num_embeddings'] = 32
    with pytest.raises(ValueError, match='architecture'):
        autoencoder_protocol(cfg)


def test_flow_fine_tune_cannot_silently_replace_the_frozen_autoencoder(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / 'flow.pth'
    torch.save(dict(model=model.state_dict(), extra=dict(provenance=dict(
        method='flow_uniform', autoencoder_sha256='parent-autoencoder'))), path)
    cfg = load_protocol()
    cfg['training']['init_from'] = dict(path=str(path), reset=['optimizer', 'scheduler', 'rng', 'early_stopping'])
    with pytest.raises(ValueError, match='same frozen autoencoder'):
        fine_tune(model, cfg, None, method='flow_uniform', autoencoder_sha256='another-autoencoder')


def test_pruning_handshake_ignores_stale_decision_and_reports_before_wait(tmp_path, monkeypatch):
    cfg = load_protocol()
    report, decision = tmp_path / 'report.json', tmp_path / 'decision.json'
    cfg['study_trial'] = dict(report_path=str(report), decision_path=str(decision), trial_number=3)
    write_json(decision, dict(step=250, prune=True))

    def controller(_):
        assert json.loads(report.read_text()) == dict(step=500, score=0.3, trial_number=3)
        write_json(decision, dict(step=500, prune=False))
    monkeypatch.setattr('scientific.training_control.time.sleep', controller)
    assert study_pruned(cfg, 500, 0.3) is False
    write_json(decision, dict(step=750, prune=True))
    assert study_pruned(cfg, 750, 0.4) is True


def test_pruning_controller_timeout_is_explicit(tmp_path, monkeypatch):
    cfg = load_protocol()
    cfg['study_trial'] = dict(report_path=str(tmp_path / 'report.json'), decision_path=str(tmp_path / 'decision.json'), trial_number=0)
    times = iter([0, 121])
    monkeypatch.setattr('scientific.training_control.time.monotonic', lambda: next(times))
    with pytest.raises(TimeoutError, match='checkpoint preserved'):
        study_pruned(cfg, 7500, 0.2)
    assert json.loads((tmp_path / 'report.json').read_text())['step'] == 7500


def test_fixed_objective_is_repeatable_and_preserves_all_rngs():
    cfg = load_protocol()
    cfg['diagnostics'] = dict(train_windows=2)
    class Dataset:
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return dict(frames=torch.ones(36, 3, 4, 4), alpha=torch.ones(36, 1, 4, 4))
    model = torch.nn.Dropout(0.5).train()
    def loss(model, *args):
        return torch.rand(()) + random.random() + np.random.random(), None
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    before = capture_rng_state()
    first = objective_snapshot(model, 'flow_uniform', Dataset(), cfg, torch.device('cpu'), loss, train_panel=True)
    second = objective_snapshot(model, 'flow_uniform', Dataset(), cfg, torch.device('cpu'), loss, train_panel=True)
    assert first == second and model.training
    after = capture_rng_state()
    assert before['python'] == after['python']
    torch.testing.assert_close(before['numpy'][1], after['numpy'][1], rtol=0, atol=0)
    torch.testing.assert_close(before['torch'], after['torch'], rtol=0, atol=0)


@pytest.fixture
def tiny_autoencoder_training(monkeypatch):
    from lutils.dict_wrapper import DictWrapper
    from scientific import runner
    import model.vqgan.discriminator as discriminator_module

    class Dataset:
        def __init__(self, *args):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return dict(frames=torch.ones(36, 3, 4, 4) * 0.2, alpha=torch.ones(36, 1, 4, 4))
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
        def forward(self, frames):
            return DictWrapper(reconstructed_images=frames * self.weight, vq_loss=self.weight.square())
    class Metric(torch.nn.Module):
        def forward(self, first, second):
            return (first - second).square().mean((1, 2, 3), keepdim=True)
    monkeypatch.setattr(runner, 'WindowDataset', Dataset)
    monkeypatch.setattr(runner, 'build_vqvae', lambda *_: Autoencoder())
    monkeypatch.setattr(discriminator_module, 'build_discriminator', lambda **_: torch.nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(runner, 'build_lpips', lambda *_: Metric())
    monkeypatch.setattr(runner, 'provenance', lambda *_: {})
    cfg = load_protocol()
    cfg['training'].update(max_updates=3, discriminator_start=0, autoencoder_seed=17)
    cfg['training']['autoencoder'].update(schedule='cosine', validation_every=1)
    return runner, cfg


def test_pruned_training_is_saved_and_never_marked_complete(tmp_path, monkeypatch, tiny_autoencoder_training):
    runner, cfg = tiny_autoencoder_training
    monkeypatch.setattr(runner, 'study_pruned', lambda *_: True)
    run = tmp_path / 'pruned'
    runner.train_autoencoder(cfg, tmp_path, run, torch.device('cpu'))
    completion = json.loads((run / 'completion.json').read_text())
    status = json.loads((run / 'run.json').read_text())
    assert completion == dict(status='pruned', termination_reason='pruning', completed_updates=1, planned_updates=3)
    assert status['status'] == 'pruned' and status['seed'] == 17
    checkpoint = torch.load(run / 'checkpoints/step_1.pth', weights_only=True)
    assert checkpoint['extra']['last_validated_step'] == 1
    assert not torch.load(run / 'best.pth', weights_only=True)['extra'].get('selected_artifact', False)


@pytest.mark.parametrize('overflow', [False, True])
def test_autoencoder_amp_cosine_resume_matches_uninterrupted(tmp_path, monkeypatch, tiny_autoencoder_training, overflow):
    runner, cfg = tiny_autoencoder_training
    cfg['training']['amp'] = True
    import model.vqgan.discriminator as discriminator_module
    original_discriminator = discriminator_module.build_discriminator
    inject = [overflow]
    def build_discriminator(**options):
        model = original_discriminator(**options)
        def fault(gradient):
            if inject[0]:
                inject[0] = False
                return torch.full_like(gradient, float('inf'))
            return gradient
        next(model.parameters()).register_hook(fault)
        return model
    monkeypatch.setattr(discriminator_module, 'build_discriminator', build_discriminator)
    runner.train_autoencoder(cfg, tmp_path, tmp_path / 'reference', torch.device('cpu'))
    inject[0] = overflow
    original = runner.study_pruned
    monkeypatch.setattr(runner, 'study_pruned', lambda *_: (_ for _ in ()).throw(InterruptedError('controller interrupted')))
    with pytest.raises(InterruptedError, match='interrupted'):
        runner.train_autoencoder(cfg, tmp_path, tmp_path / 'resumed', torch.device('cpu'))
    monkeypatch.setattr(runner, 'study_pruned', original)
    runner.train_autoencoder(cfg, tmp_path, tmp_path / 'resumed', torch.device('cpu'), resume=tmp_path / 'resumed/checkpoints/step_1.pth')
    reference = torch.load(tmp_path / 'reference/checkpoints/step_3.pth', weights_only=True)
    resumed = torch.load(tmp_path / 'resumed/checkpoints/step_3.pth', weights_only=True)
    for key in reference['model']:
        torch.testing.assert_close(reference['model'][key], resumed['model'][key], rtol=0, atol=0)
    for key in reference['extra']['discriminator']:
        torch.testing.assert_close(reference['extra']['discriminator'][key], resumed['extra']['discriminator'][key], rtol=0, atol=0)
    assert resumed['extra']['optimizer_d']['param_groups'][0]['lr'] == pytest.approx(2e-5)
    assert resumed['scheduler'] == reference['scheduler']
    assert resumed['extra']['scaler'] == reference['extra']['scaler']
    assert resumed['extra']['amp_skipped_updates'] == reference['extra']['amp_skipped_updates'] == int(overflow)
    torch.testing.assert_close(reference['rng']['torch'], resumed['rng']['torch'], rtol=0, atol=0)


def test_finite_loss_with_scaled_overflow_backs_off_before_any_step():
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.SGD([parameter], lr=1e-36)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536)
    loss = parameter * 1e35
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    assert optimizer_step(optimizer, [parameter], scaler, 1) is None
    scaler.update()
    assert parameter.item() == 0.5 and scaler.get_scale() == 32768
    assert not optimizer.state


def test_optimizer_commit_interruption_keeps_interrupt_identity(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.square().backward()
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    monkeypatch.setattr(optimizer, 'step', lambda **_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt) as interrupted:
        optimizer_step(optimizer, [parameter], scaler, 1)
    assert interrupted.value.checkpoint_unsafe is True


@pytest.mark.parametrize('overflow_in', ['generator', 'discriminator'])
def test_joint_amp_overflow_replays_rng_and_buffers_before_both_steps(monkeypatch, overflow_in):
    from lutils.dict_wrapper import DictWrapper
    from scientific import runner

    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.dropout = torch.nn.Dropout(0.1)
        def forward(self, images):
            return DictWrapper(reconstructed_images=self.dropout(images) * self.weight, vq_loss=self.weight.square())
    class Discriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.register_buffer('calls', torch.zeros(()))
        def forward(self, images):
            self.calls.add_(1)
            return images.mean((1, 2, 3), keepdim=True) * self.weight + torch.rand(()) * 0.01
    class Metric(torch.nn.Module):
        def forward(self, first, second):
            return (first - second).square().mean()
    def batch(*_):
        return torch.ones(1, 36, 3, 4, 4) * 0.2, None
    monkeypatch.setattr(runner, 'batch_at', batch)
    cfg = load_protocol()
    cfg['training'].update(amp=True, discriminator_start=0)
    settings = autoencoder_settings(cfg)
    generator, discriminator = Generator(), Discriminator()
    original = copy.deepcopy(generator.state_dict()), copy.deepcopy(discriminator.state_dict())
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0002)
    scaler = torch.amp.GradScaler('cpu', init_scale=65536)
    inject = [True]
    def fault(gradient):
        if inject[0]:
            inject[0] = False
            return torch.full_like(gradient, float('inf'))
        return gradient
    (generator if overflow_in == 'generator' else discriminator).weight.register_hook(fault)
    torch.manual_seed(42)
    before = capture_rng_state()
    result = runner._autoencoder_update(generator, discriminator, optimizer, optimizer_d, scaler,
        Metric(), None, 0, cfg, settings, torch.device('cpu'), True)
    assert result is None and scaler.get_scale() == 32768
    assert not optimizer.state and not optimizer_d.state
    for module, state in zip((generator, discriminator), original):
        for key, value in state.items():
            torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(before['torch'], capture_rng_state()['torch'], rtol=0, atol=0)
    # Retry from the recovered boundary, then compare with an otherwise identical
    # update that started at the reduced scale and saw the same stochastic draws.
    result = runner._autoencoder_update(generator, discriminator, optimizer, optimizer_d, scaler,
        Metric(), None, 0, cfg, settings, torch.device('cpu'), True)
    assert result is not None
    after = capture_rng_state()
    reference_g, reference_d = Generator(), Discriminator()
    reference_g.load_state_dict(original[0])
    reference_d.load_state_dict(original[1])
    reference_optimizer = torch.optim.AdamW(reference_g.parameters(), lr=0.0002)
    reference_optimizer_d = torch.optim.AdamW(reference_d.parameters(), lr=0.0002)
    reference_scaler = torch.amp.GradScaler('cpu', init_scale=32768)
    restore_rng_state(before)
    runner._autoencoder_update(reference_g, reference_d, reference_optimizer, reference_optimizer_d,
        reference_scaler, Metric(), None, 0, cfg, settings, torch.device('cpu'), True)
    for actual, reference in ((generator, reference_g), (discriminator, reference_d)):
        for key in actual.state_dict():
            torch.testing.assert_close(actual.state_dict()[key], reference.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(after['torch'], capture_rng_state()['torch'], rtol=0, atol=0)


def test_non_amp_discriminator_invalid_grad_does_not_commit_generator(tmp_path, monkeypatch, tiny_autoencoder_training):
    runner, cfg = tiny_autoencoder_training
    import model.vqgan.discriminator as discriminator_module
    class Discriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.register_buffer('calls', torch.zeros(()))
            self.weight.register_hook(lambda value: torch.full_like(value, float('inf')))
        def forward(self, images):
            self.calls.add_(1)
            return images.mean((1, 2, 3), keepdim=True) * self.weight
    monkeypatch.setattr(discriminator_module, 'build_discriminator', lambda **_: Discriminator())
    with pytest.raises(FloatingPointError, match='Non-finite training gradients'):
        runner.train_autoencoder(cfg, tmp_path, tmp_path / 'failed', torch.device('cpu'))
    saved = torch.load(tmp_path / 'failed/checkpoints/step_0.pth', weights_only=True)
    assert saved['step'] == 0 and saved['model']['weight'] == 0.5
    assert saved['extra']['discriminator']['weight'] == 0.5
    assert saved['extra']['discriminator']['calls'] == 0


def test_persistent_amp_overflow_exits_with_coherent_checkpoint(tmp_path, monkeypatch, tiny_autoencoder_training):
    runner, cfg = tiny_autoencoder_training
    cfg['training']['amp'] = True
    import model.vqgan.discriminator as discriminator_module
    original = discriminator_module.build_discriminator
    def build(**options):
        model = original(**options)
        next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float('inf')))
        return model
    monkeypatch.setattr(discriminator_module, 'build_discriminator', build)
    with pytest.raises(FloatingPointError, match='bounded scaler recovery'):
        runner.train_autoencoder(cfg, tmp_path, tmp_path / 'failed', torch.device('cpu'))
    state = torch.load(tmp_path / 'failed/checkpoints/step_0.pth', weights_only=True)
    assert state['step'] == 0 and state['model']['weight'] == 0.5
    assert state['scheduler']['last_epoch'] == 0
    assert not state['optimizer']['state'] and not state['extra']['optimizer_d']['state']
    assert state['extra']['amp_skipped_updates'] == 32
    assert json.loads((tmp_path / 'failed/run.json').read_text())['completed_updates'] == 0
