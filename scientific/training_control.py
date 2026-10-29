"""Opt-in study controls shared by both training loops.

These helpers do not alter a frozen legacy recipe unless its configuration
explicitly enables a scheduler, mixed precision, fine-tuning or study handshake.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import numpy as np

from scientific.hyperparameters import schedule_factor
from scientific.monitoring import preserve_rng
from scientific.protocol import file_hash, load_protocol, write_json
from training.checkpoints import capture_rng_state, restore_rng_state


def scheduler_for(optimizer, cfg, settings, *, warmup=0):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(
        step, cfg['training']['max_updates'], settings, warmup=warmup))


def scaler_for(cfg, device):
    return torch.amp.GradScaler(torch.device(device).type, enabled=cfg['training'].get('amp', False))


def autocast_for(cfg, device):
    return torch.autocast(device_type=torch.device(device).type, enabled=cfg['training'].get('amp', False))


def optimizer_step(optimizer, parameters, scaler, maximum):
    """Step only after preflight; None asks the caller to back off the scaler."""
    result = optimizer_steps([(optimizer, parameters, maximum)], scaler)
    return None if result is None else result[0]


class OptimizerCommitError(RuntimeError):
    """An optimizer failed during commit; only an earlier checkpoint is safe."""


def optimizer_steps(groups, scaler):
    """Preflight every optimizer before applying any of their parameter changes.

    Scaled-gradient overflow is recoverable with AMP. The caller updates the
    scaler and retries the same batch; a true non-AMP invalid gradient is fatal.
    """
    groups = [(optimizer, list(parameters), maximum) for optimizer, parameters, maximum in groups]
    for optimizer, _, _ in groups:
        scaler.unscale_(optimizer)
    gradients = [parameter.grad for _, parameters, _ in groups for parameter in parameters if parameter.grad is not None]
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        if scaler.is_enabled():
            return None
        raise FloatingPointError('Non-finite training gradients; update was not applied')
    norms = []
    for _, parameters, _ in groups:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        norm = torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=False)
        if not torch.isfinite(norm):
            # Finite large values can overflow the float32 norm reduction even
            # after unscaling. Accumulate this uncommon case in float64.
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients]))
        if not torch.isfinite(norm):
            raise FloatingPointError('Non-finite gradient norm; update was not applied')
        norms.append(norm)
    for (_, parameters, maximum), norm in zip(groups, norms):
        torch.nn.utils.clip_grads_with_norm_(parameters, maximum, norm)
    try:
        for optimizer, _, _ in groups:
            scaler.step(optimizer)
    except BaseException as exc:
        # Optimizer allocation/OOM can fail after a preceding optimizer has
        # changed weights. Never serialize that partial joint update as a step.
        if not isinstance(exc, Exception):
            exc.checkpoint_unsafe = True
            raise  # Keep user interruption distinguishable from a failed trial.
        raise OptimizerCommitError('Optimizer commit failed; recover from the previous checkpoint') from exc
    return norms


class UpdateTransaction:
    """Replay stochastic draws and mutable buffers when an update is discarded."""

    def __init__(self, *modules):
        self.modules = modules
        self.committed = False

    def __enter__(self):
        self.rng = capture_rng_state()
        self.buffers = [(buffer, buffer.detach().clone()) for module in self.modules for buffer in module.buffers()]
        return self

    def commit(self):
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if not self.committed:
            restore_rng_state(self.rng)
            with torch.no_grad():
                for buffer, before in self.buffers:
                    buffer.copy_(before)


def fine_tune(model, cfg, resume, *, method=None, autoencoder_sha256=None):
    """Load model weights only, with explicit reset policy and a parent receipt."""
    setting = cfg['training'].get('init_from')
    if setting is None:
        return None
    resets = {'optimizer', 'scheduler', 'rng', 'early_stopping'}
    if (not isinstance(setting, dict) or set(setting) != {'path', 'reset'}
            or not isinstance(setting['reset'], list) or not all(isinstance(name, str) for name in setting['reset'])
            or len(setting['reset']) != len(set(setting['reset']))
            or not resets <= set(setting['reset']) <= resets | {'discriminator'}):
        raise ValueError('init_from requires path and explicit reset of optimizer, scheduler, rng and early_stopping')
    if resume:
        # A resumed fine-tuning experiment still has init_from in its immutable
        # configuration; its own checkpoint, not the parent, is restored below.
        return dict(parent_checkpoint=str(Path(setting['path']).resolve()),
                    parent_sha256=file_hash(setting['path']), reset=setting['reset'])
    path = Path(setting['path']).resolve()
    state = torch.load(path, map_location='cpu', weights_only=True)
    origin = state.get('extra', {}).get('provenance', {})
    if origin.get('method') is not None and method is not None and origin['method'] != method:
        raise ValueError('Fine-tuning parent belongs to a different method')
    if autoencoder_sha256 is not None and origin.get('autoencoder_sha256') != autoencoder_sha256:
        raise ValueError('Flow fine-tuning requires the same frozen autoencoder artifact as its parent')
    weights = state.get('model', state)
    model.load_state_dict({name.removeprefix('module.'): value for name, value in weights.items()}, strict=True)
    return dict(parent_checkpoint=str(path), parent_sha256=file_hash(path), reset=setting['reset'])


def autoencoder_protocol(cfg):
    path = cfg['training'].get('autoencoder_protocol')
    if path is None:
        return cfg
    if not Path(path).is_absolute():
        raise ValueError('autoencoder_protocol must identify an absolute frozen protocol path')
    parent = load_protocol(path)
    architecture = lambda value: {key: item for key, item in value['model']['autoencoder'].items() if key != 'ckpt_path'}
    if architecture(parent) != architecture(cfg):
        raise ValueError('Frozen autoencoder architecture differs from the predictor protocol')
    return parent


def study_pruned(cfg, step, score):
    """Cooperate with the sequential Optuna controller at saved validations."""
    trial = cfg.get('study_trial')
    if not trial or 'report_path' not in trial:
        return False
    timeout = trial.get('decision_timeout_seconds', 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError('Study decision timeout must be positive')
    report_path, decision_path = Path(trial['report_path']), Path(trial['decision_path'])
    write_json(report_path, dict(step=step, score=float(score), trial_number=trial['trial_number']))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text())
            if decision.get('step') == step:
                if type(decision.get('prune')) is not bool:
                    raise ValueError('Study decision requires a boolean prune value')
                return decision['prune']
        time.sleep(0.1)
    raise TimeoutError(f'Study controller did not acknowledge validation at update {step}; checkpoint preserved')


@preserve_rng()
@torch.inference_mode()
def objective_snapshot(model, method, dataset, cfg, device, loss_function, *, train_panel=False):
    """A fixed independent objective panel; all stochastic draws are isolated."""
    from scientific.protocol import stratified_selection

    count = cfg.get('diagnostics', {}).get('train_windows', 60) if train_panel else cfg['selection']['validation_windows']
    records = getattr(dataset, 'records', None)
    if records:
        selected = stratified_selection(records, min(count, len(records)), cfg['selection_seed'])
        by_id = {record['window_id']: index for index, record in enumerate(records)}
        indices = [by_id[record['window_id']] for record in selected]
    else:
        indices = list(range(min(count, len(dataset))))
    modes = [(module, module.training) for module in model.modules()]
    values = []
    try:
        model.eval()
        for rank, index in enumerate(indices):
            random.seed(84000 + rank)
            np.random.seed(84000 + rank)
            torch.manual_seed(84000 + rank)
            if torch.device(device).type == 'cuda':
                torch.cuda.manual_seed_all(84000 + rank)
            item = dataset[index]
            value = loss_function(model, method, item['frames'][None].to(device), item['alpha'][None].to(device), cfg)[0]
            if not torch.isfinite(value):
                raise FloatingPointError('Non-finite independent validation objective')
            values.append(float(value))
    finally:
        for module, was_training in modes:
            module.training = was_training
    if not values:
        raise ValueError('Independent objective requires a nonempty fixed panel')
    return sum(values) / len(values)
