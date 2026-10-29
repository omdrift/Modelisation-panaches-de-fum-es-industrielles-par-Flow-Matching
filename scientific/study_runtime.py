"""Durable subprocess trials and Optuna's validation/pruning handshake."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import time
import warnings

import yaml

from scientific.pipeline import execute_process
from scientific.protocol import digest, file_hash, write_json


def _remove_redundant(path):
    try:
        path.unlink()
    except OSError as exc:
        warnings.warn(f'Checkpoint cleanup retained {path}: {exc}', RuntimeWarning)


def trial_protocol(base, settings, output, phase, candidate, method, seed, *, autoencoder=None, handshake=False, trial_number=None):
    """Resolve an immutable protocol; old CLI defaults are never modified."""
    cfg = copy.deepcopy(base)
    root = Path(output).resolve()
    directory = root / 'runs' / f'{phase}_{candidate["id"]}_{seed}'
    cfg['study'] = dict(root=str(root), phase=phase, candidate=candidate['id'])
    cfg['training']['max_updates'] = settings['search'][
        'exploration_updates' if phase.endswith('exploration') else
        'probe_updates' if phase == 'transfer' else 'confirmation_updates']
    cfg['training']['early_stopping'] = dict(enabled=True, patience=10 if method == 'vqgan' else 4,
        min_updates=5000, min_delta=0.0001)
    cfg['training'].pop('init_from', None)
    if method == 'vqgan':
        cfg['training']['autoencoder_seed'] = seed
        cfg['training']['autoencoder'].update(learning_rate=candidate['learning_rate'],
            discriminator_learning_rate=candidate['learning_rate'], schedule=candidate['schedule'],
            schedule_min_factor=0.1, validation_every=250)
        cfg['training']['discriminator_start'] = 1000
    else:
        cfg['training'].update(learning_rate=candidate['learning_rate'], schedule=candidate['schedule'],
            schedule_min_factor=0.1, validation_every=2500, warmup_updates=1000,
            autoencoder_protocol=str(Path(autoencoder['protocol']).resolve()))
    cfg['diagnostics'] = dict(settings['diagnostics'], forecast_realizations=1 if phase.endswith('exploration') else 4,
        reconstruction_all_frames=True, independent_objective=True)
    if handshake:
        cfg['study_trial'] = dict(report_path=str(directory / 'validation_report.json'),
            decision_path=str(directory / 'prune_decision.json'), decision_timeout_seconds=120,
            trial_number=candidate['index'] if trial_number is None else trial_number)
    return cfg, directory


def verify_result(record):
    """Do not silently reuse replaced checkpoints or modified trial protocols."""
    for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('protocol', 'protocol_sha256')]:
        path = Path(record[path_key])
        if not path.is_file() or file_hash(path) != record[hash_key]:
            raise ValueError(f'Completed trial artifact changed: {path}')


def _tracking(args, label):
    if not args.wandb:
        return ()
    options = ('--wandb', '--wandb-project', args.wandb_project, '--wandb-mode', args.wandb_mode,
        '--wandb-group', args.wandb_group or Path(args.output).name, '--wandb-name', label,
        '--wandb-log-every', str(args.wandb_log_every), '--wandb-media-every', str(args.wandb_media_every))
    return options + (('--wandb-entity', args.wandb_entity) if args.wandb_entity else ())


def validation_heartbeat(directory, trial, state, persist):
    """Report only newly completed, finite LPIPS validations at matching steps."""
    report = Path(directory) / 'validation_report.json'
    if trial is None or not report.is_file():
        return
    value = json.loads(report.read_text())
    step, score = value['step'], value['score']
    if type(step) is not int or step < 1 or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('Invalid trial validation report')
    # Re-acknowledge a report after supervisor restart, without duplicate reporting.
    if step not in trial.storage.get_trial(trial._trial_id).intermediate_values:
        trial.report(score, step)
    prune = bool(trial.should_prune())
    write_json(Path(directory) / 'prune_decision.json', dict(step=step, prune=prune))
    state.update(last_validation_step=step, last_validation_lpips=score)
    persist()


def run_trial(base, settings, args, state, persist, phase, candidate, method, seed,
              *, autoencoder=None, trial=None, executor=execute_process):
    cfg, directory = trial_protocol(base, settings, args.output, phase, candidate, method, seed,
        autoencoder=autoencoder, handshake=trial is not None, trial_number=trial.number if trial is not None else None)
    key = directory.name
    runs = state.setdefault('runs', {})
    previous = runs.get(key, {})
    if previous.get('status') in ('complete', 'pruned'):
        if previous['status'] == 'complete':
            verify_result(previous)
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'protocol.yaml'
    if frozen.exists() and yaml.safe_load(frozen.read_text()) != cfg:
        raise ValueError(f'Trial protocol changed: {key}')
    if not frozen.exists():
        frozen.write_text(yaml.safe_dump(cfg, sort_keys=False))
    run = directory / ('autoencoder' if method == 'vqgan' else f'{method}_{seed}')
    record = runs[key] = previous | dict(id=key, phase=phase, candidate=candidate['id'], method=method, seed=seed,
        learning_rate=candidate['learning_rate'], schedule=candidate['schedule'],
        protocol=str(frozen), protocol_sha256=file_hash(frozen), status='running',
        planned_updates=cfg['training']['max_updates'], autoencoder=autoencoder)
    if trial is not None:
        record['optuna_trial_number'] = trial.number
    persist()
    base_command = (sys.executable, '-u', '-m', 'scientific.cli', '--protocol', str(frozen))
    log = directory / 'process.log'
    started = time.monotonic()

    def heartbeat():
        validation_heartbeat(directory, trial, record, persist)

    try:
        if method == 'vqgan' and not (directory / 'pilot_autoencoder.json').exists():
            record['pilot_command'] = list(base_command + ('pilot-autoencoder', '--campaign', str(directory)))
            persist()
            executor(tuple(record['pilot_command']), log, lambda: None)
        if method != 'vqgan' and not (directory / 'pilot.json').exists():
            record['pilot_command'] = list(base_command + ('pilot', '--campaign', str(directory)))
            persist()
            executor(tuple(record['pilot_command']), log, lambda: None)
        command = base_command + ('autoencoder' if method == 'vqgan' else 'train',
            '--campaign', str(directory), '--data', str(Path(args.data).resolve()))
        if method != 'vqgan':
            command += ('--method', method, '--seed', str(seed), '--autoencoder', autoencoder['checkpoint'],
                '--pilot', str(directory / 'pilot.json'))
        command += _tracking(args, key)
        if (run / 'run.json').exists():
            if not any((run / 'checkpoints').glob('step_*.pth')):
                raise RuntimeError(f'Interrupted trial has no recovery checkpoint: {run}')
            command += ('--resume', 'latest')
        record['command'] = list(command)
        persist()
        executor(command, log, heartbeat)
        heartbeat()
        metadata = json.loads((run / 'run.json').read_text())
        if metadata['status'] not in ('complete', 'pruned'):
            raise RuntimeError(f'Trial did not complete: {key}')
        if metadata.get('protocol_sha256') != digest(cfg):
            raise ValueError('Trial produced another protocol identity')
        record.update(status=metadata['status'], completed_updates=metadata['completed_updates'],
            best_validation_lpips=metadata.get('best_validation_lpips'), run_dir=str(run),
            source_sha256=metadata.get('source_sha256'), dataset_sha256=metadata.get('dataset_sha256'),
            gpu=metadata.get('gpu'), torch=metadata.get('torch'))
        history = run / 'metrics.jsonl'
        if history.is_file():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            best_steps = [row['selection/best_step'] for row in rows if 'selection/best_step' in row]
            memory = [row['gpu/peak_allocated_bytes'] for row in rows if 'gpu/peak_allocated_bytes' in row]
            if best_steps:
                record['best_step'] = int(best_steps[-1])
            if memory:
                record['peak_memory_bytes'] = max(memory)
        if metadata['status'] == 'complete':
            score = metadata.get('best_validation_lpips')
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('Completed trial must have a finite validation score')
            checkpoint = run / 'best.pth'
            record.update(checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint))
            best = run / 'diagnostics' / 'best.json'
            if best.is_file():
                record['best_step'] = json.loads(best.read_text()).get('step')
        return record
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = record.get('seconds', 0) + time.monotonic() - started
        if sys.exc_info()[0] is None:
            persist()
        else:
            try:
                persist()
            except Exception as cleanup_error:
                warnings.warn(f'Trial status cleanup also failed: {cleanup_error}', RuntimeWarning)
        if record.get('status') == 'complete':
            # The durable registry now owns terminal resume. Finalists retain
            # their full selected state; numbered recovery copies are redundant.
            for checkpoint in (run / 'checkpoints').glob('step_*.pth'):
                _remove_redundant(checkpoint)


def compact_closed_trial(record, persist=lambda: None):
    """Keep a closed exploratory model, not its disposable optimizer snapshots."""
    if record['status'] != 'complete':
        return
    import torch
    from training.checkpoints import atomic_save
    path = Path(record['checkpoint'])
    if not record.get('retention'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        original_hash = file_hash(path)
        extra = {k: v for k, v in state.get('extra', {}).items()
            if k not in ('discriminator', 'optimizer_d', 'scheduler_d', 'scaler')}
        compact = path.with_name('weights.pth')
        atomic_save(dict(model=state['model'], step=state['step'], extra=extra), compact)
        receipt = dict(policy='closed_weights_only', original_checkpoint=str(path),
            original_sha256=original_hash, retained_sha256=file_hash(compact), exact_resume=False)
        write_json(path.parent / 'retention.json', receipt)
        record.update(checkpoint=str(compact), checkpoint_sha256=receipt['retained_sha256'],
            retention=receipt['policy'], original_full_checkpoint_sha256=original_hash)
        # Commit the pointer before deleting anything: every crash window is recoverable.
        persist()
        path = compact
    receipt = json.loads((path.parent / 'retention.json').read_text())
    original = Path(receipt['original_checkpoint'])
    if original.is_file():
        if file_hash(original) != receipt['original_sha256']:
            raise ValueError('Refusing to remove a modified full checkpoint')
        _remove_redundant(original)
    # Only files owned by this new, completed trial are eligible for deletion.
    for checkpoint in (path.parent / 'checkpoints').glob('step_*.pth'):
        _remove_redundant(checkpoint)
