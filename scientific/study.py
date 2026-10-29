"""Validation-only, staged VQGAN and Flow Matching configuration comparisons."""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import csv
import fcntl
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import warnings

import yaml

from scientific.pipeline import execute_process
from scientific.protocol import _source_manifest, digest, file_hash, load_protocol, write_json, validate_prepared_data
from scientific.study_runtime import compact_closed_trial, run_trial, verify_result


def load_study(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or cfg.get('version') != 1:
        raise ValueError('Unsupported study version')
    if set(cfg) != {'version', 'name', 'search', 'pruning', 'diagnostics', 'final_test'}:
        raise ValueError('Unknown or missing study fields; extend the version before changing the experiment space')
    search = cfg['search']
    if set(search) != {'learning_rates', 'schedules', 'exploration_updates', 'confirmation_updates',
            'probe_updates', 'exploration_seed', 'confirmation_seeds', 'finalists'}:
        raise ValueError('Unknown or missing search factors; study v1 varies only learning rate and schedule')
    if (search['confirmation_seeds'] != [17, 42, 2026] or search['exploration_seed'] != 42
            or search['finalists'] != 2):
        raise ValueError('Study v1 fixes three confirmation seeds and two finalists')
    rates, schedules = search['learning_rates'], search['schedules']
    if (len(rates) != 3 or len(set(rates)) != 3 or any(type(r) not in (int, float) or not math.isfinite(r) or r <= 0 for r in rates)
            or schedules != ['constant', 'cosine']):
        raise ValueError('Predeclare three distinct positive rates and constant/cosine schedules')
    for key in ('exploration_updates', 'confirmation_updates', 'probe_updates'):
        if type(search[key]) is not int or search[key] < 7500:
            raise ValueError(f'{key} must include the 7500-update pruning warm-up')
    pruning = cfg['pruning']
    if pruning != dict(startup_trials=3, min_updates=7500):
        raise ValueError('Study v1 fixes pruning startup=3 and min_updates=7500')
    if cfg['diagnostics'] != dict(train_windows=60, visual_windows=8, export_sequences=True):
        raise ValueError('Study v1 uses 60 training diagnostic windows and eight visual examples')
    if type(cfg.get('final_test')) is not bool:
        raise ValueError('final_test must be an explicit boolean')
    return cfg


def candidates(settings):
    return [dict(id=f'c{index:02d}', index=index, learning_rate=rate, schedule=schedule)
        for index, (rate, schedule) in enumerate(itertools.product(
            settings['search']['learning_rates'], settings['search']['schedules']), 1)]


def study_plan(base, settings):
    if (base['evaluation']['context_frames'], base['evaluation']['horizon']) != (5, 15):
        raise ValueError('Study v1 fixes five observations and fifteen predictions')
    return dict(name=settings['name'], task=dict(clip_frames=36, resolution=64, context_frames=5, horizon=15),
        candidates=candidates(settings), maximum_training_runs=29,
        phases=[dict(name='vqgan_exploration', runs=6), dict(name='vqgan_confirmation', runs=6),
                dict(name='transfer', runs=2), dict(name='flow_exploration', runs=6),
                dict(name='flow_confirmation', runs=6), dict(name='alpha_confirmation', runs=3)],
        selection='validation LPIPS only; fixed initialization and generation seeds',
        test_access='after final_selection.json is frozen', final_test=settings['final_test'])


def select_finalists(records, count=2):
    eligible = [r for r in records if r.get('status') == 'complete']
    if len(eligible) < count:
        raise RuntimeError('Not enough completed, finite candidates to select finalists')
    if any(not math.isfinite(r['best_validation_lpips']) for r in eligible):
        raise ValueError('Non-finite finalist score')
    return sorted(eligible, key=lambda r: (r['best_validation_lpips'], r['candidate']))[:count]


def confirmation_means(records, seeds):
    groups = defaultdict(list)
    for record in records:
        if record.get('status') != 'complete':
            raise ValueError('Every confirmation must complete before selection')
        groups[record['candidate']].append(record)
    result = {}
    for candidate, values in groups.items():
        if sorted(r['seed'] for r in values) != sorted(seeds):
            raise ValueError('Confirmation must contain each predeclared seed exactly once')
        scores = [r['best_validation_lpips'] for r in values]
        if any(not math.isfinite(value) for value in scores):
            raise ValueError('Non-finite confirmation score')
        result[candidate] = dict(mean=statistics.mean(scores), std=statistics.stdev(scores),
            by_seed={str(record['seed']): record['best_validation_lpips'] for record in values})
    return result


def _optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Study execution requires optuna==4.6.0; dry-run needs no Optuna') from exc
    if optuna.__version__ != '4.6.0':
        raise RuntimeError('Use the pinned optuna==4.6.0 to execute or resume this study')
    return optuna


def _failure_category(directory):
    receipt = Path(directory) / 'failure.json'
    return json.loads(receipt.read_text()).get('category') if receipt.is_file() else None


def _search(base, settings, args, state, persist, method, *, autoencoder=None, executor=execute_process):
    optuna = _optuna()
    phase = 'vqgan_exploration' if method == 'vqgan' else 'flow_exploration'
    study = optuna.create_study(study_name=phase,
        storage=f'sqlite:///{Path(args.output).resolve() / "optuna.sqlite3"}', direction='minimize',
        # All six tuples are explicitly enqueued below; no adaptive suggestions
        # or GridSampler.stop() side effects enter this ask/tell controller.
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=7500, n_min_trials=3),
        load_if_exists=True)
    result = []
    for candidate in candidates(settings):
        trials = [t for t in study.get_trials(deepcopy=False) if t.user_attrs.get('candidate') == candidate['id']]
        if trials:
            frozen = trials[-1]
            # Optuna 4.6's public Trial constructor takes its storage ID, not its number.
            trial = optuna.trial.Trial(study, frozen._trial_id)
            if frozen.state == optuna.trial.TrialState.WAITING:
                trial = study.ask()
                if trial.user_attrs.get('candidate') != candidate['id']:
                    raise ValueError('Queued Optuna candidate differs from the declared matrix')
                trial.suggest_categorical('learning_rate', settings['search']['learning_rates'])
                trial.suggest_categorical('schedule', settings['search']['schedules'])
            if frozen.state == optuna.trial.TrialState.FAIL:
                record = state['runs'][f'{phase}_{candidate["id"]}_42']
                if record.get('failure_category') is None:
                    record['failure_category'] = _failure_category(Path(args.output) / 'runs' / record['id'])
                if record.get('failure_category') not in ('out_of_memory', 'non_finite'):
                    raise RuntimeError(f'Failed Optuna trial {frozen.number}: inspect its recorded error')
                result.append(record)
                persist()
                continue
        else:
            params = dict(learning_rate=candidate['learning_rate'], schedule=candidate['schedule'])
            study.enqueue_trial(params, user_attrs=dict(candidate=candidate['id']))
            trial = study.ask()
            trial.suggest_categorical('learning_rate', settings['search']['learning_rates'])
            trial.suggest_categorical('schedule', settings['search']['schedules'])
        if study.get_trials(deepcopy=False)[trial.number].state == optuna.trial.TrialState.RUNNING:
            if (trial.suggest_categorical('learning_rate', settings['search']['learning_rates']) != candidate['learning_rate']
                    or trial.suggest_categorical('schedule', settings['search']['schedules']) != candidate['schedule']):
                raise ValueError('Optuna trial parameters differ from its frozen candidate')
        try:
            record = run_trial(base, settings, args, state, persist, phase, candidate, method, 42,
                autoencoder=autoencoder, trial=trial, executor=executor)
        except subprocess.CalledProcessError:
            directory = Path(args.output).resolve() / 'runs' / f'{phase}_{candidate["id"]}_42'
            category = _failure_category(directory)
            if category not in ('out_of_memory', 'non_finite'):
                raise
            record = state['runs'][directory.name]
            record['failure_category'] = category
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            persist()
            result.append(record)
            continue
        frozen = study.get_trials(deepcopy=False)[trial.number]
        if frozen.state == optuna.trial.TrialState.RUNNING:
            if record['status'] == 'pruned':
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(trial, record['best_validation_lpips'])
        if record['status'] == 'complete':
            compact_closed_trial(record, persist)
        persist()
        result.append(record)
    return result


def _confirm(base, settings, args, state, persist, phase, method, selected, *, autoencoder=None, executor=execute_process):
    matrix = {candidate['id']: candidate for candidate in candidates(settings)}
    return [run_trial(base, settings, args, state, persist, phase, matrix[record['candidate']], method, seed,
        autoencoder=autoencoder, executor=executor)
        for record in selected for seed in settings['search']['confirmation_seeds']]


def export_report(root, state):
    """Human-readable rankings share the exact durable JSON records."""
    root = Path(root)
    rows = []
    for record in state.get('runs', {}).values():
        rows.append({key: record.get(key) for key in ('id', 'phase', 'candidate', 'method', 'seed',
            'learning_rate', 'schedule', 'status', 'best_validation_lpips', 'best_step',
            'completed_updates', 'planned_updates', 'seconds', 'peak_memory_bytes', 'gpu', 'torch',
            'optuna_trial_number', 'checkpoint', 'checkpoint_sha256', 'protocol_sha256', 'source_sha256', 'dataset_sha256')})
    if rows:
        with (root / 'experiments.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(root / 'report.json', dict(status=state['status'], identity=state['identity'], experiments=rows,
        selection=state.get('selection'), uncertainty=state.get('confirmation_scores'),
        scientific_results_available=state['status'] == 'complete' and bool(state.get('evaluations'))))
    lines = ['# Industrial Plume Forecasting — comparaison des configurations', '',
        f"État : {state['status']}. Les classements utilisent exclusivement la validation.", '',
        '| Essai | Méthode | Graine | LR | Scheduler | LPIPS validation | État |',
        '|---|---|---:|---:|---|---:|---|']
    for row in rows:
        score = row['best_validation_lpips']
        lines.append(f"| {row['id']} | {row['method']} | {row['seed']} | {row['learning_rate']} | {row['schedule']} | {score if score is not None else '—'} | {row['status']} |")
    lines += ['', 'Les scores exploratoires Flow utilisent un tirage ; les confirmations en utilisent quatre.',
        'Le test est réservé aux modèles gelés. Un faible écart train/validation ne prouve pas une absence de surapprentissage.',
        'Le transfert entre VQGAN est comparé avec une seule graine Flow (42) ; cette comparaison reste exploratoire.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


def _freeze(root, state, autoencoder, uniforms, alphas):
    records = [autoencoder, *uniforms, *alphas]
    selection = dict(version=1, study_identity_sha256=digest(state['identity']),
        selected_before_test=True, artifacts=[{key: record[key] for key in
            ('id', 'method', 'seed', 'protocol', 'protocol_sha256', 'checkpoint', 'checkpoint_sha256')} for record in records])
    path = root / 'final_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError('Final model selection is immutable after test access')
    if not path.exists():
        write_json(path, selection)
    return selection


def run_study(base, args, *, executor=execute_process):
    settings = load_study(args.config)
    plan = study_plan(base, settings)
    if args.dry_run:
        return plan
    _optuna()  # Fail before creating study artifacts when the dependency is missing.
    root, data = Path(args.output).resolve(), Path(args.data).resolve()
    manifest = data / 'dataset_manifest.json'
    validate_prepared_data(json.loads(manifest.read_text()), base)
    identity = dict(study_sha256=digest(settings), protocol_sha256=digest(base),
        dataset_sha256=file_hash(manifest), source_sha256=digest(_source_manifest()),
        data=str(data), python=platform.python_version(), optuna='4.6.0',
        tracking=dict(enabled=args.wandb, project=args.wandb_project, entity=args.wandb_entity,
            mode=args.wandb_mode, group=args.wandb_group))
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'study.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns this study') from exc
        path = root / 'study.json'
        if path.exists():
            if not args.resume:
                raise FileExistsError('Study exists; use --resume')
            state = json.loads(path.read_text())
            if state['identity'] != identity:
                raise ValueError('Study resume requires identical code, data, configuration and tracking destination')
        else:
            if args.resume:
                raise FileNotFoundError('No study to resume')
            state = dict(identity=identity, plan=plan, runs={}, status='running')
            (root / 'study.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))

        def persist():
            write_json(path, state)
            export_report(root, state)

        state['status'] = 'running'
        state.pop('error', None)
        persist()
        tracking = nullcontext(None)
        if args.wandb:
            from scientific.monitoring import WandbMonitor, WandbOptions
            tracking = WandbMonitor(root / 'study_monitor', base, 'study', 42,
                WandbOptions(project=args.wandb_project, entity=args.wandb_entity,
                    mode=args.wandb_mode, group=args.wandb_group or root.name,
                    name=args.wandb_name or root.name + '-study'), resume=args.resume)
        try:
            with tracking as monitor:
                def observed_persist():
                    persist()
                    if monitor is not None:
                        completed = sum(r['status'] in ('complete', 'pruned') for r in state['runs'].values())
                        monitor.log(completed, {'study/completed_trials': completed, 'study/maximum_trials': 29})

                _execute_study(base, settings, args, root, state, observed_persist, executor)
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                error=f'{type(exc).__name__}: {exc}')
            try:
                persist()
            except Exception as cleanup_error:
                warnings.warn(f'Study status cleanup also failed: {cleanup_error}', RuntimeWarning)
            raise
        state['status'] = 'complete'
        state.pop('error', None)
        persist()
        return dict(status=state['status'], training_runs=len(state['runs']), report=str(root / 'report.md'))


def _execute_study(base, settings, args, root, state, persist, executor):
    from scientific.data_audit import DatasetLeakageError, audit_prepared_data
    try:
        audit = audit_prepared_data(args.data, base)
    except DatasetLeakageError as exc:
        write_json(root / 'data_audit_train_val.json', exc.report)
        raise
    write_json(root / 'data_audit_train_val.json', audit)
    matrix = {candidate['id']: candidate for candidate in candidates(settings)}
    vq_search = _search(base, settings, args, state, persist, 'vqgan', executor=executor)
    vq_confirm = _confirm(base, settings, args, state, persist, 'vqgan_confirmation', 'vqgan',
        select_finalists(vq_search), executor=executor)
    vq_scores = confirmation_means(vq_confirm, settings['search']['confirmation_seeds'])
    probes = []
    for candidate in vq_scores:
        parent = next(r for r in vq_confirm if r['candidate'] == candidate and r['seed'] == 42)
        probe = matrix[candidate] | dict(learning_rate=base['training']['learning_rate'], schedule='constant')
        probes.append(run_trial(base, settings, args, state, persist, 'transfer', probe,
            'flow_uniform', 42, autoencoder=parent, executor=executor))
    chosen_ae = min(probes, key=lambda r: (r['best_validation_lpips'], vq_scores[r['candidate']]['mean'], r['candidate']))['autoencoder']
    flow_search = _search(base, settings, args, state, persist, 'flow_uniform', autoencoder=chosen_ae, executor=executor)
    flow_confirm = _confirm(base, settings, args, state, persist, 'flow_confirmation', 'flow_uniform',
        select_finalists(flow_search), autoencoder=chosen_ae, executor=executor)
    flow_scores = confirmation_means(flow_confirm, settings['search']['confirmation_seeds'])
    chosen_flow = min(flow_scores, key=lambda key: (flow_scores[key]['mean'], key))
    selected_uniforms = [r for r in flow_confirm if r['candidate'] == chosen_flow]
    alphas = _confirm(base, settings, args, state, persist, 'alpha_confirmation', 'flow_alpha',
        [selected_uniforms[0]], autoencoder=chosen_ae, executor=executor)
    state.update(selection=dict(vqgan_candidate=chosen_ae['candidate'], flow_candidate=chosen_flow),
        confirmation_scores=dict(vqgan=vq_scores, flow=flow_scores))
    state['final_selection'] = _freeze(root, state, chosen_ae, selected_uniforms, alphas)
    persist()
    if settings['final_test']:
        from scientific.study_evaluation import run_final_evaluation
        run_final_evaluation(base, args, state, persist, executor=executor)
