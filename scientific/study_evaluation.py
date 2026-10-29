"""Held-out evaluation available only after the study freezes its artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from scientific.pipeline import execute_process
from scientific.protocol import digest, file_hash, load_protocol, write_json


def verify_selection(cfg, checkpoint, data):
    """Bind test access to the selected weights, protocol, dataset and study."""
    if not cfg.get('study'):
        raise ValueError('Final study evaluation requires a study protocol')
    root = Path(cfg['study']['root'])
    if not (root / 'final_selection.json').is_file():
        raise ValueError('Final selection has not been frozen; test access is forbidden')
    selection = json.loads((root / 'final_selection.json').read_text())
    state = json.loads((root / 'study.json').read_text())
    if (not selection.get('selected_before_test')
            or selection['study_identity_sha256'] != digest(state['identity'])
            or state.get('final_selection') != selection
            or state['identity']['dataset_sha256'] != file_hash(Path(data) / 'dataset_manifest.json')):
        raise ValueError('Missing or changed final selection proof')
    # Deterministic baselines use a selected predictor's protocol, not its weights.
    match = [record for record in selection['artifacts'] if
        (Path(record['checkpoint']).resolve() == Path(checkpoint).resolve() if checkpoint is not None
         else digest(load_protocol(record['protocol'])) == digest(cfg))]
    if not match or checkpoint is not None and len(match) != 1:
        raise ValueError('Checkpoint is not part of the frozen final selection')
    # Equivalent seed-specific protocols may share a digest; deterministic
    # baselines need any valid selected reference under that exact protocol.
    record = match[0]
    checkpoint = record['checkpoint'] if checkpoint is None else checkpoint
    if (file_hash(checkpoint) != record['checkpoint_sha256']
            or file_hash(record['protocol']) != record['protocol_sha256']
            or digest(load_protocol(record['protocol'])) != digest(cfg)):
        raise ValueError('Selected checkpoint or protocol changed after freeze')
    return record


def evaluate_autoencoder(cfg, data, checkpoint, split, output, device, metric=None):
    """Score all held-out clip frames; export the same eight fixed sequences."""
    import torch
    from scientific.data import WindowDataset
    from scientific.diagnostics import reconstruction_snapshot
    from scientific.metrics import build_lpips
    from scientific.runner import validate_artifact
    from lutils.dict_wrapper import DictWrapper
    from model.vqgan.vqvae import build_vqvae
    from scientific.visualizations import export_panel_media
    if split not in ('test_dates', 'test_views'):
        raise ValueError('Final reconstruction evaluation requires a test partition')
    verify_selection(cfg, checkpoint, data)
    state = validate_artifact(checkpoint, cfg, data, 'vqgan')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace final evaluation: {output}')
    model = build_vqvae(DictWrapper(cfg['model']['autoencoder'])).to(device)
    model.load_state_dict(state['model'], strict=True)
    dataset = WindowDataset(data, split)
    metric_identity = 'injected_test_metric_not_publication' if metric is not None else 'official alex v0.1'
    metric = metric if metric is not None else build_lpips(device)
    output.mkdir(parents=True)
    identity = dict(method='vqgan', split=split, checkpoint_sha256=file_hash(checkpoint),
        protocol_sha256=digest(cfg), lpips=metric_identity, status='incomplete')
    write_json(output / 'evaluation.json', identity)
    with torch.inference_mode():
        snapshot = reconstruction_snapshot(model, dataset, metric, cfg, device)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(snapshot.rows[0]))
        writer.writeheader()
        writer.writerows(snapshot.rows)
    export_panel_media(output, state['step'], 'vqgan', snapshot, split, final=True)
    write_json(output / 'evaluation.json', identity | dict(status='complete', summary=snapshot.summary()))
    return snapshot.summary()


def run_final_evaluation(base, args, state, persist, *, executor=execute_process):
    from scientific.data_audit import DatasetLeakageError, audit_prepared_data
    root = Path(args.output).resolve()
    # The freeze is persisted before any test content, images or scores are read.
    selected = state['final_selection']['artifacts']
    for record in selected:
        verify_selection(load_protocol(record['protocol']), record['checkpoint'], args.data)
    try:
        audit = audit_prepared_data(args.data, base, include_test=True, final_selection_frozen=True)
    except DatasetLeakageError as exc:
        write_json(root / 'data_audit_final.json', exc.report)
        raise
    write_json(root / 'data_audit_final.json', audit)
    evaluations = state.setdefault('evaluations', {})
    for record in selected:
        for split in ('test_dates', 'test_views'):
            method = record['method']
            command = (sys.executable, '-u', '-m', 'scientific.cli', '--protocol', record['protocol'],
                'study-evaluate-autoencoder' if method == 'vqgan' else 'evaluate',
                '--campaign', str(root / 'evaluation' / record['id']), '--data', str(Path(args.data).resolve()),
                '--checkpoint', record['checkpoint'], '--split', split)
            if method != 'vqgan':
                cfg = load_protocol(record['protocol'])
                parent_protocol = cfg['training']['autoencoder_protocol']
                parent = next(r for r in selected if r['protocol'] == parent_protocol)
                command += ('--method', method, '--seed', str(record['seed']), '--autoencoder', parent['checkpoint'])
            directory = root / 'evaluation' / record['id'] / 'evaluation' / (
                f'vqgan_{split}' if method == 'vqgan' else f'{method}_{record["seed"]}_{split}')
            key = f'{record["id"]}_{split}'
            _evaluate_once(command, directory, key, record['protocol'], evaluations, persist, executor, root)
    _baselines(args, state, persist, executor, root)
    write_final_report(root, state)


def _evaluate_once(command, directory, key, protocol, evaluations, persist, executor, root):
    saved = evaluations.get(key)
    if saved:
        if any(not Path(path).is_file() or file_hash(path) != value for path, value in saved['outputs'].items()):
            raise ValueError(f'Completed evaluation artifacts changed: {key}')
        _verify_predictions(directory)
        _export_forecast_media(directory, load_protocol(protocol))
        return
    already_complete = False
    if directory.exists():
        manifest = directory / 'evaluation.json'
        already_complete = manifest.is_file() and json.loads(manifest.read_text()).get('status') == 'complete'
        if not already_complete:
            # Keep failed-attempt evidence, including mkdir-before-manifest failures.
            import time
            directory.rename(directory.with_name(directory.name + f'_incomplete_{time.time_ns()}'))
    if not already_complete:
        executor(command, root / f'{key}.log', lambda: None)
    metadata = json.loads((directory / 'evaluation.json').read_text())
    if metadata['status'] != 'complete' or metadata['protocol_sha256'] != digest(load_protocol(protocol)):
        raise ValueError(f'Final evaluation is incomplete or uses another protocol: {key}')
    for flag, field in [('--method', 'method'), ('--split', 'split'), ('--seed', 'seed')]:
        if flag in command and str(metadata.get(field)) != command[command.index(flag) + 1]:
            raise ValueError(f'Final evaluation identity changed: {field}')
    if '--checkpoint' in command and metadata.get('checkpoint_sha256') != file_hash(command[command.index('--checkpoint') + 1]):
        raise ValueError('Final evaluation used another checkpoint')
    _verify_predictions(directory)
    _export_forecast_media(directory, load_protocol(protocol))
    paths = [directory / 'evaluation.json', directory / 'metrics.csv']
    if (directory / 'runtime.csv').exists():
        paths.append(directory / 'runtime.csv')
    evaluations[key] = dict(directory=str(directory), protocol=protocol,
        outputs={str(path): file_hash(path) for path in paths})
    persist()


def _verify_predictions(directory):
    metadata = json.loads((directory / 'evaluation.json').read_text())
    for window, expected in metadata.get('prediction_sha256', {}).items():
        path = directory / 'predictions' / f'{window}.npz'
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Final prediction archive changed: {window}')


def _export_forecast_media(directory, cfg):
    """Use archived realization zero; never regenerate or select a favorable draw."""
    metadata = json.loads((directory / 'evaluation.json').read_text())
    if not metadata['method'].startswith('flow'):
        return
    import numpy as np
    import torch
    from scientific.data import WindowDataset
    from scientific.diagnostics import Snapshot, fixed_indices
    from scientific.visualizations import export_panel_media
    state = json.loads((Path(cfg['study']['root']) / 'study.json').read_text())
    dataset = WindowDataset(state['identity']['data'], metadata['split'])
    context, horizon = cfg['evaluation']['context_frames'], cfg['evaluation']['horizon']
    snapshot = Snapshot()
    for index in fixed_indices(dataset, seed=cfg['selection_seed'])[:cfg['diagnostics']['visual_windows']]:
        item = dataset[index]
        record = dataset.records[index]
        with np.load(directory / 'predictions' / f'{record["window_id"]}.npz', allow_pickle=False) as archive:
            generated = archive['generated']
            if generated.shape != (4, horizon, 64, 64, 3) or generated.dtype != np.uint8:
                raise ValueError('Invalid final prediction archive shape or dtype')
            prediction = torch.from_numpy(generated[0].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1
        snapshot.sequences.append(dict(window_id=record['window_id'], context=item['frames'][:context],
            target=item['frames'][context:context + horizon], prediction=prediction))
    selected = next(r for r in state['final_selection']['artifacts'] if r['method'] == metadata['method'] and r['seed'] == metadata['seed'])
    checkpoint = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    export_panel_media(directory, checkpoint['step'], metadata['method'], snapshot, metadata['split'], final=True)


def _baselines(args, state, persist, executor, root):
    record = next(r for r in state['final_selection']['artifacts'] if r['method'] == 'flow_uniform' and r['seed'] == 42)
    for method in ('persistence', 'optical_flow'):
        for seed in (17, 42, 2026):
            for split in ('test_dates', 'test_views'):
                command = (sys.executable, '-u', '-m', 'scientific.cli', '--protocol', record['protocol'],
                    'evaluate', '--campaign', str(root / 'evaluation' / 'baselines'), '--data', str(Path(args.data).resolve()),
                    '--method', method, '--seed', str(seed), '--split', split)
                directory = root / 'evaluation' / 'baselines' / 'evaluation' / f'{method}_{seed}_{split}'
                _evaluate_once(command, directory, f'{method}_{seed}_{split}', record['protocol'],
                    state['evaluations'], persist, executor, root)


def write_final_report(root, state):
    """Validate each model's frozen protocol before combining common test panels."""
    from scientific.cli import (_read_evaluation_rows, _horizon_summaries, _paired_comparisons,
        _quality_cost_scores, _write_analysis_plots)
    all_rows, runtimes, identities, panels = [], [], set(), {}
    for entry in state['evaluations'].values():
        directory = Path(entry['directory'])
        metadata = json.loads((directory / 'evaluation.json').read_text())
        if metadata['method'] == 'vqgan':
            continue
        cfg = load_protocol(entry['protocol'])
        rows, times = _read_evaluation_rows([directory], cfg, 15)
        identities.add(tuple(metadata[key] for key in ('dataset_sha256', 'metric_weights_sha256', 'source_sha256')))
        split = metadata['split']
        panel = sorted(metadata['expected_window_ids'])
        if split in panels and panels[split] != panel:
            raise ValueError('Selected methods do not share the same final test windows')
        panels[split] = panel
        all_rows.extend(rows)
        runtimes.extend(times)
    if len(identities) != 1 or not all_rows:
        raise ValueError('Final evaluations require identical data, sources and metric weights')
    output = Path(root) / 'evaluation_summary'
    output.mkdir(exist_ok=True)
    summaries = _horizon_summaries(all_rows)
    scores = _quality_cost_scores(all_rows, runtimes, 15)
    comparisons = _paired_comparisons(all_rows, cfg)
    from scientific.metrics import paired_bootstrap
    for split in ('test_dates', 'test_views'):
        subset = [row for row in all_rows if row['split'] == split]
        for reference in ('persistence', 'optical_flow'):
            comparisons.append(dict(split=split, **paired_bootstrap(subset, 'flow_uniform', reference,
                replicates=cfg['evaluation']['bootstrap_replicates'])))
    write_json(output / 'summary.json', dict(status='complete_selected_study', primary_scores=scores,
        horizons=summaries, comparisons=comparisons,
        uncertainty='Three predictor seeds, four draws; day-clustered bootstrap conditional on the selected seed-42 autoencoder.'))
    for name, rows in [('horizons.csv', summaries), ('quality_cost.csv', scores)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_analysis_plots(output, summaries, scores, 15)
