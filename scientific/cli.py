"""Command-line workflow: python -m scientific.cli --help."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import re

from scientific.protocol import load_protocol, partition, eligible, write_json, write_jsonl, digest, file_hash, LEARNED, METHODS, forecast_task, validate_prepared_data


def inventory(cfg, output):
    records = json.loads(Path(cfg['metadata']).read_text())
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate source video identifiers')
    result = dict(protocol_sha256=digest(cfg), metadata_sha256=file_hash(cfg['metadata']), metadata_entries=len(records), eligible_positive=sum(map(eligible, records)), partitions=dict(Counter(map(partition, records))), counts_are='metadata candidates, not downloaded or usable videos', source='https://github.com/CMU-CREATE-Lab/deep-smoke-machine#dataset')
    write_json(output, result)
    return result


def download(cfg, root):
    """Explicit download command; never invoked by inventory, doctor, or prepare."""
    import requests

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    events = []
    for record in sorted(json.loads(Path(cfg['metadata']).read_text()), key=lambda r: r['id']):
        if partition(record).startswith('excluded'):
            continue
        path = root / f"{record['id']}_{record['file_name']}.mp4"
        event = dict(video_id=record['id'], url=record['url_root'] + record['url_part'])
        if path.exists():
            events.append(event | dict(status='existing', sha256=file_hash(path)))
            continue
        temporary = path.with_suffix('.part')
        try:
            with requests.get(event['url'], stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size == 0:
                raise ValueError('Empty video download')
            temporary.replace(path)
            event.update(status='downloaded', sha256=file_hash(path))
        except (requests.RequestException, OSError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            event.update(status='failed', error=str(exc))
        events.append(event)
        write_jsonl(root / 'download_events.jsonl', events)
    write_jsonl(root / 'download_events.jsonl', events)
    return dict(statuses=dict(Counter(e['status'] for e in events)))


def doctor():
    import torch

    dependencies = {name: importlib.util.find_spec(name) is not None for name in ('cv2', 'transformers', 'huggingface_hub', 'safetensors', 'lpips', 'skimage', 'yaml', 'torchvision', 'torchdiffeq')}
    return dict(cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, dependencies=dependencies, ready=all(dependencies.values()) and torch.cuda.is_available(), note='This read-only check does not download weights or start a training job.')


def check_data(root, cfg=None):
    """Validate all selected clips automatically; no reviewer files are required."""
    from scientific.data import WindowDataset
    if cfg is not None:
        validate_prepared_data(json.loads((Path(root) / 'dataset_manifest.json').read_text()), cfg)
    datasets = {split: WindowDataset(root, split) for split in ('train', 'val', 'test_dates', 'test_views')}
    video_sets = [set(r['video_id'] for r in d.records) for d in datasets.values()]
    import itertools
    if any(a & b for a, b in itertools.combinations(video_sets, 2)):
        raise ValueError('Video identity crosses partitions')
    for dataset in datasets.values():
        for index in range(len(dataset)):
            dataset[index]  # Checks RGB/mask shapes, dtypes, finite alpha and range.
    return dict(counts={s: len(d) for s, d in datasets.items()}, validation='automatic', clips_checked=sum(len(d) for d in datasets.values()))


def summarize(inputs, output, cfg):
    import numpy as np
    from scientific.metrics import paired_bootstrap

    context, horizon = forecast_task(cfg)
    rows, runtimes = [], []
    identities = set()
    panels = {}
    for folder in map(Path, inputs):
        metadata = json.loads((folder / 'evaluation.json').read_text())
        if metadata['status'] != 'complete':
            raise ValueError(f'Incomplete evaluation: {folder}')
        if metadata.get('lpips') != 'official alex v0.1':
            raise ValueError('Injected test metrics cannot be aggregated as publication results')
        if metadata['protocol_sha256'] != digest(cfg) or not metadata.get('training_complete'):
            raise ValueError('Wrong protocol or incomplete predictor training')
        if metadata['method'] not in METHODS or metadata['split'] not in ('test_dates', 'test_views') or metadata['seed'] not in cfg['seeds']:
            raise ValueError('Evaluation identity is outside the frozen experiment matrix')
        identities.add((metadata['protocol_sha256'], metadata['dataset_sha256'], metadata['metric_weights_sha256'], metadata['source_sha256']))
        expected_ids = metadata['expected_window_ids']
        panel = frozenset(expected_ids)
        if metadata['split'] in panels and panels[metadata['split']] != panel:
            raise ValueError('Every method and seed must evaluate the same frozen test panel')
        panels[metadata['split']] = panel
        if metadata['windows_completed'] != metadata['expected_windows'] or metadata['expected_windows'] != len(set(expected_ids)):
            raise ValueError('Evaluation window coverage is incomplete')
        observed = set()
        with (folder / 'metrics.csv').open() as stream:
            for row in csv.DictReader(stream):
                for key in ('lpips', 'mse', 'psnr', 'ssim', 'smoke_mse', 'background_mse', 'smoke_fraction', 'diversity_mse'):
                    row[key] = float(row[key]) if row[key] else None
                    if row[key] is None and key not in ('smoke_mse', 'background_mse') or row[key] is not None and not math.isfinite(row[key]):
                        raise ValueError(f'Missing or non-finite metric: {key}')
                for key in ('seed', 'horizon', 'realization'):
                    row[key] = int(row[key])
                if (row['method'], row['split'], row['seed']) != (metadata['method'], metadata['split'], metadata['seed']):
                    raise ValueError('Row identity differs from evaluation metadata')
                observed.add((row['window_id'], row['horizon'], row['realization']))
                rows.append(row)
        expected_grid = {(window, frame_horizon, realization) for window in expected_ids for frame_horizon in range(1, horizon + 1) for realization in range(4 if metadata['method'].startswith('flow') else 1)}
        if not expected_grid or observed != expected_grid:
            raise ValueError('Evaluation must cover the full frozen window/horizon/realization grid')
        with (folder / 'runtime.csv').open() as stream:
            measurements = list(csv.DictReader(stream))
        runtime_grid = [(r['window_id'], int(r['realization'])) for r in measurements]
        expected_runtime = {(w, r) for w, _, r in expected_grid}
        if len(runtime_grid) != len(set(runtime_grid)) or set(runtime_grid) != expected_runtime:
            raise ValueError('Runtime measurements do not cover all window realizations')
        for measurement in measurements:
            if any(not math.isfinite(float(measurement[key])) or float(measurement[key]) < 0 for key in ('seconds', 'nfe', 'peak_memory_bytes')):
                raise ValueError('Runtime measurements must be finite and nonnegative')
            runtimes.append(measurement | dict(method=metadata['method'], split=metadata['split'], seed=metadata['seed']))
    if len(identities) != 1:
        raise ValueError('Aggregation requires identical protocol, data, source and perceptual weights')
    unique = [(r['method'], r['split'], r['seed'], r['window_id'], r['horizon'], r['realization']) for r in rows]
    if len(unique) != len(set(unique)):
        raise ValueError('Duplicate metric rows / evaluations')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['method'], row['split'], row['horizon'])].append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        entry = dict(zip(('method', 'split', 'horizon'), key))
        for metric in ('lpips', 'mse', 'psnr', 'ssim', 'smoke_mse', 'background_mse', 'diversity_mse'):
            # Equal weight per window/seed, independent of realization count.
            units = defaultdict(list)
            for row in values:
                if row[metric] is not None:
                    units[(row['window_id'], row['seed'])].append(row[metric])
            entry[metric] = float(np.mean([np.mean(v) for v in units.values()])) if units else None
            entry[metric + '_valid_window_seeds'] = len(units)
        seed_means = [np.mean([r['lpips'] for r in values if r['seed'] == seed]) for seed in sorted({r['seed'] for r in values})]
        entry['lpips_seed_means'] = seed_means
        entry['lpips_seed_std'] = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None
        summaries.append(entry)
    comparisons = []
    for split in ('test_dates', 'test_views'):
        subset = [r for r in rows if r['split'] == split]
        methods = {r['method'] for r in subset}
        for method, reference in [('flow_uniform', 'simvp'), ('flow_alpha', 'flow_uniform')]:
            if {method, reference} <= methods:
                comparisons.append({'split': split} | paired_bootstrap(subset, method, reference, replicates=cfg['evaluation']['bootstrap_replicates']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    complete_campaign = all({r['seed'] for r in rows if r['method'] == method and r['split'] == split} == set(cfg['seeds']) for method in METHODS for split in ('test_dates', 'test_views'))
    primary_scores = []
    for method, split in sorted({(r['method'], r['split']) for r in rows}):
        units = defaultdict(list)
        for row in rows:
            if (row['method'], row['split']) == (method, split):
                units[(row['window_id'], row['seed'])].append(row['lpips'])
        costs = [r for r in runtimes if (r['method'], r['split']) == (method, split)]
        primary_scores.append(dict(method=method, split=split, horizon_count=horizon, lpips_mean_future_horizons=float(np.mean([np.mean(v) for v in units.values()])), generation_seconds_per_realization=float(np.mean([float(r['seconds']) for r in costs])), nfe_per_realization=float(np.mean([int(r['nfe']) for r in costs])), peak_memory_bytes=max(int(r['peak_memory_bytes']) for r in costs)))
    write_json(output / 'summary.json', dict(status='complete_campaign' if complete_campaign else 'partial_campaign', primary_scores=primary_scores, summaries=summaries, comparisons=comparisons, uncertainty='Day-clustered conditional on one trained autoencoder; three predictor seeds reported separately.'))
    with (output / 'quality_cost.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary_scores[0]))
        writer.writeheader()
        writer.writerows(primary_scores)
    with (output / 'horizons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    failures = defaultdict(list)
    for row in rows:
        failures[(row['method'], row['split'], row['window_id'])].append(row['lpips'])
    with (output / 'failure_review.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'split', 'window_id', 'mean_lpips', 'disappearance', 'immobility', 'background_smoke', 'temporal_drift', 'reviewer', 'notes'])
        for key, values in sorted(failures.items(), key=lambda kv: -np.mean(kv[1])):
            writer.writerow([*key, float(np.mean(values)), '', '', '', '', '', ''])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in ('test_dates', 'test_views'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in METHODS:
            values = [s for s in summaries if s['method'] == method and s['split'] == split]
            for axis, metric in zip(axes, ('lpips', 'smoke_mse')):
                axis.plot([v['horizon'] for v in values], [v[metric] for v in values], label=method)
                axis.set(xlabel='Horizon (images)', ylabel=metric, title=split)
                axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f'{split}_horizons.png', dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in primary_scores:
        axis.scatter(row['generation_seconds_per_realization'], row['lpips_mean_future_horizons'])
        axis.annotate(f"{row['method']} / {row['split']}", (row['generation_seconds_per_realization'], row['lpips_mean_future_horizons']), fontsize=7)
    axis.set(xlabel=f'Generation seconds per realization ({horizon} frames)', ylabel=f'LPIPS, mean over {horizon} horizons')
    fig.tight_layout()
    fig.savefig(output / 'quality_cost.png', dpi=160)
    plt.close(fig)
    return dict(output=str(output), rows=len(rows), comparisons=comparisons, complete_campaign=complete_campaign)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/scientific_protocol.yaml')
    parser.add_argument('--context-frames', type=int, help='Explicit observation count, distinct from the 36-frame training clip')
    parser.add_argument('--horizon', type=int, help='Explicit number of future images; observations + horizon must fit in 36')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    inv = commands.add_parser('inventory')
    inv.add_argument('--output', required=True)
    down = commands.add_parser('download')
    down.add_argument('--videos', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--videos', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='CLIPSeg inference device')
    prep.add_argument('--cache-dir', help='Hugging Face model cache directory')
    prep.add_argument('--local-files-only', action='store_true', help='Use cached pinned model files without network access')
    prep.add_argument('--batch-frames', type=int, default=1, help='CLIPSeg frames per batch; default 1, fastest measured on RTX 5060')
    check = commands.add_parser('check-data')
    check.add_argument('--data', required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--inputs', nargs='+', required=True)
    summary.add_argument('--output', required=True)
    for name in ('pilot', 'autoencoder', 'train', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--campaign', required=True)
        command.add_argument('--data', required=name != 'pilot')
        if name in ('train', 'evaluate'):
            command.add_argument('--method', choices=LEARNED if name == 'train' else METHODS, required=True)
            command.add_argument('--seed', choices=(17, 42, 2026), type=int, required=True)
            command.add_argument('--autoencoder')
        if name in ('train', 'autoencoder'):
            command.add_argument('--resume', help='Checkpoint path, or latest to continue the last saved update')
            command.add_argument('--wandb', action='store_true', help='Enable WandB training monitoring')
            command.add_argument('--wandb-project', default='smoke-science')
            command.add_argument('--wandb-entity')
            command.add_argument('--wandb-name')
            command.add_argument('--wandb-mode', choices=('online', 'offline'), default='online')
            command.add_argument('--wandb-log-every', type=int, default=25)
            command.add_argument('--wandb-media-every', type=int, default=250)
        if name == 'train':
            command.add_argument('--pilot', required=True)
        if name == 'evaluate':
            command.add_argument('--checkpoint')
            command.add_argument('--split', choices=('test_dates', 'test_views'), required=True)
            command.add_argument('--retry-incomplete', action='store_true', help='Archive the failed attempt and regenerate the same frozen panel with the same frozen predictions')
    args = parser.parse_args(argv)
    cfg = load_protocol(args.protocol)
    if args.context_frames is not None or args.horizon is not None:
        cfg['evaluation']['context_frames'] = args.context_frames
        cfg['evaluation']['horizon'] = args.horizon
    forecast_task(cfg, required=args.command in ('pilot', 'autoencoder', 'train', 'evaluate', 'summarize'))
    if args.command == 'doctor':
        result = doctor()
        print(json.dumps(result, indent=2))
        return 0 if result['ready'] else 1
    if args.command == 'inventory':
        result = inventory(cfg, args.output)
    elif args.command == 'download':
        result = download(cfg, args.videos)
    elif args.command == 'prepare':
        from scientific.data import prepare
        result = prepare(json.loads(Path(cfg['metadata']).read_text()), Path(args.videos), Path(args.output), cfg,
            metadata_path=Path(cfg['metadata']), device=args.device, cache_dir=args.cache_dir, local_files_only=args.local_files_only,
            batch_frames=args.batch_frames)
    elif args.command == 'check-data':
        result = check_data(args.data, cfg)
    elif args.command == 'summarize':
        result = summarize(args.inputs, args.output, cfg)
    else:
        import torch
        from scientific import runner
        if not torch.cuda.is_available():
            raise RuntimeError('A working CUDA GPU is required; no CPU run is promoted to a GPU campaign')
        device = torch.device('cuda:0')
        campaign = Path(args.campaign)
        if args.command in ('train', 'evaluate') and args.method.startswith('flow') and not args.autoencoder:
            parser.error('--autoencoder is required for Flow Matching')
        if args.command == 'evaluate' and args.method in LEARNED and not args.checkpoint:
            parser.error('--checkpoint is required for a learned predictor')
        if args.command != 'pilot':
            check_data(args.data, cfg)
        campaign.mkdir(parents=True, exist_ok=True)
        with (campaign / 'campaign.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another command is already running in this campaign') from exc
            if args.command in ('autoencoder', 'train'):
                directory = campaign / ('autoencoder' if args.command == 'autoencoder' else f'{args.method}_{args.seed}')
                if args.resume == 'latest':
                    candidates = [(int(match[1]), path) for path in (directory / 'checkpoints').glob('step_*.pth') if (match := re.fullmatch(r'step_(\d+)\.pth', path.name))]
                    if not candidates:
                        parser.error(f'No checkpoint to resume in {directory / "checkpoints"}')
                    args.resume = str(max(candidates)[1])
                if (directory / 'run.json').exists() and not args.resume:
                    parser.error('Existing training run: add --resume latest to continue it')
                tracking = nullcontext(None)
                if args.wandb:
                    from scientific.monitoring import WandbMonitor, WandbOptions
                    options = WandbOptions(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                        mode=args.wandb_mode, log_every=args.wandb_log_every, media_every=args.wandb_media_every)
                    tracking = WandbMonitor(directory, cfg, 'vqgan' if args.command == 'autoencoder' else args.method,
                        cfg['training']['autoencoder_seed'] if args.command == 'autoencoder' else args.seed, options, resume=bool(args.resume))
                with tracking as monitor:
                    kwargs = {'monitor': monitor} if monitor is not None else {}
                    if args.command == 'autoencoder':
                        result = runner.train_autoencoder(cfg, args.data, directory, device, args.resume, **kwargs)
                    else:
                        result = runner.train_predictor(cfg, args.data, args.method, args.seed, args.autoencoder, args.pilot, directory, device, args.resume, **kwargs)
            elif args.command == 'pilot':
                result = runner.profile(cfg, campaign / 'pilot.json', device)
            else:
                destination = campaign / 'evaluation' / f'{args.method}_{args.seed}_{args.split}'
                if args.retry_incomplete:
                    import time
                    previous = json.loads((destination / 'evaluation.json').read_text())
                    if previous['status'] != 'incomplete' or (previous['method'], previous['seed'], previous['split'], previous['protocol_sha256']) != (args.method, args.seed, args.split, digest(cfg)):
                        raise ValueError('Only an incomplete evaluation with matching identity can be retried')
                    archived = campaign / 'evaluation_incomplete' / f'{destination.name}_{time.time_ns()}'
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    destination.rename(archived)
                rows = runner.evaluate(cfg, args.data, args.method, args.seed, args.checkpoint, args.autoencoder, args.split, destination, device)
                result = dict(metric_rows=len(rows))
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
