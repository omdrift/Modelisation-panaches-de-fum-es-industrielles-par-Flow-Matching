"""Sequential campaign execution with durable state and isolated GPU processes."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import codecs
import fcntl
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
import time

import yaml

from scientific.protocol import digest, file_hash, LEARNED, write_json, validate_prepared_data, sources_compatible, _source_manifest, _protocols_compatible


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...] = ()
    reusable: bool = True


def source_fingerprint() -> str:
    return digest(_source_manifest())


def build_stages(cfg: dict, args, protocol: Path) -> list[Stage]:
    """Freeze the same ordered matrix for execution and --dry-run."""
    campaign, data = Path(args.campaign), Path(args.data)
    base = (sys.executable, '-u', '-m', 'scientific.cli', '--protocol', str(protocol))
    common = ('--campaign', str(campaign), '--data', str(data))
    tracking = ()
    if args.wandb:
        tracking = ('--wandb', '--wandb-project', args.wandb_project, '--wandb-mode', args.wandb_mode,
                    '--wandb-group', args.wandb_group or campaign.name,
                    '--wandb-log-every', str(args.wandb_log_every), '--wandb-media-every', str(args.wandb_media_every))
        if args.wandb_entity:
            tracking += ('--wandb-entity', args.wandb_entity)
    prep = ('prepare', '--videos', args.videos, '--output', str(data), '--device', 'cuda', '--batch-frames', str(args.batch_frames))
    if args.cache_dir:
        prep += ('--cache-dir', args.cache_dir)
    if args.local_files_only:
        prep += ('--local-files-only',)
    stages = [Stage('environment', base + ('doctor',), reusable=False),
              Stage('prepare', base + prep, (data / 'dataset_manifest.json',)),
              Stage('check_data', base + ('check-data', '--data', str(data)), reusable=False),
              Stage('pilot', base + ('pilot', '--campaign', str(campaign)), (campaign / 'pilot.json',)),
              Stage('vqgan', base + ('autoencoder',) + common + tracking,
                    (campaign / 'autoencoder/run.json', campaign / 'autoencoder/best.pth'))]
    autoencoder = campaign / 'autoencoder/best.pth'
    for method in args.methods:
        for seed in args.seeds:
            command = base + ('train',) + common + ('--method', method, '--seed', str(seed), '--pilot', str(campaign / 'pilot.json')) + tracking
            if method.startswith('flow'):
                command += ('--autoencoder', str(autoencoder))
            directory = campaign / f'{method}_{seed}'
            stages.append(Stage(f'train_{method}_{seed}', command, (directory / 'run.json', directory / 'best.pth')))
    evaluations = []
    for method in ('persistence', 'optical_flow', *args.methods):
        for seed in args.seeds:
            for split in ('test_dates', 'test_views'):
                command = base + ('evaluate',) + common + ('--method', method, '--seed', str(seed), '--split', split)
                if method in LEARNED:
                    command += ('--checkpoint', str(campaign / f'{method}_{seed}/best.pth'))
                if method.startswith('flow'):
                    command += ('--autoencoder', str(autoencoder))
                directory = campaign / 'evaluation' / f'{method}_{seed}_{split}'
                evaluations.append(directory)
                stages.append(Stage(f'evaluate_{method}_{seed}_{split}', command,
                                    tuple(directory / name for name in ('evaluation.json', 'metrics.csv', 'runtime.csv'))))
    outputs = tuple(campaign / 'summary' / name for name in
                    ('summary.json', 'quality_cost.csv', 'horizons.csv', 'test_dates_horizons.png', 'test_views_horizons.png', 'quality_cost.png'))
    stages.append(Stage('summarize', base + ('summarize', '--inputs', *map(str, evaluations), '--output', str(campaign / 'summary')), outputs))
    return stages


def execute_process(command: tuple[str, ...], log: Path, heartbeat) -> None:
    """Tee child output; failure and interruption stop all dependent stages."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log.open('ab') as stream, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            last_heartbeat = 0.0
            while selector.get_map():
                for key, _ in selector.select(timeout=1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    stream.write(block)
                    stream.flush()
                    sys.stdout.write(decoder.decode(block))
                    sys.stdout.flush()
                if time.monotonic() - last_heartbeat >= 10:
                    heartbeat()
                    last_heartbeat = time.monotonic()
            sys.stdout.write(decoder.decode(b'', final=True))
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()


def output_hashes(stage: Stage) -> dict:
    return {str(path): file_hash(path) for path in stage.outputs}


def validate_stage(stage: Stage, cfg: dict) -> None:
    """A zero process exit alone cannot mark an incomplete artifact as finished."""
    for path in stage.outputs:
        if not path.is_file():
            raise FileNotFoundError(f'Stage {stage.name} did not produce {path}')
    if stage.name == 'prepare':
        validate_prepared_data(json.loads(stage.outputs[0].read_text()), cfg)
    elif stage.name == 'vqgan' or stage.name.startswith('train_'):
        metadata = json.loads(stage.outputs[0].read_text())
        if metadata.get('status') != 'complete' or not _protocols_compatible(metadata.get('protocol_sha256'), digest(cfg), stage.outputs[0].parent.parent):
            raise ValueError(f'Training stage is incomplete or uses another protocol: {stage.name}')
    elif stage.name.startswith('evaluate_'):
        metadata = json.loads(stage.outputs[0].read_text())
        if metadata.get('status') != 'complete' or metadata.get('protocol_sha256') != digest(cfg):
            raise ValueError(f'Evaluation is incomplete or uses another protocol: {stage.name}')


def resume_command(stage: Stage, args) -> tuple[str, ...]:
    """Only explicit campaign resume may continue existing partial outputs."""
    command = stage.command
    if not args.resume:
        return command
    if stage.name == 'prepare' and Path(args.data).exists():
        return command + ('--resume',)
    if stage.name == 'vqgan' or stage.name.startswith('train_'):
        directory = stage.outputs[0].parent
        if (directory / 'run.json').exists():
            if any((directory / 'checkpoints').glob('step_*.pth')):
                return command + ('--resume', 'latest')
            raise RuntimeError(f'Interrupted training has no checkpoint: {directory}; choose a new campaign directory')
    if stage.name.startswith('evaluate_') and stage.outputs[0].parent.exists():
        metadata = json.loads(stage.outputs[0].read_text())
        if metadata['status'] == 'incomplete':
            return command + ('--retry-incomplete',)
    return command


def run_pipeline(cfg: dict, args, *, executor=execute_process) -> dict:
    """Run one campaign; children release GPU memory when each stage exits."""
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Methods and seeds must not contain duplicates')
    if args.batch_frames < 1 or min(args.wandb_log_every, args.wandb_media_every) < 1:
        raise ValueError('Batch size and monitoring intervals must be positive')
    campaign, data = Path(args.campaign), Path(args.data)
    frozen = campaign / 'protocol.yaml'
    stages = build_stages(cfg, args, frozen)
    if args.dry_run:
        for i, stage in enumerate(stages, 1):
            if stage.name == 'prepare' and (data / 'dataset_manifest.json').is_file():
                print(f'{i:02d}. prepare: reuse prepared dataset at {data}; check_data still verifies the NPZ files')
            else:
                print(f'{i:02d}. {stage.name}: {shlex.join(stage.command)}')
        print('protocol.yaml will contain the resolved configuration when execution starts.')
        return dict(stages=len(stages), context_frames=cfg['evaluation']['context_frames'], horizon=cfg['evaluation']['horizon'])
    manifest = data / 'dataset_manifest.json'
    if manifest.is_file():
        # Preparation is already complete. Source MP4s and CLIPSeg weights are
        # no longer inputs to training; check_data still verifies every NPZ.
        validate_prepared_data(json.loads(manifest.read_text()), cfg)
    elif not Path(args.videos).is_dir():
        raise FileNotFoundError(f'No completed dataset manifest at {manifest}; source videos are required to prepare data: {args.videos}')
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / 'pipeline.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another command is already running in this campaign') from exc
        return _run_locked(cfg, args, stages, frozen, executor)


def _run_locked(cfg, args, stages, frozen, executor):
    campaign = Path(args.campaign)
    state_path = campaign / 'pipeline.json'
    identity = dict(protocol_sha256=digest(cfg), source_sha256=source_fingerprint(),
                    videos=str(Path(args.videos).resolve()), data=str(Path(args.data).resolve()),
                    methods=args.methods, seeds=args.seeds)
    if state_path.exists():
        if not args.resume:
            raise FileExistsError('Campaign already exists; add --resume to verify and continue it')
        state = json.loads(state_path.read_text())
        previous = dict(state['identity'])
        if _protocols_compatible(previous.get('protocol_sha256'), identity['protocol_sha256'], campaign):
            previous['protocol_sha256'] = identity['protocol_sha256']
        if sources_compatible(previous.get('source_sha256'), identity['source_sha256'], campaign, identity['protocol_sha256']):
            previous['source_sha256'] = identity['source_sha256']
        if (Path(args.data) / 'dataset_manifest.json').is_file():
            # Preserve the recorded video origin, but it is no longer a runtime
            # input once preparation is finalized. Data/artifact checks remain.
            previous['videos'] = identity['videos']
        if previous != identity:
            raise ValueError('Pipeline resume requires identical sources, data paths, methods, seeds and protocol')
        if not frozen.is_file() or yaml.safe_load(frozen.read_text()) != cfg:
            raise ValueError('Frozen campaign protocol was changed')
        state['identity'].update(protocol_sha256=identity['protocol_sha256'], source_sha256=identity['source_sha256'])
    else:
        occupied = [campaign / name for name in ('autoencoder', 'pilot.json', 'evaluation')]
        occupied.extend(campaign / f'{m}_{s}' for m in args.methods for s in args.seeds)
        if any(path.exists() for path in occupied):
            raise FileExistsError('Choose a new campaign directory for the managed pipeline')
        frozen.write_text(yaml.safe_dump(cfg, sort_keys=False))
        state = dict(identity=identity, stages={}, status='running')
        write_json(state_path, state)
    (campaign / 'logs').mkdir(exist_ok=True)
    tracking = nullcontext(None)
    if args.wandb:
        from scientific.monitoring import WandbMonitor, WandbOptions
        tracking = WandbMonitor(campaign / 'pipeline_monitor', cfg, 'pipeline', cfg['selection_seed'],
            WandbOptions(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name or f'{campaign.name}-pipeline',
                         group=args.wandb_group or campaign.name, mode=args.wandb_mode), resume=args.resume)
    # The supervisor holds pipeline.lock; children hold campaign.lock separately.
    with tracking as monitor:
        for index, stage in enumerate(stages, 1):
            saved = state['stages'].get(stage.name, {})
            if saved.get('status') == 'complete' and stage.reusable:
                validate_stage(stage, cfg)
                if output_hashes(stage) != saved['outputs']:
                    raise ValueError(f'Completed stage artifacts changed: {stage.name}')
                print(f'[{index}/{len(stages)}] {stage.name}: verified, reused', flush=True)
                continue
            reused_data = stage.name == 'prepare' and (Path(args.data) / 'dataset_manifest.json').is_file()
            if reused_data:
                validate_stage(stage, cfg)
            command = resume_command(stage, args) if not reused_data else stage.command
            started = time.monotonic()
            state.update(status='running', current_stage=stage.name)
            state['stages'][stage.name] = dict(status='running', command=list(command))
            write_json(state_path, state)
            note = ': prepared data reused; source videos not required' if reused_data else ''
            print(f'[{index}/{len(stages)}] {stage.name}{note}', flush=True)

            def heartbeat():
                if monitor is not None:
                    values = {'pipeline/stage': stage.name, 'pipeline/completed_stages': index - 1,
                              'pipeline/total_stages': len(stages), 'pipeline/stage_seconds': time.monotonic() - started}
                    progress_path = Path(args.data) / 'progress.json' if stage.name == 'prepare' else None
                    if progress_path is not None and progress_path.exists():
                        progress = json.loads(progress_path.read_text())
                        values.update({'preparation/clips_completed': progress['completed'], 'preparation/clips_total': progress['total']})
                    monitor.log(index, values)

            try:
                heartbeat()
                if not reused_data:
                    if executor is execute_process:
                        execute_process(command, campaign / 'logs' / f'{stage.name}.log', heartbeat)
                    else:
                        executor(command, campaign / 'logs' / f'{stage.name}.log', heartbeat)
                validate_stage(stage, cfg)
                state['stages'][stage.name].update(status='complete', seconds=time.monotonic() - started, outputs=output_hashes(stage))
                write_json(state_path, state)
                if monitor is not None:
                    monitor.log(index, {'pipeline/completed_stages': index, 'pipeline/stage': stage.name, 'pipeline/stage_seconds': time.monotonic() - started})
            except BaseException as exc:
                state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
                state['stages'][stage.name].update(status=state['status'], seconds=time.monotonic() - started)
                write_json(state_path, state)
                raise
        state.update(status='complete', current_stage=None)
        state.pop('error', None)
        write_json(state_path, state)
    return dict(status='complete', campaign=str(campaign), summary=str(campaign / 'summary/summary.json'))
