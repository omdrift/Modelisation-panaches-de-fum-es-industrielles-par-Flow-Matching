"""Observe campaign artifacts in W&B without executing training or loading tensors."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
from html import escape
import json
import math
from pathlib import Path
from statistics import fmean
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import uuid4

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scientific.pipeline import build_stages
from scientific.protocol import digest, write_json


def _json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _recent_metrics(path, limit=1024 * 1024):
    """Read a bounded suffix and exclude only an unfinished trailing JSONL record."""
    if not path.is_file():
        return [], {}
    with path.open('rb') as stream:
        offset = max(0, path.stat().st_size - limit)
        stream.seek(offset)
        if offset:
            stream.readline()
        data = stream.read()
    if data and not data.endswith(b'\n'):
        data = data.rsplit(b'\n', 1)[0] if b'\n' in data else b''
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    validation = next((row for row in reversed(rows) if 'validation/lpips' in row), {})
    return rows, validation


def _horizons(run, step, cfg):
    if step is None:
        return []
    diagnostic = _json(run / 'diagnostics' / f'step_{step}.json', {})
    groups = defaultdict(list)
    for row in diagnostic.get('validation_panel', []):
        horizon = row['frame_index'] - cfg['evaluation']['context_frames'] + 1
        score = row['lpips']
        if (not 1 <= horizon <= cfg['evaluation']['horizon'] or isinstance(score, bool)
                or not isinstance(score, (float, int)) or not math.isfinite(score)):
            raise ValueError('Invalid horizon or non-finite diagnostic LPIPS')
        groups[horizon].append(score)
    return [dict(horizon=h, lpips=fmean(values), frames=len(values)) for h, values in sorted(groups.items())]


def _training_row(campaign, method, seed, cfg, stages):
    name = 'vqgan' if method == 'vqgan' else f'train_{method}_{seed}'
    run = campaign / ('autoencoder' if method == 'vqgan' else f'{method}_{seed}')
    metadata, tracking = _json(run / 'run.json', {}), _json(run / 'wandb.json', {})
    rows, val = _recent_metrics(run / 'metrics.jsonl')
    step = rows[-1].get('global_step') if rows else metadata.get('completed_updates')
    val_step = val.get('global_step')
    return dict(method=method, seed=seed, status=metadata.get('status', stages.get(name, {}).get('status', 'pending')),
        updates=step, validation_step=val_step, validation_lpips=val.get('validation/lpips'),
        train_eval_lpips=val.get('train_eval/lpips'),
        best_lpips=val.get('selection/best_validation_lpips', metadata.get('best_validation_lpips')),
        best_step=val.get('selection/best_step'), bad_checks=val.get('early_stopping/bad_checks'),
        patience=val.get('early_stopping/patience'), termination_reason=metadata.get('termination_reason'),
        url=tracking.get('url'), realizations=1,
        horizons=_horizons(run, val_step, cfg) if method != 'vqgan' else [])


def collect_snapshot(campaign):
    """Read status, committed summary and diagnostics; never decode NPZ or checkpoints."""
    campaign = Path(campaign).resolve()
    cfg = yaml.safe_load((campaign / 'protocol.yaml').read_text())
    state = _json(campaign / 'pipeline.json')
    identity, saved = state['identity'], state['stages']
    args = SimpleNamespace(campaign=str(campaign), data=identity['data'], videos=identity['videos'],
        methods=identity['methods'], seeds=identity['seeds'], wandb=False, batch_frames=1,
        cache_dir=None, local_files_only=False)
    planned = build_stages(cfg, args, campaign / 'protocol.yaml')
    stages = [dict(name=s.name, status=saved.get(s.name, {}).get('status', 'pending'),
                   seconds=saved.get(s.name, {}).get('seconds')) for s in planned]
    training = [_training_row(campaign, 'vqgan', cfg['training']['autoencoder_seed'], cfg, saved)]
    training += [_training_row(campaign, method, seed, cfg, saved) for method in args.methods for seed in args.seeds]
    evaluations = []
    for stage in planned:
        if not stage.name.startswith('evaluate_'):
            continue
        name = stage.name.removeprefix('evaluate_')
        directory = campaign / 'evaluation' / name
        meta = _json(directory / 'evaluation.json', {})
        evaluations.append(dict(name=name, status=meta.get('status', saved.get(stage.name, {}).get('status', 'pending')),
            windows_completed=meta.get('windows_completed'),
            expected_windows=meta.get('expected_windows', len(meta.get('expected_window_ids', [])) or None),
            archives_present=sum(p.is_file() for p in (directory / 'predictions').glob('*.npz'))))
    scalars = {'campaign/completed_stages': sum(s['status'] == 'complete' for s in stages),
               'campaign/total_stages': len(stages),
               'campaign/failed_stages': sum(s['status'] == 'failed' for s in stages),
               'training/completed_runs': sum(s['status'] == 'complete' for s in stages if s['name'] == 'vqgan' or s['name'].startswith('train_')),
               'training/total_runs': len(training),
               'evaluation/completed_runs': sum(r['status'] == 'complete' for r in evaluations),
               'evaluation/total_runs': len(evaluations)}
    for row in training:
        if row['updates'] is not None:
            scalars[f"training/{row['method']}_{row['seed']}/updates"] = row['updates']
    summary = _json(campaign / 'summary/summary.json') if saved.get('summarize', {}).get('status') == 'complete' else None
    dataset = _json(Path(args.data) / 'dataset_manifest.json', {})
    return dict(pipeline_status=state['status'], current_stage=state.get('current_stage'), error=state.get('error'),
        observed_at_utc=datetime.now(timezone.utc).isoformat(), methods=args.methods, stages=stages,
        training=training, evaluations=evaluations, scalars=scalars, summary=summary,
        dataset={k: dataset[k] for k in ('statuses', 'eligible_metadata', 'selected') if k in dataset},
        pilot={k: v for k, v in _json(campaign / 'pilot.json', {}).items() if k != 'provenance'},
        context_frames=cfg['evaluation']['context_frames'], horizon=cfg['evaluation']['horizon'])


def _cell(value):
    if value is None:
        return 'non évalué'
    if isinstance(value, float):
        return escape(f'{value:.6g}')
    return escape(str(value))


def _table(rows, columns):
    if not rows:
        return '<p>non évalué — résultat en attente</p>'
    headings = ''.join(f'<th>{escape(label)}</th>' for _, label in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{_cell(row.get(key))}</td>' for key, _ in columns) + '</tr>' for row in rows)
    return f'<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>'


def render_status(snapshot):
    """Render escaped, script-free operational status with explicit scientific scope."""
    parts = ['<!doctype html><html><head><meta charset="utf-8"><style>',
        'body{font:14px system-ui;color:#18212b;padding:16px}table{border-collapse:collapse;width:100%;margin:12px 0}',
        'td,th{border:1px solid #d5dce3;padding:7px;text-align:left}th{background:#edf3f7}h2{margin-top:24px}',
        '</style></head><body><h1>Industrial Plume Forecasting — campagne</h1>',
        f"<p>État enregistré : {_cell(snapshot['pipeline_status'])} — étape : {_cell(snapshot['current_stage'])}</p>",
        f"<p>Lecture UTC : {_cell(snapshot['observed_at_utc'])}. Prévision : {snapshot['context_frames']} observations → {snapshot['horizon']} images futures.</p>"]
    if snapshot['error']:
        parts.append(f"<p>Erreur : {_cell(snapshot['error'])}</p>")
    links = []
    for row in snapshot['training']:
        url = row.get('url')
        parsed = urlparse(url or '')
        if parsed.scheme == 'https' and parsed.hostname in ('wandb.ai', 'forge.coreweave.com'):
            links.append(f'<a href="{escape(url, quote=True)}">{escape(row["method"])} {row["seed"]}</a>')
    parts += ['<p>' + ' · '.join(links) + '</p>', '<h2>Étapes</h2>',
        '<p>pending = en attente ; running = en cours ; complete = terminé. La durée correspond à la dernière exécution enregistrée, pas au temps GPU cumulé.</p>',
        _table(snapshot['stages'], [('name', 'Étape'), ('status', 'État'), ('seconds', 'Dernière durée (s)')]),
        '<h2>Données préparées</h2>', f'<p>{_cell(snapshot["dataset"])}</p>',
        '<h2>Mesures du pilote synthétique</h2><p>Mesures de ressources, pas de qualité prédictive.</p>',
        _table([dict(method=m, **values) for m, values in snapshot['pilot'].get('methods', {}).items()],
               [('method', 'Méthode'), ('micro_batch', 'Micro-lot'), ('update_seconds', 'Mise à jour (s)'),
                ('inference_seconds', 'Génération (s)'), ('peak_memory_bytes', 'Mémoire maximale (octets)')]),
        '<h2>Entraînements</h2><p>Les métriques correspondent à la dernière validation, dont le step est indiqué séparément de la progression.</p>',
        _table(snapshot['training'], [('method', 'Méthode'), ('seed', 'Graine'), ('status', 'État'), ('updates', 'Mises à jour'),
            ('validation_step', 'Step validation'), ('train_eval_lpips', 'LPIPS apprentissage'), ('validation_lpips', 'LPIPS validation'),
            ('best_lpips', 'Meilleur LPIPS'), ('best_step', 'Meilleur step'), ('bad_checks', 'Patience consommée'), ('patience', 'Seuil'),
            ('termination_reason', 'Motif de fin')]), '<h2>Validation selon l’horizon</h2>',
        '<p>1 réalisation fixe par fenêtre, sur les diagnostics existants. Ce tableau ne représente pas la moyenne des 4 réalisations des tests Flow Matching.</p>']
    for row in snapshot['training']:
        if row['horizons']:
            parts += [f'<h3>{escape(row["method"])} — graine {row["seed"]}, validation {row["validation_step"]}</h3>',
                      _table(row['horizons'], [('horizon', 'Horizon'), ('lpips', 'LPIPS moyen'), ('frames', 'Images mesurées')])]
    parts += ['<h2>Tests réservés</h2><p>Les archives présentes indiquent seulement la progression des écritures. Les sorties incomplètes ne constituent pas un résultat final.</p>',
        _table(snapshot['evaluations'], [('name', 'Évaluation'), ('status', 'État'), ('archives_present', 'Archives présentes'),
            ('windows_completed', 'Fenêtres terminées'), ('expected_windows', 'Fenêtres attendues')]), '<h2>Résultats finaux</h2>']
    summary = snapshot['summary']
    if summary is None:
        parts.append('<p>non évalué — synthèse finale en attente</p>')
    else:
        parts.append(f'<p>Portée : {_cell(summary.get("status"))}</p>')
        scores = summary.get('primary_scores', [])
        parts.append(_table(scores, [(key, key) for key in dict.fromkeys(k for row in scores for k in row)]))
    return ''.join(parts) + '</body></html>'


def _load_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError('Install tools/requirements_dashboard.txt in the separate dashboard environment') from exc
    return wandb


def _configure_views(identity, snapshot, directory):
    from tools.configure_wandb import configure
    return {method: configure(identity, method, directory, apply=True)['url']
            for method in ('vqgan', *snapshot['methods'], 'campaign_monitor')}


def _tracking_identity(campaign):
    for name in ('pipeline_monitor', 'autoencoder', *sorted(p.name for p in campaign.iterdir() if p.is_dir() and p.name != 'campaign_monitor')):
        identity = _json(campaign / name / 'wandb.json')
        if identity is not None:
            return identity
    raise ValueError('Campaign has no W&B identity; start its pipeline with --wandb')


def observe(campaign, *, follow=False, interval=30.0, configure_views=False):
    """Publish a separate observer run; its failures never stop the campaign."""
    if isinstance(interval, bool) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('Observer interval must be finite and positive')
    campaign = Path(campaign).resolve()
    snapshot = collect_snapshot(campaign)
    origin = _tracking_identity(campaign)
    directory = campaign / 'campaign_monitor'
    directory.mkdir(exist_ok=True)
    with (directory / 'observer.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another campaign observer is already running') from exc
        sdk = _load_wandb()
        old_identity = _json(directory / 'wandb.json', {})
        state = _json(directory / 'observer_state.json', {})
        run_id = old_identity.get('id', uuid4().hex[:16])
        started, elapsed = time.monotonic(), state.get('elapsed_seconds', 0.0)
        run = sdk.init(project=origin['project'], entity=origin['entity'], id=run_id, resume='allow',
            name=f'{campaign.name}-suivi-global', group=campaign.name, job_type='campaign_monitor',
            dir=str(directory), config=dict(method='campaign_monitor', campaign=campaign.name,
                scope='read-only artifact observation; no training or test selection'),
            save_code=False, settings=sdk.Settings(disable_code=True, disable_git=True, console='off', init_timeout=30))
        if run is None:
            raise RuntimeError('W&B observer did not initialize')
        identity = dict(id=run_id, project=origin['project'], entity=run.entity, url=run.url, mode='online')
        write_json(directory / 'wandb.json', identity)
        write_json(directory / 'run.json', dict(method='campaign_monitor', status='running'))
        failed = True
        try:
            run.define_metric('observer/observation', hidden=True)
            run.define_metric('*', step_metric='observer/observation', hidden=True)
            if configure_views:
                views = _configure_views(identity, snapshot, campaign / 'dashboard')
                write_json(directory / 'dashboards.json', views)
            while True:
                snapshot = collect_snapshot(campaign)
                observation = state.get('observation', -1) + 1
                duration = elapsed + time.monotonic() - started
                payload = snapshot['scalars'] | {'observer/observation': observation, 'observer/elapsed_seconds': duration}
                content_digest = digest({key: value for key, value in snapshot.items() if key != 'observed_at_utc'})
                if content_digest != state.get('content_digest'):
                    payload['observer/status'] = sdk.Html(render_status(snapshot))
                    if snapshot['summary'] is not None:
                        for name in ('quality_cost', 'test_dates_horizons', 'test_views_horizons'):
                            path = campaign / 'summary' / f'{name}.png'
                            if path.is_file():
                                payload[f'observer/{name}'] = sdk.Image(str(path))
                run.log(payload)
                state = dict(observation=observation, elapsed_seconds=duration, content_digest=content_digest)
                write_json(directory / 'observer_state.json', state)
                write_json(directory / 'snapshot.json', snapshot)
                print(json.dumps(dict(url=run.url, pipeline_status=snapshot['pipeline_status'],
                    current_stage=snapshot['current_stage'], completed_stages=snapshot['scalars']['campaign/completed_stages']), ensure_ascii=False), flush=True)
                if not follow or snapshot['pipeline_status'] in ('complete', 'failed', 'interrupted'):
                    break
                time.sleep(interval)
            failed = snapshot['pipeline_status'] in ('failed', 'interrupted')
            return dict(url=run.url, pipeline_status=snapshot['pipeline_status'])
        finally:
            write_json(directory / 'run.json', dict(method='campaign_monitor', status='failed' if failed else 'complete',
                pipeline_status=snapshot['pipeline_status']))
            run.finish(exit_code=1 if failed else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--follow', action='store_true', help='Observe all future stages until the pipeline stops')
    parser.add_argument('--interval', type=float, default=30.0, help='Polling interval in seconds')
    parser.add_argument('--configure-views', action='store_true', help='Create or update campaign and model dashboards')
    parser.add_argument('--dry-run', action='store_true', help='Print an artifact snapshot without contacting W&B')
    args = parser.parse_args(argv)
    result = collect_snapshot(args.campaign) if args.dry_run else observe(args.campaign,
        follow=args.follow, interval=args.interval, configure_views=args.configure_views)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
