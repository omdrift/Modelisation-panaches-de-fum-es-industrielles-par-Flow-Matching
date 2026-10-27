"""Protocol validation, metadata partitioning and immutable experiment identities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

TRAIN_DATES = set('2018-05-11 2018-06-11 2018-06-12 2018-06-14 2018-07-07 2018-08-06 2018-08-24 2018-09-03 2018-09-19 2018-10-07 2018-11-10 2018-11-12 2018-12-11 2018-12-13 2018-12-28 2019-01-11 2019-01-17 2019-01-18'.split())
VAL_DATES = {'2019-01-22', '2019-02-02'}
TEST_DATES = set('2019-02-03 2019-02-04 2019-03-14 2019-04-01 2019-04-07 2019-04-09 2019-05-15 2019-06-24 2019-07-26 2019-08-11'.split())
TRAIN_VIEWS = {(0, v) for v in (0, 2, 3, 6, 8, 10, 12, 14)}
VAL_VIEWS = {(0, v) for v in (4, 9, 13)}
TEST_VIEWS = {(0, v) for v in (1, 5, 7, 11)} | {(1, 0), (2, 0), (2, 1), (2, 2)}
METHODS = ('persistence', 'optical_flow', 'simvp', 'flow_uniform', 'flow_alpha')
LEARNED = METHODS[2:]
_EARLY_STOPPING_FILES = {'scientific/early_stopping.py', 'scientific/runner.py', 'scientific/protocol.py',
                         'scientific/hyperparameters.py', 'scientific/monitoring.py', 'scientific/pipeline.py',
                         'scientific/cli.py'}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _source_manifest(root=Path('.')) -> dict[str, str]:
    """Hash execution sources consistently for runs, pipelines and study plans."""
    root = Path(root)
    return {
        str(path.relative_to(root)): file_hash(path)
        for folder in ('scientific', 'model', 'training', 'lutils')
        for path in sorted((root / folder).rglob('*.py'))
    }


def sources_compatible(previous, current, campaign=None, protocol_sha256=None) -> bool:
    """Accept identical code or one explicitly recorded campaign transition.

    A receipt binds the full before/after file manifests and the frozen protocol.
    The early-stopping opt-in may only add that policy. Data, model, seed,
    optimizer and runtime identities remain checked independently.
    """
    if previous == current:
        return True
    if campaign is None:
        return False
    path = Path(campaign) / 'maintenance.json'
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    if receipt.get('kind') == 'validation_early_stopping':
        upgrade = _early_stopping_upgrade(campaign)
        return (upgrade is not None and upgrade['protocol_sha256'] == protocol_sha256
                and digest(upgrade['before_sources']) == previous and digest(upgrade['after_sources']) == current)
    return (receipt.get('kind') in ('checkpoint_io_and_resume_checks', 'observational_monitoring_and_checkpoint_io')
            and receipt.get('protocol_sha256') == protocol_sha256
            and any(digest(sources) == previous for sources in [receipt.get('before_sources', {}), *receipt.get('intermediate_sources', [])])
            and digest(receipt.get('after_sources', {})) == current)


def _early_stopping_upgrade(campaign):
    """Validate a narrowly scoped, explicit change to an existing campaign."""
    if campaign is None or not (Path(campaign) / 'maintenance.json').is_file():
        return None
    receipt = json.loads((Path(campaign) / 'maintenance.json').read_text())
    if receipt.get('kind') != 'validation_early_stopping':
        return None
    try:
        before, after = receipt['before_protocol'], receipt['after_protocol']
        expected = copy.deepcopy(before)
        expected['training']['early_stopping'] = after['training']['early_stopping']
        from scientific.early_stopping import settings
        if settings(before)['enabled'] or not settings(after)['enabled'] or expected != after:
            return None
        if digest(after) != receipt['protocol_sha256']:
            return None
        old_sources, new_sources = receipt['before_sources'], receipt['after_sources']
        changed = {name for name in old_sources.keys() | new_sources.keys() if old_sources.get(name) != new_sources.get(name)}
        if changed - _EARLY_STOPPING_FILES:
            return None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return receipt


def _protocols_compatible(previous, current, campaign=None):
    if previous == current:
        return True
    receipt = _early_stopping_upgrade(campaign)
    return (receipt is not None and digest(receipt['before_protocol']) == previous
            and receipt['protocol_sha256'] == current)


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records) -> None:
    Path(path).write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n' for r in records))


def load_protocol(path='configs/scientific_protocol.yaml') -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('version') != '1.1':
        raise ValueError('Unsupported scientific protocol version')
    task = cfg['task']
    if (task['resolution'], task['clip_frames']) != (64, 36):
        raise ValueError('Protocol v1.1 preserves complete 36-frame clips at the current 64x64 resolution')
    from scientific.segmentation import validate_settings
    validate_settings(cfg['preprocessing'])
    from scientific.hyperparameters import autoencoder_settings
    autoencoder_settings(cfg)
    from scientific.early_stopping import settings as early_stopping_settings
    early_stopping_settings(cfg)
    if cfg['seeds'] != [17, 42, 2026]:
        raise ValueError('Protocol v1 fixes the predictor seeds')
    if cfg['evaluation']['realizations'] != 4 or cfg['evaluation']['ode_grid_points'] < 2:
        raise ValueError('Four realizations and at least two ODE grid points are required')
    if cfg['model']['min_target_index'] != 2 or cfg['model']['mask_source'] != 'alpha' or cfg['model']['autoencoder']['type'] != 'vqgan':
        raise ValueError('Scientific models require RIVER target sampling from the third frame, alpha masks and explicit architecture settings')
    if cfg['training']['effective_batch_size'] != 16 or (type(cfg['training']['max_updates']) is not int or cfg['training']['max_updates'] < 1) or cfg['training']['autoencoder_seed'] != 42:
        raise ValueError('Invalid fixed batch, update limit or autoencoder seed')
    if min(cfg['training']['smoke_weight'], cfg['training']['background_weight'], cfg['training']['warmup_updates'], cfg['training']['validation_every']) <= 0:
        raise ValueError('Loss weights and update intervals must be positive')
    if any(not 1 <= cfg['selection'][key] <= maximum for key, maximum in [('validation_windows', 60), ('test_windows_per_split', 200)]):
        raise ValueError('Invalid frozen panel size')
    forecast_task(cfg, required=False)
    return cfg


def forecast_task(cfg, required=True):
    """Keep the full training clip distinct from the explicitly chosen forecast task."""
    context = cfg['evaluation'].get('context_frames')
    horizon = cfg['evaluation'].get('horizon')
    if context is None and horizon is None:
        if required:
            raise ValueError('Choose --context-frames and --horizon explicitly before a GPU campaign; preserving 36 frames does not fix the forecast task')
        return None
    if type(context) is not int or type(horizon) is not int:
        raise ValueError('context_frames and horizon must both be integers')
    minimum = 2  # RIVER requires an earlier condition in addition to its reference.
    if context < minimum or horizon < 1 or context + horizon > cfg['task']['clip_frames']:
        raise ValueError('Require context_frames >= 2, horizon >= 1 and context_frames + horizon <= 36')
    return context, horizon


def data_spec(cfg):
    """Identify prepared data independently of forecast and training choices."""
    return {key: cfg[key] for key in ('version', 'metadata', 'selection_seed', 'task', 'preprocessing', 'selection')}


def validate_prepared_data(manifest, cfg):
    """Require an intact preparation specification matching this protocol."""
    prepared = manifest['protocol']
    current_spec = data_spec(prepared)
    if manifest.get('data_spec_sha256') != digest(current_spec):
        raise ValueError('Prepared dataset has an inconsistent data specification fingerprint')
    if digest(current_spec) != digest(data_spec(cfg)):
        raise ValueError('Prepared data does not match the frozen protocol')


def eligible(item: dict) -> bool:
    labels = (item.get('label_state'), item.get('label_state_admin'))
    return 23 in labels and -2 not in labels


def partition(item: dict) -> str:
    """Intersection of official RISE temporal split S3 and spatial split S0."""
    if not eligible(item):
        return 'excluded_label'
    date = '-'.join(item['file_name'].split('-')[2:5])
    view = (int(item['camera_id']), int(item['view_id']))
    if date in TRAIN_DATES and view in TRAIN_VIEWS:
        return 'train'
    if date in VAL_DATES and view in TRAIN_VIEWS | VAL_VIEWS:
        return 'val'
    if date in TEST_DATES and view in TRAIN_VIEWS:
        return 'test_dates'
    if date in TEST_DATES and view in TEST_VIEWS:
        return 'test_views'
    return 'excluded_partition'


def stratified_selection(records: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin date/camera/view strata, ranked by stable seeded hashes."""
    groups = {}
    for record in records:
        key = (record['date'], record['camera_id'], record['view_id'])
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        group.sort(key=lambda r: digest([seed, r['window_id']]))
    keys = sorted(groups, key=lambda k: digest([seed, k]))
    result = []
    while keys and len(result) < count:
        for key in keys:
            if len(result) == count:
                break
            result.append(groups[key].pop(0))
        keys = [key for key in keys if groups[key]]
    return result
