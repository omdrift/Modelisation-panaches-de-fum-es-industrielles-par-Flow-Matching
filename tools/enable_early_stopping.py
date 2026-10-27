"""Opt an inactive, untested campaign into validation early stopping without rewriting weights."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scientific.early_stopping import DEFAULTS, settings
from scientific.protocol import (_EARLY_STOPPING_FILES, _early_stopping_upgrade,
    _source_manifest, digest, load_protocol, write_json)


def _write_protocol(path, cfg):
    temporary = path.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
    temporary.replace(path)


def _apply_upgrade(campaign, receipt):
    pipeline_path = campaign / 'pipeline.json'
    state = json.loads(pipeline_path.read_text()) if pipeline_path.is_file() else None
    if state is not None and state['identity'] not in (receipt['before_pipeline_identity'], receipt['after_pipeline_identity']):
        raise ValueError('Pipeline identity changed outside this recorded upgrade')
    _write_protocol(campaign / 'protocol.yaml', receipt['after_protocol'])
    if state is not None:
        state['identity'] = receipt['after_pipeline_identity']
        write_json(pipeline_path, state)


def enable(campaign, previous_source_root, *, source_root=REPOSITORY, **policy):
    """Record a verifiable policy/source transition; never launch training."""
    campaign = Path(campaign)
    path = campaign / 'protocol.yaml'
    cfg = load_protocol(path)
    with ExitStack() as stack:
        for name in ('pipeline.lock', 'campaign.lock'):
            lock = stack.enter_context((campaign / name).open('a+'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Campaign is still running; stop it and wait for its checkpoint before enabling early stopping') from exc
        after_sources = _source_manifest(source_root)
        after = copy.deepcopy(cfg)
        after['training']['early_stopping'] = DEFAULTS | dict(enabled=True) | policy
        if not settings(after)['enabled']:
            raise ValueError('This upgrade must enable early stopping')
        existing = campaign / 'maintenance.json'
        if existing.exists():
            receipt = _early_stopping_upgrade(campaign)
            if (receipt is None or receipt['after_sources'] != after_sources
                    or receipt['after_protocol']['training']['early_stopping'] != after['training']['early_stopping']
                    or cfg not in (receipt['before_protocol'], receipt['after_protocol'])):
                raise ValueError('Existing maintenance receipt does not authorize this upgrade')
            _apply_upgrade(campaign, receipt)
            return dict(status='already_enabled', policy=after['training']['early_stopping'])
        if settings(cfg)['enabled']:
            raise ValueError('Campaign already declares early stopping; its policy remains frozen')
        evaluation = campaign / 'evaluation'
        if evaluation.exists() and any(evaluation.iterdir()):
            raise ValueError('Cannot change the stopping rule after test evaluation has begun')
        before_sources = _source_manifest(previous_source_root)
        previous_sha = digest(before_sources)
        changed = {name for name in before_sources.keys() | after_sources.keys()
                   if before_sources.get(name) != after_sources.get(name)}
        if not before_sources or not after_sources or changed - _EARLY_STOPPING_FILES:
            raise ValueError('Upgrade may only change the reviewed early-stopping and orchestration files')
        origins = [json.loads(p.read_text()) for p in campaign.glob('*/run.json')]
        if (campaign / 'pilot.json').exists():
            origins.append(json.loads((campaign / 'pilot.json').read_text())['provenance'])
        if not origins or any(origin.get('source_sha256') != previous_sha or origin.get('protocol_sha256') != digest(cfg) for origin in origins):
            raise ValueError('Previous source directory does not match the recorded campaign identity')
        pipeline_path = campaign / 'pipeline.json'
        old_identity = json.loads(pipeline_path.read_text())['identity'] if pipeline_path.is_file() else None
        if old_identity is not None and old_identity.get('protocol_sha256') != digest(cfg):
            raise ValueError('Supervisor protocol does not match the recorded training protocol')
        new_identity = (old_identity | dict(protocol_sha256=digest(after), source_sha256=digest(after_sources))) if old_identity is not None else None
        receipt = dict(kind='validation_early_stopping', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            before_protocol=cfg, after_protocol=after, protocol_sha256=digest(after),
            before_sources=before_sources, after_sources=after_sources,
            before_pipeline_identity=old_identity, after_pipeline_identity=new_identity,
            source_identity_note='Training and pilot sources are verified against the snapshot; the original supervisor identity is preserved separately.',
            patience_history='starts on resume, anchored to the saved best validation LPIPS; no retrospective stop',
            changed_files=sorted(changed))
        write_json(existing, receipt)
        _apply_upgrade(campaign, receipt)
        return dict(status='enabled', policy=after['training']['early_stopping'], protocol_sha256=digest(after))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--previous-source-root', required=True, help='Original source snapshot matching the campaign fingerprint')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=0.0001)
    parser.add_argument('--min-updates', type=int, default=5000)
    args = parser.parse_args(argv)
    print(json.dumps(enable(args.campaign, args.previous_source_root, patience=args.patience,
                            min_delta=args.min_delta, min_updates=args.min_updates), indent=2))


if __name__ == '__main__':
    main()
