"""Pre-register an equal-update VQGAN learning-rate comparison; never train."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shlex
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scientific.hyperparameters import autoencoder_settings, training_description
from scientific.protocol import digest, file_hash, load_protocol, validate_prepared_data, write_json, _source_manifest


def prepare_study(cfg, data, output, learning_rates, updates):
    if type(updates) is not int or updates < 1:
        raise ValueError('Predeclare a positive common update count')
    if (len(learning_rates) < 2
            or any(type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0 for rate in learning_rates)
            or len(set(learning_rates)) != len(learning_rates)):
        raise ValueError('Predeclare at least two distinct positive finite learning rates')
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to replace a registered study: {output}')
    validate_prepared_data(json.loads((data / 'dataset_manifest.json').read_text()), cfg)
    training_windows = sum(bool(line.strip()) for line in (data / 'train.jsonl').read_text().splitlines())
    if training_windows == 0:
        raise ValueError('Calibration requires a nonempty training partition')
    if not any(line.strip() for line in (data / 'val.jsonl').read_text().splitlines()):
        raise ValueError('Calibration requires a nonempty validation partition')
    settings = autoencoder_settings(cfg)
    if updates <= cfg['training']['discriminator_start']:
        raise ValueError('This comparison must include adversarial training, not only pretraining')
    candidates, commands = [], ['#!/usr/bin/env bash', 'set -euo pipefail', shlex.join(['cd', str(REPOSITORY)])]
    protocols = []
    for index, rate in enumerate(learning_rates):
        candidate = copy.deepcopy(cfg)
        candidate['training']['max_updates'] = updates
        candidate['training']['autoencoder'] = settings | dict(learning_rate=rate, discriminator_learning_rate=rate)
        path = output / f'candidate_{index + 1}.yaml'
        protocols.append((path, yaml.safe_dump(candidate, sort_keys=False)))
        campaign = output / f'candidate_{index + 1}'
        command = [sys.executable, '-m', 'scientific.cli', '--protocol', str(path), 'autoencoder',
                   '--campaign', str(campaign), '--data', str(data), '--wandb',
                   '--wandb-project', 'smoke-science', '--wandb-group', output.name,
                   '--wandb-name', f'calibration-vqgan-lr-{rate:g}']
        commands.append(shlex.join(command))
        candidates.append(dict(protocol=str(path), protocol_sha256=digest(candidate),
            learning_rate=rate, discriminator_learning_rate=rate,
            campaign=str(campaign), command=command, effective_training=training_description(candidate, 'vqgan')))
    sources = _source_manifest(REPOSITORY)
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'lpips', 'wandb', 'PyYAML'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    manifest = dict(status='registered_not_executed', factor='shared_generator_discriminator_learning_rate',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=cfg['training']['autoencoder_seed'], updates=updates,
        expected_image_draws=updates * cfg['training']['effective_batch_size'],
        training_windows=training_windows, sampling='with replacement; draw count is not unique coverage',
        dataset_sha256=file_hash(data / 'dataset_manifest.json'),
        registered_sources=sources, registered_source_sha256=digest(sources),
        registration_tool_sha256=file_hash(__file__),
        registration_runtime=dict(python=platform.python_version(), executable=sys.executable, packages=versions),
        selection_partitions=['train', 'val'], test_used_for_selection=False,
        criterion='minimum validation LPIPS among completed equal-update candidates; exact ties choose smaller learning rate',
        guardrails=['finite training and validation', 'report smoke/background MSE and smoke-present frame counts',
                    'report fixed-panel codebook occupancy and qualitative errors', 'retain incomplete attempts'],
        limitations=['one autoencoder initialization; no uncertainty across autoencoder seeds',
                     'ranking applies to the declared update count, not guaranteed eventual convergence',
                     'finite repeatedly consulted validation panel can be overfit',
                     'registered source/runtime fingerprints must be compared with actual run provenance before analysis'],
        candidates=candidates)
    output.mkdir(parents=True)
    for path, contents in protocols:
        path.write_text(contents)
    (output / 'commands.sh').write_text('\n'.join(commands) + '\n')
    write_json(output / 'study.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default='configs/scientific_protocol.yaml')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--learning-rates', type=float, nargs='+', required=True)
    parser.add_argument('--updates', type=int, required=True, help='Common exposure chosen before comparing validation scores')
    args = parser.parse_args(argv)
    result = prepare_study(load_protocol(args.protocol), args.data, args.output, args.learning_rates, args.updates)
    print(json.dumps(dict(status=result['status'], candidates=len(result['candidates']), output=str(Path(args.output).resolve())), indent=2))


if __name__ == '__main__':
    main()
