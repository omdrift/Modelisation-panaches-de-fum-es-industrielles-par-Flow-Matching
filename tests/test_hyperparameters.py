"""Stage settings and calibration registration must match executable recipes."""
import copy
import json
import shlex

import pytest

from scientific.hyperparameters import AUTOENCODER_DEFAULTS, autoencoder_settings, training_description
from scientific.protocol import data_spec, digest, load_protocol, write_json
from tools.plan_calibration import prepare_study


def test_stage_defaults_do_not_inherit_predictor_settings_or_mutate_configuration():
    cfg = load_protocol()
    before = copy.deepcopy(cfg)
    explicit = autoencoder_settings(cfg)
    assert explicit == AUTOENCODER_DEFAULTS
    cfg['training'].pop('autoencoder')
    cfg['training']['learning_rate'] = 0.1
    cfg['training']['validation_every'] = 3
    assert autoencoder_settings(cfg) == explicit
    explicit['betas'][0] = 0.0
    assert AUTOENCODER_DEFAULTS['betas'] == [0.9, 0.999]
    assert before == load_protocol()
    assert training_description(cfg, 'vqgan')['settings_source'] == 'legacy_frozen_protocol_defaults'
    assert training_description(cfg, 'flow_uniform')['learning_rate'] == 0.1


@pytest.mark.parametrize('override', [
    {'learning_rate': 0}, {'learning_rate': float('nan')}, {'epsilon': True},
    {'weight_decay': -1}, {'betas': [0.9, 1.0]}, {'betas': [True, 0.9]},
    {'validation_every': 2.5}, {'validation_every': 0}, {'schedule': 'unknown'},
    {'adversarial_weight': float('inf')}, {'typo_lr': 0.1},
    {'reconstruction_weight': 0, 'perceptual_weight': 0, 'quantization_weight': 0},
])
def test_invalid_settings_are_rejected(override):
    cfg = load_protocol()
    cfg['training']['autoencoder'].update(override)
    with pytest.raises(ValueError):
        autoencoder_settings(cfg)


def test_study_freezes_equal_exposure_inputs_and_only_varies_coupled_lr(tmp_path):
    cfg = load_protocol()
    before = copy.deepcopy(cfg)
    data = tmp_path / 'prepared data'
    data.mkdir()
    write_json(data / 'dataset_manifest.json', dict(protocol=cfg, data_spec_sha256=digest(data_spec(cfg))))
    (data / 'train.jsonl').write_text('{"window_id":"train_1"}\n')
    (data / 'val.jsonl').write_text('{"window_id":"val_1"}\n')
    output = tmp_path / 'study'
    result = prepare_study(cfg, data, output, [5e-5, 1e-4, 2e-4], 1500)
    assert cfg == before
    assert result['status'] == 'registered_not_executed'
    assert result['test_used_for_selection'] is False
    assert result['expected_image_draws'] == 1500 * 16
    assert result['registered_source_sha256'] == digest(result['registered_sources'])
    assert result['registration_runtime']['packages']['torch'] is not None
    assert not any((output / f'candidate_{i}').exists() for i in (1, 2, 3))
    for item, rate in zip(result['candidates'], [5e-5, 1e-4, 2e-4]):
        candidate = load_protocol(item['protocol'])
        expected = copy.deepcopy(before)
        expected['training']['max_updates'] = 1500
        expected['training']['autoencoder'].update(learning_rate=rate, discriminator_learning_rate=rate)
        expected['training']['early_stopping']['enabled'] = False
        assert candidate == expected
        assert digest(candidate) == item['protocol_sha256']
        assert shlex.join(item['command']) in (output / 'commands.sh').read_text()
    assert json.loads((output / 'study.json').read_text()) == result
    with pytest.raises(FileExistsError):
        prepare_study(cfg, data, output, [5e-5, 1e-4], 1500)


@pytest.mark.parametrize('rates,updates', [([1e-4], 1500), ([1e-4, 1e-4], 1500),
    ([0, 1e-4], 1500), ([float('nan'), 1e-4], 1500), ([1e-4, 2e-4], 0), ([1e-4, 2e-4], 1.5),
    ([True, 1e-4], 1500), (['0.0002', 1e-4], 1500), ([[0.0002], 1e-4], 1500)])
def test_invalid_study_does_not_create_output(tmp_path, rates, updates):
    with pytest.raises(ValueError):
        prepare_study(load_protocol(), tmp_path / 'data', tmp_path / 'output', rates, updates)
    assert not (tmp_path / 'output').exists()
