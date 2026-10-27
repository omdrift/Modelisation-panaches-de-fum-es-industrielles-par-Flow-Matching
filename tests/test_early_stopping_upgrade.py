"""Opt-in campaign upgrades preserve the checkpoint and bind the only policy change."""
import copy
import fcntl
import json

import pytest
import yaml

from scientific.protocol import _source_manifest, digest, load_protocol, sources_compatible, write_json


@pytest.fixture
def old_campaign(tmp_path):
    before = tmp_path / 'previous_sources'
    current = tmp_path / 'new_sources'
    for directory, text in [(before, 'old'), (current, 'new')]:
        (directory / 'scientific').mkdir(parents=True)
        (directory / 'scientific/runner.py').write_text(text)
        (directory / 'model').mkdir()
        (directory / 'model/model.py').write_text('unchanged')
    cfg = load_protocol()
    cfg['training'].pop('early_stopping', None)
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'protocol.yaml').write_text(yaml.safe_dump(cfg))
    source = digest(_source_manifest(before))
    origin = dict(protocol=cfg, protocol_sha256=digest(cfg), source_sha256=source, gpu=None,
                  torch='test', cuda=None, dataset_sha256='dataset', method='vqgan', seed=42)
    (campaign / 'autoencoder/checkpoints').mkdir(parents=True)
    write_json(campaign / 'autoencoder/run.json', origin | dict(status='incomplete'))
    (campaign / 'autoencoder/checkpoints/step_15000.pth').write_bytes(b'checkpoint must not be rewritten')
    write_json(campaign / 'pilot.json', dict(provenance=origin))
    write_json(campaign / 'pipeline.json', dict(identity=dict(source_sha256=source, protocol_sha256=digest(cfg)), status='interrupted'))
    return cfg, campaign, before, current, origin


@pytest.mark.parametrize('distinct_supervisor', [False, True])
def test_upgrade_only_adds_policy_and_binds_sources_without_rewriting_weights(old_campaign, distinct_supervisor):
    from tools.enable_early_stopping import enable
    from scientific.runner import check_resume_provenance
    cfg, campaign, before, current, origin = old_campaign
    pipeline_path = campaign / 'pipeline.json'
    pipeline = json.loads(pipeline_path.read_text())
    if distinct_supervisor:
        pipeline['identity']['source_sha256'] = '0' * 64
        write_json(pipeline_path, pipeline)
    old_identity = copy.deepcopy(pipeline['identity'])
    result = enable(campaign, before, source_root=current)
    after = load_protocol(campaign / 'protocol.yaml')
    expected = copy.deepcopy(cfg)
    expected['training']['early_stopping'] = dict(enabled=True, patience=10, min_delta=0.0001, min_updates=5000)
    assert after == expected
    assert result['status'] == 'enabled'
    assert (campaign / 'autoencoder/checkpoints/step_15000.pth').read_bytes() == b'checkpoint must not be rewritten'
    assert json.loads((campaign / 'autoencoder/run.json').read_text())['protocol_sha256'] == digest(cfg)
    receipt = json.loads((campaign / 'maintenance.json').read_text())
    assert receipt['before_protocol'] == cfg and receipt['after_protocol'] == after
    new_source = digest(_source_manifest(current))
    assert receipt['before_pipeline_identity'] == old_identity
    assert json.loads(pipeline_path.read_text())['identity'] == old_identity | dict(source_sha256=new_source, protocol_sha256=digest(after))
    assert sources_compatible(origin['source_sha256'], new_source, campaign, digest(after))
    resumed = origin | dict(protocol=after, protocol_sha256=digest(after), source_sha256=new_source)
    check_resume_provenance(origin, resumed, campaign)
    with pytest.raises(ValueError):
        check_resume_provenance(origin, resumed | {'seed': 17}, campaign)
    assert enable(campaign, before, source_root=current)['status'] == 'already_enabled'


@pytest.mark.parametrize('lock_name', ['pipeline.lock', 'campaign.lock'])
def test_active_campaign_cannot_be_modified(old_campaign, lock_name):
    from tools.enable_early_stopping import enable
    cfg, campaign, before, current, _ = old_campaign
    with (campaign / lock_name).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='running|active|stop'):
            enable(campaign, before, source_root=current)
    assert load_protocol(campaign / 'protocol.yaml') == cfg
    assert not (campaign / 'maintenance.json').exists()


@pytest.mark.parametrize('change', ['weights', 'optimizer', 'tested', 'source_identity'])
def test_upgrade_rejects_unrelated_changes_and_tested_campaigns(old_campaign, change):
    from tools.enable_early_stopping import enable
    cfg, campaign, before, current, _ = old_campaign
    if change == 'weights':
        (current / 'model/model.py').write_text('changed architecture')
    elif change == 'optimizer':
        (current / 'training').mkdir()
        (current / 'training/optimizer.py').write_text('changed optimizer')
    elif change == 'tested':
        (campaign / 'evaluation').mkdir()
        (campaign / 'evaluation/results.json').write_text('{}')
    else:
        (before / 'scientific/runner.py').write_text('not the code that produced this campaign')
    with pytest.raises(ValueError):
        enable(campaign, before, source_root=current)
    assert not (campaign / 'maintenance.json').exists()


def test_receipt_cannot_authorize_a_learning_rate_change(old_campaign):
    from tools.enable_early_stopping import enable
    cfg, campaign, before, current, origin = old_campaign
    enable(campaign, before, source_root=current)
    receipt = json.loads((campaign / 'maintenance.json').read_text())
    receipt['after_protocol']['training']['learning_rate'] *= 2
    receipt['protocol_sha256'] = digest(receipt['after_protocol'])
    write_json(campaign / 'maintenance.json', receipt)
    assert not sources_compatible(origin['source_sha256'], digest(_source_manifest(current)), campaign, receipt['protocol_sha256'])


@pytest.mark.parametrize('field,value', [('data', 'other_dataset'), ('methods', ['other_model']), ('seeds', [999])])
def test_retry_cannot_change_unrelated_pipeline_identity(old_campaign, field, value):
    from tools.enable_early_stopping import enable
    _, campaign, before, current, _ = old_campaign
    enable(campaign, before, source_root=current)
    receipt = json.loads((campaign / 'maintenance.json').read_text())
    state = json.loads((campaign / 'pipeline.json').read_text())
    state['identity'] = receipt['before_pipeline_identity']
    write_json(campaign / 'pipeline.json', state)
    receipt['after_pipeline_identity'][field] = value
    write_json(campaign / 'maintenance.json', receipt)
    with pytest.raises(ValueError, match='receipt|upgrade'):
        enable(campaign, before, source_root=current)
    assert json.loads((campaign / 'pipeline.json').read_text()) == state
