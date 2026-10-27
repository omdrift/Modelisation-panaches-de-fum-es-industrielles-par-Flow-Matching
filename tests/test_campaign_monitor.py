"""Campaign observation reads artifacts and never executes a scientific stage."""
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scientific.protocol import file_hash, load_protocol, write_json


@pytest.fixture
def campaign(tmp_path):
    root, data = tmp_path / 'campaign', tmp_path / 'data'
    root.mkdir()
    data.mkdir()
    cfg = load_protocol()
    (root / 'protocol.yaml').write_text(yaml.safe_dump(cfg))
    write_json(root / 'pipeline.json', dict(status='running', current_stage='train_simvp_17',
        identity=dict(data=str(data), videos='removed', methods=['simvp', 'flow_uniform', 'flow_alpha'], seeds=[17, 42, 2026]),
        stages={'vqgan': {'status': 'complete', 'seconds': 12.0}, 'train_simvp_17': {'status': 'running'}}))
    write_json(data / 'dataset_manifest.json', dict(statuses={'prepared': 100}, eligible_metadata=150,
        selected={'train': 50, 'val': 60, 'test_dates': 200, 'test_views': 200}))
    (data / 'unreadable.npz').write_bytes(b'Must never decode dataset tensors')
    write_json(root / 'pilot.json', dict(status='resource_probe_not_scientific_result',
        methods={'simvp': dict(micro_batch=4, update_seconds=0.3, inference_seconds=0.02, peak_memory_bytes=123)}))
    run = root / 'simvp_17'
    run.mkdir()
    write_json(run / 'run.json', dict(status='running', method='simvp', seed=17))
    write_json(run / 'wandb.json', dict(id='simvp-run', entity='owner', project='plumes',
        url='https://wandb.ai/owner/plumes/runs/simvp-run'))
    (run / 'metrics.jsonl').write_text(json.dumps({'global_step': 1000, 'validation/lpips': 0.2,
        'train_eval/lpips': 0.1, 'selection/best_validation_lpips': 0.2,
        'selection/best_step': 1000, 'early_stopping/bad_checks': 0, 'early_stopping/patience': 10}) + '\n' +
        json.dumps({'global_step': 1025, 'progress/updates': 1025, 'train/loss': 0.03}) + '\n' + '{"partially_written":')
    (run / 'diagnostics').mkdir()
    context = cfg['evaluation']['context_frames']
    write_json(run / 'diagnostics/step_1000.json', dict(step=1000, validation_panel=[
        dict(frame_index=context, lpips=0.1), dict(frame_index=context, lpips=0.3),
        dict(frame_index=context + 1, lpips=0.4)]))
    return root


def test_snapshot_covers_all_stages_and_keeps_unstarted_flows_pending(campaign):
    from tools.monitor_campaign import collect_snapshot
    result = collect_snapshot(campaign)
    assert len(result['stages']) == 45
    assert next(s for s in result['stages'] if s['name'] == 'train_flow_uniform_17')['status'] == 'pending'
    assert result['dataset']['selected']['train'] == 50
    assert result['pilot']['methods']['simvp']['micro_batch'] == 4
    assert result['scalars']['campaign/completed_stages'] == 1
    assert result['scalars']['campaign/total_stages'] == 45
    assert result['scalars']['training/completed_runs'] == 1
    assert result['scalars']['evaluation/total_runs'] == 30
    assert not (campaign / 'campaign_monitor').exists()


def test_snapshot_retains_latest_update_and_latest_validation_separately(campaign):
    from tools.monitor_campaign import collect_snapshot
    row = next(r for r in collect_snapshot(campaign)['training'] if r['method'] == 'simvp' and r['seed'] == 17)
    assert row['updates'] == 1025
    assert row['validation_step'] == 1000
    assert row['validation_lpips'] == 0.2
    assert row['train_eval_lpips'] == 0.1
    assert row['best_step'] == 1000
    assert row['bad_checks'] == 0
    assert row['horizons'] == [{'horizon': 1, 'lpips': pytest.approx(0.2), 'frames': 2},
                               {'horizon': 2, 'lpips': 0.4, 'frames': 1}]
    assert row['realizations'] == 1


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True])
def test_invalid_horizon_measurements_are_not_silently_averaged(campaign, invalid):
    from tools.monitor_campaign import collect_snapshot
    path = campaign / 'simvp_17/diagnostics/step_1000.json'
    obj = json.loads(path.read_text())
    obj['validation_panel'][0]['lpips'] = invalid
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='LPIPS|finite'):
        collect_snapshot(campaign)


def test_partial_test_outputs_do_not_become_final_quality_results(campaign):
    from tools.monitor_campaign import collect_snapshot
    summary = campaign / 'summary'
    summary.mkdir()
    (summary / 'summary.json').write_text('This incomplete summary must not be read')
    evaluation = campaign / 'evaluation/flow_uniform_17_test_dates'
    (evaluation / 'predictions').mkdir(parents=True)
    write_json(evaluation / 'evaluation.json', dict(status='running', expected_window_ids=['a', 'b']))
    (evaluation / 'predictions/a.npz').write_bytes(b'archive existence is only an operational indicator')
    result = collect_snapshot(campaign)
    assert result['summary'] is None
    row = next(r for r in result['evaluations'] if r['name'] == evaluation.name)
    assert row['status'] == 'running' and row['archives_present'] == 1 and row['expected_windows'] == 2
    assert result['scalars']['evaluation/completed_runs'] == 0


def test_completed_summary_is_reused_without_recomputing_test_metrics(campaign):
    from tools.monitor_campaign import collect_snapshot
    pipeline = json.loads((campaign / 'pipeline.json').read_text())
    pipeline['stages']['summarize'] = {'status': 'complete'}
    write_json(campaign / 'pipeline.json', pipeline)
    write_json(campaign / 'summary/summary.json', dict(status='partial_campaign', primary_scores=[{'method': 'simvp', 'lpips': 0.2}]))
    result = collect_snapshot(campaign)
    assert result['summary']['status'] == 'partial_campaign'
    assert result['summary']['primary_scores'] == [{'method': 'simvp', 'lpips': 0.2}]


def test_html_explains_scope_and_escapes_untrusted_values(campaign):
    from tools.monitor_campaign import collect_snapshot, render_status
    result = collect_snapshot(campaign)
    result['current_stage'] = '<script>alert(1)</script>'
    result['training'][0]['url'] = 'javascript:alert(1)'
    rendered = render_status(result)
    assert '<script>' not in rendered and '&lt;script&gt;' in rendered
    assert 'href="javascript:' not in rendered
    assert '1 réalisation fixe' in rendered
    assert 'pilote synthétique' in rendered
    assert 'non évalué' in rendered
    assert '0.2' in rendered


class FakeRun:
    entity = 'owner'
    url = 'https://wandb.ai/owner/plumes/runs/observer'

    def __init__(self, fail=False):
        self.logged, self.finished, self.fail = [], [], fail

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values):
        if self.fail:
            raise RuntimeError('transport failed')
        self.logged.append(copy.deepcopy(values))

    def finish(self, **kwargs):
        self.finished.append(kwargs)


def fake_sdk(monkeypatch, *, fail=False):
    from tools import monitor_campaign as monitor
    runs, initializations = [], []

    def initialize(**kwargs):
        initializations.append(kwargs)
        run = FakeRun(fail=fail)
        runs.append(run)
        return run

    sdk = SimpleNamespace(init=initialize, Html=lambda text: text, Image=lambda path: str(path),
                          Settings=lambda **kwargs: kwargs)
    monkeypatch.setattr(monitor, '_load_wandb', lambda: sdk)
    return runs, initializations


def test_publish_isolated_observer_resumes_its_identity_and_counter(campaign, monkeypatch):
    from tools.monitor_campaign import observe
    runs, initializations = fake_sdk(monkeypatch)
    protected = {str(p): file_hash(p) for p in campaign.rglob('*') if p.is_file()}
    for _ in range(2):
        observe(campaign)
    assert initializations[0]['id'] == initializations[1]['id']
    assert initializations[0]['config']['method'] == 'campaign_monitor'
    assert initializations[0]['settings']['disable_git'] is True
    assert runs[1].logged[0]['observer/observation'] > runs[0].logged[0]['observer/observation']
    assert runs[1].finished[-1] == {'exit_code': 0}
    assert protected == {name: file_hash(name) for name in protected}
    assert 'observer/status' in runs[0].logged[0]


def test_follow_stops_after_pipeline_completion_and_configures_future_views(campaign, monkeypatch):
    from tools import monitor_campaign as monitor
    runs, _ = fake_sdk(monkeypatch)
    configured = []
    monkeypatch.setattr(monitor, '_configure_views', lambda identity, snapshot, directory: configured.extend(snapshot['methods']))

    def finish_pipeline(seconds):
        state = json.loads((campaign / 'pipeline.json').read_text())
        state.update(status='complete', current_stage=None)
        write_json(campaign / 'pipeline.json', state)

    monkeypatch.setattr(monitor.time, 'sleep', finish_pipeline)
    observe = monitor.observe(campaign, follow=True, configure_views=True)
    assert observe['pipeline_status'] == 'complete'
    assert configured == ['simvp', 'flow_uniform', 'flow_alpha']
    assert len(runs[0].logged) == 2


@pytest.mark.parametrize('interval', [0, -1, float('inf'), float('nan')])
def test_invalid_poll_interval_is_rejected_without_loading_sdk(campaign, interval):
    from tools.monitor_campaign import observe
    with pytest.raises(ValueError, match='interval'):
        observe(campaign, interval=interval)
    assert not (campaign / 'campaign_monitor').exists()


def test_an_observer_lock_never_blocks_the_training_lock(campaign, monkeypatch):
    from tools.monitor_campaign import observe
    fake_sdk(monkeypatch)
    directory = campaign / 'campaign_monitor'
    directory.mkdir()
    with (directory / 'observer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='observer'):
            observe(campaign)
    assert not (campaign / 'campaign.lock').exists()


def test_transport_failure_is_reported_without_changing_pipeline(campaign, monkeypatch):
    from tools.monitor_campaign import observe
    runs, _ = fake_sdk(monkeypatch, fail=True)
    before = (campaign / 'pipeline.json').read_bytes()
    with pytest.raises(RuntimeError, match='transport failed'):
        observe(campaign)
    assert (campaign / 'pipeline.json').read_bytes() == before
    assert runs[0].finished[-1] == {'exit_code': 1}
    assert json.loads((campaign / 'campaign_monitor/run.json').read_text())['status'] == 'failed'
