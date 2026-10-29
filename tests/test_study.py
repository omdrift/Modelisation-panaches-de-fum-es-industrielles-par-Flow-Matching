"""The complete study matrix is exercised without GPU or pretrained weights."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scientific import cli, study
from scientific.protocol import data_spec, digest, file_hash, load_protocol, write_json, write_jsonl
from scientific.study_evaluation import verify_selection
from scientific.study_runtime import trial_protocol, validation_heartbeat


@pytest.fixture
def study_case(tmp_path):
    cfg = load_protocol()
    data = tmp_path / 'data'
    (data / 'clips').mkdir(parents=True)
    manifests = {}
    for index, split in enumerate(('train', 'val', 'test_dates', 'test_views')):
        path = data / 'clips' / f'{split}.npz'
        raw = np.full((36, 64, 64, 3), 17 + index * 50, np.uint8)
        np.savez_compressed(path, frames=raw, raw_frames=raw, alpha=np.ones((36, 64, 64), np.float32))
        write_jsonl(data / f'{split}.jsonl', [dict(window_id=split, video_id=index, split=split,
            frame_count=36, camera_id=0, view_id=1 if split == 'test_views' else 0,
            date='2019-02-01' if split.startswith('test') else f'2019-01-0{index+1}',
            path=f'clips/{split}.npz', sha256=file_hash(path),
            frame_indices=list(range(36)), timestamps=list(range(36)))])
        manifests[split] = file_hash(data / f'{split}.jsonl')
    write_json(data / 'dataset_manifest.json', dict(protocol=cfg, data_spec_sha256=digest(data_spec(cfg)), manifests=manifests))
    settings = study.load_study('configs/experiment_study.yaml')
    settings['final_test'] = False
    config = tmp_path / 'study.yaml'
    config.write_text(yaml.safe_dump(settings))
    args = SimpleNamespace(config=str(config), data=str(data), output=str(tmp_path / 'study'),
        dry_run=False, resume=False, wandb=False, wandb_project='test', wandb_entity=None,
        wandb_mode='offline', wandb_group=None, wandb_name=None, wandb_log_every=25, wandb_media_every=250)
    return cfg, settings, args


def fake_executor(calls):
    def execute(command, log, heartbeat):
        cfg = load_protocol(command[command.index('--protocol') + 1])
        campaign = Path(command[command.index('--campaign') + 1])
        name = command[6]
        calls.append((name, cfg['study']))
        if name.startswith('pilot'):
            write_json(campaign / ('pilot_autoencoder.json' if name == 'pilot-autoencoder' else 'pilot.json'), dict(status='injected_cpu_probe_not_publication'))
            return
        assert name in ('autoencoder', 'train')
        method = 'vqgan' if name == 'autoencoder' else command[command.index('--method') + 1]
        seed = cfg['training']['autoencoder_seed'] if method == 'vqgan' else int(command[command.index('--seed') + 1])
        index = int(cfg['study']['candidate'][1:])
        score = index / 100
        if cfg['study']['phase'] == 'transfer':
            score = .03 if index == 1 else .02  # The best reconstruction need not give the best forecast.
        score += (seed % 3) / 100000
        run = campaign / ('autoencoder' if method == 'vqgan' else f'{method}_{seed}')
        run.mkdir(parents=True, exist_ok=True)
        status = 'complete'
        if 'study_trial' in cfg:
            write_json(cfg['study_trial']['report_path'], dict(step=10000, score=score))
            heartbeat()
            if json.loads(Path(cfg['study_trial']['decision_path']).read_text())['prune']:
                status = 'pruned'
        write_json(run / 'run.json', dict(status=status, method=method, seed=seed,
            protocol_sha256=digest(cfg), completed_updates=cfg['training']['max_updates'], best_validation_lpips=score))
        torch.save(dict(model={'weight': torch.ones(1)}, optimizer={}, scheduler={}, rng={}, extra={}, step=10000), run / 'best.pth')
        (run / 'diagnostics').mkdir(exist_ok=True)
        write_json(run / 'diagnostics/best.json', dict(step=10000))
    return execute


def test_dry_run_is_pure_and_cli_exposes_all29_trials(study_case, monkeypatch, capsys):
    cfg, settings, args = study_case
    args.dry_run = True
    monkeypatch.setattr(study, '_optuna', lambda: pytest.fail('dry-run must not import Optuna'))
    plan = study.run_study(cfg, args, executor=lambda *a: pytest.fail('dry-run executed a command'))
    assert plan['maximum_training_runs'] == sum(p['runs'] for p in plan['phases']) == 29
    assert len(plan['candidates']) == 6
    assert not Path(args.output).exists()
    assert cli.main(['study', '--config', args.config, '--data', args.data, '--output', args.output, '--dry-run']) == 0
    assert '29' in capsys.readouterr().out


def test_trial_protocol_keeps_old_defaults_and_sets_stage_controls(study_case):
    cfg, settings, args = study_case
    original = deepcopy(cfg)
    candidate = study.candidates(settings)[1]
    ae, path = trial_protocol(cfg, settings, args.output, 'vqgan_exploration', candidate, 'vqgan', 42, handshake=True)
    assert ae['training']['max_updates'] == 15000
    assert ae['training']['autoencoder']['schedule'] == 'cosine'
    assert ae['training']['early_stopping']['patience'] == 10
    assert ae['diagnostics']['reconstruction_all_frames']
    parent = {'protocol': str(path / 'protocol.yaml'), 'checkpoint': str(path / 'autoencoder/best.pth')}
    flow, _ = trial_protocol(cfg, settings, args.output, 'flow_confirmation', candidate, 'flow_uniform', 17, autoencoder=parent)
    assert flow['training']['max_updates'] == 50000
    assert flow['training']['validation_every'] == 2500
    assert flow['training']['early_stopping']['patience'] == 4
    assert flow['diagnostics']['forecast_realizations'] == 4
    assert 'study_trial' not in flow
    assert cfg == original


def test_confirmation_requires_all_seeds_and_uses_mean_not_best_seed():
    rows = [dict(candidate='a',seed=seed,status='complete',best_validation_lpips=score)
        for seed, score in zip((17,42,2026),(.01,.09,.2))]
    result = study.confirmation_means(rows, [17,42,2026])
    assert result['a']['mean'] == pytest.approx(.1)
    assert result['a']['std'] > 0
    with pytest.raises(ValueError, match='seed'):
        study.confirmation_means(rows[:2], [17,42,2026])
    with pytest.raises(RuntimeError):
        study.select_finalists(rows[:1])


def test_full_matrix_pruning_selection_freeze_and_exact_study_resume(study_case, monkeypatch):
    pytest.importorskip('optuna')
    cfg, settings, args = study_case
    # Any attempt to open a test NPZ during search fails immediately.
    original_load = np.load
    def train_only(path, *a, **kw):
        assert not Path(path).stem.startswith('test')
        return original_load(path, *a, **kw)
    monkeypatch.setattr(np, 'load', train_only)
    calls = []
    result = study.run_study(cfg, args, executor=fake_executor(calls))
    assert result['training_runs'] == 29
    state = json.loads((Path(args.output) / 'study.json').read_text())
    assert state['selection'] == dict(vqgan_candidate='c02', flow_candidate='c01')
    assert len(state['final_selection']['artifacts']) == 7
    assert any(r['status'] == 'pruned' for r in state['runs'].values())
    assert not (Path(args.output) / 'data_audit_final.json').exists()
    assert (Path(args.output) / 'experiments.csv').is_file()
    assert len([c for c in calls if not c[0].startswith('pilot')]) == 29
    args.resume = True
    study.run_study(cfg, args, executor=lambda *a: pytest.fail('completed trial was retrained'))
    selected = next(r for r in state['final_selection']['artifacts'] if r['method'] == 'flow_uniform' and r['seed'] == 42)
    selected_cfg = load_protocol(selected['protocol'])
    verify_selection(selected_cfg, selected['checkpoint'], args.data)
    verify_selection(selected_cfg, None, args.data)  # Deterministic baseline has no checkpoint.
    Path(selected['checkpoint']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        verify_selection(selected_cfg, selected['checkpoint'], args.data)
    with pytest.raises(ValueError, match='changed'):
        study.run_study(cfg, args, executor=lambda *a: pytest.fail('tampered trial was rerun'))


def test_pruning_heartbeat_acknowledges_restart_and_rejects_nonfinite(tmp_path):
    optuna = pytest.importorskip('optuna')
    study_obj = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_startup_trials=0))
    trial = study_obj.ask()
    write_json(tmp_path / 'validation_report.json', dict(step=7500, score=.3))
    state = {}
    validation_heartbeat(tmp_path, trial, state, lambda: None)
    validation_heartbeat(tmp_path, trial, state, lambda: None)
    assert trial.storage.get_trial(trial._trial_id).intermediate_values == {7500: .3}
    assert json.loads((tmp_path / 'prune_decision.json').read_text())['step'] == 7500
    (tmp_path / 'validation_report.json').write_text('{"step":10000,"score":NaN}')
    with pytest.raises(ValueError, match='Invalid'):
        validation_heartbeat(tmp_path, trial, state, lambda: None)


def test_study_config_rejects_unbounded_or_invalid_search(study_case):
    _, settings, args = study_case
    settings['search']['learning_rates'] = [.0001] * 3
    Path(args.config).write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='distinct'):
        study.load_study(args.config)
    settings['search']['learning_rates'] = [.00005, .0001, .0002]
    settings['search']['num_embeddings'] = [512, 1024]
    Path(args.config).write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match='Unknown'):
        study.load_study(args.config)


@pytest.fixture
def frozen_study(study_case):
    pytest.importorskip('optuna')
    cfg, settings, args = study_case
    study.run_study(cfg, args, executor=fake_executor([]))
    state = json.loads((Path(args.output) / 'study.json').read_text())
    return cfg, args, state


def test_final_reconstruction_uses_selected_vqgan_and_all36_frames(frozen_study, monkeypatch, tmp_path):
    from lutils.dict_wrapper import DictWrapper
    from scientific import runner, study_evaluation
    import model.vqgan.vqvae as vqvae
    _, args, state = frozen_study
    record = next(r for r in state['final_selection']['artifacts'] if r['method'] == 'vqgan')
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, images):
            return DictWrapper(reconstructed_images=images * self.weight, vq_loss=self.weight.square())
    class Metric(torch.nn.Module):
        def forward(self, left, right):
            return (left - right).square().mean((1,2,3), keepdim=True)
    monkeypatch.setattr(vqvae, 'build_vqvae', lambda cfg: Tiny())
    monkeypatch.setattr(runner, 'validate_artifact', lambda path, *a: torch.load(path, weights_only=True))
    output = tmp_path / 'final_reconstruction'
    summary = study_evaluation.evaluate_autoencoder(load_protocol(record['protocol']), args.data,
        record['checkpoint'], 'test_dates', output, torch.device('cpu'), metric=Metric())
    assert summary['lpips'] == 0
    import csv
    with (output / 'metrics.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 36
    metadata = json.loads((output / 'evaluation.json').read_text())
    assert metadata['lpips'] == 'injected_test_metric_not_publication'
    assert list((output / 'media/final/test_dates').glob('*.gif'))


def test_final_forecast_media_reuses_realization_zero_and_archive_hashes(frozen_study, tmp_path):
    from scientific.study_evaluation import _export_forecast_media, _verify_predictions
    from PIL import Image
    _, _, state = frozen_study
    record = next(r for r in state['final_selection']['artifacts'] if r['method'] == 'flow_uniform' and r['seed'] == 42)
    output = tmp_path / 'final_forecast'
    (output / 'predictions').mkdir(parents=True)
    archive = output / 'predictions/test_dates.npz'
    np.savez_compressed(archive, generated=np.zeros((4,15,64,64,3), np.uint8))
    write_json(output / 'evaluation.json', dict(method='flow_uniform',seed=42,split='test_dates',
        prediction_sha256={'test_dates':file_hash(archive)}))
    _verify_predictions(output)
    _export_forecast_media(output, load_protocol(record['protocol']))
    media = output / 'media/final/test_dates'
    panel = json.loads((media / 'manifest.json').read_text())
    assert panel['displayed_realization'] == 0
    assert panel['examples'][0]['context_frames'] == 5
    with Image.open(next(media.glob('*.gif'))) as gif:
        assert gif.n_frames == 15
    archive.write_bytes(b'changed')
    with pytest.raises(ValueError, match='archive changed'):
        _verify_predictions(output)


def test_successful_evaluation_is_adopted_after_supervisor_crash(study_case, tmp_path):
    from scientific.study_evaluation import _evaluate_once
    cfg, _, args = study_case
    protocol = tmp_path / 'protocol.yaml'
    protocol.write_text(yaml.safe_dump(cfg))
    directory = tmp_path / 'evaluation'
    directory.mkdir()
    write_json(directory / 'evaluation.json', dict(status='complete',protocol_sha256=digest(cfg), method='persistence',seed=42,split='test_dates'))
    (directory / 'metrics.csv').write_text('metric\n1\n')
    records = {}
    command = ('evaluate','--method','persistence','--seed','42','--split','test_dates')
    _evaluate_once(command, directory, 'test', str(protocol), records, lambda: None,
        lambda *a: pytest.fail('a successful evaluation was rerun'), tmp_path)
    assert records['test']['outputs']


def test_retention_failure_before_pointer_commit_preserves_full_checkpoint(tmp_path):
    from scientific.study_runtime import compact_closed_trial, verify_result
    protocol = tmp_path / 'protocol.yaml'
    protocol.write_text('frozen')
    checkpoint = tmp_path / 'best.pth'
    torch.save(dict(model={'x':torch.ones(1)},extra={},step=1,optimizer={'large':torch.ones(8)}), checkpoint)
    original = dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        protocol=str(protocol),protocol_sha256=file_hash(protocol))
    pending = deepcopy(original)
    def interrupted():
        raise OSError('disk unavailable')
    with pytest.raises(OSError):
        compact_closed_trial(pending, interrupted)
    verify_result(original)  # Durable state still references its untouched full artifact.
    assert checkpoint.is_file()
    compact_closed_trial(original)
    assert Path(original['checkpoint']).name == 'weights.pth'
    assert not checkpoint.exists()
    verify_result(original)


def test_optuna_enqueued_or_partly_suggested_trial_resumes(study_case):
    optuna = pytest.importorskip('optuna')
    cfg, settings, args = study_case
    root = Path(args.output)
    root.mkdir()
    opt = optuna.create_study(study_name='vqgan_exploration',storage=f'sqlite:///{root / "optuna.sqlite3"}')
    opt.enqueue_trial(dict(learning_rate=.00005,schedule='constant'), user_attrs=dict(candidate='c01'))
    state = dict(status='running',runs={})
    records = study._search(cfg, settings, args, state, lambda: None, 'vqgan', executor=fake_executor([]))
    assert len(records) == 6
    opt = optuna.load_study(study_name='vqgan_exploration',storage=f'sqlite:///{root / "optuna.sqlite3"}')
    assert all(t.state.is_finished() and len(t.params) == 2 for t in opt.trials)


def test_autoencoder_resource_pilot_is_synthetic_and_preserves_rng(study_case, monkeypatch, tmp_path):
    from lutils.dict_wrapper import DictWrapper
    from scientific import study_pilot
    from training.checkpoints import capture_rng_state, restore_rng_state
    cfg, _, _ = study_case
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
        def forward(self, image):
            return DictWrapper(reconstructed_images=image*self.weight, vq_loss=self.weight.square())
    class Metric(torch.nn.Module):
        def forward(self,left,right):
            return (left-right).square().mean((1,2,3),keepdim=True)
    monkeypatch.setattr(study_pilot,'build_vqvae',lambda cfg:Tiny())
    monkeypatch.setattr(study_pilot,'build_discriminator',lambda **kw:torch.nn.Conv2d(3,1,1))
    monkeypatch.setattr(study_pilot,'build_lpips',lambda device:Metric())
    saved = capture_rng_state()
    expected = torch.rand(2)
    restore_rng_state(saved)
    result = study_pilot.profile_autoencoder(cfg,tmp_path/'pilot.json','cpu')
    torch.testing.assert_close(torch.rand(2),expected,rtol=0,atol=0)
    assert result['synthetic'] and result['status']=='resource_probe_not_scientific_result'
    assert result['update_seconds']>0 and result['effective_batch_size']==16


def test_complete_cpu_simulation_including26_final_evaluations(study_case, monkeypatch):
    """Fixture scores verify orchestration/aggregation, not RISE model quality."""
    import csv
    from scientific.protocol import read_jsonl, _source_manifest
    cfg, settings, args = study_case
    pytest.importorskip('optuna')
    settings['final_test'] = True
    Path(args.config).write_text(yaml.safe_dump(settings))
    data = Path(args.data)
    manifest = json.loads((data/'dataset_manifest.json').read_text())
    for split in ('test_dates','test_views'):
        rows = read_jsonl(data/f'{split}.jsonl')
        second = dict(rows[0],window_id=split+'_second',video_id=rows[0]['video_id']+10,
            date='2019-02-02',path=f'clips/{split}_second.npz')
        raw = np.full((36,64,64,3), 215 if split=='test_dates' else 245,np.uint8)
        np.savez_compressed(data/second['path'],raw_frames=raw,frames=raw,alpha=np.ones((36,64,64),np.float32))
        second['sha256']=file_hash(data/second['path'])
        write_jsonl(data/f'{split}.jsonl',[*rows,second])
        manifest['manifests'][split]=file_hash(data/f'{split}.jsonl')
    write_json(data/'dataset_manifest.json',manifest)
    original_load=np.load
    def gated_load(path,*a,**kw):
        if Path(path).name.startswith('test'):
            assert (Path(args.output)/'final_selection.json').is_file()
        return original_load(path,*a,**kw)
    monkeypatch.setattr(np,'load',gated_load)
    calls=[]
    train=fake_executor(calls)
    def execute(command,log,heartbeat):
        name=command[6]
        if name not in ('evaluate','study-evaluate-autoencoder'):
            return train(command,log,heartbeat)
        protocol=load_protocol(command[command.index('--protocol')+1])
        checkpoint=command[command.index('--checkpoint')+1] if '--checkpoint'in command else None
        verify_selection(protocol,checkpoint,args.data)
        method='vqgan' if name=='study-evaluate-autoencoder' else command[command.index('--method')+1]
        seed=42 if method=='vqgan' else int(command[command.index('--seed')+1])
        split=command[command.index('--split')+1]
        campaign=Path(command[command.index('--campaign')+1])
        output=campaign/'evaluation'/(f'vqgan_{split}' if method=='vqgan' else f'{method}_{seed}_{split}')
        output.mkdir(parents=True)
        calls.append((name,dict(method=method,seed=seed,split=split)))
        records=read_jsonl(data/f'{split}.jsonl')
        metadata=dict(status='complete',protocol_sha256=digest(protocol),method=method,seed=seed,split=split,
            checkpoint_sha256=file_hash(checkpoint) if checkpoint else None,synthetic_fixture=True)
        if method=='vqgan':
            (output/'metrics.csv').write_text('lpips\n0.1\n')
            write_json(output/'evaluation.json',metadata)
            return
        samples=4 if method.startswith('flow') else 1
        metric_rows=[]
        runtime=[]
        hashes={}
        for record in records:
            if method.startswith('flow'):
                (output/'predictions').mkdir(exist_ok=True)
                path=output/'predictions'/f'{record["window_id"]}.npz'
                np.savez_compressed(path,generated=np.zeros((4,15,64,64,3),np.uint8))
                hashes[record['window_id']]=file_hash(path)
            for realization in range(samples):
                runtime.append(dict(window_id=record['window_id'],realization=realization,seconds=.01,nfe=960,peak_memory_bytes=0))
                for horizon in range(1,16):
                    score=.05 if method.startswith('flow') else .15
                    metric_rows.append(dict(method=method,seed=seed,split=split,window_id=record['window_id'],
                        date=record['date'],camera_id=0,view_id=record['view_id'],horizon=horizon,realization=realization,
                        lpips=score,mse=.01,psnr=20,ssim=.8,smoke_mse=.01,background_mse=None,
                        smoke_fraction=1,diversity_mse=0))
        for filename,rows in [('metrics.csv',metric_rows),('runtime.csv',runtime)]:
            with(output/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        metadata.update(lpips='official alex v0.1',training_complete=True,
            dataset_sha256=file_hash(data/'dataset_manifest.json'),source_sha256=digest(_source_manifest()),
            metric_weights_sha256='synthetic_fixture_not_publication',expected_window_ids=[r['window_id']for r in records],
            windows_completed=len(records),expected_windows=len(records),prediction_sha256=hashes)
        write_json(output/'evaluation.json',metadata)
    study.run_study(cfg,args,executor=execute)
    state=json.loads((Path(args.output)/'study.json').read_text())
    assert state['status']=='complete' and len(state['evaluations'])==26
    result=json.loads((Path(args.output)/'evaluation_summary/summary.json').read_text())
    assert len(result['comparisons'])==6
    assert all(row['days']==2 for row in result['comparisons'])
    assert list((Path(args.output)/'evaluation').rglob('*.gif'))
    assert len([c for c in calls if c[0]in('evaluate','study-evaluate-autoencoder')])==26


def test_failed_optuna_trial_recovers_classification_from_receipt(study_case):
    """SQLite may commit FAIL before the supervisor persists its classification."""
    import subprocess
    pytest.importorskip('optuna')
    cfg, settings, args = study_case
    Path(args.output).mkdir()
    calls = []
    ordinary = fake_executor(calls)
    def execute(command, log, heartbeat):
        protocol = load_protocol(command[command.index('--protocol')+1])
        if command[6]=='autoencoder' and protocol['study']['candidate']=='c01':
            campaign=Path(command[command.index('--campaign')+1])
            write_json(campaign/'failure.json',dict(category='non_finite',error='synthetic fault'))
            raise subprocess.CalledProcessError(1,command)
        ordinary(command,log,heartbeat)
    state=dict(status='running',runs={})
    records=study._search(cfg,settings,args,state,lambda:None,'vqgan',executor=execute)
    assert records[0]['status']=='failed'
    failed=state['runs']['vqgan_exploration_c01_42']
    failed.pop('failure_category')  # Emulate the uncommitted JSON update.
    calls_before=len(calls)
    records=study._search(cfg,settings,args,state,lambda:None,'vqgan',executor=execute)
    assert len(calls)==calls_before
    assert failed['failure_category']=='non_finite'
    assert records[0]['status']=='failed'
