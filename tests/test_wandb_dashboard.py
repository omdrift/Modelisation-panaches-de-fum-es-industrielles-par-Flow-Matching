"""Workspace cleanup must change presentation without dropping recorded metrics."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools.configure_wandb import curate_spec, section_plan


def test_main_workspace_is_small_and_detailed_groups_are_collapsed():
    plan = section_plan('vqgan')
    assert 'Lire les courbes' in plan[0]['markdown'][0]
    assert 'quatre fois' in plan[0]['markdown'][0]
    assert len(plan[1]['charts']) == 7
    assert any(keys == ['early_stopping/bad_checks', 'early_stopping/patience'] for _, keys in plan[1]['charts'])
    assert 'patience' in plan[0]['markdown'][0]
    assert all(section['open'] for section in plan[:3])
    assert all(not section['open'] for section in plan[3:])
    slices = plan[3]
    assert len(slices['regex']) == 2
    assert all('_valid_frames' not in key for section in plan for _, keys in section.get('charts', []) for key in keys)
    assert all('codebook' not in key for section in section_plan('flow_uniform') for _, keys in section.get('charts', []) for key in keys)


@pytest.mark.parametrize('method', ['vqgan', 'simvp', 'flow_uniform', 'flow_alpha'])
def test_generalization_line_charts_use_numeric_trends_instead_of_boolean_flags(method):
    plan = section_plan(method)
    charts = [chart for section in plan for chart in section.get('charts', [])]
    assert all('generalization/possible_overfit' not in keys for _, keys in charts)
    slopes = ['generalization/train_lpips_slope_per_1000', 'generalization/validation_lpips_slope_per_1000']
    title, = [title for title, keys in charts if keys == slopes]
    assert '1 000' in title
    guide = plan[0]['markdown'][0]
    assert 'variation de LPIPS par 1 000 mises à jour' in guide
    assert 'négative' in guide and 'positive' in guide


def test_layout_preserves_run_selection_filters_and_original_spec():
    original = dict(section=dict(runSets=[dict(filters={'method': 'vqgan'}, selections=['active-run'])],
        customRunColors={'active-run': 'green'}, workspaceSettings={'linePlot': {'smoothing': 0}},
        panelBankConfig=dict(sections=[dict(name='validation')], panelConfigOverrides={'old': 'panel'})))
    preserved = copy.deepcopy(original)
    sections = [dict(name='Qualité', panels=[])]
    result = curate_spec(original, sections)
    assert original == preserved
    assert result['section']['runSets'] == original['section']['runSets']
    assert result['section']['customRunColors'] == original['section']['customRunColors']
    assert not result['section']['workspaceSettings']['shouldAutoGeneratePanels']
    assert not result['section']['settings']['shouldAutoGeneratePanels']
    assert result['section']['panelBankConfig']['sections'] == sections
    assert result['section']['workspaceSettings']['linePlot'] == {'smoothing': 0}


@pytest.mark.parametrize('apply', [False, True])
def test_remote_change_is_explicit_and_backup_precedes_save(tmp_path, monkeypatch, apply):
    from tools import configure_wandb as dashboard
    original = dict(section=dict(runSets=[{'filters': 'retained'}], panelBankConfig={'sections': []}))
    view = dict(id='owned-view', name='nw-monitor-v', type='project-view', user={'username': 'owner'},
                displayName='Suivi scientifique - VQGAN', updatedAt='2026-10-27T12:00:00Z', spec=json.dumps(original))
    current_view = [view]
    mutations = []

    def execute(api, query, variables=None):
        if 'viewer{' in query:
            return {'viewer': {'username': 'owner'}}
        if query == dashboard.VIEWS_QUERY:
            assert variables == dict(entity='team', project='smoke', username='owner')
            return {'project': {'allViews': {'edges': [{'node': current_view[0]}]}}}
        assert query == dashboard.SAVE_MUTATION
        payload = variables['input']
        backups = list(tmp_path.glob('workspace_before_*.json'))
        assert len(backups) == 1
        assert json.loads(backups[0].read_text())['view'] == view
        assert payload['id'] == view['id'] and payload['name'] == view['name']
        assert payload['lastUpdatedAt'] == view['updatedAt']
        assert json.loads(payload['spec'])['section']['runSets'] == original['section']['runSets']
        mutations.append(payload)
        current_view[0] = view | {'spec': payload['spec']}
        return {'upsertView': {'view': {'id': payload['id'], 'spec': payload['spec']}}}

    monkeypatch.setattr(dashboard, 'load_sdk', lambda: (object(), None, None, execute))
    monkeypatch.setattr(dashboard, 'build_sections', lambda plan, *a: [
        {'name': entry['name'], 'isOpen': entry['open'],
         'panels': [{} for kind in ('charts', 'regex', 'media', 'markdown') for _ in entry.get(kind, [])]} for entry in plan])
    result = dashboard.configure(dict(entity='team', project='smoke', url=None), 'vqgan', tmp_path, apply=apply)
    assert result['applied'] == apply and len(mutations) == int(apply)
    assert (tmp_path / 'workspace_vqgan.json').exists() == apply


def test_remote_change_rejects_a_different_owner(tmp_path, monkeypatch):
    from tools import configure_wandb as dashboard

    def execute(api, query, variables=None):
        if 'viewer{' in query:
            return {'viewer': {'username': 'owner'}}
        assert query == dashboard.VIEWS_QUERY
        return {'project': {'allViews': {'edges': [{'node': {
            'name': 'nw-nwuserowner-w', 'user': {'username': 'someone-else'}}}]}}}

    monkeypatch.setattr(dashboard, 'load_sdk', lambda: (object(), None, None, execute))
    with pytest.raises(ValueError, match='Expected only workspaces owned'):
        dashboard.configure(dict(entity='team', project='smoke'), 'vqgan', tmp_path, apply=True)
    assert not list(tmp_path.iterdir())


def test_new_named_view_does_not_replace_the_automatic_personal_view(tmp_path, monkeypatch):
    from tools import configure_wandb as dashboard
    personal = dict(id='personal-id', displayName='Personal workspace', user={'username': 'owner'})
    views = [personal]

    def sections(plan, *args, **kwargs):
        return [dict(name=e['name'], isOpen=e['open'],
                     panels=[{} for kind in ('charts', 'regex', 'media', 'markdown') for _ in e.get(kind, [])]) for e in plan]

    class Workspace:
        _internal_id = 'named-id'
        _internal_name = 'nw-named-v'
        url = 'https://example.invalid/team/smoke?nw=named'

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def save(self):
            assert self.kwargs['auto_generate_panels'] is False
            assert self.kwargs['runset_settings']['filters'] == "Config('method') == 'vqgan'"
            backup = next(tmp_path.glob('workspace_before_*.json'))
            assert json.loads(backup.read_text())['view'] is None
            spec = dict(section=dict(settings={'shouldAutoGeneratePanels': False},
                panelBankConfig={'sections': self.kwargs['sections']}))
            views.append(dict(id=self._internal_id, displayName=self.kwargs['name'],
                              spec=json.dumps(spec), user={'username': 'owner'}))

    def execute(api, query, variables=None):
        if 'viewer{' in query:
            return {'viewer': {'username': 'owner'}}
        assert query == dashboard.VIEWS_QUERY  # No mutation to the existing personal view.
        return {'project': {'allViews': {'edges': [{'node': v} for v in views]}}}

    sdk = SimpleNamespace(Workspace=Workspace, RunsetSettings=lambda **kwargs: kwargs)
    monkeypatch.setattr(dashboard, 'load_sdk', lambda: (object(), sdk, None, execute))
    monkeypatch.setattr(dashboard, 'build_sections', sections)
    result = dashboard.configure(dict(entity='team', project='smoke'), 'vqgan', tmp_path, apply=True)
    assert result['view_id'] == 'named-id' and result['url'].endswith('?nw=named')
    assert views[0] is personal and len(views) == 2
