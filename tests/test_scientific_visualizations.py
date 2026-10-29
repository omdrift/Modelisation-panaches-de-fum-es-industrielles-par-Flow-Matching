"""Fixed complete sequences remain readable without a remote dashboard."""
import json

from PIL import Image
import pytest
import torch

from scientific.diagnostics import Snapshot
from scientific.visualizations import export_panel_media


def sequence(window_id, frames=36, context=None):
    target = torch.linspace(-1, 1, frames)[:, None, None, None].expand(frames, 3, 8, 8)
    return dict(window_id=window_id, target=target, prediction=target * .8,
                alpha=torch.ones(frames, 1, 8, 8), context=context)


def test_reconstruction_media_contains_all_frames_and_bounded_latest_best(tmp_path):
    snapshot = Snapshot(sequences=[sequence('a/b'), sequence('other')])
    paths = export_panel_media(tmp_path, 250, 'vqgan', snapshot, 'validation', best=True)
    manifest = json.loads(paths['manifest'].read_text())
    assert manifest['step'] == 250 and manifest['displayed_realization'] == 0
    assert len(manifest['examples']) == 2
    with Image.open(paths['gifs'][0]) as animation:
        assert animation.n_frames == 36
    assert paths['pngs'][0].exists()
    assert (tmp_path / 'media/best/validation/manifest.json').exists()
    updated = export_panel_media(tmp_path, 500, 'vqgan', Snapshot(sequences=[sequence('new')]), 'validation')
    assert len(list(updated['manifest'].parent.glob('*.gif'))) == 1
    assert len(list((tmp_path / 'media/best/validation').glob('*.gif'))) == 2


def test_forecast_animation_keeps_observations_and_all_15_future_images(tmp_path):
    context = torch.ones(5, 3, 8, 8)
    snapshot = Snapshot(sequences=[sequence('forecast', 15, context)])
    paths = export_panel_media(tmp_path, 2500, 'flow_uniform', snapshot, 'train_eval')
    manifest = json.loads(paths['manifest'].read_text())
    assert manifest['examples'][0]['context_frames'] == 5
    assert manifest['examples'][0]['future_frames'] == 15
    with Image.open(paths['gifs'][0]) as animation:
        assert animation.n_frames == 15


def test_test_images_require_frozen_final_selection(tmp_path):
    with pytest.raises(ValueError, match='final'):
        export_panel_media(tmp_path, 1000, 'vqgan', Snapshot(sequences=[sequence('x')]), 'test_dates')
    paths = export_panel_media(tmp_path, 1000, 'vqgan', Snapshot(sequences=[sequence('x')]), 'test_dates', final=True)
    assert 'final' in str(paths['manifest'])


def test_last_validation_improvement_refreshes_best_media(tmp_path):
    snapshot = Snapshot(sequences=[sequence('fixed')])
    export_panel_media(tmp_path, 250, 'vqgan', snapshot, 'validation', best=True)
    export_panel_media(tmp_path, 500, 'vqgan', snapshot, 'validation', best=True, final=True)
    assert json.loads((tmp_path/'media/best/validation/manifest.json').read_text())['step']==500
