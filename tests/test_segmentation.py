"""CLIPSeg masks are per-frame semantic estimates, not physical opacity."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scientific.protocol import load_protocol


def test_stationary_smoke_is_retained_without_temporal_background(synthetic_segmenter):
    settings = load_protocol()['preprocessing']
    frames = np.full((36, 24, 24, 3), 0.7, dtype=np.float32)
    smoke, alpha = synthetic_segmenter(settings)(frames)
    np.testing.assert_allclose(alpha, 1, atol=1e-6)
    np.testing.assert_allclose(smoke, frames, atol=1e-6)


def test_no_smoke_returns_zero_without_inventing_a_foreground(synthetic_segmenter):
    frames = np.zeros((36, 24, 24, 3), dtype=np.float32)
    smoke, alpha = synthetic_segmenter(load_protocol()['preprocessing'])(frames)
    assert not smoke.any() and not alpha.any()


def test_spatial_refinement_respects_structure_exclusion():
    from scientific.segmentation import refine_mask
    settings = load_protocol()['preprocessing']
    frame = np.full((24, 24, 3), 0.6, dtype=np.float32)
    scores = np.zeros((3, 24, 24), dtype=np.float32)
    scores[0] = 0.5
    scores[1, :, :8] = 0.9
    alpha = refine_mask(frame, scores, settings)
    assert not alpha[:, :8].any()
    assert alpha[:, 16:].min() > 0.99
    assert alpha.dtype == np.float32


def test_guided_filter_preserves_constant_mask_and_image_edge():
    from scientific.segmentation import guided_filter
    guide = np.zeros((32, 32), dtype=np.float32)
    guide[:, 16:] = 1
    constant = np.full_like(guide, 0.3)
    np.testing.assert_allclose(guided_filter(guide, constant, 4, 1e-3), constant, atol=1e-6)
    refined = guided_filter(guide, guide, 4, 1e-3)
    assert refined[:, 15].max() < 0.01
    assert refined[:, 16].min() > 0.99


@pytest.mark.parametrize('bad', [np.nan, np.inf, -0.1, 1.1])
def test_invalid_model_scores_are_rejected(bad):
    from scientific.segmentation import refine_mask
    scores = np.zeros((3, 16, 16), dtype=np.float32)
    scores[0, 0, 0] = bad
    with pytest.raises(ValueError, match='scores'):
        refine_mask(np.zeros((16, 16, 3), dtype=np.float32), scores, load_protocol()['preprocessing'])


def test_missing_clipseg_dependency_is_not_replaced_by_another_mask(monkeypatch):
    from scientific.segmentation import ClipSegExtractor
    monkeypatch.setitem(sys.modules, 'transformers', None)
    with pytest.raises(RuntimeError, match='transformers'):
        ClipSegExtractor(load_protocol()['preprocessing'], device='cpu', local_files_only=True)


def test_prepare_initialization_failure_leaves_no_dataset(tmp_path, monkeypatch):
    from scientific import data
    def fail(*args, **kwargs):
        raise FileNotFoundError('Missing pinned model weights')
    monkeypatch.setattr(data, 'ClipSegExtractor', fail)
    output = tmp_path / 'prepared'
    with pytest.raises(FileNotFoundError, match='weights'):
        data.prepare([], tmp_path, output, load_protocol())
    assert not output.exists()


def test_unrecognized_weights_fail_before_loading_model(tmp_path, monkeypatch):
    from scientific.segmentation import ClipSegExtractor, MODEL_FILES
    for name in MODEL_FILES:
        (tmp_path / name).write_text('unrecognized model content')
    def unexpected_load(*args, **kwargs):
        pytest.fail('Unverified weights must not be loaded')
    factory = SimpleNamespace(from_pretrained=unexpected_load)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(CLIPSegProcessor=factory, CLIPSegForImageSegmentation=factory))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=lambda *a, **kw: tmp_path))
    with pytest.raises(ValueError, match='SHA-256'):
        ClipSegExtractor(load_protocol()['preprocessing'], device='cpu')


@pytest.mark.parametrize('native_shape', [(96, 128), (180, 180)])
def test_prepare_segments_native_frames_before_reducing_to_64(tmp_path, synthetic_segmenter, native_shape, capsys):
    import cv2
    from scientific.data import prepare
    name = '0-0-2018-05-11-0-0-128-96-180-180-0-1000-1175'
    height, width = native_shape
    writer = cv2.VideoWriter(str(tmp_path / f'10_{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, (width, height))
    assert writer.isOpened()
    for _ in range(36):
        writer.write(np.full((height, width, 3), 180, dtype=np.uint8))
    writer.release()
    record = dict(id=10, file_name=name, camera_id=0, view_id=0, start_time=1000, label_state=23, label_state_admin=-1)
    excluded = record | dict(id=11, label_state=0)
    missing = record | dict(id=12)
    result = prepare([record, excluded, missing], tmp_path, tmp_path / 'prepared', load_protocol())
    assert result['statuses'] == {'prepared': 1, 'excluded_label': 1, 'missing_video': 1}
    assert 'source_clips=1' in capsys.readouterr().out
    assert synthetic_segmenter.instances[0].shapes == [(height, width, 3)] * 36
    with np.load(tmp_path / 'prepared/clips/10_0000.npz') as pack:
        assert pack['raw_frames'].shape == pack['frames'].shape == (36, 64, 64, 3)
        assert pack['alpha'].shape == (36, 64, 64)
        assert pack['alpha'].min() >= 0 and pack['alpha'].max() <= 1
        np.testing.assert_allclose(pack['alpha'], 1, atol=1e-6)


@pytest.mark.parametrize('change', [
    {'revision': 'main'}, {'smoke_foreground_threshold': 0.05},
    {'guided_radius': 0}, {'guided_epsilon': float('nan')},
    {'negative_prompts': []}, {'weights_sha256': 'unverified'},
])
def test_invalid_segmentation_settings_fail_before_model_loading(change):
    from scientific.segmentation import validate_settings
    settings = copy.deepcopy(load_protocol()['preprocessing'])
    settings.update(change)
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_shared_vision_batches_preserve_frame_and_prompt_order(monkeypatch):
    from scientific.segmentation import ClipSegExtractor
    from scientific.protocol import load_protocol
    calls = []
    for flag in ('enabled', 'benchmark', 'deterministic', 'allow_tf32'):
        monkeypatch.setattr(torch.backends.cudnn, flag, True)
    class Inputs(dict):
        def to(self, device):
            return self
    def processor(images, **kwargs):
        return Inputs(pixel_values=torch.tensor([np.asarray(im).mean() / 255 for im in images], dtype=torch.float32)[:, None, None])
    def vision(pixel_values, **kwargs):
        assert torch.backends.cudnn.enabled and torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.allow_tf32
        calls.append(len(pixel_values))
        return SimpleNamespace(hidden_states=(None, pixel_values))
    def decoder(activations, conditional_embeddings, **kwargs):
        values = activations[0][:, 0, 0] + conditional_embeddings[:, 0]
        return SimpleNamespace(logits=values[:, None, None].expand(-1, 2, 2))
    extractor = ClipSegExtractor.__new__(ClipSegExtractor)
    extractor.settings = load_protocol()['preprocessing']
    extractor.device = torch.device('cpu')
    extractor.prompts = ['smoke', 'chimney', 'buildings']
    extractor.conditional_embeddings = torch.tensor([[0.0], [0.1], [0.2]])
    extractor.processor = processor
    extractor.model = SimpleNamespace(clip=SimpleNamespace(vision_model=vision), extract_layers=[0], decoder=decoder)
    frames = np.stack([np.full((8, 12, 3), value, dtype=np.float32) for value in [0.0, 0.2, 0.8]])
    batched = extractor.predict_scores_batch(frames)
    assert calls == [3]  # Three unique frames, not nine image/prompt pairs.
    assert batched.shape == (3, 3, 8, 12)
    individual = np.stack([extractor.predict_scores(frame) for frame in frames])
    np.testing.assert_array_equal(batched, individual)
    changed = frames.copy()
    changed[2] = 0.1
    np.testing.assert_array_equal(extractor.predict_scores_batch(changed)[:2], batched[:2])
    assert torch.backends.cudnn.allow_tf32  # Local policy must not leak to callers.
    def fail(**kwargs):
        raise RuntimeError('vision failed')
    extractor.model.clip.vision_model = fail
    with pytest.raises(RuntimeError, match='vision failed'):
        extractor.predict_scores_batch(frames)
    assert all(getattr(torch.backends.cudnn, flag) for flag in ('enabled', 'benchmark', 'deterministic', 'allow_tf32'))


@pytest.mark.parametrize('batch', [0, -1, 1.5, True])
def test_invalid_frame_batch_fails_before_loading_model(batch):
    from scientific.segmentation import ClipSegExtractor
    with pytest.raises(ValueError, match='batch_frames'):
        ClipSegExtractor(load_protocol()['preprocessing'], batch_frames=batch)
