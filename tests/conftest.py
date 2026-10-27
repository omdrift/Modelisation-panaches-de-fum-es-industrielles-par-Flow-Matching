"""Small deterministic semantic scorer for tests without pretrained weights."""
import numpy as np
import pytest


@pytest.fixture
def synthetic_segmenter(monkeypatch):
    from scientific import data
    from scientific.segmentation import ClipSegExtractor

    class SyntheticSegmenter(ClipSegExtractor):
        instances = []

        def __init__(self, settings, **kwargs):
            self.settings = settings
            import torch
            self.device = torch.device('cpu')
            self.batch_frames = kwargs.get('batch_frames') or 1
            self.provenance = {'backend': 'injected_test_segmenter_not_publication'}
            self.shapes = []
            self.instances.append(self)

        def predict_scores(self, frame):
            self.shapes.append(frame.shape)
            smoke = frame.max(axis=-1)
            return np.stack((smoke, np.zeros_like(smoke), np.zeros_like(smoke)))

        def predict_scores_batch(self, frames):
            return np.stack([self.predict_scores(frame) for frame in frames])

    monkeypatch.setattr(data, 'ClipSegExtractor', SyntheticSegmenter)
    return SyntheticSegmenter
