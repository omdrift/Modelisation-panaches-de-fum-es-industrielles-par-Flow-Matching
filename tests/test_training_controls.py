import pytest
import torch
from torch import nn

from evaluation.evaluator import Evaluator
from lutils.dict_wrapper import DictWrapper
from training.trainer import Trainer


def test_trainer_refuses_dataset_with_no_complete_batch():
    config = DictWrapper({"batching": {"batch_size": 2, "num_workers": 0}})
    with pytest.raises(ValueError, match="No training batches"):
        Trainer(0, 1, config, [torch.zeros(3, 3, 4, 4)], torch.device("cpu"))


def test_validation_uses_configured_ode_steps(monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.steps = None

        def forward(self, observations):
            vectors = torch.zeros_like(observations[:, 0])
            return DictWrapper(observations=observations, reconstructed_vectors=vectors, target_vectors=vectors)

        def generate_frames(self, observations, num_frames, steps, verbose):
            self.steps = steps
            return observations

    class Logger:
        def log(self, *args):
            pass

        def finalize_logs(self, **kwargs):
            pass

    config = DictWrapper({
        "batching": {"batch_size": 1, "num_workers": 0},
        "num_observations": 3, "condition_frames": 1,
        "frames_to_generate": 2, "steps": 7,
    })
    evaluator = Evaluator(0, config, [torch.zeros(5, 3, 4, 4)], torch.device("cpu"))
    monkeypatch.setattr(evaluator, "log_media", lambda *args: None)
    model = Model()
    evaluator.evaluate(model, Logger(), global_step=1, max_num_batches=1)
    assert model.steps == 7
