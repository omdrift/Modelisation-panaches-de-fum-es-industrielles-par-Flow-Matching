import torch
import torch.nn as nn
import pytest

from lutils.dict_wrapper import DictWrapper
from model.autoencoder_adapter import AutoencoderAdapter
from model.model import Model


class _IdentityBackbone:
    def decode_from_latents(self, latents):
        return latents


class _IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _IdentityBackbone()


class _ZeroVectorField(nn.Module):
    def forward(self, input_latents, **kwargs):
        return torch.zeros_like(input_latents)


@pytest.mark.parametrize("condition_frames", [1, 2])
def test_generate_frames_accepts_one_or_multiple_condition_frames(condition_frames):
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = DictWrapper(
        {
            "autoencoder": {"type": "ours", "encoder": {"out_channels": 2}},
        }
    )
    model.sigma = 0.001
    model.ae = _IdentityAutoencoder()
    model.autoencoder = AutoencoderAdapter(model.ae, "ours")
    model.vector_field_regressor = _ZeroVectorField()
    observations = torch.randn(1, condition_frames, 2, 4, 4)

    generated = model.generate_frames(observations, num_frames=2, steps=2, verbose=False)

    assert generated.shape == (1, condition_frames + 2, 2, 4, 4)
