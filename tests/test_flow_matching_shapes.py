import torch
import torch.nn as nn

from lutils.dict_wrapper import DictWrapper
from model.conditioning import ConditioningSampler
from model.flow_matching import FlowMatchingObjective
from model.model import Model
from model.vector_field_regressor import VectorFieldRegressor


def test_conditioning_indices_obey_temporal_order():
    sampled = ConditioningSampler().sample(batch_size=256, num_observations=10, device=torch.device("cpu"))

    assert torch.all(sampled.target >= 2)
    assert torch.all(sampled.target < 10)
    assert torch.equal(sampled.reference, sampled.target - 1)
    assert torch.all(sampled.condition < sampled.target)
    assert torch.all(sampled.distance > 0)


def test_flow_matching_target_has_same_shape_as_predicted_vector():
    target = torch.randn(2, 256, 8, 8)
    noise = torch.randn_like(target)
    timestamps = torch.tensor([0.2, 0.8])
    result = FlowMatchingObjective(sigma=0.001, smoke_threshold=0.1).sample(
        target,
        noise=noise,
        timestamps=timestamps,
    )

    assert result.input_latents.shape == target.shape
    assert result.target_vectors.shape == target.shape
    assert result.smoke_mask.shape == (2, 1, 8, 8)
    assert result.timestamps.shape == (2, 1, 1, 1)


def test_vector_field_output_matches_latent_shape():
    model = VectorFieldRegressor(
        depth=1,
        mid_depth=1,
        state_size=256,
        state_res=(8, 8),
        inner_dim=64,
    ).eval()
    latents = torch.randn(1, 256, 8, 8)

    with torch.inference_mode():
        prediction = model(
            input_latents=latents,
            reference_latents=latents,
            conditioning_latents=latents,
            index_distances=torch.ones(1),
            timestamps=torch.zeros(1),
        )

    assert prediction.shape == latents.shape


def test_model_forward_prediction_matches_flow_target_shape():
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.config = DictWrapper({"autoencoder": {"type": "ours"}, "smoke_threshold": 0.1})
    model.sigma = 0.001
    model.conditioning_sampler = ConditioningSampler()
    model.flow_matching_objective = FlowMatchingObjective(0.001, 0.1)
    model.vector_field_regressor = VectorFieldRegressor(
        depth=1,
        mid_depth=1,
        state_size=4,
        state_res=(2, 2),
        inner_dim=32,
    )
    observations = torch.randn(2, 8, 4, 2, 2)

    output = model(observations, observations_are_latents=True)

    assert output.reconstructed_vectors.shape == output.target_vectors.shape == (2, 4, 2, 2)
