from typing import Any

import torch
import torch.nn as nn
try:
    from torchdiffeq import odeint
except ImportError:
    from model.ode import odeint
from tqdm import tqdm

from lutils.configuration import Configuration
from lutils.dict_wrapper import DictWrapper
from model.autoencoder_adapter import AutoencoderAdapter
from model.conditioning import ConditioningSampler
from model.flow_matching import FlowMatchingObjective
from model.vector_field_regressor import build_vector_field_regressor
from model.vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f8_small_ddconfig, vq_f16_ddconfig, VQModelInterface
from model.vqgan.vqvae import build_vqvae


class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.config = config
        self.sigma = config["sigma"]
        self.conditioning_sampler = ConditioningSampler()
        self.flow_matching_objective = FlowMatchingObjective(
            sigma=self.sigma,
            smoke_threshold=float(config.get("smoke_threshold", 0.1)),
        )

        if config["autoencoder"]["type"] == "ours":
            self.ae = build_vqvae(
                config=config["autoencoder"],
                convert_to_sequence=True)
            self.ae.backbone.load_from_ckpt(config["autoencoder"]["ckpt_path"])
        else:
            if config["autoencoder"]["config"] == "f8":
                ae_config = vq_f8_ddconfig
            elif config["autoencoder"]["config"] == "f8_small":
                ae_config = vq_f8_small_ddconfig
            else:
                ae_config = vq_f16_ddconfig
            self.ae = VQModelInterface(ae_config, config["autoencoder"]["ckpt_path"])
        self.autoencoder = AutoencoderAdapter(self.ae, config["autoencoder"]["type"])

        self.vector_field_regressor = build_vector_field_regressor(
            config=self.config["vector_field_regressor"])

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_ddp = True
                break
        if is_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}

        dmodel = self.module if isinstance(self, torch.nn.parallel.DistributedDataParallel) else self
        dmodel.load_state_dict(state)

    def forward(
            self,
            observations: torch.Tensor,
            observations_are_latents: bool = False) -> DictWrapper[str, Any]:
        """

        :param observations: [b, num_observations, num_channels, height, width]
        """

        batch_size = observations.size(0)
        num_observations = observations.size(1)
        assert num_observations > 2

        sampled_indices = self.conditioning_sampler.sample(
            batch_size=batch_size,
            num_observations=num_observations,
            device=observations.device,
        )
        batch_indices = torch.arange(batch_size, device=observations.device)
        target_frames = observations[batch_indices, sampled_indices.target]
        reference_frames = observations[batch_indices, sampled_indices.reference]
        conditioning_frames = observations[batch_indices, sampled_indices.condition]

        # Encode observations to latent codes (or assume observations already latents)
        if observations_are_latents:
            # observations are expected as [b, n, c, h, w] in latent space
            latents = observations
        else:
            with torch.no_grad():
                self.ae.eval()
                input_frames = torch.stack([target_frames, reference_frames, conditioning_frames], dim=1)
                latents = self.autoencoder.encode(input_frames)
        target_latents = latents[:, 0]
        reference_latents = latents[:, 1]
        conditioning_latents = latents[:, 2]

        flow_sample = self.flow_matching_objective.sample(target_latents)

        # Predict vectors
        reconstructed_vectors = self.vector_field_regressor(
            input_latents=flow_sample.input_latents,
            reference_latents=reference_latents,
            conditioning_latents=conditioning_latents,
            index_distances=sampled_indices.distance,
            timestamps=flow_sample.timestamps.flatten())

        return DictWrapper(
            # Inputs
            observations=observations,

            # Data for loss calculation
            reconstructed_vectors=reconstructed_vectors,
            target_vectors=flow_sample.target_vectors,
            smoke_mask=flow_sample.smoke_mask)

    @torch.no_grad()
    def generate_frames(
            self,
            observations: torch.Tensor,
            num_frames: int = None,
            steps: int = 100,
            warm_start: float = 0.0,
            past_horizon: int = -1,
            verbose: bool = False) -> torch.Tensor:
        """
        Generates num_frames frames conditioned on observations

        :param observations: [b, num_observations, num_channels, height, width]
        :param num_frames: number of frames to generate
        :param warm_start: part of the integration path to jump to
        :param steps: number of steps for sampling
        :param past_horizon: number of frames to condition on
        :param verbose: whether to display loading bar
        """

        # Encode observations to latents (observations may already be latents)
        self.ae.eval()
        if getattr(self, 'config', None) is not None and isinstance(observations, torch.Tensor):
            # Heuristic: if channel count equals autoencoder output channels and spatial size equals state_res,
            # assume observations are latents. Otherwise encode via AE.
            try:
                ae_out_ch = self.config["autoencoder"].get("encoder", {}).get("out_channels", None)
            except Exception:
                ae_out_ch = None

        # Decide whether to treat observations as latents
        treat_as_latents = False
        if observations.ndim == 5:
            c = observations.size(2)
            if ae_out_ch is not None and c == ae_out_ch:
                treat_as_latents = True

        if treat_as_latents:
            latents = observations
        else:
            latents = self.autoencoder.encode(observations)

        b, n, c, h, w = latents.shape
        if n == 1:
            latents = latents[:, [0, 0]]

        # Generate future latents
        gen = tqdm(range(num_frames), desc="Generating frames", disable=not verbose, leave=False)
        for _ in gen:
            def f(t: torch.Tensor, y: torch.Tensor):
                lower_bound = 0 if past_horizon == -1 else min(0, latents.size(1) - past_horizon)
                higher_bound = latents.size(1) - 1

                # Sample conditioning and reference
                conditioning_latents_indices = torch.randint(low=lower_bound, high=higher_bound, size=[b])
                conditioning_latents = latents[torch.arange(b), conditioning_latents_indices]
                reference_latents = latents[:, -1]

                # Calculate index distances
                index_distances = (higher_bound - conditioning_latents_indices).to(y.device)

                # Calculate vectors
                return self.vector_field_regressor(
                    input_latents=y,
                    reference_latents=reference_latents,
                    conditioning_latents=conditioning_latents,
                    index_distances=index_distances,
                    timestamps=t * torch.ones(b).to(latents.device))

            # Initialize with noise
            noise = torch.randn([b, c, h, w]).to(latents.device)
            y0 = (1 - (1 - self.sigma) * warm_start) * noise + warm_start * latents[:, -1]

            # Solve ODE
            next_latents = odeint(
                f,
                y0,
                t=torch.linspace(warm_start, 1, int((1 - warm_start) * steps)).to(y0.device),
                method="rk4"
            )[-1]
            latents = torch.cat([latents, next_latents.unsqueeze(1)], dim=1)

        # Close loading bar
        gen.close()

        if n == 1:
            latents = latents[:, 1:]

        # Decode to image space
        reconstructed_observations = self.autoencoder.decode(latents)

        return reconstructed_observations
