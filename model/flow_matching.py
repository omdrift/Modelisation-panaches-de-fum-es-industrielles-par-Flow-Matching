"""Flow Matching interpolation and target construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowMatchingSample:
    input_latents: torch.Tensor
    timestamps: torch.Tensor
    target_vectors: torch.Tensor
    smoke_mask: torch.Tensor


class FlowMatchingObjective:
    def __init__(self, sigma: float, smoke_threshold: float = 0.1):
        if not 0.0 <= sigma < 1.0:
            raise ValueError("sigma must be in [0, 1)")
        self.sigma = float(sigma)
        self.smoke_threshold = float(smoke_threshold)

    def sample(
        self,
        target_latents: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> FlowMatchingSample:
        if target_latents.ndim != 4:
            raise ValueError("target_latents must have shape [batch, channels, height, width]")
        if noise is None:
            noise = torch.randn_like(target_latents)
        elif noise.shape != target_latents.shape:
            raise ValueError("noise and target_latents must have the same shape")
        if timestamps is None:
            timestamps = torch.rand(
                target_latents.size(0),
                1,
                1,
                1,
                dtype=target_latents.dtype,
                device=target_latents.device,
            )
        else:
            timestamps = timestamps.to(device=target_latents.device, dtype=target_latents.dtype)
            if timestamps.numel() != target_latents.size(0):
                raise ValueError("timestamps must contain one value per batch item")
            timestamps = timestamps.reshape(target_latents.size(0), 1, 1, 1)

        input_latents = (1 - (1 - self.sigma) * timestamps) * noise + timestamps * target_latents
        target_vectors = (target_latents - (1 - self.sigma) * input_latents) / (
            1 - (1 - self.sigma) * timestamps
        )
        smoke_mask = (target_latents.norm(dim=1, keepdim=True) > self.smoke_threshold).to(target_latents.dtype)
        return FlowMatchingSample(
            input_latents=input_latents,
            timestamps=timestamps,
            target_vectors=target_vectors,
            smoke_mask=smoke_mask,
        )
