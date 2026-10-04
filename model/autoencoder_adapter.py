"""Common sequence-shaped interface for the project's supported autoencoders."""

from __future__ import annotations

import torch


class AutoencoderAdapter:
    """Adapt existing VQGAN and taming autoencoders to [B, T, C, H, W] tensors."""

    def __init__(self, autoencoder, implementation: str):
        self.autoencoder = autoencoder
        self.implementation = implementation

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 4:
            frames = frames.unsqueeze(1)
        if frames.ndim != 5:
            raise ValueError("frames must have shape [batch, time, channels, height, width]")

        batch_size, frame_count = frames.shape[:2]
        if self.implementation == "ours":
            return self.autoencoder(frames).latents

        flat_frames = frames.reshape(batch_size * frame_count, *frames.shape[2:])
        latents = self.autoencoder.encode(flat_frames)
        return latents.reshape(batch_size, frame_count, *latents.shape[1:])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            latents = latents.unsqueeze(1)
        if latents.ndim != 5:
            raise ValueError("latents must have shape [batch, time, channels, height, width]")

        batch_size, frame_count = latents.shape[:2]
        flat_latents = latents.reshape(batch_size * frame_count, *latents.shape[2:])
        if self.implementation == "ours":
            frames = self.autoencoder.backbone.decode_from_latents(flat_latents)
        else:
            frames = self.autoencoder.decode(flat_latents)
        return frames.reshape(batch_size, frame_count, *frames.shape[1:])
