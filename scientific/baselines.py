"""Deterministic references and a compact, explicitly adapted SimVP model."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def persistence(context: torch.Tensor, horizon: int) -> torch.Tensor:
    return context[:, -1:].expand(-1, horizon, -1, -1, -1).clone()


def optical_flow(context: torch.Tensor, horizon: int) -> torch.Tensor:
    """Constant-velocity Farneback estimate from the last two observations."""
    import cv2
    outputs = []
    for clip in context.detach().cpu():
        images = ((clip[-2:].permute(0, 2, 3, 1) + 1) * 127.5).clamp(0, 255).byte().numpy()
        previous, last = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in images]
        flow = cv2.calcOpticalFlowFarneback(previous, last, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        y, x = np.mgrid[:last.shape[0], :last.shape[1]].astype(np.float32)
        frames = [cv2.remap(images[-1], x - h * flow[..., 0], y - h * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for h in range(1, horizon + 1)]
        outputs.append(torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1)
    return torch.stack(outputs).to(context.device)


class Spatial(nn.Module):
    def __init__(self, incoming, outgoing, stride, transpose=False):
        super().__init__()
        if transpose and stride == 2:
            conv = nn.ConvTranspose2d(incoming, outgoing, 3, 2, 1, output_padding=1)
        else:
            conv = nn.Conv2d(incoming, outgoing, 3, stride, 1)
        self.block = nn.Sequential(conv, nn.GroupNorm(2, outgoing), nn.LeakyReLU(0.2))

    def forward(self, value):
        return self.block(value)


class Inception(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.project = nn.Conv2d(incoming, outgoing // 2, 1)
        self.branches = nn.ModuleList(nn.Sequential(nn.Conv2d(outgoing // 2, outgoing, k, padding=k // 2, groups=8), nn.GroupNorm(8, outgoing), nn.LeakyReLU(0.2)) for k in (3, 5, 7, 11))

    def forward(self, value):
        projected = self.project(value)
        return sum(branch(projected) for branch in self.branches)


class SimVP(nn.Module):
    """Spatial encoder, Inception temporal U-network, spatial decoder.

    Adaptation of Gao et al. (CVPR 2022), not a claim of reproducing their
    published scores: 32 spatial/128 temporal channels, 4+4 depths, and
    autoregressive chunks matching the explicitly selected observation count.
    """
    def __init__(self, spatial_channels=32, temporal_channels=128, spatial_depth=4, temporal_depth=4, context_frames=None):
        super().__init__()
        if spatial_depth != 4 or temporal_depth < 2 or spatial_channels < 4 or spatial_channels % 4 or temporal_channels % 16:
            raise ValueError('Require spatial_depth=4, temporal_depth>=2 and group-compatible channels')
        s, t, d = spatial_channels, temporal_channels, temporal_depth
        if type(context_frames) is not int or context_frames < 2 or context_frames * s % 16:
            raise ValueError('Require context_frames >= 2 and a group-compatible temporal bottleneck')
        self.context_frames = context_frames
        self.encoder = nn.ModuleList([Spatial(3, s, 1), Spatial(s, s, 2), Spatial(s, s, 1), Spatial(s, s, 2)])
        self.temporal_encoder = nn.ModuleList([Inception(context_frames * s if i == 0 else t, t) for i in range(d)])
        self.temporal_decoder = nn.ModuleList([Inception(t if i == 0 else 2 * t, context_frames * s if i == d - 1 else t) for i in range(d)])
        self.decoder = nn.ModuleList([Spatial(s, s, 2, True), Spatial(s, s, 1), Spatial(s, s, 2, True), Spatial(2 * s, s, 1)])
        self.output = nn.Conv2d(s, 3, 1)
        self.last_nfe = 0

    def chunk(self, frames):
        b, _, _, h, w = frames.shape
        value = frames.reshape(b * self.context_frames, 3, h, w)
        spatial_skip = self.encoder[0](value)
        value = spatial_skip
        for layer in self.encoder[1:]:
            value = layer(value)
        shape = value.shape
        value = value.reshape(b, -1, *shape[-2:])
        skips = []
        for layer in self.temporal_encoder:
            value = layer(value)
            skips.append(value)
        for i, layer in enumerate(self.temporal_decoder):
            if i:
                value = torch.cat((value, skips[-1 - i]), dim=1)
            value = layer(value)
        value = value.reshape(shape)
        for layer in self.decoder[:-1]:
            value = layer(value)
        value = self.decoder[-1](torch.cat((value, spatial_skip), dim=1))
        return self.output(value).tanh().reshape(b, self.context_frames, 3, h, w)

    def forward(self, context, horizon):
        if context.shape[1] != self.context_frames or horizon < 1:
            raise ValueError('SimVP requires its explicitly configured observation count and positive horizon')
        chunks = []
        self.last_nfe = 0
        for _ in range((horizon + self.context_frames - 1) // self.context_frames):
            context = self.chunk(context)
            chunks.append(context)
            self.last_nfe += 1
        return torch.cat(chunks, dim=1)[:, :horizon]
