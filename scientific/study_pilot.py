"""Small synthetic VQGAN resource probe, never a model-quality experiment."""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from lutils.dict_wrapper import DictWrapper
from model.vqgan.discriminator import build_discriminator
from model.vqgan.vqvae import build_vqvae
from scientific.hyperparameters import autoencoder_settings
from scientific.metrics import build_lpips
from scientific.monitoring import preserve_rng
from scientific.protocol import write_json


@preserve_rng()
def profile_autoencoder(cfg, output, device):
    """Measure three complete batch-16 updates with adversarial losses active."""
    from scientific.runner import provenance, synchronize
    settings = autoencoder_settings(cfg)
    device = torch.device(device)
    torch.manual_seed(42)
    model = build_vqvae(DictWrapper(cfg['model']['autoencoder'])).to(device).train()
    discriminator = build_discriminator(input_nc=3, ndf=64, n_layers=3).to(device).train()
    metric = build_lpips(device)
    options = dict(lr=settings['learning_rate'], weight_decay=settings['weight_decay'],
        betas=tuple(settings['betas']), eps=settings['epsilon'])
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), **options)
    image = torch.rand(1, 3, 64, 64, device=device) * 2 - 1
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for iteration in range(3):
        synchronize(device)
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        pairs = []
        for _ in range(cfg['training']['effective_batch_size']):
            result = model(image)
            reconstruction = settings['reconstruction_weight'] * (result.reconstructed_images - image).square().mean()
            perceptual = settings['perceptual_weight'] * metric(result.reconstructed_images, image).mean()
            loss = reconstruction + perceptual + settings['quantization_weight'] * result.vq_loss.mean()
            loss -= settings['adversarial_weight'] * discriminator(result.reconstructed_images).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite synthetic resource probe')
            (loss / cfg['training']['effective_batch_size']).backward()
            pairs.append(result.reconstructed_images.detach())
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        for generated in pairs:
            loss_d = (F.relu(1 - discriminator(image)).mean() + F.relu(1 + discriminator(generated)).mean()) / 2
            (loss_d / cfg['training']['effective_batch_size']).backward()
        for parameters in (model.parameters(), discriminator.parameters()):
            torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        optimizer_d.step()
        synchronize(device)
        if iteration:
            durations.append(time.monotonic() - start)
    result = dict(status='resource_probe_not_scientific_result', method='vqgan', synthetic=True,
        provenance=provenance(cfg), micro_batch=1, effective_batch_size=16, update_seconds=max(durations),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        note='FP32 adversarial updates; generation/validation/export time is not included in training-time estimates')
    write_json(output, result)
    return result
