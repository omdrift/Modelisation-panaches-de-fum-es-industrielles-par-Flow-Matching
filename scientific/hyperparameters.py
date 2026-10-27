"""Resolve stage settings explicitly; inherited values remain exploratory."""
from __future__ import annotations

import copy
import math


AUTOENCODER_DEFAULTS = {
    'learning_rate': 0.0002,
    'discriminator_learning_rate': 0.0002,
    'weight_decay': 1e-6,
    'betas': [0.9, 0.999],
    'epsilon': 1e-8,
    'validation_every': 250,
    'gradient_clip_norm': 1.0,
    'schedule': 'constant',
    'reconstruction_weight': 1.0,
    'perceptual_weight': 1.0,
    'quantization_weight': 1.0,
    'adversarial_weight': 0.1,
}


def autoencoder_settings(cfg: dict) -> dict:
    """Old frozen protocols retain their exact settings; new ones declare them."""
    overrides = cfg['training'].get('autoencoder', {})
    if not isinstance(overrides, dict) or overrides.keys() - AUTOENCODER_DEFAULTS.keys():
        raise ValueError('Unknown or malformed autoencoder training settings')
    result = copy.deepcopy(AUTOENCODER_DEFAULTS) | copy.deepcopy(overrides)
    for key in ('learning_rate', 'discriminator_learning_rate', 'epsilon', 'gradient_clip_norm'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Autoencoder {key} must be finite and positive')
    for key in ('weight_decay', 'reconstruction_weight', 'perceptual_weight', 'quantization_weight', 'adversarial_weight'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Autoencoder {key} must be finite and nonnegative')
    betas = result['betas']
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v < 1 for v in betas)):
        raise ValueError('Autoencoder AdamW requires two betas in [0,1)')
    if type(result['validation_every']) is not int or result['validation_every'] < 1:
        raise ValueError('Autoencoder validation_every must be a positive integer')
    if result['schedule'] != 'constant':
        raise ValueError('Only an explicitly constant autoencoder schedule is implemented')
    if not any(result[k] for k in ('reconstruction_weight', 'perceptual_weight', 'quantization_weight')):
        raise ValueError('Autoencoder requires a reconstruction, perceptual or quantization objective')
    return result


def training_description(cfg: dict, method: str) -> dict:
    """Describe executed settings without pretending they were calibrated."""
    training = cfg['training']
    common = dict(method=method, max_updates=training['max_updates'],
                  effective_batch_size=training['effective_batch_size'],
                  status='exploratory_initial_settings_not_smoke_calibrated')
    if method == 'vqgan':
        settings = autoencoder_settings(cfg)
        return common | dict(optimizer='AdamW', settings=settings,
            discriminator_start=training['discriminator_start'],
            scheduler='constant; predictor warmup is not applied',
            settings_source='training.autoencoder' if 'autoencoder' in training else 'legacy_frozen_protocol_defaults',
            training_reconstruction_range=[-1, 1], evaluation_mse_range=[0, 1],
            loss='reconstruction_weight*MSE[-1,1] + perceptual_weight*LPIPS + quantization_weight*VQ - adversarial_weight*mean(D(reconstruction))',
            regional_weighting='none; smoke/background weights belong to flow_alpha')
    if method not in ('simvp', 'flow_uniform', 'flow_alpha'):
        raise ValueError('Expected VQGAN or a learned predictor')
    return common | dict(optimizer='AdamW', learning_rate=training['learning_rate'],
        weight_decay=training['weight_decay'], betas=[0.9, 0.999], epsilon=1e-8,
        validation_every=training['validation_every'], gradient_clip_norm=1.0,
        scheduler=dict(kind='linear_warmup_then_constant', warmup_updates=training['warmup_updates']),
        regional_weighting=dict(smoke=training['smoke_weight'], background=training['background_weight']) if method == 'flow_alpha' else 'none')
