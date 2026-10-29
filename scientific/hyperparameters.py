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
    if not isinstance(overrides, dict) or overrides.keys() - (AUTOENCODER_DEFAULTS.keys() | {'schedule_min_factor'}):
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
    validate_schedule(result)
    if not any(result[k] for k in ('reconstruction_weight', 'perceptual_weight', 'quantization_weight')):
        raise ValueError('Autoencoder requires a reconstruction, perceptual or quantization objective')
    return result


def validate_schedule(settings: dict) -> None:
    """Validate the two predeclared schedules without changing legacy defaults."""
    if settings.get('schedule', 'constant') not in ('constant', 'cosine'):
        raise ValueError('schedule must be constant or cosine')
    factor = settings.get('schedule_min_factor', 0.1)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('schedule_min_factor must be finite and in (0, 1]')


def schedule_factor(step: int, total: int, settings: dict, *, warmup: int = 0) -> float:
    """Factor at the next update; cosine reaches its declared floor at total."""
    if warmup and step < warmup:
        return min(1.0, (step + 1) / warmup)
    if settings.get('schedule', 'constant') == 'constant':
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    minimum = settings.get('schedule_min_factor', 0.1)
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def training_description(cfg: dict, method: str) -> dict:
    """Describe executed settings without pretending they were calibrated."""
    training = cfg['training']
    common = dict(method=method, max_updates=training['max_updates'],
                  effective_batch_size=training['effective_batch_size'],
                  status='exploratory_initial_settings_not_smoke_calibrated')
    from scientific.early_stopping import settings as early_stopping_settings
    common['early_stopping'] = early_stopping_settings(cfg)
    if method == 'vqgan':
        settings = autoencoder_settings(cfg)
        return common | dict(optimizer='AdamW', settings=settings,
            discriminator_start=training['discriminator_start'],
            scheduler=dict(kind=settings['schedule'], minimum_factor=settings.get('schedule_min_factor', 0.1), warmup_updates=0),
            settings_source='training.autoencoder' if 'autoencoder' in training else 'legacy_frozen_protocol_defaults',
            training_reconstruction_range=[-1, 1], evaluation_mse_range=[0, 1],
            loss='reconstruction_weight*MSE[-1,1] + perceptual_weight*LPIPS + quantization_weight*VQ - adversarial_weight*mean(D(reconstruction))',
            regional_weighting='none; smoke/background weights belong to flow_alpha')
    if method not in ('simvp', 'flow_uniform', 'flow_alpha'):
        raise ValueError('Expected VQGAN or a learned predictor')
    return common | dict(optimizer='AdamW', learning_rate=training['learning_rate'],
        weight_decay=training['weight_decay'], betas=[0.9, 0.999], epsilon=1e-8,
        validation_every=training['validation_every'], gradient_clip_norm=1.0,
        scheduler=dict(kind='linear_warmup_then_' + training.get('schedule', 'constant'), warmup_updates=training['warmup_updates'], minimum_factor=training.get('schedule_min_factor', 0.1)),
        regional_weighting=dict(smoke=training['smoke_weight'], background=training['background_weight']) if method == 'flow_alpha' else 'none')
