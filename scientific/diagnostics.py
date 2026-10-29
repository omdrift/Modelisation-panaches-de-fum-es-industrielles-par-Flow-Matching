"""Fixed-panel diagnostics independent of training gradients and test data."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scientific.monitoring import preserve_rng
from scientific.protocol import digest, write_json


@contextmanager
def evaluation_mode(model):
    """Diagnostic forwards must not advance RNG or leave dropout in eval mode."""
    modes = [(module, module.training) for module in model.modules()]
    with preserve_rng():
        model.eval()
        try:
            with torch.inference_mode():
                yield
        finally:
            for module, was_training in modes:
                module.training = was_training


def fixed_indices(dataset, limit=None, seed=42):
    indices = list(range(len(dataset)))
    if limit is not None:
        records = getattr(dataset, 'records', None)
        indices.sort(key=lambda i: digest([seed, records[i]['window_id'] if records else i]))
        indices = indices[:limit]
    if not indices:
        raise ValueError('Diagnostics require a nonempty fixed panel')
    return indices


def image_errors(prediction, target, alpha, lpips_values):
    """MSE on [0,1] RGB; estimated smoke regions use alpha >= 0.1."""
    from skimage.metrics import structural_similarity

    if prediction.shape != target.shape or alpha.shape != (len(target), 1, *target.shape[-2:]):
        raise ValueError('Diagnostic images and masks must be aligned')
    if not all(torch.isfinite(x).all() for x in (prediction, target, alpha, lpips_values)):
        raise FloatingPointError('Non-finite diagnostic image, mask or LPIPS')
    errors = ((prediction.clamp(-1, 1) - target) / 2).square()
    values = []
    if lpips_values.numel() != len(target):
        raise ValueError('Diagnostics require one LPIPS score per image')
    for index, error in enumerate(errors):
        smoke = (alpha[index] >= 0.1).expand_as(error)
        mse = float(error.mean())
        left, right = [((image[index].detach().float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
                       for image in (prediction, target)]
        window = min(7, left.shape[0], left.shape[1])
        window -= 1 - window % 2
        if window < 3:
            raise ValueError('SSIM diagnostics require spatial dimensions of at least three pixels')
        values.append(dict(lpips=float(lpips_values.reshape(-1)[index]), mse=mse,
            psnr=100.0 if mse <= 1e-10 else -10 * math.log10(mse),
            ssim=float(structural_similarity(left, right, channel_axis=2, data_range=1.0, win_size=window)),
            smoke_mse=float(error[smoke].mean()) if smoke.any() else None,
            background_mse=float(error[~smoke].mean()) if (~smoke).any() else None,
            smoke_fraction=float(smoke.float().mean())))
    return values


def codebook_statistics(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('Codebook counts must be finite, nonnegative and one-dimensional')
    total = float(counts.sum())
    if not total:
        return dict(perplexity=None, active_codes=0, used_fraction=None, assignments=0)
    probabilities = counts[counts > 0] / total
    return dict(perplexity=float(torch.exp(-(probabilities * probabilities.log()).sum())),
                active_codes=int((counts > 0).sum()), used_fraction=float((counts > 0).double().mean()), assignments=int(total))


@dataclass
class Snapshot:
    rows: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    real: list = field(default_factory=list)
    reconstructed: list = field(default_factory=list)
    alpha: list = field(default_factory=list)
    code_counts: torch.Tensor | None = None
    smoke_code_counts: torch.Tensor | None = None
    sequences: list[dict] = field(default_factory=list)
    visual_limit: int = 4
    realizations: int = 1
    diversity_values: list[float] = field(default_factory=list)

    @property
    def score(self):
        return float(np.mean(self.scores))

    def add(self, dataset, index, rank, prediction, target, alpha, perceptual, baseline, frame_indices):
        self._add_realization(dataset, index, rank, prediction, target, alpha, perceptual, baseline, frame_indices)

    def _add_realization(self, dataset, index, rank, prediction, target, alpha, perceptual, baseline, frame_indices, *, realization=0):
        prediction = prediction.clamp(-1, 1)
        lpips_values = perceptual(prediction, target)
        # Same float32 per-clip mean as checkpoint selection before diagnostics.
        self.scores.append(float(lpips_values.mean()))
        baseline_values = perceptual(baseline, target).reshape(-1)
        rows = image_errors(prediction, target, alpha, lpips_values)
        if not torch.isfinite(baseline_values).all() or baseline_values.numel() != len(rows):
            raise FloatingPointError('Invalid baseline LPIPS')
        records = getattr(dataset, 'records', None)
        record = records[index] if records else {}
        for offset, values in enumerate(rows):
            self.rows.append(dict(window_id=record.get('window_id', str(index)), date=record.get('date'),
                camera_id=record.get('camera_id'), view_id=record.get('view_id'), frame_index=frame_indices[offset],
                realization=realization, baseline_lpips=float(baseline_values[offset]), **values))
        if rank < self.visual_limit and realization == 0:
            self.real.append(target[0].detach().cpu())
            self.reconstructed.append(prediction[0].detach().cpu())
            self.alpha.append(alpha[0].detach().cpu())

    def add_codes(self, codes, alpha):
        if codes is None:
            return  # Non-quantized test doubles and non-VQ predictors.
        if codes.ndim != 4 or not torch.isfinite(codes).all():
            raise ValueError('Expected finite [B,H,W,K] VQ assignment codes')
        counts = codes.sum((0, 1, 2)).double().cpu()
        smoke = F.interpolate(alpha, size=codes.shape[1:3], mode='area')[:, 0] >= 0.1
        smoke_counts = codes[smoke].sum(0).double().cpu()
        self.code_counts = counts if self.code_counts is None else self.code_counts + counts
        self.smoke_code_counts = smoke_counts if self.smoke_code_counts is None else self.smoke_code_counts + smoke_counts

    def summary(self):
        result = {}
        for metric in ('lpips', 'mse', 'psnr', 'ssim', 'smoke_mse', 'background_mse', 'smoke_fraction', 'baseline_lpips'):
            values = [row[metric] for row in self.rows if row[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
            result[metric + '_valid_frames'] = len(values)
        result['lpips'] = self.score
        result['windows'] = len({row['window_id'] for row in self.rows})
        result['realizations'] = self.realizations
        if self.diversity_values:
            result['diversity_mse'] = float(np.mean(self.diversity_values))
        result['lpips_p90'] = float(np.quantile([row['lpips'] for row in self.rows], 0.9))
        for key in ('date', 'view_id'):
            groups = defaultdict(list)
            for row in self.rows:
                if row[key] is not None:
                    name = str(row[key]) if key == 'date' else f"{row['camera_id']}_{row['view_id']}"
                    groups[name].append(row['lpips'])
            result.update({f'by_{key}/{name}/lpips': float(np.mean(values)) for name, values in groups.items()})
        if self.code_counts is not None:
            for region, counts in (('codebook', self.code_counts), ('smoke_codebook', self.smoke_code_counts)):
                result.update({f'{region}/{key}': value for key, value in codebook_statistics(counts).items()})
        return result


def reconstruction_snapshot(model, dataset, metric, cfg, device, *, train_panel=False):
    settings = cfg.get('diagnostics', {})
    snapshot = Snapshot(visual_limit=settings.get('visual_windows', 4))
    indices = fixed_indices(dataset, limit=settings.get('train_windows', 60) if train_panel else None, seed=cfg['selection_seed'])
    with evaluation_mode(model):
        for rank, index in enumerate(indices):
            item = dataset[index]
            frame_index = rank % cfg['task']['clip_frames']
            all_frames = settings.get('reconstruction_all_frames', False)
            selected = list(range(len(item['frames']))) if all_frames else [frame_index]
            predictions = []
            # Keep the old single-image validation score unless the study explicitly opts in.
            for offset in selected:
                target = item['frames'][offset:offset + 1].to(device)
                alpha = item['alpha'][offset:offset + 1].to(device)
                output = model(target)
                if all_frames and rank < snapshot.visual_limit:
                    predictions.append(output.reconstructed_images.detach().cpu())
                snapshot.add(dataset, index, rank if offset == selected[0] else snapshot.visual_limit,
                             output.reconstructed_images, target, alpha, metric,
                             torch.full_like(target, -1), [offset])
                snapshot.add_codes(output.get('quantized_latents_ids'), alpha)
            if settings.get('export_sequences', False) and rank < snapshot.visual_limit:
                frames = item['frames'].to(device)
                reconstructed = torch.cat(predictions) if all_frames else torch.cat([model(chunk).reconstructed_images for chunk in frames.split(4)])
                snapshot.sequences.append(_sequence(dataset, index, frames, reconstructed, item['alpha']))
    return snapshot


def forecast_snapshot(model, dataset, metric, cfg, device, predict, *, train_panel=False):
    from scientific.metrics import diversity

    settings = cfg.get('diagnostics', {})
    realizations = settings.get('forecast_realizations', 1)
    if type(realizations) is not int or realizations < 1:
        raise ValueError('forecast_realizations must be a positive integer')
    snapshot = Snapshot(visual_limit=settings.get('visual_windows', 4), realizations=realizations)
    context, horizon = cfg['evaluation']['context_frames'], cfg['evaluation']['horizon']
    indices = fixed_indices(dataset, limit=settings.get('train_windows', 8) if train_panel else None, seed=cfg['selection_seed'])
    with evaluation_mode(model):
        for rank, index in enumerate(indices):
            item = dataset[index]
            frames, alpha = item['frames'].to(device), item['alpha'].to(device)
            target = frames[context:context + horizon]
            predictions = []
            for realization in range(realizations):
                generator = torch.Generator(device=device).manual_seed(42000 + rank + realization * len(indices))
                prediction = predict(frames[None, :context], generator)[0]
                if prediction.shape != target.shape:
                    raise ValueError('Forecast diagnostics require exactly the declared future horizon')
                predictions.append(prediction.clamp(-1, 1))
                snapshot._add_realization(dataset, index, rank, prediction, target, alpha[context:context + horizon], metric,
                             frames[context - 1:context].expand_as(target), list(range(context, context + horizon)),
                             realization=realization)
            snapshot.diversity_values.extend(diversity(predictions))
            if settings.get('export_sequences', False) and rank < snapshot.visual_limit:
                snapshot.sequences.append(_sequence(dataset, index, target, predictions[0], alpha[context:context + horizon],
                                                    context=frames[:context]))
    return snapshot


def _sequence(dataset, index, target, prediction, alpha, *, context=None):
    records = getattr(dataset, 'records', None)
    record = records[index] if records else {}
    return dict(window_id=record.get('window_id', str(index)), target=target.detach().cpu(),
                prediction=prediction.detach().clamp(-1, 1).cpu(), alpha=alpha.detach().cpu(),
                context=None if context is None else context.detach().cpu())


def sequence_snapshot(model, dataset, cfg, device, predict=None, *, split='validation', final_selection_frozen=False):
    """Collect only the fixed visual examples, without scoring a whole test split.

    A predictor receives the observed prefix only. This helper is intended for
    final test exports after selection, or a lightweight local inspection.
    """
    if split.startswith('test') and not final_selection_frozen:
        raise ValueError('Test sequence collection requires frozen final selection')
    settings = cfg.get('diagnostics', {})
    panel_limit = settings.get('train_windows', 60) if split in ('train', 'train_eval') else None
    indices = fixed_indices(dataset, limit=panel_limit, seed=cfg['selection_seed'])[:settings.get('visual_windows', 8)]
    snapshot = Snapshot()
    context, horizon = cfg['evaluation']['context_frames'], cfg['evaluation']['horizon']
    with evaluation_mode(model):
        for rank, index in enumerate(indices):
            item = dataset[index]
            frames, alpha = item['frames'].to(device), item['alpha'].to(device)
            if predict is None:
                prediction = torch.cat([model(chunk).reconstructed_images for chunk in frames.split(4)])
                snapshot.sequences.append(_sequence(dataset, index, frames, prediction, alpha))
            else:
                generator = torch.Generator(device=device).manual_seed(42000 + rank)
                prediction = predict(frames[None, :context], generator)[0]
                if prediction.shape != frames[context:context + horizon].shape:
                    raise ValueError('Visual forecast differs from the declared future horizon')
                snapshot.sequences.append(_sequence(dataset, index, frames[context:context + horizon], prediction,
                                                    alpha[context:context + horizon], context=frames[:context]))
    return snapshot


def export_snapshot(run, step, method, training, validation, best, best_step, origin=None):
    """Write lightweight diagnostics; never store additional model weights."""
    train_metrics, val_metrics = training.summary(), validation.summary()
    values = {**{f'train_eval/{key}': value for key, value in train_metrics.items()},
              **{f'validation/{key}': value for key, value in val_metrics.items()}}
    values.update({'generalization/lpips_gap': validation.score - training.score,
                   'generalization/train_lpips': training.score, 'generalization/validation_lpips': validation.score,
                   'selection/best_validation_lpips': best, 'selection/best_step': best_step,
                   'selection/updates_since_best': step - best_step,
                   'selection/relative_lpips_regret': (validation.score - best) / max(abs(best), 1e-8)})
    values = {key: value for key, value in values.items() if value is not None}
    values.update(trend_diagnostics(run, step, training.score, validation.score, best))
    write_json(Path(run) / 'diagnostics' / f'step_{step}.json', dict(step=step, method=method,
        train_panel=training.rows, validation_panel=validation.rows, metrics=values,
        codebook_counts=validation.code_counts.tolist() if validation.code_counts is not None else None,
        smoke_codebook_counts=validation.smoke_code_counts.tolist() if validation.smoke_code_counts is not None else None,
        baseline='black image' if method == 'vqgan' else 'persistence',
        provenance=origin,
        interpretation='Descriptive train/validation gap; not proof of overfitting. Test data is excluded.'))
    write_json(Path(run) / 'diagnostics' / 'metric_definitions.json', METRIC_DEFINITIONS)
    return values


METRIC_DEFINITIONS = {
    'lpips': dict(definition='AlexNet LPIPS image distance; average per frame, realization and window',
                  unit='LPIPS distance', direction='lower', warning='No best-of-realizations selection'),
    'mse': dict(definition='Mean squared RGB error on [0,1]', unit='squared intensity', direction='lower'),
    'psnr': dict(definition='-10 log10(MSE), capped at 100 for near-exact reconstruction', unit='dB', direction='higher'),
    'ssim': dict(definition='RGB structural similarity, data_range=1, local window up to 7 pixels',
                 unit='SSIM', direction='higher'),
    'smoke_mse': dict(definition='RGB MSE where estimated alpha >= 0.1; missing if no smoke pixels',
                      unit='squared intensity', direction='lower'),
    'background_mse': dict(definition='RGB MSE where estimated alpha < 0.1; missing if no background pixels',
                           unit='squared intensity', direction='lower'),
    'diversity_mse': dict(definition='Mean pairwise [0,1] RGB MSE between generated realizations',
                         unit='squared intensity', direction='descriptive', warning='Diversity alone does not show accuracy'),
    'codebook/perplexity': dict(definition='exp(entropy) of observed quantizer assignment counts',
                                unit='effective codes', direction='descriptive'),
    'generalization/lpips_gap': dict(definition='Validation LPIPS minus fixed-train-panel LPIPS',
                                    unit='LPIPS distance', direction='descriptive', warning='Not proof of overfitting'),
    'frequency': 'Every configured validation event on the same fixed panel IDs',
}


def trend_diagnostics(run, step, train_lpips, validation_lpips, best):
    """Advisory trend over five validations; never changes model selection."""
    import json
    paths = sorted((Path(run) / 'diagnostics').glob('step_*.json'), key=lambda p: int(p.stem.split('_')[1]))
    paths = [p for p in paths if int(p.stem.split('_')[1]) < step][-4:]
    previous = [json.loads(path.read_text()) for path in paths]
    points = [(entry['step'], entry['metrics']['train_eval/lpips'], entry['metrics']['validation/lpips']) for entry in previous]
    points.append((step, train_lpips, validation_lpips))
    values = {'generalization/trend_points': len(points), 'generalization/trend_available': len(points) == 5}
    if len(points) == 5:
        x, train, val = np.asarray(points, dtype=float).T
        train_slope = float(np.polyfit(x / 1000, train, 1)[0])
        val_slope = float(np.polyfit(x / 1000, val, 1)[0])
        values.update({'generalization/train_lpips_slope_per_1000': train_slope,
                       'generalization/validation_lpips_slope_per_1000': val_slope,
                       'generalization/possible_overfit': train_slope < 0 and val_slope > 0 and validation_lpips > best * 1.1})
    return values


def selection_details(run, best):
    path = Path(run) / 'best.pth'
    if best is None or not path.is_file():
        return 0, None
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    return int(state['step']), state['extra'].get('best_diagnostics')


@preserve_rng()
def plot_history(run):
    """Compact local counterpart to the WandB scalar charts."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(run) / 'diagnostics'
    paths = sorted(directory.glob('step_*.json'), key=lambda p: int(p.stem.split('_')[1]))
    if not paths:
        return
    snapshots = [json.loads(path.read_text()) for path in paths]
    steps = [entry['step'] for entry in snapshots]
    with preserve_rng():
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        charts = [('LPIPS on fixed panels', ('train_eval/lpips', 'validation/lpips')),
                  ('Smoke region MSE', ('train_eval/smoke_mse', 'validation/smoke_mse')),
                  ('Validation and trivial baseline', ('validation/lpips', 'validation/baseline_lpips')),
                  ('RGB MSE on [0,1] (lower)', ('train_eval/mse', 'validation/mse')),
                  ('PSNR in dB (higher)', ('train_eval/psnr', 'validation/psnr')),
                  ('Structural similarity (higher)', ('train_eval/ssim', 'validation/ssim')),
                  ('Validation codebook usage', ('validation/codebook/used_fraction', 'validation/smoke_codebook/used_fraction')),
                  ('Realization diversity (descriptive)', ('train_eval/diversity_mse', 'validation/diversity_mse')),
                  ('Validation minus train LPIPS (descriptive)', ('generalization/lpips_gap',))]
        for axis, (title, keys) in zip(axes.flat, charts):
            for key in keys:
                values = [entry['metrics'].get(key, np.nan) for entry in snapshots]
                if any(math.isfinite(value) for value in values):
                    axis.plot(steps, values, marker='.', label=key)
            axis.set(title=title, xlabel='Optimizer update')
            if axis.lines:
                axis.legend(fontsize=7)
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(directory / 'summary.png', dpi=140)
        plt.close(fig)
        history = Path(run) / 'metrics.jsonl'
        if history.is_file():
            records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            fig, axes = plt.subplots(2, 3, figsize=(15, 7))
            controls = [('Flow objective: fixed independent panels (lower)', ('train_eval/objective', 'validation/objective')),
                        ('Learning rate per update', ('train/lr', 'train/lr_generator', 'train/lr_discriminator')),
                        ('Gradient norm before clipping', ('train/gradient_norm_before_clip', 'train/generator_gradient_norm_before_clip')),
                        ('Plateau patience: completed validation checks', ('early_stopping/bad_checks', 'early_stopping/patience')),
                        ('Selected checkpoint update', ('selection/best_step',)),
                        ('Measured optimizer update duration (seconds)', ('performance/update_seconds',))]
            for axis, (title, keys) in zip(axes.flat, controls):
                for key in keys:
                    points = [(row['global_step'], row[key]) for row in records
                        if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
                    if points:
                        axis.plot(*zip(*points), marker='.', label=key)
                axis.set(title=title, xlabel='Optimizer update')
                if axis.lines:
                    axis.legend(fontsize=7)
                axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(directory / 'optimization.png', dpi=140)
            plt.close(fig)
        last = snapshots[-1]
        rows = last.get('validation_panel', [])
        horizons = sorted({row['frame_index'] for row in rows})
        if len(horizons) > 1 and last.get('method') != 'vqgan':
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            for axis, key in zip(axes, ('lpips', 'mse', 'ssim')):
                values = [np.mean([row[key] for row in rows if row['frame_index'] == index]) for index in horizons]
                axis.plot(range(1, len(horizons) + 1), values, marker='.')
                axis.set(title=f'Validation {key}; all realizations averaged', xlabel='Future horizon', ylabel=key)
                axis.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(directory / 'horizons.png', dpi=140)
            plt.close(fig)
