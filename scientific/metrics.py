"""Frame metrics, realization diversity and paired day-clustered uncertainty."""
from __future__ import annotations

import itertools

import numpy as np
import torch


def build_lpips(device):
    """Build calibrated official LPIPS with the AlexNet backbone."""
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError('Install environment_scientific.yml: official lpips==0.1.4 is required') from exc
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


def frame_metrics(prediction, target, alpha, perceptual) -> list[dict]:
    from skimage.metrics import structural_similarity

    if prediction.shape != target.shape or alpha.shape != (target.size(0), 1, *target.shape[-2:]):
        raise ValueError('Metrics require aligned [T,3,H,W] images and [T,1,H,W] alpha')
    if not all(torch.isfinite(value).all() for value in (prediction, target, alpha)):
        raise ValueError('Metrics require finite images and alpha masks')
    prediction, target = prediction.clamp(-1, 1), target.clamp(-1, 1)
    squared = ((prediction - target) / 2).square()
    lpips_values = perceptual(prediction, target).reshape(-1).detach().cpu().tolist()
    if len(lpips_values) != len(target) or not np.isfinite(lpips_values).all():
        raise ValueError('LPIPS must return one finite value per future frame')
    rows = []
    for index in range(len(target)):
        error = squared[index]
        mse = float(error.mean())
        mask = (alpha[index] >= 0.1).expand_as(error)
        left, right = [((x[index] + 1) / 2).permute(1, 2, 0).cpu().numpy() for x in (prediction, target)]
        rows.append(dict(horizon=index + 1, lpips=float(lpips_values[index]), mse=mse, psnr=100.0 if mse <= 1e-10 else float(-10 * np.log10(mse)), ssim=float(structural_similarity(left, right, channel_axis=2, data_range=1.0)), smoke_mse=float(error[mask].mean()) if mask.any() else None, background_mse=float(error[~mask].mean()) if (~mask).any() else None, smoke_fraction=float(mask.float().mean())))
    return rows


def diversity(predictions: list[torch.Tensor]) -> list[float]:
    if len(predictions) == 1:
        return [0.0] * predictions[0].size(0)
    pairwise = [((a - b) / 2).square().mean(dim=(1, 2, 3)) for a, b in itertools.combinations(predictions, 2)]
    return torch.stack(pairwise).mean(0).cpu().tolist()


def paired_bootstrap(rows: list[dict], method: str, reference: str, metric='lpips', replicates=2000, seed=42) -> dict:
    """Average realizations, horizons and seeds within window; resample whole days.

    Fail on mismatched coverage, rather than silently comparing easier subsets.
    """
    import collections
    values = {m: collections.defaultdict(list) for m in (method, reference)}
    for row in rows:
        if row['method'] in values and row.get(metric) is not None:
            key = (row['split'], row['date'], row['window_id'], int(row['seed']), int(row['horizon']))
            values[row['method']][key].append(float(row[metric]))
    if not values[method] or set(values[method]) != set(values[reference]):
        raise ValueError('Paired comparison requires identical windows, horizons and seeds')
    days = collections.defaultdict(list)
    for key in sorted(values[method]):
        days[key[1]].append(np.mean(values[method][key]) - np.mean(values[reference][key]))
    if len(days) < 2:
        raise ValueError('At least two distinct days are required for clustered uncertainty')
    day_values = [np.asarray(v) for v in days.values()]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        selected = rng.integers(len(day_values), size=len(day_values))
        estimates.append(float(np.concatenate([day_values[i] for i in selected]).mean()))
    return dict(method=method, reference=reference, metric=metric, difference=float(np.concatenate(day_values).mean()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist(), days=len(days), bootstrap_replicates=replicates, unit='day; realizations and seeds averaged within matched window/horizon')
