"""Training and measurement; all publication metrics use held-out windows."""
from __future__ import annotations

import copy
import csv
from importlib import metadata as package_metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lutils.dict_wrapper import DictWrapper
from model.model import Model
from model.vqgan.vqvae import build_vqvae
from scientific.baselines import SimVP, persistence, optical_flow
from scientific.data import WindowDataset
from scientific.metrics import build_lpips, frame_metrics, diversity
from scientific.monitoring import gpu_metrics, image_panel, preserve_rng, record_metrics
from scientific.protocol import digest, file_hash, write_json, LEARNED, forecast_task
from training.checkpoints import capture_rng_state, restore_rng_state, save_checkpoint, load_checkpoint


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device) -> None:
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def provenance(cfg, root=None) -> dict:
    def git(*args):
        result = subprocess.run(['git', *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    sources = {str(p): file_hash(p) for folder in ('scientific', 'model', 'training', 'lutils') for p in sorted(Path(folder).rglob('*.py'))}
    versions = {}
    for name in ('numpy', 'scipy', 'torchvision', 'transformers', 'huggingface-hub', 'safetensors', 'lpips', 'scikit-image', 'torchdiffeq', 'PyYAML'):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return dict(protocol=cfg, protocol_sha256=digest(cfg), commit=git('rev-parse', 'HEAD'), tracked_diff_sha256=digest(git('diff', 'HEAD')), source_sha256=digest(sources), command=sys.argv, python=platform.python_version(), package_versions=versions, torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, dataset_sha256=file_hash(Path(root) / 'dataset_manifest.json') if root else None, deterministic_algorithms='warn_only; unsupported CUDA kernels may remain nondeterministic')


def check_resume_provenance(previous, current):
    """Resume only with the same sources, data, configuration and runtime."""
    previous = {k: v for k, v in previous.items() if k != 'command'}
    current = {k: v for k, v in current.items() if k != 'command'}
    if previous != current:
        raise ValueError('Resume requires identical training source, dataset, protocol, seed and artifacts')


def model_for(cfg, method, autoencoder=None, pilot=False):
    if method == 'simvp':
        return SimVP(**cfg['simvp'], context_frames=forecast_task(cfg)[0])
    if method not in ('flow_uniform', 'flow_alpha'):
        raise ValueError(f'Unknown learned method: {method}')
    model_cfg = copy.deepcopy(cfg['model'])
    model_cfg['autoencoder']['ckpt_path'] = str(autoencoder)
    model = Model(DictWrapper(model_cfg), load_autoencoder=not pilot)
    for parameter in model.ae.parameters():
        parameter.requires_grad_(False)
    return model


def finalize_selection(run, status, step, planned_updates):
    path = Path(run) / 'best.pth'
    if not path.exists():
        return
    state = torch.load(path, map_location='cpu', weights_only=True)
    state['extra']['completion'] = dict(status=status, completed_updates=step, planned_updates=planned_updates)
    state['extra']['selected_artifact'] = True
    state['extra']['selection_step'] = state['step']
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def validate_artifact(path, cfg, root, method):
    state = torch.load(path, map_location='cpu', weights_only=True)
    extra = state.get('extra', {})
    origin = extra.get('provenance', {})
    completion = extra.get('completion', {})
    allowed = ('complete',)
    if (origin.get('method') != method or origin.get('protocol_sha256') != digest(cfg)
            or origin.get('dataset_sha256') != file_hash(Path(root) / 'dataset_manifest.json')
            or completion.get('status') not in allowed or extra.get('best_validation_lpips') is None):
        raise ValueError('Artifact must be a selected, validated scientific checkpoint for this dataset/protocol')
    if not extra.get('selected_artifact') or state.get('step') != extra.get('selection_step'):
        raise ValueError('Selection proof is not bound to this checkpoint step')
    if method == 'vqgan' and origin.get('seed') != cfg['training']['autoencoder_seed']:
        raise ValueError('Unexpected common autoencoder seed')
    if origin.get('source_sha256') != provenance(cfg, root)['source_sha256']:
        raise ValueError('Scientific sources changed since the checkpoint was produced')
    if method != 'vqgan' and completion['completed_updates'] != completion['planned_updates']:
        raise ValueError('Incomplete predictor training')
    return state


def predict(model, method, context, cfg, generator=None):
    # No function in this dispatch receives the target sequence.
    context_frames, horizon = forecast_task(cfg)
    if context.size(1) != context_frames:
        raise ValueError("Observation count does not match the frozen evaluation task")
    if method == 'persistence':
        return persistence(context, horizon), 0
    if method == 'optical_flow':
        return optical_flow(context, horizon), 0
    if method == 'simvp':
        result = model(context, horizon)
    else:
        result = model.generate_frames(context, num_frames=horizon, steps=cfg['evaluation']['ode_grid_points'], warm_start=cfg['evaluation']['warm_start'], generator=generator)[:, context_frames:]
    return result, model.last_nfe


def batch_at(dataset, seed, step, offset, count, device, augment=True):
    """Indexed schedule avoids DataLoader cursor loss when resuming mid-epoch."""
    generator = torch.Generator().manual_seed(seed * 1000003 + step)
    indices = torch.randint(len(dataset), (16,), generator=generator)
    flips = torch.rand(16, generator=generator) < 0.5
    frames, masks = [], []
    for i in range(offset, offset + count):
        item = dataset[int(indices[i])]
        frame, mask = item['frames'], item['alpha']
        if augment and flips[i]:
            frame, mask = frame.flip(-1), mask.flip(-1)
        frames.append(frame)
        masks.append(mask)
    return torch.stack(frames).to(device), torch.stack(masks).to(device)


def parameters(model, method):
    return model.parameters() if method == 'simvp' else model.vector_field_regressor.parameters()


def loss_for(model, method, frames, alpha, cfg):
    if method == 'simvp':
        context, horizon = forecast_task(cfg)
        start = int(torch.randint(0, frames.size(1) - context - horizon + 1, ()).item())
        return (model(frames[:, start:start + context], horizon) - frames[:, start + context:start + context + horizon]).square().mean(), None
    output = model(frames, alpha_masks=alpha)
    error = (output.reconstructed_vectors - output.target_vectors).square().mean(1, keepdim=True)
    if method == 'flow_alpha':
        train = cfg['training']
        weights = output.smoke_mask * train['smoke_weight'] + (1 - output.smoke_mask) * train['background_weight']
        # Normalize each target independently so accumulation/micro-batch size
        # cannot change the scientific objective.
        normalizer = weights.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        error = error * weights / normalizer
    return error.mean(), float(output.latent_mask_fraction.detach())


@torch.no_grad()
def validation_score(model, method, dataset, metric, cfg, device):
    context, horizon = forecast_task(cfg)
    rng = capture_rng_state()
    was_training = model.training
    model.eval()
    scores = []
    try:
        for i in range(len(dataset)):
            frames = dataset[i]['frames'].unsqueeze(0).to(device)
            # One fixed validation realization, four realizations in final test evaluation.
            generator = torch.Generator(device=device).manual_seed(42000 + i)
            prediction, _ = predict(model, method, frames[:, :context], cfg, generator)
            score = float(metric(prediction[0].clamp(-1, 1), frames[0, context:context + horizon]).mean())
            if not math.isfinite(score):
                raise FloatingPointError('Non-finite validation LPIPS; checkpoint selection aborted')
            scores.append(score)
        return float(np.mean(scores))
    finally:
        restore_rng_state(rng)
        model.train(was_training)


def profile(cfg, output, device):
    """Synthetic TRAIN-ONLY throughput probe: no quality result is inferred."""
    context, horizon = forecast_task(cfg)
    seed_all(42)
    results = {}
    # Keep the same resident calibrated metric and optimizer states as real training.
    metric = build_lpips(device)
    for method in LEARNED:
        model = model_for(cfg, method, pilot=True).to(device).train()
        micro = None
        for candidate in (4, 2, 1):
            optimizer = torch.optim.AdamW(parameters(model, method), lr=cfg['training']['learning_rate'])
            try:
                frames = torch.rand(candidate, cfg['task']['clip_frames'], 3, 64, 64, device=device) * 2 - 1
                alpha = torch.rand(candidate, cfg['task']['clip_frames'], 1, 64, 64, device=device)
                loss_for(model, method, frames, alpha, cfg)[0].backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro = candidate
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                del optimizer
                torch.cuda.empty_cache()
        if micro is None:
            raise RuntimeError(f'{method} cannot fit a micro-batch of one')
        timings = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(4):
            synchronize(device)
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(16 // micro):
                (loss_for(model, method, frames, alpha, cfg)[0] / (16 // micro)).backward()
            optimizer.step()
            synchronize(device)
            if iteration:
                timings.append(time.monotonic() - start)
        model.eval()
        with torch.inference_mode():
            synchronize(device)
            start = time.monotonic()
            predict(model, method, frames[:1, :context], cfg, torch.Generator(device=device).manual_seed(42))
            synchronize(device)
            inference_seconds = time.monotonic() - start
        results[method] = dict(micro_batch=micro, update_seconds=max(timings), inference_seconds=inference_seconds, peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        del model, optimizer, frames, alpha
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    result = dict(provenance=provenance(cfg), methods=results, common_updates=cfg['training']['max_updates'], status='resource_probe_not_scientific_result')
    write_json(output, result)
    return result


def train_predictor(cfg, root, method, seed, ae_path, pilot_path, run, device, resume=None, monitor=None):
    forecast_task(cfg)
    pilot = json.loads(Path(pilot_path).read_text())
    if pilot['provenance']['protocol_sha256'] != digest(cfg):
        raise ValueError('Pilot belongs to a different protocol')
    current = provenance(cfg)
    if (pilot['provenance']['source_sha256'] != current['source_sha256']
            or any(pilot['provenance'][key] != current[key] for key in ('gpu', 'torch', 'cuda'))):
        raise ValueError('Pilot source/runtime/GPU changed; do not train from a stale resource measurement')
    if pilot['common_updates'] != cfg['training']['max_updates'] or pilot['methods'][method]['micro_batch'] not in (1, 2, 4):
        raise ValueError('Invalid frozen pilot update or micro-batch count')
    training, validation = WindowDataset(root, 'train'), WindowDataset(root, 'val')
    if method.startswith('flow'):
        validate_artifact(ae_path, cfg, root, 'vqgan')
    seed_all(seed)
    model = model_for(cfg, method, ae_path).to(device)
    optimizer = torch.optim.AdamW(parameters(model, method), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / cfg['training']['warmup_updates']))
    metric = build_lpips(device)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    metadata = provenance(cfg, root) | dict(method=method, seed=seed, autoencoder_sha256=file_hash(ae_path) if method.startswith('flow') else None, pilot_sha256=file_hash(pilot_path))
    extra = dict(provenance=metadata, best_validation_lpips=None, last_validated_step=0)
    step = 0
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        check_resume_provenance(checkpoint['extra']['provenance'], metadata)
        extra = checkpoint['extra']
        extra['provenance'] = metadata
        extra.pop('completion', None)
        extra.pop('selected_artifact', None)
        step = load_checkpoint(model, optimizer, scheduler, resume, device)
    elif (run / 'run.json').exists():
        raise FileExistsError('Existing run: use --resume explicitly')
    write_json(run / 'run.json', metadata | {'status': 'running'})
    total = cfg['training']['max_updates']
    micro = pilot['methods'][method]['micro_batch']
    status = 'incomplete'
    try:
        model.train()
        while step < total or extra.get('last_validated_step', 0) < total:
            ratios = []
            if step < total:
                report = step + 1 == total or (monitor.should_log(step + 1) if monitor else (step + 1) % 25 == 0)
                start_time = time.monotonic()
                update_losses = []
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, 16, micro):
                    frames, alpha = batch_at(training, seed, step, offset, micro, device)
                    loss, ratio = loss_for(model, method, frames, alpha, cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    (loss / (16 // micro)).backward()
                    if report:
                        update_losses.append(loss.detach())
                    if ratio is not None:
                        ratios.append(ratio)
                nn.utils.clip_grad_norm_(parameters(model, method), 1)
                optimizer.step()
                scheduler.step()
                step += 1
                if report:
                    synchronize(device)
                    record_metrics(run, step, {'train/loss': torch.stack(update_losses).mean(), 'train/lr': scheduler.get_last_lr()[0],
                        'progress/updates': step, 'progress/total_updates': total, 'performance/update_seconds': time.monotonic() - start_time,
                        **gpu_metrics(device)}, monitor)
            if step % cfg['training']['validation_every'] == 0 or step == total:
                score = validation_score(model, method, validation, metric, cfg, device)
                images = None
                if monitor is not None and monitor.should_media(step, step == total):
                    with preserve_rng(), torch.inference_mode():
                        was_training = model.training
                        model.eval()
                        try:
                            context, horizon = forecast_task(cfg)
                            frames = validation[0]['frames'].unsqueeze(0).to(device)
                            generated, _ = predict(model, method, frames[:, :context], cfg, torch.Generator(device=device).manual_seed(42000))
                            images = {'validation/forecasts': (image_panel(frames[0, context:context + min(2, horizon)], generated[0, :min(2, horizon)]), 'Left: true future; right: prediction; fixed validation clip')}
                        finally:
                            model.train(was_training)
                record_metrics(run, step, {'validation/lpips': score}, monitor, images)
                extra['last_validated_step'] = step
                if extra['best_validation_lpips'] is None or score < extra['best_validation_lpips']:
                    extra['best_validation_lpips'] = score
                    extra['selection_step'] = step
                    selected = save_checkpoint(model, optimizer, scheduler, run, step, extra)
                    import shutil
                    shutil.copy2(selected, run / 'best.pth')
                extra['latent_mask_fraction'] = float(np.mean(ratios)) if ratios else None
                save_checkpoint(model, optimizer, scheduler, run, step, extra)
                print(json.dumps(dict(method=method, seed=seed, step=step, validation_lpips=score)), flush=True)
        status = 'complete'
    finally:
        final = save_checkpoint(model, optimizer, scheduler, run, step, extra)
        if status == 'complete':
            finalize_selection(run, status, step, total)
        write_json(run / 'run.json', metadata | dict(status=status, completed_updates=step, planned_updates=total, last_checkpoint=str(final), best_validation_lpips=extra['best_validation_lpips']))


def train_autoencoder(cfg, root, run, device, resume=None, monitor=None):
    """Common VQGAN trained with reconstruction, calibrated LPIPS, VQ and hinge GAN."""
    forecast_task(cfg)
    from model.vqgan.discriminator import build_discriminator
    from torch.nn import functional as F

    training, validation = WindowDataset(root, 'train'), WindowDataset(root, 'val')
    seed_all(cfg['training']['autoencoder_seed'])
    model = build_vqvae(DictWrapper(cfg['model']['autoencoder'])).to(device)
    discriminator = build_discriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    metric = build_lpips(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-6)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0002, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    metadata = provenance(cfg, root) | dict(method='vqgan', seed=42)
    step, best, status = 0, None, 'incomplete'
    last_validated_step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=True)
        check_resume_provenance(state['extra']['provenance'], metadata)
        step = load_checkpoint(model, optimizer, scheduler, resume, device)
        discriminator.load_state_dict(state['extra']['discriminator'])
        optimizer_d.load_state_dict(state['extra']['optimizer_d'])
        best = state['extra']['best_validation_lpips']
        last_validated_step = state['extra'].get('last_validated_step', 0)
    elif (run / 'run.json').exists():
        raise FileExistsError('Existing autoencoder run; use --resume')
    write_json(run / 'run.json', metadata | dict(status='running'))
    print(json.dumps(dict(method='vqgan', step=step, total_updates=cfg['training']['max_updates'], status='training')), flush=True)
    try:
        while step < cfg['training']['max_updates'] or last_validated_step < cfg['training']['max_updates']:
            if step < cfg['training']['max_updates']:
                report = step + 1 == cfg['training']['max_updates'] or (monitor.should_log(step + 1) if monitor else (step + 1) % 25 == 0)
                start_time = time.monotonic()
                scalars = {key: [] for key in ('generator', 'reconstruction', 'perceptual', 'vq', 'adversarial', 'discriminator')}
                optimizer.zero_grad(set_to_none=True)
                optimizer_d.zero_grad(set_to_none=True)
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                discriminator_batches = []
                for offset in range(16):
                    frames, _ = batch_at(training, 42, step, offset, 1, device)
                    frame_index = (step * 17 + offset) % cfg['task']['clip_frames']
                    image = frames[:, frame_index]
                    result = model(image)
                    reconstruction = (result.reconstructed_images - image).square().mean()
                    perceptual = metric(result.reconstructed_images, image).mean()
                    quantization = result.vq_loss.mean()
                    loss = reconstruction + perceptual + quantization
                    adversarial = image.new_zeros(())
                    if step >= cfg['training']['discriminator_start']:
                        adversarial = discriminator(result.reconstructed_images).mean()
                        loss = loss - 0.1 * adversarial
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite autoencoder loss')
                    (loss / 16).backward()
                    if report:
                        for key, value in [('generator', loss), ('reconstruction', reconstruction), ('perceptual', perceptual), ('vq', quantization), ('adversarial', -adversarial)]:
                            scalars[key].append(value.detach())
                    if step >= cfg['training']['discriminator_start']:
                        discriminator_batches.append((image.detach(), result.reconstructed_images.detach()))
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                if step >= cfg['training']['discriminator_start']:
                    for real, generated in discriminator_batches:
                        loss_d = (F.relu(1 - discriminator(real)).mean() + F.relu(1 + discriminator(generated)).mean()) / 2
                        (loss_d / 16).backward()
                        if report:
                            scalars['discriminator'].append(loss_d.detach())
                    optimizer_d.step()
                scheduler.step()
                step += 1
                if report:
                    synchronize(device)
                    values = {f'train/{key}': torch.stack(values).mean() if values else 0.0 for key, values in scalars.items()}
                    values.update({'train/lr_generator': optimizer.param_groups[0]['lr'], 'train/lr_discriminator': optimizer_d.param_groups[0]['lr'],
                        'train/discriminator_active': step > cfg['training']['discriminator_start'], 'progress/updates': step,
                        'progress/total_updates': cfg['training']['max_updates'], 'performance/update_seconds': time.monotonic() - start_time, **gpu_metrics(device)})
                    record_metrics(run, step, values, monitor)
            if step % 250 == 0 or step == cfg['training']['max_updates']:
                model.eval()
                scores = []
                real_images, reconstructions = [], []
                show_images = monitor is not None and monitor.should_media(step, step == cfg['training']['max_updates'])
                with torch.inference_mode():
                    for index in range(len(validation)):
                        frame_index = index % cfg['task']['clip_frames']
                        frame = validation[index]['frames'][frame_index:frame_index + 1].to(device)
                        reconstruction = model(frame).reconstructed_images
                        score = float(metric(reconstruction.clamp(-1, 1), frame).mean())
                        if show_images and index < 2:
                            real_images.append(frame[0].detach().cpu())
                            reconstructions.append(reconstruction[0].detach().cpu())
                        if not math.isfinite(score):
                            raise FloatingPointError('Non-finite autoencoder validation LPIPS')
                        scores.append(score)
                score = float(np.mean(scores))
                images = {'validation/reconstructions': (image_panel(torch.stack(real_images), torch.stack(reconstructions)), 'Left: true RGB; right: reconstruction; fixed validation samples')} if show_images else None
                record_metrics(run, step, {'validation/lpips': score}, monitor, images)
                last_validated_step = step
                improved = best is None or score < best
                best = score if improved else best
                extra = dict(provenance=metadata, best_validation_lpips=best, last_validated_step=last_validated_step, discriminator=discriminator.state_dict(), optimizer_d=optimizer_d.state_dict())
                path = save_checkpoint(model, optimizer, scheduler, run, step, extra)
                if improved:
                    import shutil
                    shutil.copy2(path, run / 'best.pth')
                model.train()
        status = 'complete'
    finally:
        extra = dict(provenance=metadata, best_validation_lpips=best, last_validated_step=last_validated_step, discriminator=discriminator.state_dict(), optimizer_d=optimizer_d.state_dict())
        path = save_checkpoint(model, optimizer, scheduler, run, step, extra)
        if status == 'complete':
            finalize_selection(run, status, step, cfg['training']['max_updates'])
        write_json(run / 'run.json', metadata | dict(status=status, completed_updates=step, best_validation_lpips=best, last_checkpoint=str(path)))


@torch.inference_mode()
def evaluate(cfg, root, method, seed, checkpoint, autoencoder, split, output, device, metric=None):
    context, horizon = forecast_task(cfg)
    dataset = WindowDataset(root, split)
    seed_all(seed)
    model = None
    if method in LEARNED:
        state = validate_artifact(checkpoint, cfg, root, method)
        if method.startswith('flow'):
            validate_artifact(autoencoder, cfg, root, 'vqgan')
        model = model_for(cfg, method, autoencoder).to(device).eval()
        expected = state['extra']['provenance']
        if expected['method'] != method or expected['seed'] != seed or expected['protocol_sha256'] != digest(cfg) or expected['dataset_sha256'] != file_hash(Path(root) / 'dataset_manifest.json'):
            raise ValueError('Checkpoint identity does not match this evaluation')
        if method.startswith('flow') and expected['autoencoder_sha256'] != file_hash(autoencoder):
            raise ValueError('Autoencoder differs from the training artifact')
        model.load_state_dict(state['model'], strict=True)
    metric_identity = 'injected_test_metric_not_publication' if metric is not None else 'official alex v0.1'
    metric = metric if metric is not None else build_lpips(device)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    metadata = provenance(cfg, root) | dict(method=method, seed=seed, split=split, checkpoint_sha256=file_hash(checkpoint) if checkpoint else None, lpips=metric_identity, status='running', expected_window_ids=[r['window_id'] for r in dataset.records], expected_realizations=4 if method.startswith('flow') else 1, expected_horizons=list(range(1, horizon + 1)), context_frames=context, horizon=horizon, training_complete=method not in LEARNED or state['extra']['completion']['status'] == 'complete')
    # Hash actual calibrated metric tensors, including the feature backbone.
    import hashlib
    weights = hashlib.sha256()
    for name, tensor in sorted(metric.state_dict().items()):
        weights.update(name.encode())
        weights.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata['metric_weights_sha256'] = weights.hexdigest()
    write_json(output / 'evaluation.json', metadata)
    rows, runtimes, artifact_hashes = [], [], {}
    status = 'incomplete'
    try:
        for i, record in enumerate(dataset.records):
            item = dataset[i]
            frames, alpha = item['frames'].to(device), item['alpha'].to(device)
            generated = []
            count = cfg['evaluation']['realizations'] if method.startswith('flow') else 1
            for realization in range(count):
                realization_seed = int(digest([seed, record['window_id'], realization])[:12], 16)
                generator = torch.Generator(device=device).manual_seed(realization_seed)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize(device)
                start = time.monotonic()
                prediction, nfe = predict(model, method, frames[None, :context], cfg, generator)
                synchronize(device)
                elapsed = time.monotonic() - start
                prediction = prediction[0].clamp(-1, 1)
                generated.append(prediction)
                measurements = frame_metrics(prediction, frames[context:context + horizon], alpha[context:context + horizon], metric)
                for measurement in measurements:
                    rows.append(measurement | dict(method=method, seed=seed, split=split, window_id=record['window_id'], date=record['date'], camera_id=record['camera_id'], view_id=record['view_id'], realization=realization, realization_seed=realization_seed))
                runtimes.append(dict(window_id=record['window_id'], realization=realization, seconds=elapsed, nfe=nfe, peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0))
            diversity_values = diversity(generated)
            for row in rows[-horizon * count:]:
                row['diversity_mse'] = diversity_values[row['horizon'] - 1]
            archive = output / 'predictions' / f"{record['window_id']}.npz"
            samples = ((torch.stack(generated).detach().cpu().permute(0, 1, 3, 4, 2) + 1) * 127.5).round().byte().numpy()
            np.savez_compressed(archive, generated=samples)
            artifact_hashes[record['window_id']] = file_hash(archive)
            if i < 8:
                from PIL import Image
                originals = ((frames.detach().cpu().permute(0, 2, 3, 1) + 1) * 127.5).round().byte().numpy()
                originals = originals[:context + horizon]
                blank = np.zeros_like(originals[context:])
                # First eight fixed panel windows, every draw; no visual cherry-picking.
                panel = [np.concatenate((originals[:context], blank)), originals]
                panel.extend(np.concatenate((originals[:context], sample)) for sample in samples)
                Image.fromarray(np.concatenate([np.concatenate(row, axis=1) for row in panel], axis=0)).save(output / f"panel_{record['window_id']}.png")
        status = 'complete'
    finally:
        for name, values in [('metrics.csv', rows), ('runtime.csv', runtimes)]:
            if values:
                with (output / name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
        write_json(output / 'evaluation.json', metadata | dict(status=status, windows_completed=len(artifact_hashes), expected_windows=len(dataset), prediction_sha256=artifact_hashes))
    return rows
