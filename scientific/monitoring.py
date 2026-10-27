"""Optional WandB monitoring with local metrics and preserved training RNG state."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import uuid
import warnings

import numpy as np
import torch

from scientific.protocol import write_json
from training.checkpoints import capture_rng_state, restore_rng_state


@contextmanager
def preserve_rng():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


@dataclass(frozen=True)
class WandbOptions:
    project: str = 'smoke-science'
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    mode: str = 'online'
    log_every: int = 25
    media_every: int = 250


class WandbMonitor:
    """The SDK owns transport; checkpoints alone own model/optimizer recovery."""
    def __init__(self, directory, cfg, method, seed, options: WandbOptions, resume=False):
        self.directory = Path(directory)
        self.cfg, self.method, self.seed = cfg, method, seed
        self.options, self.resume = options, resume
        self.run = None
        if options.mode not in ('online', 'offline') or min(options.log_every, options.media_every) < 1:
            raise ValueError('WandB requires online/offline mode and positive logging intervals')

    def __enter__(self):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('WandB requested: install wandb==0.19.11 in smoke-science') from exc
        self.sdk = wandb
        self.directory.mkdir(parents=True, exist_ok=True)
        identity_file = self.directory / 'wandb.json'
        previous = json.loads(identity_file.read_text()) if self.resume and identity_file.exists() else None
        if previous and (previous['project'] != self.options.project or self.options.entity not in (None, previous['entity'])):
            raise ValueError('Resume WandB with the original project/entity')
        run_id = previous['id'] if previous else uuid.uuid4().hex[:16]
        entity = self.options.entity or (previous['entity'] if previous else None)
        session = self.directory / 'tracking' / uuid.uuid4().hex[:12]
        session.mkdir(parents=True)
        configuration = {'protocol': self.cfg, 'method': self.method, 'seed': self.seed}
        if self.method in ('vqgan', 'simvp', 'flow_uniform', 'flow_alpha'):
            from scientific.hyperparameters import training_description
            configuration['effective_training'] = training_description(self.cfg, self.method)
        try:
            with preserve_rng():
                self.run = wandb.init(
                    project=self.options.project, entity=entity, id=run_id,
                    name=self.options.name or f'{self.directory.parent.name}-{self.method}-{self.seed}',
                    group=self.options.group or self.directory.parent.name, job_type=self.method,
                    mode=self.options.mode, resume='allow' if self.options.mode == 'online' else None,
                    dir=str(session), config=configuration,
                    save_code=False,
                    settings=wandb.Settings(disable_code=True, disable_git=True, disable_job_creation=True, console='off', init_timeout=30),
                )
                if self.run is None:
                    raise RuntimeError('WandB did not initialize a run')
                self.run.define_metric('global_step', hidden=True)
                self.run.define_metric('*', step_metric='global_step', hidden=True)
                # Keep new automatic workspaces small. The curated workspace
                # can still query every logged metric, including hidden ones.
                for name in ('train_eval/lpips', 'validation/lpips', 'validation/smoke_mse',
                             'validation/background_mse', 'validation/errors_and_masks',
                             'generalization/lpips_gap', 'validation/codebook/perplexity',
                             'validation/codebook/used_fraction'):
                    self.run.define_metric(name, step_metric='global_step', hidden=False)
                self.identity = dict(id=run_id, project=self.options.project, entity=self.run.entity,
                                     mode=self.options.mode, directory=str(session), url=self.run.url if self.options.mode == 'online' else None,
                                     sdk_version=wandb.__version__)
                write_json(identity_file, self.identity)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        print(f"WandB: {self.identity['url'] or session} ({self.options.mode})", flush=True)
        return self

    def should_log(self, step):
        return step % self.options.log_every == 0

    def should_media(self, step, final=False):
        return final or step % self.options.media_every == 0

    def log(self, step, values, images=None):
        with preserve_rng():
            payload = dict(values, global_step=step)
            for key, (pixels, caption) in (images or {}).items():
                payload[key] = self.sdk.Image(pixels, caption=caption)
            # Separate optimizer-update axis permits a checkpoint rollback online.
            self.run.log(payload)

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            try:
                with preserve_rng():
                    self.run.finish(exit_code=0 if exc_type is None else 1)
            except Exception as finish_error:
                if exc_type is None:
                    raise
                warnings.warn(f'WandB finalization also failed: {finish_error}', RuntimeWarning)
            finally:
                self.run = None


def image_panel(real, reconstructed):
    """Rows are examples, left/right are true/reconstructed RGB; explicit uint8."""
    rows = [torch.cat((left, right), dim=-1) for left, right in zip(real[:2], reconstructed[:2])]
    pixels = torch.cat(rows, dim=-2).detach().float().clamp(-1, 1)
    return ((pixels.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


def error_panel(real, reconstructed, alpha):
    """Columns: true RGB, reconstruction, absolute error, estimated mask."""
    real, reconstructed = real[:4], reconstructed[:4].clamp(-1, 1)
    errors = ((real - reconstructed) / 2).abs().mean(1, keepdim=True).expand(-1, 3, -1, -1) * 2 - 1
    masks = alpha[:4].expand(-1, 3, -1, -1) * 2 - 1
    rows = [torch.cat(images, dim=-1) for images in zip(real, reconstructed, errors, masks)]
    pixels = torch.cat(rows, dim=-2).detach().float().clamp(-1, 1)
    return ((pixels.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)


def record_metrics(directory, step, values, monitor=None, images=None):
    """Always keep scalar history locally; forward to WandB only when requested."""
    values = {key: float(value.detach().cpu()) if torch.is_tensor(value) else value for key, value in values.items()}
    with (Path(directory) / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(values, global_step=step), allow_nan=False) + '\n')
    print(json.dumps(dict(values, step=step), allow_nan=False), flush=True)
    if monitor is not None:
        monitor.log(step, values, images)


def gpu_metrics(device):
    if torch.device(device).type != 'cuda':
        return {}
    return {'gpu/allocated_bytes': torch.cuda.memory_allocated(device),
            'gpu/reserved_bytes': torch.cuda.memory_reserved(device),
            'gpu/peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
