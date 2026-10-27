"""Checkpoint serialization including reproducible Python/NumPy/Torch RNG states."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def capture_rng_state() -> dict:
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], torch.from_numpy(state[1].astype(np.int64)), state[2], state[3], state[4]), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(model, optimizer, scheduler, run_dir: Path, step: int, extra: dict | None = None) -> Path:
    directory = Path(run_dir) / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'step_{step}.pth'
    state = {'step': step, 'model': model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng_state(), 'extra': extra or {}}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, device: torch.device) -> int:
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = loaded.get('model', loaded)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    dmodel = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    dmodel.load_state_dict(state, strict=True)
    if 'optimizer' in loaded:
        optimizer.load_state_dict(loaded['optimizer'])
    if 'scheduler' in loaded:
        scheduler.load_state_dict(loaded['scheduler'])
    if 'rng' in loaded:
        restore_rng_state(loaded['rng'])
    return int(loaded.get('step', 0))
