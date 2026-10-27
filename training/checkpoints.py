"""Checkpoint serialization including reproducible Python/NumPy/Torch RNG states."""
from __future__ import annotations

import random
import re
import shutil
import tempfile
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


class CheckpointWriteError(OSError):
    """Checkpoint transaction failed without replacing an existing artifact."""


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
        return Path(stream.name)


def _write_error(path: Path) -> CheckpointWriteError:
    directory = path.parent
    while not directory.exists():
        directory = directory.parent
    free = shutil.disk_usage(directory).free / 1024 ** 3
    return CheckpointWriteError(f'Cannot write checkpoint {path}; free disk space: {free:.2f} GiB. Previous checkpoints are preserved.')


def atomic_save(state: dict, path: Path) -> None:
    """Commit a complete serialization or leave the previous file untouched."""
    path = Path(path)
    temporary = None
    try:
        temporary = _temporary_path(path)
        torch.save(state, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as exc:
        raise _write_error(path) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_checkpoint(source: Path, destination: Path) -> None:
    """Update best.pth atomically, including when disk space is exhausted."""
    destination = Path(destination)
    temporary = None
    try:
        temporary = _temporary_path(destination)
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except OSError as exc:
        raise _write_error(destination) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    return sorted((int(match[1]), path) for path in (Path(run_dir) / 'checkpoints').glob('step_*.pth')
                  if (match := re.fullmatch(r'step_(\d+)\.pth', path.name)) and path.is_file())


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = _checkpoints(run_dir)
    return candidates[-1][1] if candidates else None


def save_checkpoint(model, optimizer, scheduler, run_dir: Path, step: int, extra: dict | None = None, *, keep_last: int = 2) -> Path:
    """Retain the two newest recovery points; best.pth is stored separately.

    Pruning happens only after a successful atomic save. A failed write cannot
    remove a recovery point or replace the previously selected best artifact.
    """
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError('keep_last must be a positive integer')
    directory = Path(run_dir) / 'checkpoints'
    path = directory / f'step_{step}.pth'
    state = {'step': step, 'model': model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': capture_rng_state(), 'extra': extra or {}}
    atomic_save(state, path)
    candidates = _checkpoints(run_dir)
    retained = {entry[1] for entry in candidates[-keep_last:]} | {path}
    for _, obsolete in candidates:
        if obsolete not in retained:
            obsolete.unlink()
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
