"""Flow Matching checkpoint serialization and restoration."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, scheduler, run_dir: Path, step: int) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"step_{step}.pth"
    state = {
        "step": step,
        "model": model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path, device: torch.device) -> int:
    loaded = torch.load(checkpoint_path, map_location=device)
    state = loaded.get("model", loaded)
    dmodel = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    dmodel.load_state_dict(state, strict=False)
    if "optimizer" in loaded:
        optimizer.load_state_dict(loaded["optimizer"])
    if "scheduler" in loaded:
        scheduler.load_state_dict(loaded["scheduler"])
    return int(loaded.get("step", 0))
