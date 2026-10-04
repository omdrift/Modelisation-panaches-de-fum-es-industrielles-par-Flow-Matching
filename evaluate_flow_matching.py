#!/usr/bin/env python3
"""Offline evaluation for flow matching checkpoints on the smoke dataset."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.text_based_video_dataset import TextBasedVideoDataset
from lutils.configuration import Configuration
from model.model import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to evaluate")
    parser.add_argument("--num-test-videos", type=int, default=None, help="Optional max number of test sequences")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs", help="Directory for metrics")
    parser.add_argument("--steps", type=int, default=None, help="Override ODE integration steps")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu or cuda")
    return parser.parse_args()


def resolve_split_file(data_root: str, split: str) -> str:
    candidates = [f"{split}_files.txt", f"{split}.txt"]
    for c in candidates:
        if (Path(data_root) / c).exists():
            return c
    raise FileNotFoundError(f"Missing split file for '{split}' in {data_root}")


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def psnr_metric(mse_value: float) -> float:
    if mse_value <= 1e-12:
        return 100.0
    max_pixel = 2.0
    return 10.0 * math.log10((max_pixel ** 2) / mse_value)


def main():
    args = parse_args()
    config = Configuration(args.config)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(config["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_state = ckpt.get("model", ckpt)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    data_cfg = config["data"]
    eval_cfg = config["evaluation"]

    frames_per_sample = max(
        int(data_cfg.get("frames_per_sample", 0)),
        int(eval_cfg["num_observations"]) + int(eval_cfg["frames_to_generate"]),
    )

    test_dataset = TextBasedVideoDataset(
        data_path=data_cfg["data_root"],
        file_list=resolve_split_file(data_cfg["data_root"], "test"),
        input_size=data_cfg["input_size"],
        crop_size=data_cfg["crop_size"],
        frames_per_sample=frames_per_sample,
        random_horizontal_flip=False,
        random_time=False,
    )

    max_items = len(test_dataset)
    if args.num_test_videos is not None:
        max_items = min(max_items, int(args.num_test_videos))
    if args.num_samples is not None:
        max_items = min(max_items, int(args.num_samples))

    subset = Subset(test_dataset, list(range(max_items)))
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=int(eval_cfg.get("batching", {}).get("num_workers", 0)),
        pin_memory=True,
    )

    num_observations = int(eval_cfg["num_observations"])
    condition_frames = int(eval_cfg["condition_frames"])
    frames_to_generate = int(eval_cfg["frames_to_generate"])
    steps = int(args.steps if args.steps is not None else eval_cfg.get("steps", 100))

    mse_values = []
    psnr_values = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device, non_blocking=True)
            observations = batch[:, :num_observations]
            targets = batch[:, num_observations:num_observations + frames_to_generate]

            generated = model.generate_frames(
                observations=observations[:, :condition_frames],
                num_frames=frames_to_generate,
                steps=steps,
                verbose=False,
            )

            if generated.size(1) != targets.size(1):
                min_len = min(generated.size(1), targets.size(1))
                generated = generated[:, :min_len]
                targets = targets[:, :min_len]

            mse = float(mse_metric(generated, targets).item())
            mse_values.append(mse)
            psnr_values.append(psnr_metric(mse))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": len(mse_values),
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_std": float(np.std(mse_values)) if mse_values else None,
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "psnr_std": float(np.std(psnr_values)) if psnr_values else None,
        "steps": steps,
        "condition_frames": condition_frames,
        "frames_to_generate": frames_to_generate,
    }

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
