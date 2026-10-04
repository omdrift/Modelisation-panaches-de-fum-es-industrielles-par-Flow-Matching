#!/usr/bin/env python3
"""Evaluate a Flow Matching checkpoint without requiring a training logger."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dataset.text_based_video_dataset import TextBasedVideoDataset
from lutils.configuration import Configuration
from model.model import Model


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--num-test-videos",
        type=int,
        default=None,
        help="Optional upper bound on the number of videos evaluated",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override ODE integration steps")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_outputs/flow_matching"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_split_file(data_root: Path, split: str) -> str:
    for candidate in (f"{split}_files.txt", f"{split}.txt"):
        if (data_root / candidate).is_file():
            return candidate
    raise FileNotFoundError(f"Missing split file for '{split}' in {data_root}")


def mse_metric(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((prediction - target) ** 2).item())


def psnr_metric(mse_value: float) -> float:
    if mse_value <= 1e-12:
        return 100.0
    return 10.0 * math.log10(4.0 / mse_value)


def ssim_metric(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    prediction = ((prediction.detach().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
    target = ((target.detach().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
    scores = [
        structural_similarity(left, right, data_range=1.0, channel_axis=2)
        for left, right in zip(prediction, target)
    ]
    return float(np.mean(scores)) if scores else None


def save_video(frames: torch.Tensor, output_dir: Path, fps: int = 7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = ((frames.detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    pil_frames = []
    for index, frame in enumerate(frames):
        array = frame.permute(1, 2, 0).numpy()
        image = Image.fromarray(array, mode="RGB")
        image.save(output_dir / f"frame_{index:04d}.png")
        pil_frames.append(image)
    if pil_frames:
        pil_frames[0].save(
            output_dir / "video.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=round(1000 / fps),
            loop=0,
        )


def load_model(config: Configuration, checkpoint_path: Path, device: torch.device) -> Model:
    # Resolve local relative paths from the repository root, independent of the shell cwd.
    data_root = Path(config["data"]["data_root"])
    config["data"]["data_root"] = str(resolve_path(data_root))
    autoencoder_path = Path(config["model"]["autoencoder"]["ckpt_path"])
    config["model"]["autoencoder"]["ckpt_path"] = str(resolve_path(autoencoder_path))
    config["model"]["smoke_threshold"] = float(config["training"].get("smoke_threshold", 0.1))

    model = Model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.num_test_videos is not None and args.num_test_videos <= 0:
        raise ValueError("--num-test-videos must be positive")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = Configuration(str(config_path))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(config, checkpoint_path, device)

    data_config = config["data"]
    eval_config = config["evaluation"]
    observations_count = int(eval_config["num_observations"])
    condition_frames = int(eval_config["condition_frames"])
    frames_to_generate = int(eval_config["frames_to_generate"])
    steps = int(args.steps if args.steps is not None else eval_config.get("steps", 50))
    frames_per_sample = max(
        int(data_config.get("frames_per_sample", 0)),
        observations_count + frames_to_generate,
    )
    dataset = TextBasedVideoDataset(
        data_path=data_config["data_root"],
        file_list=resolve_split_file(Path(data_config["data_root"]), args.split),
        input_size=int(data_config["input_size"]),
        crop_size=int(data_config["crop_size"]),
        frames_per_sample=frames_per_sample,
        random_horizontal_flip=False,
        random_time=False,
        skip_short_videos=True,
    )
    sample_count = min(args.num_samples, len(dataset))
    if args.num_test_videos is not None:
        sample_count = min(sample_count, args.num_test_videos)
    if sample_count == 0:
        raise ValueError(f"No usable videos in split '{args.split}'")
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=1,
        shuffle=False,
        num_workers=int(eval_config.get("batching", {}).get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    for name in ("real_videos", "generated_videos", "comparison_grids"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    with torch.inference_mode():
        for sample_index, batch in enumerate(tqdm(loader, desc=f"Evaluating {args.split}")):
            batch = batch.to(device, non_blocking=True)
            condition = batch[:, :condition_frames]
            target = batch[:, condition_frames : condition_frames + frames_to_generate]
            generated_full = model.generate_frames(
                observations=condition,
                num_frames=frames_to_generate,
                steps=steps,
                verbose=False,
            )
            generated = generated_full[:, condition_frames : condition_frames + frames_to_generate]
            length = min(target.size(1), generated.size(1))
            target = target[:, :length]
            generated = generated[:, :length]
            mse = mse_metric(generated, target)
            mse_values.append(mse)
            psnr_values.append(psnr_metric(mse))
            ssim = ssim_metric(generated[0], target[0])
            if ssim is not None:
                ssim_values.append(ssim)

            sample_name = f"sample_{sample_index:04d}"
            save_video(target[0], output_dir / "real_videos" / sample_name)
            save_video(generated[0], output_dir / "generated_videos" / sample_name)
            grid_frames = torch.cat((condition[0].cpu(), target[0].cpu(), generated[0].cpu()), dim=0)
            grid = make_grid(grid_frames, nrow=max(1, grid_frames.size(0) // 3), normalize=True, value_range=(-1, 1))
            save_image(grid, output_dir / "comparison_grids" / f"{sample_name}.png")

    summary = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split": args.split,
        "num_samples": len(mse_values),
        "steps": steps,
        "condition_frames": condition_frames,
        "frames_to_generate": frames_to_generate,
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_std": float(np.std(mse_values)) if mse_values else None,
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "psnr_std": float(np.std(psnr_values)) if psnr_values else None,
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "metrics_note": None if ssim_values else "SSIM omitted; install scikit-image to enable it.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
