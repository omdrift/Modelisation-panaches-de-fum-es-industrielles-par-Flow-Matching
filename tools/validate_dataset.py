#!/usr/bin/env python3
"""Check split integrity and sample image health for the frame dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


FRAME_SUFFIX = re.compile(r"^(?P<video>.+)_frame_(?P<frame>\d+)$")


def video_name(path_text: str) -> str:
    stem = Path(path_text).stem
    match = FRAME_SUFFIX.match(stem)
    if match:
        return match.group("video")
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def read_split(root: Path, split: str) -> tuple[list[str], Counter[str], list[str]]:
    candidates = (root / f"{split}_files.txt", root / f"{split}.txt")
    split_file = next((candidate for candidate in candidates if candidate.is_file()), None)
    if split_file is None:
        raise FileNotFoundError(f"Missing split file for '{split}' in {root}")

    paths: list[str] = []
    errors: list[str] = []
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            errors.append(f"{split_file}:{line_number}: empty line")
            continue
        relative_path = fields[0]
        candidate_path = (root / relative_path).resolve()
        try:
            candidate_path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{split_file}:{line_number}: path escapes dataset root: {relative_path}")
            continue
        paths.append(relative_path)
        counts[video_name(relative_path)] += 1
    return paths, counts, errors


def inspect_image(path: Path, black_threshold: int) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"unreadable image {path}: {exc}") from exc

    return {
        "size": [int(pixels.shape[1]), int(pixels.shape[0])],
        "mean_luminance": float(pixels.mean() / 255.0),
        "is_black": bool(int(pixels.max()) <= black_threshold),
    }


def validate_dataset(
    root: Path,
    requested_split: str,
    num_samples: int,
    min_sequence_frames: int,
    black_threshold: int,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    split_data: dict[str, tuple[list[str], Counter[str]]] = {}
    for split in ("train", "val", "test"):
        try:
            paths, counts, split_errors = read_split(root, split)
        except FileNotFoundError as exc:
            errors.append(str(exc))
            continue
        split_data[split] = (paths, counts)
        errors.extend(split_errors)
        if not paths:
            errors.append(f"split '{split}' contains no samples")
        short_sequences = [name for name, count in counts.items() if count < min_sequence_frames]
        if short_sequences:
            shortest = min(counts[name] for name in short_sequences)
            errors.append(
                f"split '{split}' has {len(short_sequences)} sequences shorter than "
                f"{min_sequence_frames} frames (shortest: {shortest})"
            )

    overlap_errors = []
    split_names = list(split_data)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = set(split_data[left][1]).intersection(split_data[right][1])
            if overlap:
                overlap_errors.append(f"{left}/{right} share {len(overlap)} video IDs")
    errors.extend(overlap_errors)

    if requested_split not in split_data:
        errors.append(f"requested split '{requested_split}' is unavailable")
        sample_paths: list[str] = []
    else:
        all_paths = split_data[requested_split][0]
        count = min(num_samples, len(all_paths))
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(all_paths)), count))
        sample_paths = [all_paths[index] for index in indices]

    inspected = []
    expected_size = None
    for relative_path in sample_paths:
        image_path = root / relative_path
        try:
            record = inspect_image(image_path, black_threshold)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if expected_size is None:
            expected_size = record["size"]
        elif record["size"] != expected_size:
            errors.append(
                f"sample image dimensions differ: expected {expected_size}, "
                f"found {record['size']} at {image_path}"
            )
        inspected.append(record)

    report = {
        "root": str(root),
        "sampled_split": requested_split,
        "sampled_images": len(inspected),
        "splits": {
            split: {
                "frames": len(paths),
                "videos": len(counts),
                "min_frames_per_video": min(counts.values()) if counts else None,
                "max_frames_per_video": max(counts.values()) if counts else None,
            }
            for split, (paths, counts) in split_data.items()
        },
        "sample_image_size": expected_size,
        "mean_luminance": (
            float(np.mean([record["mean_luminance"] for record in inspected]))
            if inspected
            else None
        ),
        "black_frame_fraction": (
            float(np.mean([record["is_black"] for record in inspected]))
            if inspected
            else None
        ),
        "errors": errors,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-sequence-frames", type=int, default=16)
    parser.add_argument("--black-threshold", type=int, default=2, help="Maximum RGB value (0-255) for an all-black frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.min_sequence_frames <= 0:
        parser.error("--min-sequence-frames must be positive")
    if not 0 <= args.black_threshold <= 255:
        parser.error("--black-threshold must be between 0 and 255")

    root = args.root.resolve()
    report, errors = validate_dataset(
        root=root,
        requested_split=args.split,
        num_samples=args.num_samples,
        min_sequence_frames=args.min_sequence_frames,
        black_threshold=args.black_threshold,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
