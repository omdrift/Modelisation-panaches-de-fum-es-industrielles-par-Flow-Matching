#!/usr/bin/env python3
"""Check experiment YAML dimensions, paths, and train/evaluation compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PERSONAL_PATH_MARKERS = ("/home/", "/Users/", "/absolute/path")


def _strings(value: Any, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            path = f"{key}.{child_key}" if key else str(child_key)
            yield from _strings(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _strings(child, f"{key}[{index}]")
    elif isinstance(value, str):
        yield key, value


def validate_config(
    config_path: Path,
    *,
    check_data: bool = True,
    check_checkpoints: bool = True,
) -> list[str]:
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot load YAML: {exc}"]
    if not isinstance(config, dict):
        return ["top-level YAML value must be a mapping"]

    errors: list[str] = []
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    evaluation = config.get("evaluation", {}) or {}

    for key, value in _strings(config):
        if any(marker in value for marker in PERSONAL_PATH_MARKERS):
            errors.append(f"{key}: machine-specific path is not portable: {value}")

    input_size = data.get("input_size")
    if input_size is not None:
        downsampling = model.get("latent_downsampling", 8)
        if not isinstance(downsampling, int) or downsampling <= 0:
            errors.append("model.latent_downsampling must be a positive integer")
        elif input_size % downsampling:
            errors.append("data.input_size must be divisible by model.latent_downsampling")
        else:
            latent_size = input_size // downsampling
            flow_model = model.get("vector_field_regressor", {}) or {}
            latent_resolution = flow_model.get("state_res", model.get("latent_resolution"))
            if latent_resolution is not None and list(latent_resolution) != [latent_size, latent_size]:
                errors.append(
                    f"latent resolution {latent_resolution} does not match "
                    f"input_size/downsampling = {[latent_size, latent_size]}"
                )
        crop_size = data.get("crop_size")
        if crop_size is not None and crop_size != input_size:
            errors.append("data.crop_size must match data.input_size")

    autoencoder = model.get("autoencoder", {}) or {}
    regressor = model.get("vector_field_regressor", {}) or {}
    state_size = regressor.get("state_size")
    embedding_dimension = (autoencoder.get("vector_quantizer", {}) or {}).get("embedding_dimension")
    if state_size is not None and embedding_dimension is not None and state_size != embedding_dimension:
        errors.append(
            "model.vector_field_regressor.state_size must equal "
            "model.autoencoder.vector_quantizer.embedding_dimension"
        )

    frames_per_sample = data.get("frames_per_sample")
    if frames_per_sample is not None:
        required_training_frames = training.get("num_observations")
        if required_training_frames is not None and frames_per_sample < required_training_frames:
            errors.append("data.frames_per_sample must be >= training.num_observations")
        eval_observations = evaluation.get("num_observations")
        eval_generated = evaluation.get("frames_to_generate")
        if eval_observations is not None and eval_generated is not None:
            if frames_per_sample < eval_observations + eval_generated:
                errors.append(
                    "data.frames_per_sample must cover evaluation.num_observations "
                    "+ evaluation.frames_to_generate"
                )

    for key in ("num_observations", "condition_frames", "frames_to_generate"):
        train_value = training.get(key)
        eval_value = evaluation.get(key)
        if train_value is not None and eval_value is not None and train_value != eval_value:
            errors.append(f"evaluation.{key} must match training.{key}")

    for name, section in (("training", training), ("evaluation", evaluation)):
        observations = section.get("num_observations")
        conditions = section.get("condition_frames")
        if observations is not None and observations < 3:
            errors.append(f"{name}.num_observations must be at least 3 for the current target sampler")
        if observations is not None and conditions is not None and not 1 <= conditions <= observations:
            errors.append(f"{name}.condition_frames must be between 1 and num_observations")

    if check_data and data.get("data_root"):
        data_root = Path(data["data_root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        if not data_root.exists():
            errors.append(f"data.data_root does not exist: {data_root}")

    checkpoint = autoencoder.get("ckpt_path")
    if check_checkpoints and checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            errors.append(f"model.autoencoder.ckpt_path does not exist: {checkpoint_path}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="YAML configuration files")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    parser.add_argument("--allow-missing-data", action="store_true")
    args = parser.parse_args()

    any_errors = False
    for raw_path in args.configs:
        config_path = Path(raw_path)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        errors = validate_config(
            config_path,
            check_data=not args.allow_missing_data,
            check_checkpoints=not args.allow_missing_checkpoints,
        )
        if errors:
            any_errors = True
            for error in errors:
                print(f"ERROR: {config_path}: {error}")
        else:
            print(f"OK: {config_path}")
    return int(any_errors)


if __name__ == "__main__":
    raise SystemExit(main())
