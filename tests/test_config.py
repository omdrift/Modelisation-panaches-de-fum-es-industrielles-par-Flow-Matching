from pathlib import Path

import yaml

from tools.validate_config import validate_config


CONFIGS = (
    "configs/flow_matching_64.yaml",
    "configs/flow_matching_128.yaml",
    "configs/vqgan_64.yaml",
)


def test_supported_configurations_are_structurally_valid():
    root = Path(__file__).resolve().parents[1]
    for relative_path in CONFIGS:
        assert validate_config(
            root / relative_path,
            check_data=True,
            check_checkpoints=False,
        ) == []


def test_validator_reports_latent_and_temporal_mismatches(tmp_path):
    invalid_config = {
        "data": {
            "data_root": ".",
            "input_size": 64,
            "crop_size": 64,
            "frames_per_sample": 8,
        },
        "model": {
            "vector_field_regressor": {"state_size": 8, "state_res": [7, 8]},
            "autoencoder": {"vector_quantizer": {"embedding_dimension": 16}},
        },
        "training": {"num_observations": 10, "condition_frames": 1, "frames_to_generate": 2},
        "evaluation": {"num_observations": 10, "condition_frames": 1, "frames_to_generate": 2},
    }
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(invalid_config), encoding="utf-8")

    errors = validate_config(config_path, check_data=False, check_checkpoints=False)
    assert any("latent resolution" in error for error in errors)
    assert any("state_size" in error for error in errors)
    assert any("frames_per_sample" in error for error in errors)
