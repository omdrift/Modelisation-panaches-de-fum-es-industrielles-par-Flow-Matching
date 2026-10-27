"""Regression checks for the documented preparation and evaluation workflow."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from PIL import Image

import quick_eval_test
from prepare_data.organize_dataset import organize_dataset
from tools.validate_config import validate_config


def test_organizer_preserves_existing_dataset(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    destination = tmp_path / "dataset"
    destination.mkdir()
    marker = destination / "train.txt"
    marker.write_text("existing split\n")

    with pytest.raises(FileExistsError, match="not empty"):
        organize_dataset(str(source), str(destination))

    assert marker.read_text() == "existing split\n"


def test_split_assignment_is_independent_of_directory_creation_order(tmp_path):
    outputs = []
    for index, names in enumerate((range(10), reversed(range(10)))):
        source = tmp_path / f"source_{index}"
        for name in names:
            directory = source / f"video_{name}"
            directory.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(directory / "frame_0000.png")
        destination = tmp_path / f"dataset_{index}"
        organize_dataset(str(source), str(destination))
        outputs.append([(destination / f"{split}.txt").read_text() for split in ("train", "val", "test")])

    assert outputs[0] == outputs[1]
    assert all(outputs[0])


def test_strict_validation_requires_vqgan_splits(tmp_path):
    config = tmp_path / "experiment.yaml"
    config.write_text(f"encoder: {{}}\ndata:\n  data_root: {tmp_path.as_posix()}\n")
    errors = validate_config(config)
    assert len(errors) == 3
    assert all("missing" in error and "split list" in error for error in errors)
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.txt").write_text(f"{split}/video_frame_0000.png 1\n")
    assert validate_config(config) == []


def test_quick_eval_forwards_selected_artifacts_and_options(tmp_path, monkeypatch):
    checkpoint = tmp_path / "selected.pth"
    config = tmp_path / "selected.yaml"
    checkpoint.touch()
    config.touch()
    monkeypatch.setattr(sys, "argv", [
        "quick_eval_test.py", "--checkpoint", str(checkpoint), "--config", str(config),
        "--steps", "7", "--num-samples", "2", "--output-dir", "chosen_output",
    ])
    calls = []
    def run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=4)
    monkeypatch.setattr(quick_eval_test.subprocess, "run", run)

    assert quick_eval_test.main() == 4
    command = calls[0]
    assert Path(command[1]).name == "evaluate_flow_matching.py"
    options = dict(zip(command[2::2], command[3::2]))
    assert options["--checkpoint"] == str(checkpoint)
    assert options["--config"] == str(config)
    assert options["--steps"] == "7"
    assert options["--num-samples"] == "2"
    assert options["--output-dir"] == "chosen_output"


def test_quick_eval_refuses_missing_checkpoint_before_launch(tmp_path, monkeypatch):
    config = tmp_path / "selected.yaml"
    config.touch()
    monkeypatch.setattr(sys, "argv", [
        "quick_eval_test.py", "--checkpoint", str(tmp_path / "missing.pth"), "--config", str(config),
    ])
    def unexpected_run(*args, **kwargs):
        pytest.fail("must not launch evaluation with a missing checkpoint")
    monkeypatch.setattr(quick_eval_test.subprocess, "run", unexpected_run)
    with pytest.raises(SystemExit) as error:
        quick_eval_test.main()
    assert error.value.code == 2


def test_extraction_without_preview_does_not_call_window_functions(tmp_path, monkeypatch):
    import cv2
    from prepare_data.prepare_dataset import generate_optimized_smoke_dataset
    source = tmp_path / "videos"
    source.mkdir()
    # A static readable video is sufficient to exercise the headless cleanup path.
    writer = cv2.VideoWriter(str(source / "static.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 7, (32, 32))
    assert writer.isOpened()
    import numpy as np
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    def window_call(*args, **kwargs):
        pytest.fail("headless extraction must not call OpenCV window functions")
    for name in ("imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, window_call)
    generate_optimized_smoke_dataset(str(source), str(tmp_path / "frames"), show_monitor=False)
