from pathlib import Path

import pytest
import torch
from PIL import Image

from dataset.text_based_video_dataset import TextBasedVideoDataset


def _write_video(root: Path, split: str, name: str, colors: list[tuple[int, int, int]]):
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for frame_index, color in enumerate(colors):
        filename = f"{name}_frame_{frame_index:04d}.png"
        Image.new("RGB", (80, 72), color).save(directory / filename)
        names.append(f"{split}/{filename} 1")
    return names


def _dataset(root: Path, names: list[str], *, frames: int, random_time: bool = False):
    (root / "train.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return TextBasedVideoDataset(
        data_path=str(root),
        file_list="train.txt",
        input_size=64,
        crop_size=64,
        frames_per_sample=frames,
        random_horizontal_flip=False,
        random_time=random_time,
    )


def test_clip_has_requested_shape_and_never_crosses_videos(tmp_path):
    names = []
    names += _write_video(tmp_path, "train", "red_video", [(200, 20, 20)] * 4)
    names += _write_video(tmp_path, "train", "blue_video", [(20, 20, 200)] * 4)
    dataset = _dataset(tmp_path, names, frames=3)

    assert len(dataset) == 2
    for clip in (dataset[0], dataset[1]):
        assert clip.shape == (3, 3, 64, 64)
        assert torch.all(clip >= -1) and torch.all(clip <= 1)
        means = clip.mean(dim=(2, 3))
        assert torch.allclose(means, means[:1].expand_as(means), atol=1e-5)


def test_short_video_raises_instead_of_repeating_the_last_frame(tmp_path):
    names = _write_video(tmp_path, "train", "short_video", [(90, 90, 90)] * 2)
    dataset = _dataset(tmp_path, names, frames=3)

    with pytest.raises(ValueError, match="available frames but needs"):
        dataset[0]


def test_flow_matching_loader_can_drop_short_videos(tmp_path):
    names = []
    names += _write_video(tmp_path, "train", "short_video", [(90, 90, 90)] * 2)
    names += _write_video(tmp_path, "train", "long_video", [(50, 100, 150)] * 4)
    (tmp_path / "train.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    dataset = TextBasedVideoDataset(
        data_path=str(tmp_path),
        file_list="train.txt",
        input_size=64,
        crop_size=64,
        frames_per_sample=3,
        random_horizontal_flip=False,
        random_time=False,
        skip_short_videos=True,
    )

    assert len(dataset) == 1
    assert dataset[0].shape[0] == 3
