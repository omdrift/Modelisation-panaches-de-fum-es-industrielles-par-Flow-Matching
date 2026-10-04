"""Shared image transforms used by the frame-based datasets."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms as T


def build_frame_transform(input_size: int, crop_size: int):
    return T.Compose(
        [
            T.Resize(size=input_size, antialias=True),
            T.CenterCrop(size=crop_size),
        ]
    )


def load_frame(path: str, transform, *, horizontal_flip: bool) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = T.functional.to_tensor(image)
    tensor = transform(tensor)
    if horizontal_flip:
        tensor = T.functional.hflip(tensor)
    return tensor.mul(2.0).sub(1.0)
