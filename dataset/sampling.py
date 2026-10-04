"""Temporal window selection for frame sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def sample_temporal_indices(
    available_indices: Sequence[int],
    frames_per_sample: int,
    *,
    random_time: bool,
    stride: int = 1,
) -> list[int]:
    """Select a temporal window from one video's sorted frame indices.

    Raises instead of repeating the last frame when the clip is too short.
    """
    indices = list(available_indices)
    if frames_per_sample <= 0:
        raise ValueError("frames_per_sample must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    required = 1 + (frames_per_sample - 1) * stride
    if len(indices) < required:
        raise ValueError(
            f"Video has {len(indices)} available frames but needs {required} "
            f"for a {frames_per_sample}-frame window with stride {stride}"
        )

    max_start = len(indices) - required
    start = int(np.random.randint(0, max_start + 1)) if random_time and max_start else 0
    return [indices[start + offset * stride] for offset in range(frames_per_sample)]
