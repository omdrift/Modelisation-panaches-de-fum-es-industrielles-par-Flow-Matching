"""Temporal index selection for Flow Matching training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConditioningIndices:
    target: torch.Tensor
    reference: torch.Tensor
    condition: torch.Tensor
    distance: torch.Tensor


class ConditioningSampler:
    """Choose a target, its immediate predecessor, and an earlier condition."""

    def sample(self, batch_size: int, num_observations: int, device: torch.device) -> ConditioningIndices:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_observations < 3:
            raise ValueError("num_observations must be at least 3")

        target = torch.randint(2, num_observations, (batch_size,), device=device)
        reference = target - 1
        # Multiplying by (target - 1) gives the integer range [0, target - 2].
        condition = (torch.rand(batch_size, device=device) * (target - 1)).floor().long()
        return ConditioningIndices(
            target=target,
            reference=reference,
            condition=condition,
            distance=reference - condition,
        )
