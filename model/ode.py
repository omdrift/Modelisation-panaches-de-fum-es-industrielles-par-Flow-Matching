"""Small RK4 ODE solver fallback for environments without torchdiffeq."""

from __future__ import annotations

import torch


def odeint(func, y0: torch.Tensor, t: torch.Tensor, method: str = "rk4") -> torch.Tensor:
    if method != "rk4":
        raise ValueError(f"Fallback ODE solver supports rk4, received {method!r}")
    if t.ndim != 1 or t.numel() < 2:
        raise ValueError("RK4 integration requires a one-dimensional time tensor with at least two values")

    values = [y0]
    y = y0
    for index in range(t.numel() - 1):
        t0 = t[index]
        dt = t[index + 1] - t0
        half_dt = dt / 2
        k1 = func(t0, y)
        k2 = func(t0 + half_dt, y + half_dt * k1)
        k3 = func(t0 + half_dt, y + half_dt * k2)
        k4 = func(t0 + dt, y + dt * k3)
        y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        values.append(y)
    return torch.stack(values)
