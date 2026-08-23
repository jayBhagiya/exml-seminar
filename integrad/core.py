from collections.abc import Callable

import torch
from torch import Tensor


def integrated_gradients(
    score_fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    baseline: Tensor | None = None,
    *,
    steps: int = 64,
    batch_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Attribute one input to a scalar score using a straight-line path.

    ``score_fn`` receives path points as a batch and must return one scalar per
    point. The second return value is the signed completeness residual:
    ``F(inputs) - F(baseline) - attributions.sum()``.
    """
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating point")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    if baseline.shape != inputs.shape:
        raise ValueError("baseline and inputs must have the same shape")
    if baseline.device != inputs.device:
        raise ValueError("baseline and inputs must use the same device")
    if baseline.dtype != inputs.dtype:
        raise ValueError("baseline and inputs must use the same dtype")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    alphas = torch.linspace(0, 1, steps + 1, device=inputs.device, dtype=inputs.dtype)
    alpha_shape = (steps + 1,) + (1,) * inputs.ndim
    path = baseline.unsqueeze(0) + alphas.reshape(alpha_shape) * (
        inputs - baseline
    ).unsqueeze(0)

    gradients = []
    scores = []
    with torch.enable_grad():
        for points in path.split(batch_size or steps + 1):
            points = points.detach().requires_grad_(True)
            chunk_scores = score_fn(points)
            if chunk_scores.shape != (points.shape[0],):
                raise ValueError("score_fn must return one scalar per path point")
            gradients.append(
                torch.autograd.grad(chunk_scores.sum(), points)[0].detach()
            )
            scores.append(chunk_scores.detach())

    gradients = torch.cat(gradients)
    scores = torch.cat(scores)
    average_gradients = torch.trapezoid(gradients, dx=1 / steps, dim=0)
    attributions = (inputs - baseline) * average_gradients
    residual = scores[-1] - scores[0] - attributions.sum()
    return attributions, residual


def adaptive_integrated_gradients(
    score_fn: Callable[[Tensor], Tensor],
    inputs: Tensor,
    baseline: Tensor | None = None,
    *,
    steps: int = 32,
    max_steps: int = 1024,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 1e-4,
    batch_size: int | None = None,
) -> tuple[Tensor, Tensor, int, bool]:
    """Double path samples until completeness reaches tolerance or limit."""
    if max_steps < steps:
        raise ValueError("max_steps must be at least steps")

    while True:
        attributions, residual = integrated_gradients(
            score_fn,
            inputs,
            baseline,
            steps=steps,
            batch_size=batch_size,
        )
        score_delta = attributions.sum() + residual
        allowed_error = absolute_tolerance + relative_tolerance * abs(score_delta)
        converged = bool(abs(residual) <= allowed_error)
        if converged or steps >= max_steps:
            return attributions, residual, steps, converged
        steps = min(steps * 2, max_steps)
