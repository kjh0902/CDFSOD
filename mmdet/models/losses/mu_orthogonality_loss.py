"""Orthogonality between the class mean and each class residual."""
import math

import torch
from torch import Tensor


def mu_orthogonality_loss(prototypes: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean squared cosine of mu and p_c - mu over images and all classes.

    Neither branch is detached. Compute in FP32 outside autocast, including
    the norms; additive epsilon keeps zero means/residuals finite. A single
    class has a zero residual and therefore a differentiable zero loss.
    """
    if (prototypes.ndim != 3 or not prototypes.is_floating_point()
            or any(size == 0 for size in prototypes.shape)):
        raise ValueError('Expected floating-point prototypes with shape [B,C,D] '
                         'and nonzero dimensions.')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive.')
    with torch.autocast(device_type=prototypes.device.type, enabled=False):
        x = prototypes.float()
        mu = x.mean(dim=1, keepdim=True)
        residuals = x - mu
        numerator = (mu * residuals).sum(dim=-1)
        denominator = (torch.linalg.vector_norm(mu, dim=-1)
                       * torch.linalg.vector_norm(residuals, dim=-1) + eps)
        return (numerator / denominator).square().mean()
