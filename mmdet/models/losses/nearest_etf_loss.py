"""Detached nearest-simplex ETF geometry, using only PyTorch.

The non-proximal objective in Guiding-Neural-Collapse's nc/ETF_distance.py
and nc/models/ddn_modules.py is ||Y - P H / sqrt(C - 1)||_F^2, where
H = I - 11^T/C and P^T P = I. The target norm is constant, so this is an
orthogonal Procrustes problem, solved exactly by SVD. No DDN is needed.

Reference: https://github.com/evanmarkou/Guiding-Neural-Collapse
"""
import math

import torch
from torch import Tensor


def _validate_prototypes(prototypes: Tensor) -> None:
    if prototypes.ndim != 3:
        raise ValueError('Expected prototypes with shape [B, C, D].')
    batch, classes, dimensions = prototypes.shape
    if batch == 0 or classes < 2 or dimensions < classes - 1:
        raise ValueError(
            'Nearest simplex ETF requires B > 0, C >= 2 and D >= C - 1.')
    if not prototypes.is_floating_point():
        raise ValueError('Prototypes must be floating-point tensors.')


def _geometry_precision(prototypes: Tensor) -> Tensor:
    return prototypes if prototypes.dtype == torch.float64 else prototypes.float()


@torch.no_grad()
def nearest_simplex_etf(normalized_prototypes: Tensor) -> Tensor:
    """Return a detached nearest unit-Frobenius simplex ETF of shape [B,C,D].

    Let Q be a Helmert basis of the class-centered subspace: Q^T Q = I and
    QQ^T = H. Write every simplex ETF as Q R^T / sqrt(C - 1), R^T R = I.
    For Z^T Q = U S V^T the optimum is R = U V^T. This removes H's redundant
    null direction and also supports the minimal dimension D = C - 1.
    Rank-deficient inputs can have multiple optima; SVD selects a valid one.
    """
    _validate_prototypes(normalized_prototypes)
    with torch.autocast(device_type=normalized_prototypes.device.type,
                        enabled=False):
        z = _geometry_precision(normalized_prototypes)
        classes = z.size(1)
        # Column j has j+1 equal positive entries, then -(j+1), then zeros.
        rows = torch.arange(classes, device=z.device)[:, None]
        columns = torch.arange(classes - 1, device=z.device)[None, :]
        counts = (columns + 1).to(z.dtype)
        basis = ((rows <= columns).to(z.dtype)
                 - (rows == columns + 1).to(z.dtype) * counts)
        basis = basis / (counts * (counts + 1)).sqrt()
        u, _, vh = torch.linalg.svd(
            z.transpose(-2, -1) @ basis, full_matrices=False)
        rotation = u @ vh
        return (basis @ rotation.transpose(-2, -1)) / math.sqrt(classes - 1)


def nearest_etf_loss(prototypes: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean per-sample squared Frobenius distance to the nearest simplex ETF.

    Center across classes and normalize the *whole* matrix, never individual
    class vectors. Gradients flow through this normalization but not through
    the target solve. Clamp makes collapsed inputs finite (their target is
    non-unique); no target or prototype is cached between calls.
    """
    _validate_prototypes(prototypes)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive.')
    with torch.autocast(device_type=prototypes.device.type, enabled=False):
        x = _geometry_precision(prototypes)
        centered = x - x.mean(dim=1, keepdim=True)
        norm = torch.linalg.vector_norm(centered, dim=(-2, -1), keepdim=True)
        normalized = centered / norm.clamp_min(eps)
        target = nearest_simplex_etf(normalized)
        return (normalized - target).square().sum(dim=(-2, -1)).mean()
