"""Batch-shared, mean-orthogonal simplex targets for final FE class tokens."""
import math

import torch
from torch import Tensor

from .raw_mean_etf_loss import _raw_mean_prototypes


@torch.no_grad()
def nearest_mu_orthogonal_simplex_etf(z: Tensor, mu: Tensor) -> Tensor:
    """Solve min ||Z-M||_F^2 over unit-Frobenius simplex ETFs with M mu=0.

    Q is a Helmert basis of the class-centered space, W spans mu-perp, and
    W.T @ Z.T @ Q = U S V.T. The constrained Procrustes optimum is
    M = Q @ (W @ U @ V.T).T / sqrt(C-1). Solving *inside* W also constrains
    SVD's arbitrary null-space completion when prototypes are rank deficient.
    The target is detached and recomputed on each call; there is no EMA.
    """
    if z.ndim != 2 or not z.is_floating_point():
        raise ValueError('Expected floating-point Z with shape [C,D].')
    classes, dimensions = z.shape
    if classes < 2 or dimensions < classes - 1:
        raise ValueError('Simplex ETF requires C >= 2 and D >= C - 1.')
    if (mu.shape != (dimensions,) or not mu.is_floating_point()
            or mu.device != z.device):
        raise ValueError('mu must be a floating-point [D] vector on Z.device.')
    with torch.autocast(device_type=z.device.type, enabled=False):
        z, mu = z.float(), mu.float()
        if not torch.isfinite(z).all() or not torch.isfinite(mu).all():
            raise ValueError('ETF geometry must be finite.')
        identity = torch.eye(dimensions, device=z.device, dtype=z.dtype)
        magnitude = mu.abs().max()
        if magnitude.item() == 0:
            complement = identity
        else:
            if dimensions < classes:
                raise ValueError('A nonzero mu requires D >= C for a simplex ETF.')
            # Rescale first: even a subnormal nonzero mu has a finite direction.
            direction = mu / magnitude
            direction = direction / torch.linalg.vector_norm(direction)
            reflector = direction.clone()
            reflector[0] += torch.copysign(torch.ones_like(direction[0]), direction[0])
            reflector = reflector / torch.linalg.vector_norm(reflector)
            complement = (identity - 2 * torch.outer(reflector, reflector))[:, 1:]

        rows = torch.arange(classes, device=z.device)[:, None]
        columns = torch.arange(classes - 1, device=z.device)[None, :]
        counts = (columns + 1).to(z.dtype)
        basis = ((rows <= columns).to(z.dtype)
                 - (rows == columns + 1).to(z.dtype) * counts)
        basis = basis / (counts * (counts + 1)).sqrt()
        u, _, vh = torch.linalg.svd(
            complement.T @ z.T @ basis, full_matrices=False)
        rotation = complement @ (u @ vh)
        return basis @ rotation.T / math.sqrt(classes - 1)


def batch_shared_mu_orth_token_etf_loss(
        memory_text: Tensor, class_token_maps, text_token_mask: Tensor,
        eps: float = 1e-6) -> Tensor:
    """Average token squared distance, equally weighted over images/classes.

    Pool raw class-name tokens per image, average those means over the batch,
    then compute a shared class mean mu and a single Frobenius scale s. All
    dataset classes participate, including those absent from GT. Only the ETF
    solve is detached: mu, s, and every selected FE token remain differentiable.
    No individual token/class normalization or averaging over D is performed.
    B is this forward's local minibatch (the launcher uses one GPU/process).
    """
    if not memory_text.is_floating_point():
        raise ValueError('memory_text must be floating point with shape [B,L,D].')
    with torch.autocast(device_type=memory_text.device.type, enabled=False):
        features = memory_text.float()
        # Reuse complete-class, unique-index and padding validation. Pooling
        # also remains in FP32 for double/half/bfloat16 inputs.
        prototypes = _raw_mean_prototypes(
            features, class_token_maps, text_token_mask, eps)
        if not torch.isfinite(prototypes).all():
            raise ValueError('Selected class tokens must be finite.')
        classes = prototypes.size(1)
        if classes == 1:
            # No inter-class simplex exists (e.g. FISH); retain a zero gradient.
            return prototypes.sum() * 0
        batch_prototypes = prototypes.mean(dim=0)
        mu = batch_prototypes.mean(dim=0)
        centered = batch_prototypes - mu
        scale = torch.linalg.vector_norm(centered).clamp_min(eps)
        target = nearest_mu_orthogonal_simplex_etf(centered / scale, mu)
        distances = []
        for b, mapping in enumerate(class_token_maps):
            for c in range(1, classes + 1):
                normalized_tokens = (features[b, mapping[c]] - mu) / scale
                distances.append(
                    (normalized_tokens - target[c - 1]).square().sum(dim=-1).mean())
        return torch.stack(distances).mean()
