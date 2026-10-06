"""Raw final-FE prototypes anchored at background, with a detached ETF solve.

Only the canonical simplex is centered. Neither the feature matrix nor the
row-scaled target is centered, and no individual token/row is normalized.
"""
import math
from numbers import Integral, Real

import torch
from torch import Tensor


def validate_nonnegative_scalar(value, name: str) -> None:
    if (not isinstance(value, Real) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative scalar.')


def _precision(features: Tensor) -> Tensor:
    return features if features.dtype == torch.float64 else features.float()


def _validate_geometry(prototypes: Tensor, presence: Tensor) -> None:
    if prototypes.ndim != 3 or not prototypes.is_floating_point():
        raise ValueError('Expected floating-point prototypes [B,C,D].')
    batch, classes, dimensions = prototypes.shape
    if batch == 0 or classes == 0 or dimensions < max(1, classes - 1):
        raise ValueError('Requires B > 0, C >= 1 and D >= max(1,C-1).')
    if (presence.shape != prototypes.shape[:2]
            or presence.dtype != torch.bool
            or presence.device != prototypes.device):
        raise ValueError('presence must be boolean [B,C] on the feature device.')
    if not torch.isfinite(prototypes).all():
        raise ValueError('Prototypes must be finite.')


def class_presence(gt_labels, classes: int, device) -> Tensor:
    """Binary presence from the image's current (post-augmentation) GT labels."""
    presence = torch.zeros(len(gt_labels), classes, dtype=torch.bool,
                           device=device)
    for b, labels in enumerate(gt_labels):
        if (not isinstance(labels, Tensor) or labels.ndim != 1
                or labels.dtype not in (torch.int8, torch.int16, torch.int32,
                                        torch.int64, torch.uint8)):
            raise ValueError('GT labels must be one-dimensional integer tensors.')
        if ((labels < 0) | (labels >= classes)).any():
            raise ValueError('GT label is outside foreground class IDs 0..C-1.')
        presence[b, labels.to(device=device, dtype=torch.long)] = True
    return presence


def raw_mean_prototypes(memory_text: Tensor, class_token_maps,
                        background_token_indices,
                        text_token_mask: Tensor) -> tuple:
    """Pool raw class-name subwords into foreground [B,C,D] and bg [B,D]."""
    if (memory_text.ndim != 3 or not memory_text.is_floating_point()
            or min(memory_text.shape) == 0):
        raise ValueError('memory_text must be nonempty floating point [B,L,D].')
    batch, length, _ = memory_text.shape
    if (text_token_mask.shape != memory_text.shape[:2]
            or text_token_mask.dtype != torch.bool
            or text_token_mask.device != memory_text.device):
        raise ValueError('text_token_mask must be boolean [B,L] on the feature device.')
    if (len(class_token_maps) != batch
            or len(background_token_indices) != batch):
        raise ValueError('Expected foreground/background token maps per sample.')
    classes = len(class_token_maps[0])
    if classes == 0:
        raise ValueError('At least one foreground class is required.')

    def mean_tokens(features, b, indices):
        if (not indices or len(set(indices)) != len(indices)
                or any(not isinstance(i, Integral) or isinstance(i, bool)
                       or i < 0 or i >= length for i in indices)):
            raise ValueError('Expected nonempty, unique, valid class-name token indices.')
        if not text_token_mask[b, indices].all():
            raise ValueError('Class-name tokens point to padding.')
        return features[b, indices].mean(dim=0)

    with torch.autocast(device_type=memory_text.device.type, enabled=False):
        features = _precision(memory_text)
        foreground, background = [], []
        for b, mapping in enumerate(class_token_maps):
            if (not isinstance(mapping, dict)
                    or set(mapping) != set(range(1, classes + 1))):
                raise ValueError('Every sample must map all foreground IDs 1..C.')
            if set(background_token_indices[b]).intersection(
                    i for indices in mapping.values() for i in indices):
                raise ValueError('Background and foreground token maps must be disjoint.')
            foreground.append(torch.stack([
                mean_tokens(features, b, mapping[c])
                for c in range(1, classes + 1)]))
            background.append(mean_tokens(features, b, background_token_indices[b]))
        return torch.stack(foreground), torch.stack(background)


@torch.no_grad()
def nearest_deformed_etf(normalized_prototypes: Tensor, presence: Tensor,
                         alpha: float = 0.5) -> Tensor:
    """Exact nearest row-deformed simplex target [B,C,D], detached.

    For M = Q_hat^T A_hat = U Sigma V^T, R = U V^T satisfies R^T R = I
    and maximizes tr(R^T M). The Helmert basis removes the simplex's redundant
    null direction, including at the minimal embedding dimension D = C-1.
    Rank-deficient M admits multiple optima; thin SVD chooses a valid one.
    """
    _validate_geometry(normalized_prototypes, presence)
    validate_nonnegative_scalar(alpha, 'alpha')
    classes = normalized_prototypes.size(1)
    if classes < 2:
        raise ValueError('A simplex ETF target requires C >= 2.')
    with torch.autocast(device_type=normalized_prototypes.device.type,
                        enabled=False):
        q = _precision(normalized_prototypes)
        rows = torch.arange(classes, device=q.device)[:, None]
        columns = torch.arange(classes - 1, device=q.device)[None, :]
        counts = (columns + 1).to(q.dtype)
        basis = ((rows <= columns).to(q.dtype)
                 - (rows == columns + 1).to(q.dtype) * counts)
        basis = basis / (counts * (counts + 1)).sqrt()
        canonical = basis / math.sqrt(classes - 1)
        scales = 1.0 + float(alpha) * presence.to(q.dtype)
        deformed = scales.unsqueeze(-1) * canonical
        deformed = deformed / torch.linalg.vector_norm(
            deformed, dim=(-2, -1), keepdim=True)
        u, _, vh = torch.linalg.svd(
            q.transpose(-2, -1) @ deformed, full_matrices=False)
        rotation = u @ vh
        return deformed @ rotation.transpose(-2, -1)


def background_anchored_prototype_loss(foreground: Tensor, background: Tensor,
                                       presence: Tensor, alpha: float = 0.5,
                                       eps: float = 1e-6) -> Tensor:
    """Mean squared Frobenius distance; gradients include the background."""
    _validate_geometry(foreground, presence)
    validate_nonnegative_scalar(alpha, 'alpha')
    if (not isinstance(eps, Real) or isinstance(eps, bool)
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError('eps must be a finite positive scalar.')
    if (background.shape != (foreground.size(0), foreground.size(2))
            or background.device != foreground.device
            or not background.is_floating_point()
            or not torch.isfinite(background).all()):
        raise ValueError('background must be finite floating point [B,D] on the feature device.')
    with torch.autocast(device_type=foreground.device.type, enabled=False):
        q = _precision(foreground) - _precision(background).unsqueeze(1)
        if q.size(1) == 1:
            # No 0-dimensional simplex: retain a differentiable zero for FISH.
            return q.sum() * 0.0
        q_hat = q / torch.linalg.vector_norm(
            q, dim=(-2, -1), keepdim=True).clamp_min(eps)
        target = nearest_deformed_etf(q_hat, presence, alpha)
        return (q_hat - target).square().sum(dim=(-2, -1)).mean()


def background_anchored_etf_loss(memory_text: Tensor, class_token_maps,
                                 background_token_indices,
                                 text_token_mask: Tensor, gt_labels,
                                 alpha: float = 0.5,
                                 eps: float = 1e-6) -> Tensor:
    """Apply the geometry only to final FE class-name raw means."""
    foreground, background = raw_mean_prototypes(
        memory_text, class_token_maps, background_token_indices, text_token_mask)
    presence = class_presence(gt_labels, foreground.size(1), foreground.device)
    return background_anchored_prototype_loss(
        foreground, background, presence, alpha, eps)
