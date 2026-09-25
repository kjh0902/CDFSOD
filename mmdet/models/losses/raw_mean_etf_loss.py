"""Raw mean class prototypes from final, token-level enhancer features."""
import math
from numbers import Integral

import torch
from torch import Tensor

from .nearest_etf_loss import nearest_etf_loss


def _raw_mean_prototypes(memory_text: Tensor, class_token_maps,
                         text_token_mask: Tensor,
                         eps: float = 1e-6) -> Tensor:
    """Build [B,C,D] raw token means without token or class normalization.

    Maps use the existing get_positive_map convention: class keys 1..C and
    zero-based token indices. Each map must contain every dataset class.
    """
    if memory_text.ndim != 3 or not memory_text.is_floating_point():
        raise ValueError(
            'memory_text must be floating point with shape [B,L,D].')
    batch, length, dimensions = memory_text.shape
    if batch == 0 or length == 0 or dimensions == 0:
        raise ValueError('memory_text dimensions must be nonzero.')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive.')
    if (text_token_mask.shape != memory_text.shape[:2]
            or text_token_mask.dtype != torch.bool
            or text_token_mask.device != memory_text.device):
        raise ValueError(
            'text_token_mask must be a boolean [B,L] mask on the feature device.')
    if len(class_token_maps) != batch:
        raise ValueError('Expected one complete class-token map per sample.')
    classes = len(class_token_maps[0])
    if classes < 2 or dimensions < classes - 1:
        raise ValueError('Raw mean ETF requires C >= 2 and D >= C - 1.')

    with torch.autocast(device_type=memory_text.device.type, enabled=False):
        features = (memory_text if memory_text.dtype == torch.float64
                    else memory_text.float())
        samples = []
        for b, mapping in enumerate(class_token_maps):
            if (not isinstance(mapping, dict)
                    or set(mapping) != set(range(1, classes + 1))):
                raise ValueError(
                    'Every sample must map the same complete class IDs 1..C.')
            prototypes = []
            for c in range(1, classes + 1):
                indices = mapping[c]
                if (not indices
                        or any(not isinstance(i, Integral) or isinstance(i, bool)
                               or i < 0 or i >= length for i in indices)
                        or len(set(indices)) != len(indices)):
                    raise ValueError(
                        f'Sample {b}, class {c}: expected nonempty, unique '
                        'valid token indices.')
                if not text_token_mask[b, indices].all():
                    raise ValueError(
                        f'Sample {b}, class {c}: class tokens point to padding.')
                tokens = features[b, indices]
                prototypes.append(tokens.mean(dim=0))
            samples.append(torch.stack(prototypes))
        return torch.stack(samples)


def raw_mean_etf_loss(memory_text: Tensor, class_token_maps,
                      text_token_mask: Tensor,
                      eps: float = 1e-6) -> Tensor:
    """Mean independent per-image ETF distance, with a detached SVD target.

    Raw token means, class centering and whole-matrix Frobenius normalization
    remain differentiable. No token or individual class vector is normalized.
    The original memory_text and detection mappings are never modified.
    """
    prototypes = _raw_mean_prototypes(
        memory_text, class_token_maps, text_token_mask, eps)
    return nearest_etf_loss(prototypes, eps=eps)
