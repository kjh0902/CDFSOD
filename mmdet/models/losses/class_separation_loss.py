"""GT-aware negative-only separation of final class-name token features."""
import math
from numbers import Integral

import torch
from torch import Tensor


def class_separation_loss(memory_text: Tensor, class_token_maps,
                          text_token_mask: Tensor, gt_labels,
                          temperature: float = 1.0) -> Tensor:
    """Average log(1 + sum_j exp(s(g,j) / temperature)) over GT anchors.

    Maps contain every dataset class with keys 1..C and zero-based token
    indices; GT labels are zero-based. Similarity is the mean raw dot product
    of all token pairs. Other GT classes are also negatives. Empty-GT images
    contribute a differentiable zero to the mean over the entire batch.
    """
    if memory_text.ndim != 3 or not memory_text.is_floating_point():
        raise ValueError('memory_text must be floating point with shape [B,L,D].')
    batch, length, dimensions = memory_text.shape
    if batch == 0 or length == 0 or dimensions == 0:
        raise ValueError('memory_text dimensions must be nonzero.')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive.')
    if (text_token_mask.shape != memory_text.shape[:2]
            or text_token_mask.dtype != torch.bool
            or text_token_mask.device != memory_text.device):
        raise ValueError(
            'text_token_mask must be a boolean [B,L] mask on the feature device.')
    if len(class_token_maps) != batch or len(gt_labels) != batch:
        raise ValueError('Expected one complete class-token map and GT tensor per sample.')
    if not isinstance(class_token_maps[0], dict) or not class_token_maps[0]:
        raise ValueError('Expected at least one dataset class.')
    classes = len(class_token_maps[0])

    with torch.autocast(device_type=memory_text.device.type, enabled=False):
        features = (memory_text if memory_text.dtype == torch.float64
                    else memory_text.float())
        samples = []
        for b, (mapping, labels) in enumerate(zip(class_token_maps, gt_labels)):
            if (not isinstance(mapping, dict)
                    or any(not isinstance(c, Integral) or isinstance(c, bool)
                           for c in mapping)
                    or set(mapping) != set(range(1, classes + 1))):
                raise ValueError('Every sample must map the same complete class IDs 1..C.')
            if (not isinstance(labels, Tensor) or labels.ndim != 1
                    or labels.dtype not in (torch.uint8, torch.int8, torch.int16,
                                            torch.int32, torch.int64)):
                raise ValueError(f'Sample {b}: GT labels must be a 1D integer tensor.')
            if ((labels < 0) | (labels >= classes)).any():
                raise ValueError(f'Sample {b}: GT labels must be in 0..C-1.')
            means = []
            for c in range(1, classes + 1):
                indices = mapping[c]
                if (not isinstance(indices, (list, tuple)) or not indices
                        or any(not isinstance(i, Integral) or isinstance(i, bool)
                               or i < 0 or i >= length for i in indices)
                        or len(set(indices)) != len(indices)):
                    raise ValueError(
                        f'Sample {b}, class {c}: expected nonempty, unique valid token indices.')
                if not text_token_mask[b, indices].all():
                    raise ValueError(f'Sample {b}, class {c}: class tokens point to padding.')
                means.append(features[b, indices].mean(dim=0))
            if labels.numel() == 0 or classes == 1:
                # An empty sum stays connected without summing large features.
                samples.append(features[b, :0].sum())
                continue
            means = torch.stack(means)
            anchors = labels.to(device=features.device, dtype=torch.long).unique()
            # mean_a,b(t_g,a dot t_j,b) == mean_a(t_g,a) dot mean_b(t_j,b).
            similarities = means[anchors] @ means.transpose(0, 1)
            negative_mask = (torch.arange(classes, device=features.device)[None, :]
                             != anchors[:, None])
            logits = similarities[negative_mask].reshape(-1, classes - 1) / temperature
            # The zero logit supplies the constant 1; self-similarity is excluded.
            logits = torch.cat((logits.new_zeros((len(anchors), 1)), logits), dim=1)
            samples.append(torch.logsumexp(logits, dim=1).mean())
        return torch.stack(samples).mean()
