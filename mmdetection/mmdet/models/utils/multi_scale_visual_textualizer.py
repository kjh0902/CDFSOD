# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from torch import Tensor

from mmdet.structures.bbox import bbox2roi


class MultiScaleVisualTextualizer(nn.Module):
    """Convert support RoIs from a feature pyramid into visual tokens."""

    def __init__(self,
                 output_size: int = 7,
                 feature_strides: Sequence[int] = (8, 16, 32, 64),
                 embed_dims: int = 256,
                 spatial_hidden_dim: int = 128,
                 sampling_ratio: int = 2) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError('output_size must be positive.')
        if len(feature_strides) != 4:
            raise ValueError('Exactly four feature strides are required.')
        if spatial_hidden_dim <= 0:
            raise ValueError('spatial_hidden_dim must be positive.')

        self.output_size = output_size
        self.embed_dims = embed_dims
        self.roi_align_layers = nn.ModuleList([
            RoIAlign(
                output_size=(output_size, output_size),
                spatial_scale=1.0 / stride,
                sampling_ratio=sampling_ratio,
                aligned=True) for stride in feature_strides
        ])
        spatial_dim = output_size * output_size
        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_hidden_dim, 1),
        )

    def extract_instance_tokens(
            self, support_features: Tuple[Tensor],
            support_bboxes: Sequence[Tensor],
            support_labels: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        """RoIAlign four levels and return one max-pooled token per RoI."""
        if len(support_features) != len(self.roi_align_layers):
            raise ValueError(
                f'Expected {len(self.roi_align_layers)} neck features, got '
                f'{len(support_features)}.')
        if len(support_bboxes) != len(support_labels):
            raise ValueError('support_bboxes and support_labels must have the '
                             'same batch length.')
        if len(support_bboxes) != support_features[0].size(0):
            raise ValueError('Support GT batch length must match the support '
                             'feature batch size.')

        support_rois = bbox2roi(list(support_bboxes)).type_as(
            support_features[0])
        if support_rois.numel() == 0:
            raise ValueError('No support GT bbox was provided.')
        roi_labels = torch.cat(list(support_labels), dim=0).to(
            device=support_features[0].device, dtype=torch.long)
        if roi_labels.numel() != support_rois.size(0):
            raise ValueError('The number of support labels must match the '
                             'number of support RoIs.')

        scale_tokens = []
        for level_idx, (feature,
                        roi_align) in enumerate(zip(support_features,
                                                  self.roi_align_layers)):
            if feature.dim() != 4 or feature.size(1) != self.embed_dims:
                raise ValueError(
                    f'Neck feature level {level_idx} must have shape '
                    f'[B, {self.embed_dims}, H, W], got '
                    f'{tuple(feature.shape)}.')
            roi_features = roi_align(feature, support_rois)
            flattened = roi_features.flatten(2)
            scale_tokens.append(
                self.spatial_projection(flattened).squeeze(-1))

        instance_tokens = torch.stack(scale_tokens, dim=0).amax(dim=0)
        return instance_tokens, roi_labels

    def aggregate_by_class(self, instance_tokens: Tensor, labels: Tensor,
                           num_classes: int) -> Tensor:
        """Average all instance tokens of each class into ``[C, D]``."""
        if instance_tokens.dim() != 2:
            raise ValueError('instance_tokens must have shape [N, D].')
        if instance_tokens.size(0) != labels.numel():
            raise ValueError('Visual token and label counts must match.')
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError('Support labels are outside the class range.')

        class_tokens = []
        shots_per_class = []
        for class_idx in range(num_classes):
            tokens = instance_tokens[labels == class_idx]
            if tokens.size(0) > 0:
                class_tokens.append(tokens.mean(dim=0))
            shots_per_class.append(tokens.size(0))
        if any(num_shots == 0 for num_shots in shots_per_class):
            raise ValueError(
                f'Every class needs at least one support RoI, got '
                f'{shots_per_class}.')

        return torch.stack(class_tokens, dim=0)
