# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from torch import Tensor


class TextualizedVisualTokenGenerator(nn.Module):
    """Generate one visual token for each RoI from four feature levels."""

    def __init__(self,
                 featmap_strides: Sequence[int] = (8, 16, 32, 64),
                 output_size: Tuple[int, int] = (7, 7),
                 in_channels: int = 256,
                 token_dim: int = 768) -> None:
        super().__init__()
        if len(featmap_strides) != 4:
            raise ValueError('Exactly four feature strides are required.')

        self.in_channels = in_channels
        self.roi_align_layers = nn.ModuleList([
            RoIAlign(
                output_size=output_size,
                spatial_scale=1.0 / stride,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True) for stride in featmap_strides
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(in_channels, token_dim)

    def forward(self, features: Sequence[Tensor], rois: Tensor) -> Tensor:
        if len(features) != 4:
            raise ValueError(
                f'Expected four feature levels, but got {len(features)}.')
        if any(feature.size(1) != self.in_channels for feature in features):
            raise ValueError(
                f'All feature levels must have {self.in_channels} channels.')
        if rois.ndim != 2 or rois.size(-1) != 5:
            raise ValueError('RoIs must have shape [N, 5].')

        rois = rois.type_as(features[0])
        roi_features = [
            roi_align(feature, rois)
            for roi_align, feature in zip(self.roi_align_layers, features)
        ]
        level_features = [
            self.global_avg_pool(roi_feature).flatten(1)
            for roi_feature in roi_features
        ]
        averaged = torch.stack(level_features, dim=0).mean(dim=0)
        averaged = averaged.to(self.projection.weight.dtype)
        return self.projection(averaged)
