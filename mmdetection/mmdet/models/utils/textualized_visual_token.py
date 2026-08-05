# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from mmengine.logging import MMLogger
from torch import Tensor


class TextualizedVisualTokenGenerator(nn.Module):
    """Generate one visual token for each RoI from four feature levels."""

    def __init__(self,
                 featmap_strides: Sequence[int] = (8, 16, 32, 64),
                 output_size: Tuple[int, int] = (7, 7),
                 in_channels: int = 256,
                 token_dim: int = 768,
                 log_shapes: bool = True) -> None:
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
        self.pool_1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.projection = nn.Linear(in_channels * 5, token_dim)
        self.log_shapes = log_shapes
        self._shapes_logged = False

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
        stacked = torch.stack(roi_features, dim=1)
        max_pooled = stacked.max(dim=1).values

        pooled_1x1 = self.pool_1x1(max_pooled)
        pooled_2x2 = self.pool_2x2(max_pooled)
        flattened_1x1 = pooled_1x1.flatten(1)
        flattened_2x2 = pooled_2x2.flatten(1)
        concatenated = torch.cat([flattened_1x1, flattened_2x2], dim=1)
        concatenated = concatenated.to(self.projection.weight.dtype)
        tokens = self.projection(concatenated)

        if self.log_shapes and not self._shapes_logged:
            shapes = dict(
                features=[tuple(feature.shape) for feature in features],
                rois=tuple(rois.shape),
                roi_features=[tuple(feature.shape) for feature in roi_features],
                stacked=tuple(stacked.shape),
                level_max=tuple(max_pooled.shape),
                pooled_1x1=tuple(pooled_1x1.shape),
                pooled_2x2=tuple(pooled_2x2.shape),
                flattened_1x1=tuple(flattened_1x1.shape),
                flattened_2x2=tuple(flattened_2x2.shape),
                concatenated=tuple(concatenated.shape),
                tokens=tuple(tokens.shape))
            MMLogger.get_current_instance().info(
                f'Textualized visual token shapes: {shapes}')
            self._shapes_logged = True

        return tokens
