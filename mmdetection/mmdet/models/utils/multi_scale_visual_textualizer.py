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
                 num_attention_heads: int = 8,
                 sampling_ratio: int = 2) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError('output_size must be positive.')
        if len(feature_strides) != 4:
            raise ValueError('Exactly four feature strides are required.')
        if embed_dims % 2 != 0:
            raise ValueError('embed_dims must be even for 2D embeddings.')
        if num_attention_heads <= 0:
            raise ValueError('num_attention_heads must be positive.')
        if embed_dims % num_attention_heads != 0:
            raise ValueError(
                'embed_dims must be divisible by num_attention_heads.')

        self.output_size = output_size
        self.embed_dims = embed_dims
        self.num_levels = len(feature_strides)
        self.roi_align_layers = nn.ModuleList([
            RoIAlign(
                output_size=(output_size, output_size),
                spatial_scale=1.0 / stride,
                sampling_ratio=sampling_ratio,
                aligned=True) for stride in feature_strides
        ])
        position_dims = embed_dims // 2
        self.row_embedding = nn.Embedding(output_size, position_dims)
        self.column_embedding = nn.Embedding(output_size, position_dims)
        self.level_embedding = nn.Embedding(self.num_levels, embed_dims)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_attention_heads,
            dropout=0.0,
            batch_first=True,
        )

    def extract_instance_tokens(
            self, support_features: Tuple[Tensor],
            support_bboxes: Sequence[Tensor],
            support_labels: Sequence[Tensor],
            class_text_prototypes: Tensor) -> Tuple[Tensor, Tensor]:
        """Attend from class text to all scale-major RoI tokens."""
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
        if (class_text_prototypes.dim() != 2 or
                class_text_prototypes.size(1) != self.embed_dims):
            raise ValueError(
                'class_text_prototypes must have shape '
                f'[C, {self.embed_dims}].')
        if ((roi_labels < 0) |
                (roi_labels >= class_text_prototypes.size(0))).any():
            raise ValueError('Support labels are outside the prototype range.')

        rows = self.row_embedding.weight[:, None, :].expand(
            self.output_size, self.output_size, -1)
        columns = self.column_embedding.weight[None, :, :].expand(
            self.output_size, self.output_size, -1)
        spatial_positions = torch.cat((rows, columns), dim=-1).reshape(
            self.output_size * self.output_size, self.embed_dims)

        scale_major_tokens = []
        for level_idx, (feature,
                        roi_align) in enumerate(zip(support_features,
                                                  self.roi_align_layers)):
            if feature.dim() != 4 or feature.size(1) != self.embed_dims:
                raise ValueError(
                    f'Neck feature level {level_idx} must have shape '
                    f'[B, {self.embed_dims}, H, W], got '
                    f'{tuple(feature.shape)}.')
            roi_features = roi_align(feature, support_rois)
            roi_tokens = roi_features.flatten(2).transpose(1, 2)
            level_position = self.level_embedding.weight[level_idx]
            roi_tokens = (roi_tokens + spatial_positions[None, :, :] +
                          level_position[None, None, :])
            scale_major_tokens.append(roi_tokens)

        multi_scale_tokens = torch.cat(scale_major_tokens, dim=1)
        text_queries = class_text_prototypes[roi_labels].unsqueeze(1)
        attended_tokens, _ = self.cross_attention(
            query=text_queries,
            key=multi_scale_tokens,
            value=multi_scale_tokens,
            need_weights=False)
        instance_tokens = attended_tokens.squeeze(1)
        return instance_tokens, roi_labels

    def aggregate_by_class(self, instance_tokens: Tensor, labels: Tensor,
                           num_classes: int) -> Tensor:
        """Average instance tokens into class prototypes ``[C, 1, D]``."""
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

        return torch.stack(class_tokens, dim=0).unsqueeze(1)
