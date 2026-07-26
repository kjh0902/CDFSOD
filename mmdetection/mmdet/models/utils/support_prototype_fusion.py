# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from torch import Tensor

from mmdet.structures.bbox import bbox2roi


class SupportPrototypeFusion(nn.Module):
    """Create and fuse single-scale support visual prototypes.

    A single self-attention module is reused for the text and visual branches,
    so both branches are transformed by the exact same parameters.
    """

    def __init__(self,
                 output_size: int = 7,
                 feature_stride: int = 8,
                 embed_dims: int = 256,
                 num_attention_heads: int = 8,
                 sampling_ratio: int = 2,
                 text_weight: float = 0.5,
                 visual_weight: float = 0.5) -> None:
        super().__init__()
        if output_size <= 0:
            raise ValueError('output_size must be positive.')
        if feature_stride <= 0:
            raise ValueError('feature_stride must be positive.')
        if num_attention_heads <= 0:
            raise ValueError('num_attention_heads must be positive.')
        if embed_dims % num_attention_heads != 0:
            raise ValueError(
                'embed_dims must be divisible by num_attention_heads.')
        assert abs(text_weight + visual_weight - 1.0) < 1e-6, \
            'text_weight and visual_weight must sum to 1.'

        self.embed_dims = embed_dims
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=1.0 / feature_stride,
            sampling_ratio=sampling_ratio,
            aligned=True)
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_attention_heads,
            dropout=0.0,
            batch_first=True)

    def extract_instance_features(
            self, support_feature: Tensor,
            support_bboxes: Sequence[Tensor],
            support_labels: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        """RoIAlign and spatially average support objects into ``[N, D]``."""
        if support_feature.dim() != 4:
            raise ValueError('support_feature must have shape [B, D, H, W].')
        if support_feature.size(1) != self.embed_dims:
            raise ValueError(
                f'Support feature channels must be {self.embed_dims}, got '
                f'{support_feature.size(1)}.')
        if len(support_bboxes) != len(support_labels):
            raise ValueError('support_bboxes and support_labels must have the '
                             'same batch length.')
        if len(support_bboxes) != support_feature.size(0):
            raise ValueError('Support GT batch length must match the support '
                             'feature batch size.')

        support_rois = bbox2roi(list(support_bboxes)).type_as(support_feature)
        if support_rois.numel() == 0:
            raise ValueError('No support GT bbox was provided.')
        labels = torch.cat(list(support_labels), dim=0).to(
            device=support_feature.device, dtype=torch.long)
        if labels.numel() != support_rois.size(0):
            raise ValueError('The support label and RoI counts must match.')

        roi_features = self.roi_align(support_feature, support_rois)
        instance_features = roi_features.mean(dim=(-2, -1))
        return instance_features, labels

    @staticmethod
    def aggregate_by_class(instance_features: Tensor, labels: Tensor,
                           num_classes: int) -> Tensor:
        """Average a variable number of instances into ``[C, D]``."""
        if instance_features.dim() != 2:
            raise ValueError('instance_features must have shape [N, D].')
        if instance_features.size(0) != labels.numel():
            raise ValueError('Support feature and label counts must match.')
        if ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError('Support labels are outside the class range.')

        class_prototypes = []
        missing_classes = []
        for class_idx in range(num_classes):
            class_features = instance_features[labels == class_idx]
            if class_features.size(0) == 0:
                missing_classes.append(class_idx)
                continue
            class_prototypes.append(class_features.mean(dim=0))
        if missing_classes:
            raise ValueError(
                'Every class needs at least one support instance. Missing '
                f'class indices: {missing_classes}.')
        return torch.stack(class_prototypes, dim=0)

    def align_and_fuse(
            self, text_prototypes: Tensor,
            visual_prototypes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Align both branches with shared attention and fuse them 1:1."""
        if (text_prototypes.dim() != 2 or
                text_prototypes.size(1) != self.embed_dims):
            raise ValueError(
                f'text_prototypes must have shape [C, {self.embed_dims}].')
        if visual_prototypes.shape != text_prototypes.shape:
            raise ValueError('visual_prototypes must match text_prototypes.')

        text_sequence = text_prototypes.unsqueeze(0)
        visual_sequence = visual_prototypes.unsqueeze(0)
        aligned_text, _ = self.shared_attention(
            text_sequence, text_sequence, text_sequence,
            need_weights=False)
        aligned_visual, _ = self.shared_attention(
            visual_sequence, visual_sequence, visual_sequence,
            need_weights=False)
        aligned_text = aligned_text.squeeze(0)
        aligned_visual = aligned_visual.squeeze(0)
        fused = (self.text_weight * aligned_text +
                 self.visual_weight * aligned_visual)
        return aligned_text, aligned_visual, fused
