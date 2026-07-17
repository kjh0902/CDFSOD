# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from torch import Tensor

from mmdet.structures.bbox import bbox2roi


class SupportImagePrototype(nn.Module):
    """Build class-wise image prototypes from Grounding DINO features.

    The first Grounding DINO neck feature is used because it has 256 channels
    and stride 8 in the CDFSOD configuration.
    """

    def __init__(self,
                 output_size: int = 7,
                 featmap_stride: int = 8,
                 embed_dims: int = 256,
                 sampling_ratio: int = 2,
                 temperature: int = 10000) -> None:
        super().__init__()
        self.output_size = output_size
        self.embed_dims = embed_dims
        self.temperature = temperature
        if embed_dims % 4 != 0:
            raise ValueError('embed_dims must be divisible by 4 for 2D sine '
                             'positional encoding.')
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=1.0 / featmap_stride,
            sampling_ratio=sampling_ratio,
            aligned=True)

    @staticmethod
    def _print_shape(name: str, tensor: Tensor) -> None:
        print(f'{name}: {tuple(tensor.shape)}')

    def _build_2d_positional_encoding(self, reference: Tensor) -> Tensor:
        """Create fixed 2D sine/cosine encoding with shape [1, HW, D]."""
        height = width = self.output_size
        num_pos_feats = self.embed_dims // 2
        y_embed = torch.arange(
            1, height + 1, dtype=torch.float32,
            device=reference.device).view(height, 1).expand(height, width)
        x_embed = torch.arange(
            1, width + 1, dtype=torch.float32,
            device=reference.device).view(1, width).expand(height, width)
        scale = 2 * math.pi
        y_embed = y_embed / height * scale
        x_embed = x_embed / width * scale

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=reference.device)
        dim_t = self.temperature**(
            2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=-1).flatten(-2)
        position_encoding_2d = torch.cat(
            (pos_y, pos_x), dim=-1).reshape(1, height * width,
                                            self.embed_dims)
        return position_encoding_2d.to(dtype=reference.dtype)

    def forward(self,
                support_features: Tuple[Tensor],
                support_bboxes: Sequence[Tensor],
                support_labels: Sequence[Tensor],
                num_classes: int) -> Tensor:
        """Generate position-enriched support image prototypes.

        Args:
            support_features: Grounding DINO neck feature pyramid.
            support_bboxes: GT boxes for each support image.
            support_labels: GT class labels for each support image.
            num_classes: Total number of support classes.

        Returns:
            Tensor: Image prototypes with shape [C, 49, 256].
        """
        if not support_features:
            raise ValueError('support_features must not be empty.')
        if num_classes <= 0:
            raise ValueError('num_classes must be greater than zero.')
        support_feature_map = support_features[0]
        if support_feature_map.size(1) != self.embed_dims:
            raise ValueError(
                f'Expected {self.embed_dims} support feature channels, got '
                f'{support_feature_map.size(1)}.')
        if len(support_bboxes) != len(support_labels):
            raise ValueError('support_bboxes and support_labels must have the '
                             'same batch length.')
        if len(support_bboxes) != support_feature_map.size(0):
            raise ValueError(
                'GT batch length must match support feature batch size.')

        support_rois = bbox2roi(list(support_bboxes)).type_as(
            support_feature_map)
        if support_rois.numel() == 0:
            raise ValueError('No support GT bbox was provided.')
        roi_labels = torch.cat(list(support_labels), dim=0).to(
            device=support_feature_map.device, dtype=torch.long)
        if roi_labels.numel() != support_rois.size(0):
            raise ValueError('The number of support GT labels must match the '
                             'number of support GT bboxes.')
        if ((roi_labels < 0) | (roi_labels >= num_classes)).any():
            raise ValueError('Support GT labels must be in the range '
                             f'[0, {num_classes - 1}].')

        self._print_shape('support_feature_map', support_feature_map)
        self._print_shape('support_rois', support_rois)
        roi_aligned_features = self.roi_align(support_feature_map,
                                              support_rois)
        self._print_shape('roi_aligned_features', roi_aligned_features)

        class_shot_mean_features: List[Tensor] = []
        missing_classes = []
        for class_idx in range(num_classes):
            class_mask = roi_labels == class_idx
            if not class_mask.any():
                missing_classes.append(class_idx)
                continue
            class_shot_mean_features.append(
                roi_aligned_features[class_mask].mean(dim=0))
        if missing_classes:
            raise ValueError('No support RoI feature found for class indices: '
                             f'{missing_classes}.')

        class_shot_mean_features = torch.stack(
            class_shot_mean_features, dim=0)
        self._print_shape('class_shot_mean_features',
                          class_shot_mean_features)
        flattened_class_prototypes = class_shot_mean_features.flatten(
            2).transpose(1, 2).contiguous()
        self._print_shape('flattened_class_prototypes',
                          flattened_class_prototypes)

        position_encoding_2d = self._build_2d_positional_encoding(
            flattened_class_prototypes)
        position_encoding_2d = position_encoding_2d.expand(
            num_classes, -1, -1)
        self._print_shape('position_encoding_2d', position_encoding_2d)
        position_enriched_image_prototypes = (
            flattened_class_prototypes + position_encoding_2d)
        self._print_shape('position_enriched_image_prototypes',
                          position_enriched_image_prototypes)
        return position_enriched_image_prototypes
