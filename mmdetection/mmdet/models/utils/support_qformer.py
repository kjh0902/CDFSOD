# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.ops import RoIAlign
from torch import Tensor

from mmdet.structures.bbox import bbox2roi


class SupportVisualTokenizer(nn.Module):
    """Convert support object RoIs into class-separated visual tokens."""

    def __init__(self,
                 output_size: int = 7,
                 featmap_stride: int = 8,
                 hidden_dim: int = 256,
                 sampling_ratio: int = 2,
                 temperature: int = 10000) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError('hidden_dim must be divisible by 4 for 2D '
                             'positional encoding.')
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.roi_align = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=1.0 / featmap_stride,
            sampling_ratio=sampling_ratio,
            aligned=True)

    def _build_2d_positional_encoding(self, reference: Tensor) -> Tensor:
        """Build fixed 2D sine/cosine positions with shape [1, HW, D]."""
        height = width = self.output_size
        num_pos_feats = self.hidden_dim // 2
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
        positions = torch.cat((pos_y, pos_x), dim=-1).reshape(
            1, height * width, self.hidden_dim)
        return positions.to(dtype=reference.dtype)

    def extract_roi_features(
            self, support_features: Tuple[Tensor],
            support_bboxes: Sequence[Tensor],
            support_labels: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        """RoIAlign the first neck feature without aggregating shots."""
        if not support_features:
            raise ValueError('support_features must not be empty.')
        feature_map = support_features[0]
        if feature_map.size(1) != self.hidden_dim:
            raise ValueError(
                f'Expected {self.hidden_dim} support feature channels, got '
                f'{feature_map.size(1)}.')
        if len(support_bboxes) != len(support_labels):
            raise ValueError('support_bboxes and support_labels must have the '
                             'same batch length.')
        if len(support_bboxes) != feature_map.size(0):
            raise ValueError(
                'Support annotation batch length must match feature batch.')

        rois = bbox2roi(list(support_bboxes)).type_as(feature_map)
        if rois.numel() == 0:
            raise ValueError('No support GT bbox was provided.')
        labels = torch.cat(list(support_labels), dim=0).to(
            device=feature_map.device, dtype=torch.long)
        if labels.numel() != rois.size(0):
            raise ValueError('Support GT label and bbox counts must match.')
        return self.roi_align(feature_map, rois), labels

    def build_class_tokens(self, roi_features: Tensor, roi_labels: Tensor,
                           num_classes: int) -> Tuple[Tensor, Tensor]:
        """Concatenate every 7x7 object token and pad only across classes."""
        if roi_features.dim() != 4:
            raise ValueError('roi_features must have shape [N, D, H, W].')
        if roi_features.size(0) != roi_labels.numel():
            raise ValueError('RoI feature and label counts must match.')
        if roi_features.size(1) != self.hidden_dim:
            raise ValueError(
                f'Expected RoI feature dimension {self.hidden_dim}.')
        if roi_features.shape[-2:] != (self.output_size, self.output_size):
            raise ValueError(
                f'Expected RoI size {self.output_size}x{self.output_size}.')
        if num_classes <= 0:
            raise ValueError('num_classes must be greater than zero.')
        if ((roi_labels < 0) | (roi_labels >= num_classes)).any():
            raise ValueError('Support labels are outside the class range.')

        object_tokens = roi_features.flatten(2).transpose(1, 2).contiguous()
        object_tokens = object_tokens + self._build_2d_positional_encoding(
            object_tokens)

        class_tokens: List[Tensor] = []
        missing_classes = []
        for class_idx in range(num_classes):
            class_mask = roi_labels == class_idx
            if not class_mask.any():
                missing_classes.append(class_idx)
                continue
            class_tokens.append(
                object_tokens[class_mask].reshape(-1, self.hidden_dim))
        if missing_classes:
            raise ValueError('No support object found for class indices: '
                             f'{missing_classes}.')

        lengths = torch.tensor(
            [tokens.size(0) for tokens in class_tokens],
            dtype=torch.long,
            device=roi_features.device)
        padded_tokens = nn.utils.rnn.pad_sequence(
            class_tokens, batch_first=True)
        token_indices = torch.arange(
            padded_tokens.size(1), device=roi_features.device).unsqueeze(0)
        padding_mask = token_indices >= lengths.unsqueeze(1)
        return padded_tokens, padding_mask


class SupportQFormerLayer(nn.Module):
    """A pre-norm query self-attention/cross-attention/FFN block."""

    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.query_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, visual_tokens: Tensor,
                visual_padding_mask: Tensor) -> Tensor:
        normalized_queries = self.query_norm(queries)
        attended_queries, _ = self.query_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False)
        queries = queries + self.residual_dropout(attended_queries)

        attended_visual, _ = self.cross_attention(
            self.cross_norm(queries),
            visual_tokens,
            visual_tokens,
            key_padding_mask=visual_padding_mask,
            need_weights=False)
        queries = queries + self.residual_dropout(attended_visual)
        queries = queries + self.residual_dropout(
            self.ffn(self.ffn_norm(queries)))
        return queries


class SupportQFormer(nn.Module):
    """Extract one text-conditioned visual representation per class."""

    def __init__(self,
                 hidden_dim: int = 256,
                 num_queries: int = 4,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_queries <= 0 or num_layers <= 0:
            raise ValueError(
                'Q-Former dimensions and counts must be positive.')
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads.')
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.learnable_queries = nn.Parameter(
            torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.learnable_queries, std=0.02)
        self.layers = nn.ModuleList([
            SupportQFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, text_prototypes: Tensor, visual_tokens: Tensor,
                visual_padding_mask: Tensor) -> Tensor:
        if text_prototypes.dim() != 2:
            raise ValueError('text_prototypes must have shape [C, D].')
        if visual_tokens.dim() != 3:
            raise ValueError('visual_tokens must have shape [C, L, D].')
        if visual_padding_mask.shape != visual_tokens.shape[:2]:
            raise ValueError('visual_padding_mask must have shape [C, L].')
        if text_prototypes.size(0) != visual_tokens.size(0):
            raise ValueError('Text and visual class counts must match.')
        if (text_prototypes.size(-1) != self.hidden_dim or
                visual_tokens.size(-1) != self.hidden_dim):
            raise ValueError(
                f'Q-Former inputs must have dimension {self.hidden_dim}.')
        if visual_padding_mask.all(dim=1).any():
            raise ValueError('Every class must contain a visual token.')

        queries = self.learnable_queries.unsqueeze(0) + \
            text_prototypes.unsqueeze(1)
        for layer in self.layers:
            queries = layer(queries, visual_tokens, visual_padding_mask)
        return queries
