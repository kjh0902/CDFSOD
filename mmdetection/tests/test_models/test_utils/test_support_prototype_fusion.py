# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.models.utils import SupportPrototypeFusion


def test_support_prototype_fusion_shapes_and_gradients():
    fusion = SupportPrototypeFusion(
        output_size=7,
        feature_stride=8,
        embed_dims=256,
        num_attention_heads=8,
        text_weight=0.5,
        visual_weight=0.5)
    support_feature = torch.randn(
        2, 256, 16, 16, requires_grad=True)
    support_bboxes = [
        torch.tensor([[8., 8., 56., 56.], [40., 40., 104., 104.]]),
        torch.tensor([[16., 24., 96., 112.]])
    ]
    support_labels = [torch.tensor([0, 0]), torch.tensor([1])]

    instance_features, labels = fusion.extract_instance_features(
        support_feature, support_bboxes, support_labels)
    assert instance_features.shape == (3, 256)
    visual_prototypes = fusion.aggregate_by_class(
        instance_features, labels, num_classes=2)
    assert visual_prototypes.shape == (2, 256)

    text_prototypes = torch.randn(2, 256, requires_grad=True)
    aligned_text, aligned_visual, fused = fusion.align_and_fuse(
        text_prototypes, visual_prototypes)
    assert aligned_text.shape == (2, 256)
    assert aligned_visual.shape == (2, 256)
    assert fused.shape == (2, 256)
    assert fusion.text_weight + fusion.visual_weight == 1.0

    fused.sum().backward()
    assert text_prototypes.grad is not None
    assert support_feature.grad is not None
    assert fusion.shared_attention.in_proj_weight.grad is not None
