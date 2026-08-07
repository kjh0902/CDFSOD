import torch
import torch.nn as nn

from mmdet.models.utils import TextualizedVisualTokenGenerator


def test_textualized_visual_token_shape():
    generator = TextualizedVisualTokenGenerator()
    features = (
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
        torch.randn(2, 256, 4, 4),
    )
    rois = torch.tensor([
        [0, 16, 16, 128, 128],
        [1, 32, 32, 192, 192],
    ], dtype=torch.float32)

    tokens = generator(features, rois)

    assert tokens.shape == (2, 256)
    assert sum(parameter.numel() for parameter in generator.parameters()) == 0


def test_textualized_visual_token_max_pools_levels_before_gap():

    class FixedRoIAlign(nn.Module):

        def __init__(self, roi_feature):
            super().__init__()
            self.register_buffer('roi_feature', roi_feature)

        def forward(self, feature, rois):
            return self.roi_feature.expand(rois.size(0), -1, -1, -1)

    generator = TextualizedVisualTokenGenerator()

    left_feature = torch.zeros(1, 256, 7, 7)
    left_feature[:, :, :, :4] = 1
    right_feature = torch.zeros(1, 256, 7, 7)
    right_feature[:, :, :, 4:] = 2
    zero_feature = torch.zeros(1, 256, 7, 7)
    generator.roi_align_layers = nn.ModuleList([
        FixedRoIAlign(left_feature),
        FixedRoIAlign(right_feature),
        FixedRoIAlign(zero_feature),
        FixedRoIAlign(zero_feature),
    ])

    features = tuple(torch.zeros(1, 256, 1, 1) for _ in range(4))
    rois = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float32)

    tokens = generator(features, rois)

    assert tokens.shape == (1, 256)
    assert torch.allclose(tokens, torch.full_like(tokens, 10 / 7))


def test_gradients_pass_through_parameter_free_textualizer():
    generator = TextualizedVisualTokenGenerator()
    features = (
        torch.randn(1, 256, 16, 16, requires_grad=True),
        torch.randn(1, 256, 8, 8, requires_grad=True),
        torch.randn(1, 256, 4, 4, requires_grad=True),
        torch.randn(1, 256, 2, 2, requires_grad=True),
    )
    rois = torch.tensor([[0, 8, 8, 96, 96]], dtype=torch.float32)
    visual_token = generator(features, rois)
    detection_head = nn.Linear(256, 1)
    loss = detection_head(visual_token).sum()
    loss.backward()

    assert any(feature.grad is not None for feature in features)
    assert detection_head.weight.grad is not None
    assert torch.count_nonzero(detection_head.weight.grad) > 0
