import torch

from mmdet.models.utils import TextualizedVisualTokenGenerator


def test_textualized_visual_token_shape():
    generator = TextualizedVisualTokenGenerator(log_shapes=False)
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

    assert tokens.shape == (2, 768)
