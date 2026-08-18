import pytest
import torch
import torch.nn as nn

from mmdet.models.utils import (SupportVisualCrossAttention,
                                SupportVisualTokenizer)


def test_support_visual_tokenizer_preserves_every_object_token():
    tokenizer = SupportVisualTokenizer(
        output_size=2, hidden_dim=8, featmap_stride=1)
    roi_features = torch.stack([
        torch.full((8, 2, 2), 1.0),
        torch.full((8, 2, 2), 2.0),
        torch.full((8, 2, 2), 3.0),
    ])
    labels = torch.tensor([0, 1, 1])

    tokens, padding_mask = tokenizer.build_class_tokens(
        roi_features, labels, num_classes=2)

    assert tokens.shape == (2, 8, 8)
    assert padding_mask.tolist() == [[False] * 4 + [True] * 4,
                                     [False] * 8]
    assert not torch.equal(tokens[1, :4], tokens[1, 4:])


@pytest.mark.parametrize('shot_count', [1, 5, 10])
def test_support_visual_tokenizer_accepts_standard_shot_counts(shot_count):
    tokenizer = SupportVisualTokenizer(
        output_size=2, hidden_dim=8, featmap_stride=1)
    roi_features = torch.randn(shot_count, 8, 2, 2)
    labels = torch.zeros(shot_count, dtype=torch.long)

    tokens, padding_mask = tokenizer.build_class_tokens(
        roi_features, labels, num_classes=1)

    assert tokens.shape == (1, shot_count * 4, 8)
    assert not padding_mask.any()


def test_cross_attention_keeps_classes_isolated_and_ignores_padding():
    torch.manual_seed(7)
    fusion = SupportVisualCrossAttention(
        hidden_dim=8, num_heads=2, dropout=0.0).eval()
    text = torch.randn(2, 8)
    visual = torch.randn(2, 8, 8)
    padding_mask = torch.tensor([[False] * 4 + [True] * 4,
                                 [False] * 8])

    original = fusion(text, visual, padding_mask)
    assert original.shape == (2, 8)

    changed_other_class = visual.clone()
    changed_other_class[1] += 100
    isolated = fusion(text, changed_other_class, padding_mask)
    torch.testing.assert_close(original[0], isolated[0])
    assert not torch.allclose(original[1], isolated[1])

    changed_padding = visual.clone()
    changed_padding[0, 4:] += 1000
    padding_ignored = fusion(text, changed_padding, padding_mask)
    torch.testing.assert_close(original, padding_ignored)


def test_cross_attention_has_only_expected_modules_and_receives_gradients():
    fusion = SupportVisualCrossAttention(
        hidden_dim=8, num_heads=2, dropout=0.0)
    module_names = dict(fusion.named_modules())
    parameter_names = dict(fusion.named_parameters())
    assert all('query_attention' not in name for name in module_names)
    assert all('ffn' not in name for name in module_names)
    assert all('learnable_queries' not in name for name in parameter_names)

    text = torch.randn(3, 8, requires_grad=True)
    visual = torch.randn(3, 5, 8, requires_grad=True)
    padding_mask = torch.zeros(3, 5, dtype=torch.bool)
    visual_representation = fusion(text, visual, padding_mask)
    gate = nn.Parameter(torch.tensor(0.5))
    fused = text + gate * visual_representation
    fused.square().mean().backward()

    assert gate.grad is not None
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert fusion.cross_attention.in_proj_weight.grad is not None
    assert fusion.cross_attention.in_proj_weight.grad.abs().sum() > 0
