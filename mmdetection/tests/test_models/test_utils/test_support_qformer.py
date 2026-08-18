import pytest
import torch
import torch.nn as nn

from mmdet.models.utils import SupportQFormer, SupportVisualTokenizer


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
    # Both class-1 objects remain as separate four-token blocks.
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


def test_qformer_keeps_classes_isolated_and_ignores_padding():
    torch.manual_seed(7)
    qformer = SupportQFormer(
        hidden_dim=8,
        num_queries=3,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0).eval()
    text = torch.randn(2, 8)
    visual = torch.randn(2, 8, 8)
    padding_mask = torch.tensor([[False] * 4 + [True] * 4,
                                 [False] * 8])

    original = qformer(text, visual, padding_mask)
    assert original.shape == (2, 3, 8)

    changed_other_class = visual.clone()
    changed_other_class[1] += 100
    isolated = qformer(text, changed_other_class, padding_mask)
    torch.testing.assert_close(original[0], isolated[0])
    assert not torch.allclose(original[1], isolated[1])

    changed_padding = visual.clone()
    changed_padding[0, 4:] += 1000
    padding_ignored = qformer(text, changed_padding, padding_mask)
    torch.testing.assert_close(original, padding_ignored)


def test_identity_gate_and_qformer_gradients():
    torch.manual_seed(11)
    qformer = SupportQFormer(
        hidden_dim=8,
        num_queries=4,
        num_layers=2,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0)
    text = torch.randn(3, 8, requires_grad=True)
    visual = torch.randn(3, 5, 8, requires_grad=True)
    padding_mask = torch.zeros(3, 5, dtype=torch.bool)
    visual_representation = qformer(text, visual, padding_mask).mean(dim=1)

    zero_gate = nn.Parameter(torch.tensor(0.0))
    identity_output = text + zero_gate * visual_representation
    assert torch.equal(identity_output, text)

    trainable_gate = nn.Parameter(torch.tensor(0.5))
    fused = text + trainable_gate * visual_representation
    fused.square().mean().backward()
    assert trainable_gate.grad is not None
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert qformer.learnable_queries.grad is not None
    assert qformer.learnable_queries.grad.abs().sum() > 0
