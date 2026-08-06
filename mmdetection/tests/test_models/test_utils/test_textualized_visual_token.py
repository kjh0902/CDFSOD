import pytest
import torch
import torch.nn as nn

from mmdet.models.language_models.bert import BertEncoder
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

    assert tokens.shape == (2, 768)
    assert generator.projection.in_features == 256


def test_textualized_visual_token_averages_gap_across_levels():
    generator = TextualizedVisualTokenGenerator(token_dim=256)
    with torch.no_grad():
        generator.projection.weight.copy_(torch.eye(256))
        generator.projection.bias.zero_()

    features = tuple(
        torch.full((1, 256, size, size), float(level))
        for level, size in zip((1, 2, 3, 4), (16, 8, 4, 2)))
    rois = torch.tensor([[0, 32, 32, 96, 96]], dtype=torch.float32)

    tokens = generator(features, rois)

    assert tokens.shape == (1, 256)
    assert torch.allclose(tokens, torch.full_like(tokens, 2.5))


def test_gradients_reach_textualizer_bert_and_detection_head():
    transformers = pytest.importorskip('transformers')
    generator = TextualizedVisualTokenGenerator()
    features = (
        torch.randn(1, 256, 16, 16),
        torch.randn(1, 256, 8, 8),
        torch.randn(1, 256, 4, 4),
        torch.randn(1, 256, 2, 2),
    )
    rois = torch.tensor([[0, 8, 8, 96, 96]], dtype=torch.float32)
    visual_token = generator(features, rois)

    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=12,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0)
    encoder = BertEncoder.__new__(BertEncoder)
    nn.Module.__init__(encoder)
    encoder.model = transformers.BertModel(config, add_pooling_layer=False)
    encoder.language_dim = config.hidden_size
    encoder.num_layers_of_embedded = 1

    input_ids = torch.tensor([[1, 2]])
    word_embeddings = encoder.model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat(
        [word_embeddings, visual_token.unsqueeze(0)], dim=1)
    position_ids = torch.arange(3).unsqueeze(0)
    token_type_ids = torch.zeros_like(position_ids)
    attention_mask = torch.ones((1, 3, 3), dtype=torch.bool)

    embeddings = encoder.model.embeddings
    expected_visual_embedding = embeddings.LayerNorm(
        inputs_embeds[:, -1] + embeddings.position_embeddings(
            position_ids[:, -1]) + embeddings.token_type_embeddings(
                token_type_ids[:, -1]))
    actual_embeddings = embeddings(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        token_type_ids=token_type_ids)
    assert torch.allclose(
        actual_embeddings[:, -1], expected_visual_embedding, atol=1e-6)

    text_features = encoder(
        dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids))['embedded']
    detection_head = nn.Linear(768, 1)
    loss = detection_head(text_features[:, 0]).sum()
    loss.backward()

    assert generator.projection.weight.grad is not None
    assert torch.count_nonzero(generator.projection.weight.grad) > 0
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in encoder.parameters())
    assert detection_head.weight.grad is not None
    assert torch.count_nonzero(detection_head.weight.grad) > 0
