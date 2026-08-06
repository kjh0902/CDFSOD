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
    assert isinstance(generator.projection[0], nn.Linear)
    assert generator.projection[0].in_features == 256
    assert generator.projection[0].out_features == 512
    assert isinstance(generator.projection[1], nn.GELU)
    assert isinstance(generator.projection[2], nn.Linear)
    assert generator.projection[2].in_features == 512
    assert generator.projection[2].out_features == 768


def test_textualized_visual_token_max_pools_levels_before_gap():

    class FixedRoIAlign(nn.Module):

        def __init__(self, roi_feature):
            super().__init__()
            self.register_buffer('roi_feature', roi_feature)

        def forward(self, feature, rois):
            return self.roi_feature.expand(rois.size(0), -1, -1, -1)

    generator = TextualizedVisualTokenGenerator()
    projection_inputs = []
    hook = generator.projection.register_forward_pre_hook(
        lambda module, inputs: projection_inputs.append(inputs[0].detach()))

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
    hook.remove()

    assert tokens.shape == (1, 768)
    assert len(projection_inputs) == 1
    assert torch.allclose(
        projection_inputs[0], torch.full_like(projection_inputs[0], 10 / 7))


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

    for layer in (generator.projection[0], generator.projection[2]):
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) > 0
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in encoder.parameters())
    assert detection_head.weight.grad is not None
    assert torch.count_nonzero(detection_head.weight.grad) > 0
