from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from mmdet.models.utils import SupportBlip2Encoder


class _FakeVisionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, pixel_values, return_dict=True):
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False)
        hidden = pooled[:, None, None].expand(-1, 4, 8) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeQFormer(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, query_embeds, encoder_hidden_states,
                encoder_attention_mask, return_dict=True):
        visual = encoder_hidden_states.mean(dim=(1, 2))[:, None, None]
        return SimpleNamespace(
            last_hidden_state=query_embeds * self.scale + visual)


class _FakeProcessor:

    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.ones(len(images), 3, 4, 4)}


def _build_encoder():
    pretrained = SimpleNamespace(
        vision_model=_FakeVisionModel(),
        qformer=_FakeQFormer(),
        query_tokens=nn.Parameter(torch.full((1, 32, 768), 0.25)))
    with patch(
            'mmdet.models.utils.support_blip2.'
            'Blip2VisionModelWithProjection.from_pretrained',
            return_value=pretrained) as model_loader, patch(
                'mmdet.models.utils.support_blip2.'
                'BlipImageProcessor.from_pretrained',
                return_value=_FakeProcessor()) as processor_loader:
        encoder = SupportBlip2Encoder('fake/blip2')
    model_loader.assert_called_once_with('fake/blip2')
    processor_loader.assert_called_once_with('fake/blip2')
    return encoder


def test_pretrained_blip2_components_are_preserved_and_trainable():
    encoder = _build_encoder()

    assert encoder.num_query_tokens == 32
    assert encoder.hidden_size == 768
    assert torch.equal(encoder.query_tokens,
                       torch.full((1, 32, 768), 0.25))
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.vision_model.checkpointing_enabled
    assert encoder.qformer.checkpointing_enabled


def test_pretrained_blip2_image_processor_is_used():
    encoder = _build_encoder()

    pixel_values = encoder.preprocess_images([
        Image.new('RGB', (7, 5)),
        Image.new('RGB', (3, 9)),
    ])

    assert pixel_values.shape == (2, 3, 4, 4)
    assert pixel_values.device.type == 'cpu'


def test_blip2_queries_and_existing_text_feat_map_receive_gradients():
    encoder = _build_encoder()
    text_feat_map = nn.Linear(768, 256)
    pixel_values = torch.randn(2, 3, 4, 4)

    queries = encoder(pixel_values)
    prototypes = text_feat_map(queries)
    prototypes.square().mean().backward()

    assert queries.shape == (2, 32, 768)
    assert encoder.vision_model.scale.grad is not None
    assert encoder.qformer.scale.grad is not None
    assert encoder.query_tokens.grad is not None
    assert text_feat_map.weight.grad is not None
