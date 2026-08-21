from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from mmdet.models.utils import SupportBlipEncoder


class _FakeVisionModel(nn.Module):

    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, pixel_values, return_dict=True):
        pooled = pixel_values.mean(dim=(1, 2, 3))
        hidden = pooled[:, None, None].expand(-1, 4, 768) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeTextEncoder(nn.Module):

    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, input_ids, attention_mask, encoder_hidden_states,
                encoder_attention_mask, return_dict=True):
        text = input_ids.float().unsqueeze(-1).expand(-1, -1, 768)
        visual = encoder_hidden_states.mean(dim=(1, 2))[:, None, None]
        return SimpleNamespace(
            last_hidden_state=text * self.scale + visual)


class _FakeUnsupportedTextEncoder(_FakeTextEncoder):

    supports_gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        raise ValueError('This text encoder does not support checkpointing.')


class _FakeImageProcessor:

    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.ones(len(images), 3, 4, 4)}


class _FakeTokenizer:

    def __call__(self, texts, padding, return_special_tokens_mask,
                 return_tensors):
        num_texts = len(texts)
        input_ids = torch.tensor([[101, 17, 18, 102]]).expand(
            num_texts, -1).clone()
        return dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            special_tokens_mask=torch.tensor([[1, 0, 0, 1]]).expand(
                num_texts, -1).clone())


def _build_encoder(text_encoder=None):
    pretrained = SimpleNamespace(
        vision_model=_FakeVisionModel(),
        text_encoder=(text_encoder if text_encoder is not None else
                      _FakeTextEncoder()),
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=768)))
    processor = SimpleNamespace(
        image_processor=_FakeImageProcessor(), tokenizer=_FakeTokenizer())
    with patch(
            'mmdet.models.utils.support_blip.'
            'BlipForImageTextRetrieval.from_pretrained',
            return_value=pretrained) as model_loader, patch(
                'mmdet.models.utils.support_blip.'
                'BlipProcessor.from_pretrained',
                return_value=processor) as processor_loader:
        encoder = SupportBlipEncoder('fake/blip')
    model_loader.assert_called_once_with('fake/blip')
    processor_loader.assert_called_once_with('fake/blip')
    return encoder


def test_pretrained_blip_components_are_preserved_and_trainable():
    encoder = _build_encoder()

    assert encoder.hidden_size == 768
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.vision_model.checkpointing_enabled
    assert encoder.text_encoder.checkpointing_enabled


def test_unsupported_text_encoder_skips_gradient_checkpointing():
    text_encoder = _FakeUnsupportedTextEncoder()

    encoder = _build_encoder(text_encoder=text_encoder)

    assert encoder.text_encoder is text_encoder
    assert not encoder.text_encoder.checkpointing_enabled
    assert encoder.vision_model.checkpointing_enabled


def test_pretrained_blip_pair_processor_is_used():
    encoder = _build_encoder()

    inputs = encoder.preprocess_pairs([
        Image.new('RGB', (7, 5)),
        Image.new('RGB', (3, 9)),
    ], ['pitted surface', 'rolled-in scale'])

    assert inputs['pixel_values'].shape == (2, 3, 4, 4)
    assert inputs['input_ids'].shape == (2, 4)
    assert inputs['attention_mask'].shape == (2, 4)
    assert inputs['special_tokens_mask'].tolist() == [
        [1, 0, 0, 1], [1, 0, 0, 1]
    ]
    assert all(value.device.type == 'cpu' for value in inputs.values())


def test_blip_pair_fusion_and_existing_text_feat_map_receive_gradients():
    encoder = _build_encoder()
    text_feat_map = nn.Linear(768, 256)
    pixel_values = torch.stack([
        torch.ones(3, 4, 4),
        torch.full((3, 4, 4), 2.0),
    ])
    input_ids = torch.tensor([[101, 17, 18, 102], [101, 17, 18, 102]])
    attention_mask = torch.ones_like(input_ids)

    tokens = encoder(pixel_values, input_ids, attention_mask)
    prototypes = text_feat_map(tokens)
    prototypes.square().mean().backward()

    assert tokens.shape == (2, 4, 768)
    assert not torch.equal(tokens[0], tokens[1])
    assert encoder.vision_model.scale.grad is not None
    assert encoder.text_encoder.scale.grad is not None
    assert text_feat_map.weight.grad is not None
