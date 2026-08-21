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
        self.last_input_ids = None

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, input_ids, attention_mask, encoder_hidden_states,
                encoder_attention_mask, return_dict=True):
        self.last_input_ids = input_ids.detach().clone()
        text = input_ids.float().unsqueeze(-1).expand(-1, -1, 768)
        visual = encoder_hidden_states.mean(dim=(1, 2))[:, None, None]
        return SimpleNamespace(
            last_hidden_state=text * self.scale + visual)

    def get_input_embeddings(self):
        return SimpleNamespace(num_embeddings=30524)


class _FakeUnsupportedTextEncoder(_FakeTextEncoder):

    supports_gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        raise ValueError('This text encoder does not support checkpointing.')


class _FakeImageProcessor:

    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.ones(len(images), 3, 4, 4)}


class _FakeTokenizer:

    def __init__(self):
        self.unk_token_id = 100
        self._token_ids = {'[UNK]': self.unk_token_id}
        self._next_token_id = 30522

    def add_special_tokens(self, special_tokens_dict):
        tokens = []
        if 'bos_token' in special_tokens_dict:
            tokens.append(special_tokens_dict['bos_token'])
        tokens.extend(special_tokens_dict.get(
            'additional_special_tokens', []))
        added = 0
        for token in tokens:
            if token not in self._token_ids:
                self._token_ids[token] = self._next_token_id
                self._next_token_id += 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self._token_ids.get(token, self.unk_token_id)

    def __call__(self, texts, padding, return_tensors):
        num_texts = len(texts)
        input_ids = torch.tensor([[101, 17, 18, 102]]).expand(
            num_texts, -1).clone()
        return dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids))


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
    assert encoder.enc_token_id == 30523
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

    enc_features = encoder(pixel_values, input_ids, attention_mask)
    prototypes = text_feat_map(enc_features)
    prototypes.square().mean().backward()

    assert enc_features.shape == (2, 768)
    assert not torch.equal(enc_features[0], enc_features[1])
    assert encoder.text_encoder.last_input_ids[:, 0].tolist() == [
        encoder.enc_token_id, encoder.enc_token_id
    ]
    assert encoder.vision_model.scale.grad is not None
    assert encoder.text_encoder.scale.grad is not None
    assert text_feat_map.weight.grad is not None
