from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from mmdet.models.utils import SupportBlipCaptioner


class _FakeVisionModel(nn.Module):

    supports_gradient_checkpointing = True

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def forward(self, pixel_values, return_dict=True):
        pooled = pixel_values.mean(dim=(1, 2, 3)) * self.scale
        hidden = pooled[:, None, None].expand(-1, 3, 8)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeTextDecoder(nn.Module):

    supports_gradient_checkpointing = True

    def __init__(self, checkpointing_error=False):
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.checkpointing_enabled = False
        self.checkpointing_error = checkpointing_error

    def gradient_checkpointing_enable(self):
        if self.checkpointing_error:
            raise ValueError('unsupported concrete decoder')
        self.checkpointing_enabled = True

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, encoder_hidden_states,
                encoder_attention_mask, use_cache, return_dict):
        batch_size, sequence_length = inputs_embeds.shape[:2]
        visual = encoder_hidden_states.mean(dim=(1, 2))
        vocab_axis = torch.arange(
            12, dtype=visual.dtype, device=visual.device)
        differentiable = visual[:, None] * self.scale * vocab_axis[None, :]
        bias = visual.new_full((batch_size, 12), -5.0)
        bias[:, 10] = 100.0  # [DEC] must be masked.
        bias[:, 11] = 90.0  # [ENC] must be masked.
        if sequence_length == 1:
            bias[:, 3] = 10.0
        else:
            bias[:, 4] = 10.0  # [SEP]
        step_logits = differentiable + bias
        logits = step_logits[:, None, :].expand(
            -1, sequence_length, -1)
        return SimpleNamespace(logits=logits)


class _FakeImageProcessor:

    def __call__(self, images, return_tensors):
        return {'pixel_values': torch.ones(len(images), 3, 4, 4)}


class _FakeTokenizer:

    def __init__(self):
        self.unk_token_id = 1
        self._vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            'a': 2,
            'caption': 3,
            '[SEP]': 4,
            '[CLS]': 5,
            '[MASK]': 6,
            ':': 7,
            'class': 8,
            'token': 9,
            '[DEC]': 10,
            '[ENC]': 11,
        }

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def get_vocab(self):
        return dict(self._vocab)


def _build_captioner(checkpointing_error=False):
    text_config = SimpleNamespace(
        hidden_size=768,
        vocab_size=12,
        max_length=20,
        bos_token_id=10,
        sep_token_id=4,
        pad_token_id=0)
    pretrained = SimpleNamespace(
        vision_model=_FakeVisionModel(),
        text_decoder=_FakeTextDecoder(checkpointing_error),
        config=SimpleNamespace(text_config=text_config))
    processor = SimpleNamespace(
        image_processor=_FakeImageProcessor(), tokenizer=_FakeTokenizer())
    with patch(
            'mmdet.models.utils.support_blip.'
            'BlipForConditionalGeneration.from_pretrained',
            return_value=pretrained) as model_loader, patch(
                'mmdet.models.utils.support_blip.'
                'BlipProcessor.from_pretrained',
                return_value=processor) as processor_loader:
        captioner = SupportBlipCaptioner('fake/blip')
    model_loader.assert_called_once_with('fake/blip')
    processor_loader.assert_called_once_with('fake/blip')
    return captioner


def test_pretrained_caption_components_are_preserved_and_trainable():
    captioner = _build_captioner()

    assert captioner.hidden_size == 768
    assert captioner.max_length == 20
    assert all(parameter.requires_grad for parameter in captioner.parameters())
    assert captioner.vision_model.checkpointing_enabled
    assert captioner.text_decoder.checkpointing_enabled


def test_unsupported_decoder_checkpointing_is_skipped():
    captioner = _build_captioner(checkpointing_error=True)

    assert captioner.vision_model.checkpointing_enabled
    assert not captioner.text_decoder.checkpointing_enabled


def test_pretrained_blip_image_processor_is_used():
    captioner = _build_captioner()

    pixel_values = captioner.preprocess_images([
        Image.new('RGB', (7, 5)),
        Image.new('RGB', (3, 9)),
    ])

    assert pixel_values.shape == (2, 3, 4, 4)
    assert pixel_values.device.type == 'cpu'


def test_private_tokens_are_masked_and_st_forward_is_greedy():
    captioner = _build_captioner()
    outputs = captioner(torch.ones(2, 3, 4, 4))

    assert outputs['token_ids'].tolist() == [[3, 4], [3, 4]]
    assert outputs['caption_mask'].tolist() == [[True, False], [True, False]]
    hard_ids = outputs['token_distributions'].argmax(dim=-1)
    torch.testing.assert_close(hard_ids, outputs['token_ids'])
    assert not outputs['token_distributions'][..., 10].any()
    assert not outputs['token_distributions'][..., 11].any()


def test_st_caption_gradients_reach_vision_and_decoder():
    captioner = _build_captioner()
    outputs = captioner(torch.ones(2, 3, 4, 4))
    weights = torch.arange(12, dtype=torch.float32)

    loss = (outputs['token_distributions'][:, 0] * weights).sum()
    loss.backward()

    assert captioner.vision_model.scale.grad is not None
    assert captioner.vision_model.scale.grad.abs().item() > 0
    assert captioner.text_decoder.scale.grad is not None
    assert captioner.text_decoder.scale.grad.abs().item() > 0


def test_shared_grounding_vocabulary_is_validated():
    captioner = _build_captioner()
    grounding_tokenizer = SimpleNamespace(
        get_vocab=lambda: dict(
            list(captioner.tokenizer.get_vocab().items())[:10]))

    assert captioner.validate_grounding_tokenizer(grounding_tokenizer) == 10

    bad_vocab = grounding_tokenizer.get_vocab()
    bad_vocab['caption'] = 9
    with pytest.raises(ValueError, match='not id-compatible'):
        captioner.validate_grounding_tokenizer(
            SimpleNamespace(get_vocab=lambda: bad_vocab))
