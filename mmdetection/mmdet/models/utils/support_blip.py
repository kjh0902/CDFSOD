# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from transformers import BlipForImageTextRetrieval, BlipProcessor


class SupportBlipEncoder(nn.Module):
    """BLIP-1 vision and multimodal text encoders for support pairs.

    The pretrained ITM head and projection layers are intentionally discarded.
    Grounding DINO's pretrained ``text_feat_map`` remains the only 768 -> 256
    projection used by the CDFSOD prototype path.
    """

    def __init__(self,
                 model_name: str = 'Salesforce/blip-itm-base-coco',
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        pretrained = BlipForImageTextRetrieval.from_pretrained(model_name)
        processor = BlipProcessor.from_pretrained(model_name)
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.vision_model = pretrained.vision_model
        self.text_encoder = pretrained.text_encoder
        self.hidden_size = pretrained.config.text_config.hidden_size

        if self.hidden_size != 768:
            raise ValueError(
                'Grounding DINO text_feat_map expects 768-dimensional BLIP '
                f'text features, got {self.hidden_size}.')

        self.requires_grad_(True)
        if gradient_checkpointing:
            for module in (self.vision_model, self.text_encoder):
                enable = getattr(module, 'gradient_checkpointing_enable', None)
                if enable is not None:
                    enable()

    def preprocess_pairs(self, images: Sequence[Image.Image],
                         texts: Sequence[str]) -> Dict[str, Tensor]:
        """Preprocess aligned image-text pairs and retain tensors on CPU."""
        if not images:
            raise ValueError('At least one support crop is required.')
        if len(images) != len(texts):
            raise ValueError('Support image and text counts must match.')

        image_inputs = self.image_processor(
            images=list(images), return_tensors='pt')
        text_inputs = self.tokenizer(
            list(texts),
            padding=True,
            return_special_tokens_mask=True,
            return_tensors='pt')
        inputs = dict(
            pixel_values=image_inputs['pixel_values'],
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask'],
            special_tokens_mask=text_inputs['special_tokens_mask'])

        num_pairs = len(images)
        if inputs['pixel_values'].dim() != 4 or \
                inputs['pixel_values'].size(0) != num_pairs:
            raise ValueError(
                'BLIP image processor must return [N, 3, H, W].')
        for key in ('input_ids', 'attention_mask', 'special_tokens_mask'):
            if inputs[key].dim() != 2 or inputs[key].size(0) != num_pairs:
                raise ValueError(
                    f'BLIP tokenizer must return {key} with shape [N, L].')
        if inputs['input_ids'].shape != inputs['attention_mask'].shape or \
                inputs['input_ids'].shape != \
                inputs['special_tokens_mask'].shape:
            raise ValueError(
                'BLIP text input masks must have matching shapes.')
        return {key: value.detach().cpu() for key, value in inputs.items()}

    def forward(self, pixel_values: Tensor, input_ids: Tensor,
                attention_mask: Tensor) -> Tensor:
        """Return independently fused text tokens for aligned support pairs."""
        batch_size = pixel_values.size(0)
        if input_ids.size(0) != batch_size or \
                attention_mask.shape != input_ids.shape:
            raise ValueError('BLIP image and text batch shapes must match.')

        vision_outputs = self.vision_model(
            pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device)
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True)
        multimodal_tokens = text_outputs.last_hidden_state
        expected_shape = (batch_size, input_ids.size(1), self.hidden_size)
        if tuple(multimodal_tokens.shape) != expected_shape:
            raise RuntimeError(
                f'Unexpected BLIP multimodal text shape '
                f'{tuple(multimodal_tokens.shape)}; expected '
                f'{expected_shape}.')
        return multimodal_tokens
