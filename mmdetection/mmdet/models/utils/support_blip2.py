# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from transformers import Blip2VisionModelWithProjection, BlipImageProcessor


class SupportBlip2Encoder(nn.Module):
    """Pretrained BLIP-2 vision encoder and Q-Former for support crops.

    The checkpoint's vision projection is intentionally discarded. Grounding
    DINO's pretrained ``text_feat_map`` performs the only 768 -> 256
    projection used by the CDFSOD prototype path.
    """

    def __init__(self,
                 model_name: str = 'Salesforce/blip2-itm-vit-g',
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        pretrained = Blip2VisionModelWithProjection.from_pretrained(model_name)
        self.image_processor = BlipImageProcessor.from_pretrained(model_name)
        self.vision_model = pretrained.vision_model
        self.qformer = pretrained.qformer
        self.query_tokens = pretrained.query_tokens

        if self.query_tokens.dim() != 3:
            raise ValueError('BLIP-2 query_tokens must have shape [1, Q, D].')
        self.num_query_tokens = self.query_tokens.size(1)
        self.hidden_size = self.query_tokens.size(2)
        if self.num_query_tokens != 32:
            raise ValueError(
                'CDFSOD visual prototypes require exactly 32 BLIP-2 query '
                f'tokens, got {self.num_query_tokens}.')
        if self.hidden_size != 768:
            raise ValueError(
                'Grounding DINO text_feat_map expects 768-dimensional BLIP-2 '
                f'queries, got {self.hidden_size}.')

        self.requires_grad_(True)
        if gradient_checkpointing:
            for module in (self.vision_model, self.qformer):
                enable = getattr(module, 'gradient_checkpointing_enable', None)
                if enable is not None:
                    enable()

    def preprocess_images(self, images: Sequence[Image.Image]) -> Tensor:
        """Apply the checkpoint's pretrained image preprocessing on CPU."""
        if not images:
            raise ValueError('At least one support crop is required.')
        processed = self.image_processor(
            images=list(images), return_tensors='pt')
        pixel_values = processed['pixel_values']
        if pixel_values.dim() != 4:
            raise ValueError(
                'BLIP-2 image processor must return [N, 3, H, W].')
        return pixel_values.detach().cpu()

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Return one set of 32 pretrained Q-Former tokens per crop."""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device)
        query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True)
        queries = query_outputs.last_hidden_state
        expected_shape = (image_embeds.size(0), self.num_query_tokens,
                          self.hidden_size)
        if tuple(queries.shape) != expected_shape:
            raise RuntimeError(
                f'Unexpected BLIP-2 query shape {tuple(queries.shape)}; '
                f'expected {expected_shape}.')
        return queries
