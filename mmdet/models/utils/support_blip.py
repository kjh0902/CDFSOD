# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import BlipForConditionalGeneration, BlipProcessor


class SupportBlipCaptioner(nn.Module):
    """Differentiable BLIP-1 captioner for support object crops.

    Caption tokens are selected with noise-free straight-through argmax. The
    hard forward pass matches greedy decoding with ``[DEC]`` and ``[ENC]``
    suppressed, while the backward pass follows the softmax probabilities.
    """

    def __init__(self,
                 model_name: str =
                 'Salesforce/blip-image-captioning-base',
                 gradient_checkpointing: bool = True) -> None:
        super().__init__()
        pretrained = BlipForConditionalGeneration.from_pretrained(model_name)
        processor = BlipProcessor.from_pretrained(model_name)
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.vision_model = pretrained.vision_model
        self.text_decoder = pretrained.text_decoder
        self.hidden_size = pretrained.config.text_config.hidden_size
        self.vocab_size = pretrained.config.text_config.vocab_size
        pretrained_max_length = pretrained.config.text_config.max_length
        self.max_length = 10
        self.bos_token_id = pretrained.config.text_config.bos_token_id
        self.sep_token_id = pretrained.config.text_config.sep_token_id
        self.pad_token_id = pretrained.config.text_config.pad_token_id
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[ENC]']})
        self.enc_token_id = self.tokenizer.convert_tokens_to_ids('[ENC]')

        if self.hidden_size != 768:
            raise ValueError(
                'Grounding DINO BERT expects 768-dimensional BLIP text '
                f'embeddings, got {self.hidden_size}.')
        if pretrained_max_length < self.max_length:
            raise ValueError(
                'The pretrained BLIP caption max_length must be at least '
                f'{self.max_length}, got {pretrained_max_length}.')
        if self.bos_token_id is None or self.sep_token_id is None or \
                self.pad_token_id is None:
            raise ValueError('BLIP caption special token ids must be defined.')
        if self.enc_token_id == self.tokenizer.unk_token_id:
            raise ValueError('BLIP tokenizer must contain the pretrained '
                             '[ENC] token.')
        if self.tokenizer.bos_token_id != self.bos_token_id:
            raise ValueError(
                'BLIP tokenizer [DEC] id must match the pretrained decoder '
                f'BOS id, got {self.tokenizer.bos_token_id} and '
                f'{self.bos_token_id}.')
        if self.bos_token_id != self.vocab_size - 2 or \
                self.enc_token_id != self.vocab_size - 1:
            raise ValueError(
                'BLIP pretrained [DEC]/[ENC] ids must occupy the final two '
                f'decoder vocabulary rows, got {self.bos_token_id} and '
                f'{self.enc_token_id} for vocab_size={self.vocab_size}.')

        input_embeddings = self.text_decoder.get_input_embeddings()
        if input_embeddings.num_embeddings != self.vocab_size:
            raise ValueError(
                'BLIP decoder embedding and vocabulary sizes must match, got '
                f'{input_embeddings.num_embeddings} and {self.vocab_size}.')

        self.requires_grad_(True)
        if gradient_checkpointing:
            self._enable_decoder_layer_gradient_checkpointing()

    def _enable_decoder_layer_gradient_checkpointing(self) -> None:
        """Checkpoint only the BLIP caption decoder Transformer layers."""
        text_model = getattr(self.text_decoder, 'bert', None)
        encoder = getattr(text_model, 'encoder', None)
        layers = getattr(encoder, 'layer', None)
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise ValueError(
                'BLIP caption decoder must expose '
                'text_decoder.bert.encoder.layer Transformer layers.')
        checkpoint_func = partial(checkpoint, use_reentrant=False)
        for layer in layers:
            if not hasattr(layer, 'gradient_checkpointing'):
                raise ValueError(
                    'Every BLIP caption decoder Transformer layer must '
                    'support gradient checkpointing.')
            layer._gradient_checkpointing_func = checkpoint_func
            layer.gradient_checkpointing = True

    def preprocess_images(self, images: Sequence[Image.Image]) -> Tensor:
        """Preprocess support crops once and keep the pixels on CPU."""
        if not images:
            raise ValueError('At least one support crop is required.')
        image_inputs = self.image_processor(
            images=list(images), return_tensors='pt')
        pixel_values = image_inputs['pixel_values']
        if pixel_values.dim() != 4 or pixel_values.size(0) != len(images):
            raise ValueError(
                'BLIP image processor must return [N, 3, H, W].')
        return pixel_values.detach().cpu()

    def validate_grounding_tokenizer(self, grounding_tokenizer) -> int:
        """Validate the shared BERT vocabulary and return its size."""
        blip_vocab = self.tokenizer.get_vocab()
        grounding_vocab = grounding_tokenizer.get_vocab()
        mismatched = [
            token for token, token_id in grounding_vocab.items()
            if blip_vocab.get(token) != token_id
        ]
        if mismatched:
            examples = mismatched[:5]
            raise ValueError(
                'BLIP and Grounding DINO BERT lexical vocabularies are not '
                f'id-compatible; mismatched tokens include {examples}.')
        grounding_vocab_size = len(grounding_vocab)
        if grounding_vocab_size >= self.vocab_size:
            raise ValueError(
                'BLIP caption vocabulary must extend the Grounding DINO BERT '
                'vocabulary with its private special tokens.')
        for token_id in (self.bos_token_id, self.enc_token_id):
            if token_id < grounding_vocab_size:
                raise ValueError(
                    'BLIP [DEC]/[ENC] ids must be outside the shared lexical '
                    'vocabulary.')
        return grounding_vocab_size

    def _mask_private_special_logits(self, logits: Tensor) -> Tensor:
        masked_logits = logits.clone()
        masked_logits[:, self.bos_token_id] = -torch.inf
        masked_logits[:, self.enc_token_id] = -torch.inf
        return masked_logits

    def forward(self, pixel_values: Tensor) -> Dict[str, Tensor]:
        """Generate differentiable greedy captions for aligned crop rows."""
        if pixel_values.dim() != 4 or pixel_values.size(0) == 0:
            raise ValueError('pixel_values must have shape [N, 3, H, W].')

        vision_outputs = self.vision_model(
            pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device)
        batch_size = pixel_values.size(0)
        decoder_embeddings = self.text_decoder.get_input_embeddings()
        bos_ids = torch.full(
            (batch_size, ),
            self.bos_token_id,
            dtype=torch.long,
            device=pixel_values.device)
        decoder_inputs_embeds = decoder_embeddings(bos_ids).unsqueeze(1)
        decoder_attention_mask = torch.ones(
            batch_size, 1, dtype=torch.long, device=pixel_values.device)
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=pixel_values.device)

        token_distributions = []
        token_ids = []
        caption_masks = []
        for _ in range(self.max_length - 1):
            decoder_outputs = self.text_decoder(
                inputs_embeds=decoder_inputs_embeds,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                use_cache=False,
                return_dict=True)
            logits = self._mask_private_special_logits(
                decoder_outputs.logits[:, -1, :])
            probabilities = torch.softmax(logits, dim=-1)
            greedy_ids = probabilities.argmax(dim=-1)
            active = ~finished
            selected_ids = torch.where(
                active, greedy_ids,
                torch.full_like(greedy_ids, self.pad_token_id))
            hard = F.one_hot(
                selected_ids, num_classes=self.vocab_size).to(
                    probabilities.dtype)
            straight_through = hard + probabilities - probabilities.detach()
            pad_hard = F.one_hot(
                torch.full_like(selected_ids, self.pad_token_id),
                num_classes=self.vocab_size).to(probabilities.dtype)
            straight_through = torch.where(
                active.unsqueeze(-1), straight_through, pad_hard)

            reached_sep = active & selected_ids.eq(self.sep_token_id)
            is_caption_token = active & ~reached_sep & \
                selected_ids.ne(self.pad_token_id)
            token_distributions.append(straight_through)
            token_ids.append(selected_ids)
            caption_masks.append(is_caption_token)

            next_embeddings = straight_through @ decoder_embeddings.weight
            decoder_inputs_embeds = torch.cat(
                [decoder_inputs_embeds, next_embeddings.unsqueeze(1)], dim=1)
            decoder_attention_mask = torch.cat([
                decoder_attention_mask,
                active.to(torch.long).unsqueeze(1)
            ], dim=1)
            finished = finished | reached_sep
            if finished.all():
                break

        return {
            'token_distributions': torch.stack(token_distributions, dim=1),
            'token_ids': torch.stack(token_ids, dim=1),
            'caption_mask': torch.stack(caption_masks, dim=1),
        }
