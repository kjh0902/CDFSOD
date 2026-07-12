# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 use_class_text_prototypes: bool = False,
                 use_enriched_class_tokens: Optional[bool] = None,
                 support_caption_file: Optional[str] = None,
                 support_class_names: Optional[Sequence[str]] = None,
                 support_domain_attribute: Optional[str] = None,
                 debug_text_prototype: bool = False,
                 debug_text_tokens: Optional[bool] = None,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        if use_enriched_class_tokens is None:
            use_enriched_class_tokens = use_class_text_prototypes
        if debug_text_tokens is None:
            debug_text_tokens = debug_text_prototype
        self.use_enriched_class_tokens = use_enriched_class_tokens
        self.use_class_text_prototypes = use_enriched_class_tokens
        self.support_caption_file = support_caption_file
        self.support_class_names = list(support_class_names or [])
        self.support_domain_attribute = support_domain_attribute
        self.debug_text_tokens = debug_text_tokens
        self.debug_text_prototype = debug_text_tokens
        self.support_prompt_bank = None
        self.support_prompt_labels = None
        self.support_prompt_texts = None
        self.support_prompt_token_output_indices = None
        self.support_tokenized = None
        self._cached_eval_support_token_text_dict = None
        self._printed_text_token_debug = False
        self._registered_bert_grad_debug_hook = False
        super().__init__(*args, **kwargs)
        if self.use_enriched_class_tokens:
            self.build_support_prompt_bank()

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def train(self, mode: bool = True):
        """Switch train/eval mode and clear stale eval text-token cache."""
        super().train(mode)
        if mode:
            self._cached_eval_support_token_text_dict = None
        return self

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def build_support_prompt_bank(self) -> None:
        """Read support captions once and cache class-name token positions."""
        if self.support_prompt_bank is not None:
            return
        if not self.support_caption_file:
            raise ValueError('support_caption_file is required when '
                             'support enriched class tokens are enabled.')

        with open(self.support_caption_file, 'r', encoding='utf-8') as f:
            caption_data = json.load(f)

        if isinstance(caption_data, dict):
            entries = caption_data.get('captions',
                                       caption_data.get('annotations',
                                                        caption_data))
        else:
            entries = caption_data

        class_to_idx = {name: i for i, name in enumerate(
            self.support_class_names)}
        prompt_bank = defaultdict(list)
        span_bank = defaultdict(list)

        if isinstance(entries, dict):
            iterable_entries = [
                value for value in entries.values() if isinstance(value, dict)
            ]
        else:
            iterable_entries = entries

        for item in iterable_entries:
            class_name = item.get('category_name', item.get('class_name'))
            caption = item.get('caption', '')
            if class_name not in class_to_idx or not caption:
                continue
            prompt, class_span = self._format_support_prompt(
                class_name, caption)
            class_idx = class_to_idx[class_name]
            prompt_bank[class_idx].append(prompt)
            span_bank[class_idx].append(class_span)

        for class_idx, class_name in enumerate(self.support_class_names):
            if len(prompt_bank[class_idx]) == 0:
                prompt, class_span = self._format_support_prompt(
                    class_name, '')
                prompt_bank[class_idx].append(prompt)
                span_bank[class_idx].append(class_span)

        prompt_texts = []
        prompt_labels = []
        prompt_class_spans = []
        ordered_bank = {}
        for class_idx in range(len(self.support_class_names)):
            ordered_bank[class_idx] = prompt_bank[class_idx]
            for prompt, class_span in zip(prompt_bank[class_idx],
                                          span_bank[class_idx]):
                prompt_texts.append(prompt)
                prompt_labels.append(class_idx)
                prompt_class_spans.append(class_span)

        self.support_prompt_bank = ordered_bank
        self.support_prompt_texts = prompt_texts
        self.support_prompt_labels = torch.tensor(prompt_labels,
                                                  dtype=torch.long)
        self.support_prompt_class_spans = prompt_class_spans
        self.support_tokenized = self._tokenize_support_prompts(prompt_texts)
        self.support_prompt_class_token_positions = \
            self._find_class_name_token_positions(
                self.support_tokenized['offset_mapping'],
                prompt_class_spans)
        self.support_class_token_output_indices = \
            self._build_class_token_output_indices()

    def _format_support_prompt(self, class_name: str,
                               caption: str) -> Tuple[str, Tuple[int, int]]:
        clean_class_name = clean_label_name(class_name).strip()
        parts = [clean_class_name]
        if caption:
            parts.append(caption.strip().rstrip('.'))
        if self.support_domain_attribute:
            parts.append(self.support_domain_attribute.strip().rstrip('.'))
        prompt = ', '.join(parts) + '.'
        return prompt, (0, len(clean_class_name))

    def _tokenize_support_prompts(self, prompts: Sequence[str]) -> dict:
        tokenized = self.language_model.tokenizer.batch_encode_plus(
            list(prompts),
            max_length=self.language_model.max_tokens,
            padding='max_length' if self.language_model.pad_to_max else
            'longest',
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True)
        return dict(tokenized)

    def _find_class_name_token_positions(self, offset_mapping: Tensor,
                                         class_spans: Sequence[Tuple[int,
                                                                    int]]):
        token_positions = []
        for prompt_idx, (span_start, span_end) in enumerate(class_spans):
            positions = []
            for token_idx, (token_start, token_end) in enumerate(
                    offset_mapping[prompt_idx].tolist()):
                if token_end <= token_start:
                    continue
                if token_start < span_end and token_end > span_start:
                    positions.append(token_idx)
            if len(positions) == 0:
                prompt = self.support_prompt_texts[prompt_idx]
                raise RuntimeError(
                    f'No class-name tokens found for prompt: {prompt}')
            token_positions.append(positions)
        return token_positions

    def _build_class_token_output_indices(self) -> Dict[int, List[int]]:
        class_token_indices = defaultdict(list)
        prompt_token_indices = []
        output_idx = 0
        for prompt_idx, class_idx in enumerate(
                self.support_prompt_labels.tolist()):
            current_prompt_indices = []
            for _ in self.support_prompt_class_token_positions[prompt_idx]:
                class_token_indices[class_idx].append(output_idx)
                current_prompt_indices.append(output_idx)
                output_idx += 1
            prompt_token_indices.append(current_prompt_indices)
        self.support_prompt_token_output_indices = prompt_token_indices
        return {
            class_idx: class_token_indices[class_idx]
            for class_idx in range(len(self.support_class_names))
        }

    def _prepare_cached_tokenized(self, device) -> dict:
        tokenized = {
            key: value.to(device)
            for key, value in self.support_tokenized.items()
            if key != 'offset_mapping'
        }
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'token_type_ids': tokenized.get('token_type_ids', None)
        }

    def _encode_support_prompt_features(self, tokenizer_input: dict) -> Tensor:
        """Encode enriched prompts with standard BERT row-wise attention."""
        bert = self.language_model.language_backbone.body.model
        outputs = bert(
            input_ids=tokenizer_input['input_ids'],
            attention_mask=tokenizer_input['attention_mask'],
            token_type_ids=tokenizer_input.get('token_type_ids', None),
            output_hidden_states=False,
            return_dict=True)
        return outputs.last_hidden_state

    def compute_support_class_token_features(self, return_debug: bool = False):
        """Encode prompts and keep only class-name token features."""
        self.build_support_prompt_bank()
        device = next(self.language_model.parameters()).device
        tokenizer_input = self._prepare_cached_tokenized(device)
        hidden_states = self._encode_support_prompt_features(tokenizer_input)

        selected_features = []
        selected_labels = []
        for prompt_idx, class_idx in enumerate(
                self.support_prompt_labels.tolist()):
            for token_idx in self.support_prompt_class_token_positions[
                    prompt_idx]:
                selected_features.append(hidden_states[prompt_idx, token_idx])
                selected_labels.append(class_idx)
        selected_features = torch.stack(selected_features, dim=0)
        selected_labels = torch.tensor(
            selected_labels, dtype=torch.long, device=device)

        if return_debug:
            class_token_counts = {
                self.support_class_names[class_idx]:
                len(self.support_class_token_output_indices[class_idx])
                for class_idx in range(len(self.support_class_names))
            }
            debug = dict(
                input_shape=tuple(tokenizer_input['input_ids'].shape),
                prompt_hidden_shape=tuple(hidden_states.shape),
                selected_token_shape=tuple(selected_features.shape),
                class_token_counts=class_token_counts)
            return selected_features, selected_labels, debug
        return selected_features, selected_labels

    def build_support_token_text_dict(self, batch_size: int, device) -> Dict:
        """Build text_dict from selected support class-name token features."""
        if (not self.training and self._cached_eval_support_token_text_dict
                is not None):
            cached = self._cached_eval_support_token_text_dict
            return {
                key: value.to(device)
                for key, value in cached.items()
            }

        token_features, _, debug = self.compute_support_class_token_features(
            return_debug=True)
        if self.text_feat_map is not None:
            token_features = self.text_feat_map(token_features)
        token_features = token_features.to(device)
        num_tokens = token_features.size(0)
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        if num_tokens > max_text_len:
            raise RuntimeError(
                f'Selected support class-name tokens ({num_tokens}) exceed '
                f'max_text_len ({max_text_len}). Increase max_text_len or '
                'reduce K-shot/support prompts.')

        embedded = token_features.unsqueeze(0).expand(batch_size, -1, -1)
        text_token_mask = torch.ones(
            batch_size, num_tokens, dtype=torch.bool, device=device)
        text_self_attention_masks = torch.zeros(
            num_tokens, num_tokens, dtype=torch.bool, device=device)
        for token_indices in self.support_prompt_token_output_indices:
            idx = torch.tensor(token_indices, dtype=torch.long, device=device)
            text_self_attention_masks[idx[:, None], idx[None, :]] = True
        text_self_attention_masks = text_self_attention_masks.unsqueeze(
            0).expand(batch_size, -1, -1)
        position_ids = torch.arange(
            num_tokens, dtype=torch.long,
            device=device).unsqueeze(0).expand(batch_size, -1)
        text_dict = dict(
            embedded=embedded,
            text_token_mask=text_token_mask,
            masks=text_self_attention_masks,
            position_ids=position_ids)

        if self.debug_text_tokens and not self._printed_text_token_debug:
            self._print_support_token_debug(debug, token_features, text_dict)
            self._register_bert_grad_debug_hook()
            self._debug_support_token_isolation()
            self._printed_text_token_debug = True

        if not self.training:
            self._cached_eval_support_token_text_dict = {
                key: value.detach().cpu()
                for key, value in text_dict.items()
            }
        return text_dict

    def build_support_token_positive_maps(self, gt_labels: List[Tensor],
                                          device) -> List[Tensor]:
        """Create positive maps from classes to selected class-name tokens."""
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        positive_maps = []
        for labels in gt_labels:
            positive_map = torch.zeros(
                labels.size(0), max_text_len, device=device)
            for row_idx, label in enumerate(labels.detach().cpu().tolist()):
                token_indices = self.support_class_token_output_indices[label]
                positive_map[row_idx, token_indices] = 1.0
            positive_maps.append(positive_map)
        return positive_maps

    def build_support_token_positive_map(self) -> dict:
        """Map 1-based class ids to selected class-name token positions."""
        return {
            class_idx + 1: self.support_class_token_output_indices[class_idx]
            for class_idx in range(len(self.support_class_names))
        }

    def _print_support_token_debug(self, debug: dict, token_features: Tensor,
                                   text_dict: Dict) -> None:
        print('[SupportClassTokens] support prompt counts:')
        for class_idx, class_name in enumerate(self.support_class_names):
            print(f'  - {class_name}: '
                  f'{len(self.support_prompt_bank[class_idx])}')
        print('[SupportClassTokens] class-name token counts:',
              debug['class_token_counts'])
        print('[SupportClassTokens] example prompt:',
              self.support_prompt_texts[0])
        print('[SupportClassTokens] example class token ids:',
              self.support_prompt_class_token_positions[0])
        print('[SupportClassTokens] BERT input shape:',
              debug['input_shape'])
        print('[SupportClassTokens] BERT prompt hidden shape:',
              debug['prompt_hidden_shape'])
        print('[SupportClassTokens] selected class-token feature shape:',
              debug['selected_token_shape'])
        print('[SupportClassTokens] projected selected-token shape:',
              tuple(token_features.shape))
        print('[SupportClassTokens] selected tokens require_grad:',
              token_features.requires_grad)
        first_bert_param = next(self.language_model.parameters())
        print('[SupportClassTokens] BERT parameter requires_grad:',
              first_bert_param.requires_grad)
        prompt_labels = self.support_prompt_labels.tolist()
        cross_prompt_allowed = False
        for left_idx, left_tokens in enumerate(
                self.support_prompt_token_output_indices):
            for right_idx, right_tokens in enumerate(
                    self.support_prompt_token_output_indices):
                if left_idx == right_idx or not left_tokens or \
                        not right_tokens:
                    continue
                if text_dict['masks'][0][left_tokens][:,
                                                       right_tokens].any(
                                                       ).item():
                    cross_prompt_allowed = True
                if prompt_labels[left_idx] != prompt_labels[right_idx] and \
                        text_dict['masks'][0][left_tokens][:,
                                                           right_tokens].any(
                                                           ).item():
                    cross_prompt_allowed = True
        print('[SupportClassTokens] cross-prompt/class self-attention '
              'blocked:', not cross_prompt_allowed)

    def _register_bert_grad_debug_hook(self) -> None:
        if self._registered_bert_grad_debug_hook:
            return
        for name, param in self.language_model.named_parameters():
            if param.requires_grad:
                def _hook(grad, param_name=name):
                    print('[SupportClassTokens] backward BERT gradient norm '
                          f'({param_name}): {grad.norm().item():.6f}')
                param.register_hook(_hook)
                self._registered_bert_grad_debug_hook = True
                return

    def _debug_support_token_isolation(self) -> None:
        if len(self.support_class_names) < 2:
            print('[SupportClassTokens] isolation test skipped: '
                  'num_classes < 2')
            return

        with torch.no_grad():
            original, labels = self.compute_support_class_token_features()
            modified_prompts = list(self.support_prompt_texts)
            target_class = int(labels[0].item())
            prompt_labels = self.support_prompt_labels.tolist()
            target_prompt_indices = [
                idx for idx, label in enumerate(prompt_labels)
                if label == target_class
            ]
            for idx in target_prompt_indices:
                modified_prompts[idx] = modified_prompts[idx] + \
                    ' changed caption token.'
            tokenized = self._tokenize_support_prompts(modified_prompts)
            old_tokenized = self.support_tokenized
            self.support_tokenized = tokenized
            modified, _ = self.compute_support_class_token_features()
            self.support_tokenized = old_tokenized
            diffs = (original - modified).abs().sum(dim=1)
            target_changed = bool((diffs[labels == target_class] > 0).any())
            others_same = torch.allclose(
                diffs[labels != target_class],
                diffs.new_zeros((labels != target_class).sum()),
                atol=1e-6)
        print('[SupportClassTokens] isolation changed class changed:',
              target_changed)
        print('[SupportClassTokens] isolation other classes unchanged:',
              bool(others_same))

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if self.use_enriched_class_tokens:
            text_dict = self.build_support_token_text_dict(
                len(batch_inputs), batch_inputs.device)
            positive_maps = self.build_support_token_positive_maps(
                gt_labels, batch_inputs.device)
            for i, data_samples in enumerate(batch_data_samples):
                positive_map = positive_maps[i].bool().float()
                text_token_mask = text_dict['text_token_mask'][i]
                data_samples.gt_instances.positive_maps = positive_map
                data_samples.gt_instances.text_token_mask = \
                    text_token_mask.unsqueeze(0).repeat(
                        len(positive_map), 1)

            if self.use_autocast:
                with autocast(enabled=True):
                    visual_features = self.extract_feat(batch_inputs)
            else:
                visual_features = self.extract_feat(batch_inputs)
            head_inputs_dict = self.forward_transformer(
                visual_features, text_dict, batch_data_samples)
            losses = self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)
            return losses

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        if self.use_enriched_class_tokens:
            visual_feats = self.extract_feat(batch_inputs)
            text_dict = self.build_support_token_text_dict(
                len(batch_inputs), batch_inputs.device)
            token_positive_map = self.build_support_token_positive_map()
            entities = self.support_class_names
            for data_sample in batch_data_samples:
                data_sample.token_positive_map = token_positive_map

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)
            for data_sample, pred_instances in zip(batch_data_samples,
                                                   results_list):
                if len(pred_instances) > 0:
                    pred_instances.label_names = [
                        entities[label.item()] if label.item() < len(entities)
                        else 'unobject' for label in pred_instances.labels
                    ]
                data_sample.pred_instances = pred_instances
            return batch_data_samples

        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            # text feature map layer
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
