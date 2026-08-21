# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from PIL import Image
from torch import Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from ..utils import SupportBlipCaptioner
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
                 use_blip_prototypes: bool = False,
                 support_ann_file: Optional[str] = None,
                 support_class_names: Optional[Sequence[str]] = None,
                 support_domain_attribute: Optional[str] = None,
                 support_image_root: Optional[str] = None,
                 support_image_batch_size: int = 2,
                 blip_model_name: str =
                 'Salesforce/blip-image-captioning-base',
                 blip_gradient_checkpointing: bool = True,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.use_blip_prototypes = use_blip_prototypes
        self.support_ann_file = support_ann_file
        self.support_class_names = list(support_class_names or [])
        self.support_domain_attribute = support_domain_attribute
        self.support_image_root = support_image_root
        if support_image_batch_size <= 0:
            raise ValueError('support_image_batch_size must be positive.')
        self.support_image_batch_size = support_image_batch_size
        self.blip_model_name = blip_model_name
        self.blip_gradient_checkpointing = blip_gradient_checkpointing
        self.support_entries = None
        self._support_pixel_values = None
        self._support_pixel_labels = None
        self._support_class_token_ids = None
        self._support_colon_token_ids = None
        self._cached_eval_support_prototypes = None
        super().__init__(*args, **kwargs)
        if self.use_blip_prototypes:
            self.support_blip_captioner = SupportBlipCaptioner(
                model_name=self.blip_model_name,
                gradient_checkpointing=self.blip_gradient_checkpointing)
            bert = self.language_model.language_backbone.body.model
            bert_hidden_size = bert.config.hidden_size
            if self.text_feat_map.in_features != bert_hidden_size or \
                    bert_hidden_size != \
                    self.support_blip_captioner.hidden_size:
                raise ValueError(
                    'BLIP caption decoder, Grounding DINO BERT and '
                    'text_feat_map input dimensions must match.')
            self.blip_shared_vocab_size = \
                self.support_blip_captioner.validate_grounding_tokenizer(
                    self.language_model.tokenizer)
            bert_vocab_size = bert.get_input_embeddings().num_embeddings
            if self.blip_shared_vocab_size != bert_vocab_size:
                raise ValueError(
                    'Validated shared vocabulary size must match the '
                    'Grounding DINO BERT embedding table.')
            self.build_support_object_bank()
        else:
            self.support_blip_captioner = None
            self.blip_shared_vocab_size = None

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
        """Switch mode and clear stale detached support prototypes."""
        super().train(mode)
        if mode:
            self._cached_eval_support_prototypes = None
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

    def build_support_object_bank(self) -> None:
        """Load support object paths, GT boxes and labels from COCO JSON."""
        if self.support_entries is not None:
            return
        if not self.support_ann_file:
            raise ValueError('support_ann_file is required when BLIP '
                             'caption prototypes are enabled.')

        with open(self.support_ann_file, 'r', encoding='utf-8') as f:
            support_data = json.load(f)
        if not isinstance(support_data, dict):
            raise ValueError('Support annotation file must be a COCO JSON '
                             'dictionary.')

        images = support_data.get('images')
        annotations = support_data.get('annotations')
        categories = support_data.get('categories')
        if not isinstance(images, list) or not isinstance(annotations, list) \
                or not isinstance(categories, list):
            raise ValueError('Support annotation file must contain COCO '
                             'images, annotations and categories arrays.')

        class_to_idx = {
            name: idx
            for idx, name in enumerate(self.support_class_names)
        }
        images_by_id = {}
        category_names_by_id = {}
        support_entries = []
        validation_errors = []

        for image_idx, image in enumerate(images):
            if not isinstance(image, dict):
                validation_errors.append(
                    f'image {image_idx} is not a dictionary')
                continue
            image_id = image.get('id')
            file_name = image.get('file_name')
            if image_id is None:
                validation_errors.append(
                    f'image {image_idx} has no id')
                continue
            if image_id in images_by_id:
                validation_errors.append(
                    f'duplicate image id {image_id!r}')
                continue
            if not isinstance(file_name, str) or not file_name:
                validation_errors.append(
                    f'image {image_idx} has an invalid file_name')
                continue
            images_by_id[image_id] = file_name

        for category_idx, category in enumerate(categories):
            if not isinstance(category, dict):
                validation_errors.append(
                    f'category {category_idx} is not a dictionary')
                continue
            category_id = category.get('id')
            category_name = category.get('name')
            if category_id is None:
                validation_errors.append(
                    f'category {category_idx} has no id')
                continue
            if category_id in category_names_by_id:
                validation_errors.append(
                    f'duplicate category id {category_id!r}')
                continue
            if not isinstance(category_name, str) or not category_name:
                validation_errors.append(
                    f'category {category_idx} has an invalid name')
                continue
            category_names_by_id[category_id] = category_name

        for ann_idx, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                validation_errors.append(
                    f'annotation {ann_idx} is not a dictionary')
                continue
            image_id = annotation.get('image_id')
            category_id = annotation.get('category_id')
            if image_id not in images_by_id:
                validation_errors.append(
                    f'annotation {ann_idx} references unknown image id '
                    f'{image_id!r}')
                continue
            if category_id not in category_names_by_id:
                validation_errors.append(
                    f'annotation {ann_idx} references unknown category id '
                    f'{category_id!r}')
                continue
            category_name = category_names_by_id[category_id]
            if category_name not in class_to_idx:
                validation_errors.append(
                    f'annotation {ann_idx} has unknown class '
                    f'{category_name!r}')
                continue
            bbox = annotation.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                validation_errors.append(
                    f'annotation {ann_idx} has an invalid xywh bbox')
                continue
            try:
                x, y, width, height = [float(value) for value in bbox]
            except (TypeError, ValueError):
                validation_errors.append(
                    f'annotation {ann_idx} has a non-numeric bbox')
                continue
            if width <= 0 or height <= 0:
                validation_errors.append(
                    f'annotation {ann_idx} has a non-positive bbox size')
                continue

            support_entries.append(
                dict(
                    file_name=images_by_id[image_id],
                    bbox=(x, y, width, height),
                    class_idx=class_to_idx[category_name]))

        support_counts = [0] * len(self.support_class_names)
        for entry in support_entries:
            support_counts[entry['class_idx']] += 1
        missing_classes = [
            self.support_class_names[class_idx]
            for class_idx, count in enumerate(support_counts) if count == 0
        ]
        if missing_classes:
            validation_errors.append(
                f'classes without support objects: {missing_classes}')
        if validation_errors:
            raise ValueError(
                f'Invalid support annotation file {self.support_ann_file}: '
                + '; '.join(validation_errors))

        if self.support_image_root is None:
            raise ValueError('support_image_root is required for support '
                             'images referenced by the COCO annotation file.')
        self.support_entries = support_entries

    def _prepare_support_image_inputs(self) -> None:
        """Crop and preprocess support objects once, retaining CPU pixels."""
        if self._support_pixel_values is not None:
            return
        self.build_support_object_bank()
        image_cache = {}
        crops = []
        labels = []
        for entry in self.support_entries:
            image_path = entry['file_name']
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.support_image_root, image_path)
            if image_path not in image_cache:
                try:
                    image_cache[image_path] = Image.open(image_path).convert(
                        'RGB')
                except (OSError, ValueError) as error:
                    message = f'Unable to read support image: {image_path}'
                    raise FileNotFoundError(message) from error
            image = image_cache[image_path]
            x, y, width, height = entry['bbox']
            left = max(0, int(round(x)))
            top = max(0, int(round(y)))
            right = min(image.width, int(round(x + width)))
            bottom = min(image.height, int(round(y + height)))
            if right <= left or bottom <= top:
                raise ValueError(
                    f'Support bbox becomes empty after clipping: '
                    f'{entry["bbox"]} in {image_path}')
            crops.append(
                image.crop((left, top, right, bottom)).convert('RGB'))
            class_idx = entry['class_idx']
            labels.append(class_idx)

        self._support_pixel_values = \
            self.support_blip_captioner.preprocess_images(crops)
        self._support_pixel_labels = torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def aggregate_support_caption_features(
            object_features: Tensor, object_labels: Tensor,
            num_classes: int) -> Tensor:
        """Average caption-enriched class-token features by class."""
        if object_features.dim() != 2:
            raise ValueError(
                'object_features must have shape [N, hidden].')
        if object_features.size(0) != object_labels.numel():
            raise ValueError('Support feature and label counts must match.')
        if num_classes <= 0:
            raise ValueError('num_classes must be positive.')
        if ((object_labels < 0) | (object_labels >= num_classes)).any():
            raise ValueError('Support labels are outside the class range.')

        class_prototypes = []
        missing_classes = []
        for class_idx in range(num_classes):
            class_mask = object_labels == class_idx
            if not class_mask.any():
                missing_classes.append(class_idx)
                continue

            class_prototypes.append(object_features[class_mask].mean(dim=0))
        if missing_classes:
            raise ValueError('No support object found for class indices: '
                             f'{missing_classes}.')
        return torch.stack(class_prototypes, dim=0)

    def _prepare_support_prompt_tokens(self) -> None:
        """Cache class-name and colon token ids without caption decoding."""
        if self._support_class_token_ids is not None:
            return
        tokenizer = self.language_model.tokenizer
        class_token_ids = []
        for class_name in self.support_class_names:
            clean_name = clean_label_name(class_name).strip()
            token_ids = tokenizer.encode(
                clean_name, add_special_tokens=False)
            if len(token_ids) == 0:
                raise ValueError(
                    f'No BERT tokens found for support class {class_name}.')
            class_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
        colon_token_ids = tokenizer.encode(':', add_special_tokens=False)
        if len(colon_token_ids) == 0:
            raise ValueError(
                'Grounding DINO BERT tokenizer cannot encode ":".')
        self._support_class_token_ids = class_token_ids
        self._support_colon_token_ids = torch.tensor(
            colon_token_ids, dtype=torch.long)

    def _encode_caption_enriched_class_features(
            self, caption_outputs: Dict[str, Tensor],
            object_labels: Tensor) -> Tensor:
        """Encode ``class_name: caption`` and keep class-name states only."""
        self._prepare_support_prompt_tokens()
        distributions = caption_outputs['token_distributions']
        caption_mask = caption_outputs['caption_mask']
        if distributions.dim() != 3 or \
                caption_mask.shape != distributions.shape[:2]:
            raise ValueError('Invalid differentiable BLIP caption shapes.')
        if distributions.size(0) != object_labels.numel():
            raise ValueError('Caption and support label counts must match.')

        bert = self.language_model.language_backbone.body.model
        bert_embeddings = bert.get_input_embeddings()
        embedding_weight = bert_embeddings.weight
        tokenizer = self.language_model.tokenizer
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError('Grounding DINO BERT special tokens are missing.')

        prompt_embeddings = []
        class_spans = []
        for row_idx, class_idx in enumerate(object_labels.tolist()):
            class_ids = self._support_class_token_ids[class_idx].to(
                distributions.device)
            prefix_ids = torch.cat([
                torch.tensor(
                    [cls_token_id],
                    dtype=torch.long,
                    device=distributions.device),
                class_ids,
                self._support_colon_token_ids.to(distributions.device)
            ])
            prefix_embeddings = bert_embeddings(prefix_ids)
            active_distribution = distributions[
                row_idx, caption_mask[row_idx], :self.blip_shared_vocab_size]
            active_distribution = active_distribution.to(
                embedding_weight.dtype)
            caption_embeddings = active_distribution @ embedding_weight
            sep_embedding = bert_embeddings(
                torch.tensor(
                    [sep_token_id],
                    dtype=torch.long,
                    device=distributions.device))
            row_embeddings = torch.cat([
                prefix_embeddings, caption_embeddings, sep_embedding
            ], dim=0)
            if row_embeddings.size(0) > self.language_model.max_tokens:
                raise RuntimeError(
                    'BLIP enriched support prompt exceeds Grounding DINO '
                    f'BERT max_tokens={self.language_model.max_tokens}.')
            prompt_embeddings.append(row_embeddings)
            class_spans.append((1, 1 + class_ids.numel()))

        max_length = max(row.size(0) for row in prompt_embeddings)
        hidden_size = embedding_weight.size(1)
        inputs_embeds = embedding_weight.new_zeros(
            len(prompt_embeddings), max_length, hidden_size)
        attention_mask = torch.zeros(
            len(prompt_embeddings),
            max_length,
            dtype=torch.long,
            device=distributions.device)
        for row_idx, row_embeddings in enumerate(prompt_embeddings):
            row_length = row_embeddings.size(0)
            inputs_embeds[row_idx, :row_length] = row_embeddings
            attention_mask[row_idx, :row_length] = 1

        bert_outputs = bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(attention_mask),
            output_hidden_states=False,
            return_dict=True)
        hidden_states = bert_outputs.last_hidden_state
        object_features = [
            hidden_states[row_idx, span_start:span_end].mean(dim=0)
            for row_idx, (span_start, span_end) in enumerate(class_spans)
        ]
        return torch.stack(object_features, dim=0)

    def _compute_support_batch_caption_features(
            self, pixel_values: Tensor, labels: Tensor) -> Tensor:
        """Run the differentiable support caption pipeline for one batch."""
        caption_outputs = self.support_blip_captioner(pixel_values)
        return self._encode_caption_enriched_class_features(
            caption_outputs, labels)

    def compute_support_caption_features(self, device) -> Tensor:
        """Recompute differentiable caption features with train checkpoint."""
        self._prepare_support_image_inputs()
        object_features = []
        for start_idx in range(0, self._support_pixel_values.size(0),
                               self.support_image_batch_size):
            end_idx = start_idx + self.support_image_batch_size
            pixel_values = self._support_pixel_values[start_idx:end_idx].to(
                device, non_blocking=True)
            labels = self._support_pixel_labels[start_idx:end_idx].to(device)
            if self.training:
                batch_features = gradient_checkpoint(
                    self._compute_support_batch_caption_features,
                    pixel_values,
                    labels,
                    use_reentrant=False)
            else:
                batch_features = \
                    self._compute_support_batch_caption_features(
                        pixel_values, labels)
            object_features.append(batch_features)
        caption_features = torch.cat(object_features, dim=0)
        labels = self._support_pixel_labels.to(device)
        return self.aggregate_support_caption_features(
            caption_features, labels, len(self.support_class_names))

    def compute_caption_support_prototypes(self, device) -> Tensor:
        """Build one projected differentiable caption prototype per class."""
        caption_prototypes = self.compute_support_caption_features(device)
        expected_shape = (
            len(self.support_class_names), self.text_feat_map.in_features)
        if tuple(caption_prototypes.shape) != expected_shape:
            raise RuntimeError(
                'Unexpected aggregated caption prototype shape: '
                f'{tuple(caption_prototypes.shape)}; expected '
                f'{expected_shape}.')
        return self.text_feat_map(caption_prototypes)

    def _prototypes_to_text_dict(self, prototypes: Tensor,
                                 batch_size: int) -> Dict:
        """Build a Grounding DINO text dictionary from class prototypes."""
        expected_shape = (len(self.support_class_names), self.embed_dims)
        if tuple(prototypes.shape) != expected_shape:
            raise ValueError(
                'Caption prototypes must have shape [classes, hidden], got '
                f'{tuple(prototypes.shape)}; expected {expected_shape}.')
        device = prototypes.device
        num_tokens = prototypes.size(0)
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        if num_tokens > max_text_len:
            raise RuntimeError(
                f'Class text prototypes ({num_tokens}) exceed max_text_len '
                f'({max_text_len}).')

        embedded = prototypes.unsqueeze(0).expand(
            batch_size, -1, -1)
        text_token_mask = torch.ones(
            batch_size, num_tokens, dtype=torch.bool, device=device)
        text_self_attention_masks = torch.eye(
            num_tokens, dtype=torch.bool,
            device=device).unsqueeze(0).expand(batch_size, -1, -1)
        position_ids = torch.arange(
            num_tokens, dtype=torch.long,
            device=device).unsqueeze(0).expand(batch_size, -1)
        text_dict = dict(
            embedded=embedded,
            text_token_mask=text_token_mask,
            masks=text_self_attention_masks,
            position_ids=position_ids)
        return text_dict

    def build_prototype_text_dict(self, batch_size: int, device) -> Dict:
        """Build text_dict from differentiable BLIP caption prototypes."""
        if (not self.training and
                self._cached_eval_support_prototypes is not None):
            prototypes = self._cached_eval_support_prototypes.to(device)
            return self._prototypes_to_text_dict(prototypes, batch_size)

        prototypes = self.compute_caption_support_prototypes(device)
        if not self.training:
            prototypes = prototypes.detach()
            self._cached_eval_support_prototypes = prototypes.cpu()
        return self._prototypes_to_text_dict(prototypes, batch_size)

    def build_prototype_positive_maps(self, gt_labels: List[Tensor],
                                      device) -> List[Tensor]:
        """Map each class target to its single enriched text prototype."""
        max_text_len = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        num_classes = len(self.support_class_names)
        if num_classes > max_text_len:
            raise RuntimeError(
                f'Class prototypes ({num_classes}) exceed max_text_len '
                f'({max_text_len}).')
        positive_maps = []
        for labels in gt_labels:
            positive_map = torch.zeros(
                labels.size(0), max_text_len, device=device)
            for row_idx, label in enumerate(labels.detach().cpu().tolist()):
                if label < 0 or label >= num_classes:
                    raise ValueError(
                        f'Class label {label} exceeds prototype range.')
                positive_map[row_idx, label] = 1.0
            positive_maps.append(positive_map)
        return positive_maps

    def build_prototype_token_positive_map(self) -> dict:
        """Map every class to its single enriched prototype position."""
        return {
            class_idx + 1: [class_idx]
            for class_idx in range(len(self.support_class_names))
        }

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

        if self.use_blip_prototypes:
            text_dict = self.build_prototype_text_dict(
                len(batch_inputs), batch_inputs.device)
            positive_maps = self.build_prototype_positive_maps(
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
        if self.use_blip_prototypes:
            text_dict = self.build_prototype_text_dict(
                len(batch_inputs), batch_inputs.device)
            visual_feats = self.extract_feat(batch_inputs)
            token_positive_map = self.build_prototype_token_positive_map()
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
