# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Optional, Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@MODELS.register_module()
class BertModel(BaseModel):
    """BERT model for language embedding only encoder.

    Args:
        name (str, optional): name of the pretrained BERT model from
            HuggingFace. Defaults to bert-base-uncased.
        max_tokens (int, optional): maximum number of tokens to be
            used for BERT. Defaults to 256.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
             Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. It cannot be None when `use_sub_sentence_represent`
            is True. Defaults to None.
        add_pooling_layer (bool, optional): whether to adding pooling
            layer in bert encoder. Defaults to False.
        num_layers_of_embedded (int, optional): number of layers of
            the embedded model. Defaults to 1.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
             Defaults to False.
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          BertEncoder(
                              name,
                              add_pooling_layer=add_pooling_layer,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def _build_visual_token_inputs(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
            position_ids: Optional[torch.Tensor],
            token_type_ids: Optional[torch.Tensor],
            visual_tokens: Sequence[torch.Tensor],
            visual_token_positive_maps: Sequence[torch.Tensor]) -> dict:
        """Append visual tokens and connect them to their class tokens."""
        batch_size, text_length = input_ids.shape
        if len(visual_tokens) != batch_size or len(
                visual_token_positive_maps) != batch_size:
            raise ValueError(
                'Visual tokens and positive maps must match the batch size.')

        bert = self.language_backbone.body.model
        word_embeddings = bert.get_input_embeddings()(input_ids)
        embedding_dim = word_embeddings.size(-1)
        max_visual_tokens = max(tokens.size(0) for tokens in visual_tokens)
        if max_visual_tokens == 0:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids)

        total_length = text_length + max_visual_tokens
        if total_length > bert.config.max_position_embeddings:
            raise ValueError(
                f'Text and visual tokens require {total_length} positions, '
                'but BERT supports only '
                f'{bert.config.max_position_embeddings}.')

        padded_visual_tokens = []
        for tokens, positive_map in zip(visual_tokens,
                                        visual_token_positive_maps):
            if tokens.ndim != 2 or tokens.size(-1) != embedding_dim:
                raise ValueError(
                    f'Visual tokens must have shape [N, {embedding_dim}].')
            if positive_map.ndim != 2 or positive_map.size(0) != tokens.size(0):
                raise ValueError(
                    'Each visual token must have one class positive map.')
            tokens = tokens.to(device=word_embeddings.device,
                               dtype=word_embeddings.dtype)
            padding = tokens.new_zeros(
                (max_visual_tokens - tokens.size(0), embedding_dim))
            padded_visual_tokens.append(torch.cat([tokens, padding], dim=0))
        inputs_embeds = torch.cat(
            [word_embeddings,
             torch.stack(padded_visual_tokens)], dim=1)

        if attention_mask.ndim == 2:
            pair_attention_mask = (
                attention_mask.bool().unsqueeze(1)
                & attention_mask.bool().unsqueeze(2))
        else:
            pair_attention_mask = attention_mask.bool()
        expanded_attention_mask = torch.zeros(
            (batch_size, total_length, total_length),
            dtype=torch.bool,
            device=input_ids.device)
        expanded_attention_mask[:, :text_length, :text_length] = \
            pair_attention_mask

        if position_ids is None:
            text_position_ids = torch.arange(
                text_length, device=input_ids.device).unsqueeze(0).expand(
                    batch_size, -1)
        else:
            text_position_ids = position_ids
        visual_position_ids = torch.zeros(
            (batch_size, max_visual_tokens),
            dtype=text_position_ids.dtype,
            device=input_ids.device)

        for batch_index, (tokens, positive_map) in enumerate(
                zip(visual_tokens, visual_token_positive_maps)):
            positive_map = positive_map.to(input_ids.device)
            for token_index in range(tokens.size(0)):
                class_positions = torch.nonzero(
                    positive_map[token_index, :text_length] > 0,
                    as_tuple=False).flatten()
                if class_positions.numel() == 0:
                    raise ValueError(
                        'A visual token has no matching class-name token.')
                visual_position = text_length + token_index
                if attention_mask.ndim == 3:
                    class_context = pair_attention_mask[
                        batch_index, class_positions[0]]
                else:
                    class_context = torch.zeros(
                        text_length, dtype=torch.bool, device=input_ids.device)
                    class_context[class_positions] = True
                expanded_attention_mask[
                    batch_index, visual_position, :text_length] = class_context
                expanded_attention_mask[
                    batch_index, :text_length, visual_position] = class_context
                expanded_attention_mask[
                    batch_index, visual_position, visual_position] = True
                visual_position_ids[batch_index, token_index] = torch.clamp(
                    text_position_ids[batch_index, class_positions].max() + 1,
                    max=bert.config.max_position_embeddings - 1)

            for token_index in range(tokens.size(0), max_visual_tokens):
                visual_position = text_length + token_index
                expanded_attention_mask[
                    batch_index, visual_position, visual_position] = True

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        expanded_token_type_ids = torch.cat([
            token_type_ids,
            token_type_ids.new_zeros((batch_size, max_visual_tokens))
        ], dim=1)
        expanded_position_ids = torch.cat(
            [text_position_ids, visual_position_ids], dim=1)

        # HFBertModel applies position/type embeddings and its embedding
        # LayerNorm to every row of inputs_embeds, including visual tokens.
        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_attention_mask,
            position_ids=expanded_position_ids,
            token_type_ids=expanded_token_type_ids)

    def forward(
            self,
            captions: Sequence[str],
            visual_tokens: Optional[Sequence[torch.Tensor]] = None,
            visual_token_positive_maps: Optional[
                Sequence[torch.Tensor]] = None,
            class_name_positive_maps: Optional[
                Sequence[torch.Tensor]] = None,
            **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        input_ids = tokenized.input_ids
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
            token_type_ids = tokenized['token_type_ids']

        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        visual_inputs = (visual_tokens, visual_token_positive_maps,
                         class_name_positive_maps)
        if any(item is None for item in visual_inputs) and not all(
                item is None for item in visual_inputs):
            raise ValueError(
                'Visual tokens, their positive maps, and all class-name '
                'positive maps must be provided together.')
        if visual_tokens is not None:
            tokenizer_input = self._build_visual_token_inputs(
                input_ids, attention_mask, position_ids, token_type_ids,
                visual_tokens, visual_token_positive_maps)

        language_dict_features = self.language_backbone(tokenizer_input)
        if visual_tokens is not None:
            language_dict_features['embedded'] = language_dict_features[
                'embedded'][:, :input_ids.size(1)]
            language_dict_features['hidden'] = language_dict_features[
                'hidden'][:, :input_ids.size(1)]
            class_name_token_mask = torch.stack([
                (positive_map[:, :input_ids.size(1)] > 0).any(dim=0)
                for positive_map in class_name_positive_maps
            ]).to(device)
            class_name_token_mask &= tokenized.attention_mask.bool()
            language_dict_features['embedded'] = language_dict_features[
                'embedded'] * class_name_token_mask.unsqueeze(-1)
            language_dict_features['hidden'] = language_dict_features[
                'hidden'] * class_name_token_mask.unsqueeze(-1)

            if attention_mask.ndim == 2:
                text_self_attention_mask = (
                    attention_mask.bool().unsqueeze(1)
                    & attention_mask.bool().unsqueeze(2))
            else:
                text_self_attention_mask = attention_mask.bool()
            text_self_attention_mask = (
                text_self_attention_mask
                & class_name_token_mask.unsqueeze(1)
                & class_name_token_mask.unsqueeze(2))
            text_self_attention_mask |= torch.eye(
                input_ids.size(1), dtype=torch.bool,
                device=device).unsqueeze(0)
            language_dict_features['masks'] = text_self_attention_mask
        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
            if visual_tokens is None:
                language_dict_features[
                    'text_token_mask'] = tokenized.attention_mask.bool()
            else:
                language_dict_features[
                    'text_token_mask'] = class_name_token_mask
        return language_dict_features


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        model_inputs = dict(
            attention_mask=mask,
            position_ids=x['position_ids'],
            token_type_ids=x['token_type_ids'],
            output_hidden_states=True)
        if 'inputs_embeds' in x:
            model_inputs['inputs_embeds'] = x['inputs_embeds']
        else:
            model_inputs['input_ids'] = x['input_ids']
        outputs = self.model(**model_inputs)

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results
