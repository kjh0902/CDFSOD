"""Native PyTorch LoRA with unchanged pretrained state-dict keys."""
import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Linear):
    def __init__(self, original, rank, alpha, dropout):
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(self.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(self.weight.new_empty(self.out_features, rank))
        self.reset_lora_parameters()
        self.train(original.training)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)


class LoRAConv2d(nn.Conv2d):
    """Factor a convolution into a rank-r spatial conv and a 1x1 conv."""
    def __init__(self, original, rank, alpha, dropout):
        nn.Module.__init__(self)
        if original.groups != 1:
            raise ValueError('Full LoRA requires ungrouped convolutions.')
        for name in ('in_channels', 'out_channels', 'kernel_size', 'stride',
                     'padding', 'dilation', 'groups', 'padding_mode',
                     '_reversed_padding_repeated_twice'):
            setattr(self, name, getattr(original, name))
        self.weight = original.weight
        self.bias = original.bias
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(self.weight.new_empty(
            rank, self.in_channels * math.prod(self.kernel_size)))
        self.lora_B = nn.Parameter(self.weight.new_empty(self.out_channels, rank))
        self.reset_lora_parameters()
        self.train(original.training)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self._conv_forward(x, self.weight, self.bias)
        hidden = self._conv_forward(self.lora_dropout(x), self.lora_A.view(
            -1, self.in_channels, *self.kernel_size), None)
        return base + self.scaling * F.conv2d(
            hidden, self.lora_B[:, :, None, None])


class LoRAProjection(nn.Module):
    """An adapter for a projection whose base weight lives in fused QKV."""
    def __init__(self, in_features, out_features, reference, rank, alpha, dropout):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(reference.new_empty(rank, in_features))
        self.lora_B = nn.Parameter(reference.new_empty(out_features, rank))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.scaling * F.linear(
            F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)


class LoRAMultiheadAttention(nn.Module):
    """LoRA on Q/K/V and output, including PyTorch's fused QKV weights.

    nn.MultiheadAttention calls functional linear projections directly, so
    replacing only its out_proj child would silently bypass that adapter.
    This implements the same attention with explicit LoRA projections and
    activation dropout, preserving every original parameter name.
    """
    def __init__(self, original, rank, alpha, dropout):
        super().__init__()
        for name in ('embed_dim', 'num_heads', 'head_dim', 'dropout',
                     'batch_first', 'add_zero_attn', '_qkv_same_embed_dim',
                     'kdim', 'vdim'):
            setattr(self, name, getattr(original, name))
        for name in ('in_proj_weight', 'in_proj_bias', 'q_proj_weight',
                     'k_proj_weight', 'v_proj_weight', 'bias_k', 'bias_v'):
            self.register_parameter(name, getattr(original, name))
        self.out_proj = LoRALinear(original.out_proj, rank, alpha, dropout)
        self.projections = nn.ModuleList([
            LoRAProjection(dim, self.embed_dim, original.out_proj.weight,
                           rank, alpha, dropout)
            for dim in (self.embed_dim, self.kdim, self.vdim)])
        self.train(original.training)

    def _project(self, x, index):
        if self._qkv_same_embed_dim:
            weight = self.in_proj_weight.chunk(3)[index]
        else:
            weight = (self.q_proj_weight, self.k_proj_weight,
                      self.v_proj_weight)[index]
        bias = None if self.in_proj_bias is None else self.in_proj_bias.chunk(3)[index]
        return F.linear(x, weight, bias) + self.projections[index](x)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        unbatched = query.dim() == 2
        if unbatched:
            query, key, value = (x.unsqueeze(1) for x in (query, key, value))
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        elif self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        q, k, v = (self._project(x, i) for i, x in enumerate((query, key, value)))
        target_len, batch, _ = q.shape
        if is_causal and attn_mask is None:
            raise RuntimeError('Need attn_mask if specifying the is_causal hint.')
        if self.bias_k is not None:
            k = torch.cat((k, self.bias_k.repeat(1, batch, 1)))
            v = torch.cat((v, self.bias_v.repeat(1, batch, 1)))
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        q, k, v = (x.reshape(x.shape[0], batch, self.num_heads,
                            self.head_dim).permute(1, 2, 0, 3) for x in (q, k, v))
        if self.add_zero_attn:
            k = F.pad(k, (0, 0, 0, 1))
            v = F.pad(v, (0, 0, 0, 1))
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        source_len = k.shape[-2]
        mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = attn_mask.view(1, 1, target_len, source_len)
            else:
                mask = attn_mask.reshape(batch, self.num_heads, target_len, source_len)
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))
        if key_padding_mask is not None:
            padding = key_padding_mask.view(batch, 1, 1, source_len)
            if padding.dtype == torch.bool:
                padding = torch.zeros_like(padding, dtype=q.dtype).masked_fill(padding, float('-inf'))
            mask = padding if mask is None else mask + padding
        dropout = self.dropout if self.training else 0.
        weights = None
        if need_weights:
            weights = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
            if mask is not None:
                weights = weights + mask
            weights = F.dropout(weights.softmax(dim=-1), p=dropout, training=self.training)
            output = weights @ v
            if average_attn_weights:
                weights = weights.mean(dim=1)
        else:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                   dropout_p=dropout)
        output = self.out_proj(output.permute(2, 0, 1, 3).reshape(
            target_len, batch, self.embed_dim))
        if unbatched:
            output = output.squeeze(1)
            if weights is not None:
                weights = weights.squeeze(0)
        elif self.batch_first:
            output = output.transpose(0, 1)
        return output, weights


def inject_full_lora(model, rank=32, alpha=64, dropout=0.05):
    """Freeze the entire base and adapt all Linear/Conv2d/MHA projections."""
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError('LoRA rank must be a positive integer.')
    if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError('Invalid LoRA alpha/dropout.')
    model.requires_grad_(False)
    converted = {}

    def convert(module):
        if id(module) in converted:
            return converted[id(module)]
        replacement = module
        for base, adapted in ((nn.MultiheadAttention, LoRAMultiheadAttention),
                              (nn.Linear, LoRALinear), (nn.Conv2d, LoRAConv2d)):
            if isinstance(module, base):
                replacement = adapted(module, rank, alpha, dropout)
                break
        converted[id(module)] = replacement
        if replacement is module:
            for name, child in list(module._modules.items()):
                if child is not None:
                    module._modules[name] = convert(child)
        return replacement

    convert(model)


def lora_parameter_groups(model):
    """Seven disjoint groups; reject any accidentally unfrozen base tensor."""
    groups = {name: dict(params=[], lr=lr, name=name) for name, lr in (
        ('backbone', 1e-4), ('language_model', 1e-4), ('encoder', 2e-4),
        ('decoder', 2e-4), ('bbox_head', 2e-4), ('text_feat_map', 2e-4),
        ('neck', 2e-4))}
    for name, parameter in model.named_parameters():
        adapter = name.rsplit('.', 1)[-1] in ('lora_A', 'lora_B')
        if parameter.requires_grad != adapter:
            raise RuntimeError(f'Invalid Full LoRA freeze state: {name}')
        if not adapter:
            continue
        component = name.split('.')[0]
        # The encoder proposal projection belongs to the detection head path.
        if component == 'memory_trans_fc':
            component = 'bbox_head'
        if component not in groups:
            raise RuntimeError(f'Unassigned LoRA parameter: {name}')
        groups[component]['params'].append(parameter)
    for name, group in groups.items():
        if not group['params']:
            raise RuntimeError(f'No LoRA parameters found for {name}')
    return list(groups.values())
