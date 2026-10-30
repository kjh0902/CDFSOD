"""Real repository text/fusion forwards with PyTorch MHA; no MMCV kernels.

Only MMCV construction/residual wrappers and image deformable attention are
substituted. The actual GroundingDINO encoder loop, DETR text-layer forward,
sine positions, bidirectional fusion and attention math execute unchanged.
"""
import copy
from types import SimpleNamespace as NS
import torch
from torch import nn
from torch.nn import functional as F
from visual_test_support import execute, node, source, ENV, load

env = dict(ENV, F=F, Tensor=torch.Tensor, MAX_CLAMP_VALUE=50000)
execute([node(source('mmdet/models/layers/transformer/utils.py'),
              'get_text_sine_pos_embed')], env)
for name in ('BiMultiHeadAttention', 'BiAttentionBlock', 'SingleScaleBiAttentionBlock'):
    execute([node(source('mmdet/models/utils/vlfuse_helper.py'), name)], env)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.calls = []

    def forward(self, query, key, value, query_pos=None, key_pos=None,
                key_padding_mask=None, attn_mask=None):
        self.calls.append(dict(query=query.detach().clone(), key=key.detach().clone(),
                               value=value.detach().clone(),
                               key_pos=key_pos.detach().clone() if key_pos is not None else None,
                               mask=attn_mask.detach().clone()))
        q = query if query_pos is None else query + query_pos
        k = key if key_pos is None else key + key_pos
        return query + self.attn(q, k, value, key_padding_mask=key_padding_mask,
                                 attn_mask=attn_mask, need_weights=False)[0]


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, query):
        return query + self.layers(query)


class TextLayer(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.self_attn_cfg = NS(num_heads=heads)
        self.self_attn = Attention(dim, heads)
        self.norms = nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)])
        self.ffn = FFN(dim)


layer_env = dict(env)
execute([node(source('mmdet/models/layers/transformer/detr_layers.py'),
              'DetrTransformerEncoderLayer').body[-1]], layer_env)
TextLayer.forward = layer_env['forward']


class ImageLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, query, **kwargs):
        return self.linear(query)


SupportAttention = load('mmdet/models/utils/support_cross_attention.py', 'support_attention').SupportTextCrossAttention


class Encoder(nn.Module):
    def __init__(self, dim=8, heads=2, layers=2, fusion=True, support=True):
        super().__init__()
        self.support_layers = nn.ModuleList([SupportAttention(dim, heads) for _ in range(layers)] if support else [])
        self.embed_dims = dim
        self.layers = nn.ModuleList([ImageLayer(dim) for _ in range(layers)])
        self.text_layers = nn.ModuleList([TextLayer(dim, heads) for _ in range(layers)])
        self.fusion_layers = nn.ModuleList([
            env['SingleScaleBiAttentionBlock'](dim, dim, 2 * dim, heads, dropout=0.)
            for _ in range(layers)] if fusion else [])

    @staticmethod
    def get_encoder_reference_points(spatial_shapes, valid_ratios, device):
        return torch.zeros(valid_ratios.shape[0], 1, 1, 2, device=device)


encoder_env = dict(env)
execute([node(source('mmdet/models/layers/transformer/grounding_dino_layers.py'),
              'GroundingDinoTransformerEncoder').body[-1]], encoder_env)
Encoder.forward = encoder_env['forward']


def run(encoder, text, memory=None, allowed=None, image=None, stats=None):
    if image is None:
        image = torch.randn(len(text['embedded']), 4, encoder.embed_dims)
    return encoder(query=image, query_pos=torch.zeros_like(image),
                   key_padding_mask=torch.zeros(image.shape[:2], dtype=torch.bool),
                   spatial_shapes=torch.tensor([[2, 2]]),
                   level_start_index=torch.tensor([0]),
                   valid_ratios=torch.ones(len(image), 1, 2),
                   memory_text=text['embedded'],
                   text_attention_mask=~text['text_token_mask'],
                   position_ids=text['position_ids'],
                   text_self_attention_masks=text['masks'],
                   support_memory=memory['tokens'] if memory is not None else None,
                   text_class_ids=allowed,
                   support_counts=tuple(memory['counts'].tolist()) if memory is not None else None,
                   support_stats=stats)
