"""Text queries over neck-local support, initialized from the FE text update."""
import torch
from torch import nn
from torch.nn import functional as F


class SupportMultiheadAttention(nn.Module):
    """Separate projections allow a wider attention space than the input."""

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('Support attention dimension must divide into heads.')
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, input_dim)

    def heads(self, x):
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None, need_weights=False,
                average_attn_weights=True):
        q, k, v = (self.heads(proj(x)) for proj, x in
                   ((self.q_proj, query), (self.k_proj, key), (self.v_proj, value)))
        mask = attn_mask
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unflatten(0, (len(query), self.num_heads))
            if mask.dtype == torch.bool:
                mask = ~mask  # MHA True=blocked; SDPA True=allowed.
        weights = None
        if need_weights:
            scores = q @ k.transpose(-1, -2) * self.head_dim ** -.5
            if mask is not None:
                scores = (scores.masked_fill(~mask, float('-inf'))
                          if mask.dtype == torch.bool else scores + mask)
            weights = scores.softmax(-1)
            output = weights @ v
        else:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                    dropout_p=0.)
        output = self.out_proj(output.transpose(1, 2).flatten(-2))
        if weights is not None and average_attn_weights:
            weights = weights.mean(1)
        return output, weights


class SupportTextCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, gamma_init=1e-4, attn_dims=None,
                 norm_eps=1e-5):
        super().__init__()
        self.text_norm = nn.LayerNorm(embed_dims, eps=norm_eps)
        self.support_norm = nn.LayerNorm(embed_dims, eps=norm_eps)
        self.attn = SupportMultiheadAttention(embed_dims, attn_dims or 4 * embed_dims,
                                              num_heads)
        self.gamma = nn.Parameter(torch.full((embed_dims,), gamma_init))

    def fe_parameter_map(self):
        """Destination suffix -> corresponding original fusion suffix."""
        result = {'gamma': 'gamma_l'}
        for dst, src in (('text_norm', 'layer_norm_l'),
                         ('support_norm', 'layer_norm_v'),
                         ('attn.q_proj', 'attn.l_proj'),
                         ('attn.k_proj', 'attn.v_proj'),
                         ('attn.v_proj', 'attn.values_v_proj'),
                         ('attn.out_proj', 'attn.out_l_proj')):
            for field in ('weight', 'bias'):
                result[f'{dst}.{field}'] = f'{src}.{field}'
        return result

    def forward(self, text, support, text_classes, counts, stats=None, layer_id=0):
        # Segmenting K/V is exactly a class allow-mask, without B*H*T*S storage.
        # Batch/token queries can share a segment: queries never attend each other.
        delta = torch.zeros_like(text)
        offset = 0
        if stats is not None:
            stats.record_gamma(layer_id, self.gamma)
        for class_id, count in enumerate(counts):
            batch, token = (text_classes == class_id).nonzero(as_tuple=True)
            local = support[offset:offset + count].detach()
            offset += count
            if batch.numel() == 0:
                continue
            query = text[batch, token]
            normalized_query = self.text_norm(query)
            normalized_support = self.support_norm(local)
            update = self.attn(normalized_query[None], normalized_support[None], normalized_support[None],
                               need_weights=False)[0][0]
            residual = update * self.gamma.to(update.dtype)
            delta[batch, token] = residual.to(delta.dtype)
            if stats is not None and stats.capture_support:
                stats.cross_attention(layer_id, self.attn, normalized_query,
                                      normalized_support, residual, batch,
                                      reference=query, update=update)
        return text + delta
