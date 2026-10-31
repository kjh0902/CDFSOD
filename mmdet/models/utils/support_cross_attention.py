"""Class-segmented, text-query attention over immutable local descriptors."""
import torch
from torch import nn


class SupportTextCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, gamma_init=1e-2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=0.,
                                         batch_first=True)
        self.gamma = nn.Parameter(torch.full((embed_dims,), gamma_init))
        # Constant LN initialization consumes no RNG: existing projections keep
        # exactly the same random initialization for a fixed seed.
        self.support_norm = nn.LayerNorm(embed_dims, eps=1e-5, elementwise_affine=True)

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
            # Stop-gradient precedes the trainable LN. Only K/V are normalized;
            # the query and residual reference remain the original text.
            local = self.support_norm(local)
            update = self.attn(query[None], local[None], local[None],
                               need_weights=False)[0][0]
            residual = update * self.gamma.to(update.dtype)
            delta[batch, token] = residual.to(delta.dtype)
            if stats is not None and stats.capture_support:
                stats.cross_attention(layer_id, self.attn, query, local,
                                      residual, batch)
        return text + delta
