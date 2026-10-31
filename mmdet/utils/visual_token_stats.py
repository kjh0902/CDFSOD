"""Detached support-memory statistics; only O(classes) aggregates persist."""
import math
import torch
from torch.nn import functional as F


class VisualTokenStats:
    fields = ('text_norm', 'support_token_norm', 'enhanced_text_norm')

    def __init__(self, classes, device, num_layers=0, num_heads=4):
        self.classes = classes
        self.sums = torch.zeros(len(self.fields), classes, device=device,
                                dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums)
        self.support_counts = torch.zeros(classes, device=device, dtype=torch.long)
        self.valid = None
        self.capture_support = True
        self.layer_sums = torch.zeros(num_layers, 4, device=device, dtype=torch.float64)
        self.layer_counts = torch.zeros(num_layers, 2, device=device, dtype=torch.float64)
        self.head_sums = torch.zeros(num_layers, num_heads, device=device, dtype=torch.float64)
        self.gamma = torch.zeros(num_layers, 3, device=device, dtype=torch.float64)
        self._norm_memory_id = None
        self._support_norm_sums = None

    def add(self, field, values, counts):
        i = self.fields.index(field)
        self.sums[i] += values.to(self.sums)
        self.counts[i] += counts.to(self.counts)

    @torch.no_grad()
    def text_features(self, embedded, mappings, field):
        for b, mapping in enumerate(mappings):
            if self.valid is not None and not self.valid[b]:
                continue
            norms = torch.stack([embedded[b, ids].float().norm(dim=-1).sum()
                                 for _, ids in sorted(mapping.items())])
            counts = norms.new_tensor([len(ids) for _, ids in sorted(mapping.items())])
            self.add(field, norms, counts)

    @torch.no_grad()
    def support_features(self, memory, batch_size):
        # Every rank holds the FULL support set. These are snapshot counts,
        # never summed over images, batches or ranks.
        self.support_counts.copy_(memory['counts'])
        weight = sum(self.valid) if self.valid is not None else batch_size
        if self._norm_memory_id != id(memory['tokens']):
            norms = memory['tokens'].float().norm(dim=-1)
            self._support_norm_sums = norms.new_zeros(self.classes)
            self._support_norm_sums.index_add_(0, memory['class_ids'], norms)
            self._norm_memory_id = id(memory['tokens'])
        self.add('support_token_norm', self._support_norm_sums * weight,
                 memory['counts'] * weight)

    @torch.no_grad()
    def record_gamma(self, layer_id, gamma):
        # Lazy extension also supports lightweight callers outside the runner.
        if layer_id >= len(self.gamma):
            extra = layer_id + 1 - len(self.gamma)
            for name, width in (('gamma', 3), ('layer_sums', 4), ('layer_counts', 2),
                                ('head_sums', self.head_sums.shape[1])):
                value = getattr(self, name)
                setattr(self, name, torch.cat([value, value.new_zeros(extra, width)]))
        gamma = gamma.detach().float()
        self.gamma[layer_id] = torch.stack((gamma.abs().mean(), gamma.abs().max(),
                                            gamma.std(unbiased=False))).to(self.gamma)

    @staticmethod
    @torch.no_grad()
    def attention_entropy(attn, query, support, chunk_size=1024):
        """Exact per-head entropy with bounded K chunks, never stored probabilities."""
        heads, dim = attn.num_heads, attn.embed_dim
        if len(support) == 1:
            return query.new_zeros(heads, len(query), dtype=torch.float32)
        with torch.autocast(device_type=query.device.type, enabled=False):
            if hasattr(attn, 'q_proj'):
                qw, qb = attn.q_proj.weight, attn.q_proj.bias
                kw, kb = attn.k_proj.weight, attn.k_proj.bias
            else:  # Also usable by the standard MHA reference in diagnostics.
                weight, bias = attn.in_proj_weight, attn.in_proj_bias
                qw, kw = weight[:dim], weight[dim:2*dim]
                qb = None if bias is None else bias[:dim]
                kb = None if bias is None else bias[dim:2*dim]
            q = F.linear(query.detach().float(), qw.detach().float(),
                         None if qb is None else qb.detach().float())
            q = q.view(-1, heads, dim // heads).transpose(0, 1) / math.sqrt(dim // heads)
            maximum = q.new_full(q.shape[:2], -float('inf'))
            normalizer = torch.zeros_like(maximum)
            moment = torch.zeros_like(maximum)
            for local in support.split(chunk_size):
                k = F.linear(local.detach().float(), kw.detach().float(),
                             None if kb is None else kb.detach().float())
                k = k.view(-1, heads, dim // heads).permute(1, 2, 0)
                scores = q @ k
                updated = torch.maximum(maximum, scores.amax(-1))
                rescale = (maximum - updated).exp()
                exp = (scores - updated[..., None]).exp()
                normalizer = normalizer * rescale + exp.sum(-1)
                moment = moment * rescale + (exp * scores).sum(-1)
                maximum = updated
            entropy = maximum + normalizer.log() - moment / normalizer
            return (entropy / math.log(len(support))).clamp(0., 1.)

    @torch.no_grad()
    def cross_attention(self, layer_id, attn, query, support, residual, batch,
                        reference=None, update=None):
        if self.head_sums.shape[1] != attn.num_heads:
            if self.head_sums.any():
                raise ValueError('All support layers must have the same head count.')
            self.head_sums = self.head_sums.new_zeros(len(self.gamma), attn.num_heads)
        valid = torch.ones(len(batch), dtype=torch.bool, device=batch.device)
        if self.valid is not None:
            valid = torch.tensor(self.valid, device=batch.device, dtype=torch.bool)[batch]
        reference = query if reference is None else reference
        update = residual if update is None else update
        reference, update = reference.detach()[valid], update.detach()[valid]
        query, residual = query.detach()[valid], residual.detach()[valid]
        if not len(query):
            return
        # Contribution is exact over all valid queries in sampled batches.
        self.layer_sums[layer_id, 1] += residual.float().square().sum().double()
        self.layer_sums[layer_id, 2] += reference.float().square().sum().double()
        self.layer_sums[layer_id, 3] += update.float().square().sum().double()
        self.layer_counts[layer_id, 1] += len(query)
        entropy = self.attention_entropy(attn, query[:8], support)
        self.layer_sums[layer_id, 0] += entropy.double().sum()
        self.head_sums[layer_id] += entropy.double().sum(-1)
        self.layer_counts[layer_id, 0] += entropy.numel()

    def finish_batch(self):
        self.valid = None

    def report(self, names):
        sums, counts = self.sums.cpu().tolist(), self.counts.cpu().tolist()
        support_counts = self.support_counts.cpu().tolist()
        layers = {}
        for i, (values, n, gamma) in enumerate(zip(self.layer_sums.cpu().tolist(),
                self.layer_counts.cpu().tolist(), self.gamma.cpu().tolist())):
            layers[str(i)] = dict(
                normalized_attention_entropy=dict(
                    value=values[0] / n[0] if n[0] else None, count=int(n[0])),
                residual_contribution=dict(
                    value=math.sqrt(values[1]) / max(math.sqrt(values[2]), 1e-12)
                          if n[1] else None, count=int(n[1])),
                ungated_update_contribution=dict(
                    value=math.sqrt(values[3]) / max(math.sqrt(values[2]), 1e-12)
                          if n[1] else None, count=int(n[1])),
                normalized_attention_entropy_per_head=[dict(
                    value=value / (n[0] / self.head_sums.shape[1]) if n[0] else None,
                    count=int(n[0] / self.head_sums.shape[1]))
                    for value in self.head_sums[i].cpu().tolist()],
                gamma=dict(mean_absolute=gamma[0], max_absolute=gamma[1], std=gamma[2]))

        def entry(i, c=None):
            total = sum(sums[i]) if c is None else sums[i][c]
            count = sum(counts[i]) if c is None else counts[i][c]
            return dict(value=total / count if count else None, count=int(count))

        return dict(
            support_cross_attention=layers,
            overall=dict(support_token_count=sum(support_counts),
                         **{f: entry(i) for i, f in enumerate(self.fields)}),
            classes={name: dict(support_token_count=support_counts[c],
                               **{f: entry(i, c) for i, f in enumerate(self.fields)})
                     for c, name in enumerate(names)})
