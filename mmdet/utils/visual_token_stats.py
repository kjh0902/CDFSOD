"""Detached support-memory statistics; only O(classes) aggregates persist."""
import torch


class VisualTokenStats:
    fields = ('text_norm', 'support_token_norm', 'enhanced_text_norm')

    def __init__(self, classes, device):
        self.classes = classes
        self.sums = torch.zeros(len(self.fields), classes, device=device,
                                dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums)
        self.support_counts = torch.zeros(classes, device=device, dtype=torch.long)
        self.valid = None

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
        norms = memory['tokens'].float().norm(dim=-1)
        sums = norms.new_zeros(self.classes)
        sums.index_add_(0, memory['class_ids'], norms)
        self.add('support_token_norm', sums * weight, memory['counts'] * weight)

    def finish_batch(self):
        self.valid = None

    def report(self, names):
        sums, counts = self.sums.cpu().tolist(), self.counts.cpu().tolist()
        support_counts = self.support_counts.cpu().tolist()

        def entry(i, c=None):
            total = sum(sums[i]) if c is None else sums[i][c]
            count = sum(counts[i]) if c is None else counts[i][c]
            return dict(value=total / count if count else None, count=int(count))

        return dict(
            overall=dict(support_token_count=sum(support_counts),
                         **{f: entry(i) for i, f in enumerate(self.fields)}),
            classes={name: dict(support_token_count=support_counts[c],
                               **{f: entry(i, c) for i, f in enumerate(self.fields)})
                     for c, name in enumerate(names)})
