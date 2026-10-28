"""Detached streaming statistics. Only O(classes) sums survive a batch."""
import torch


class VisualTokenStats:
    # Each entry has a per-class numerator and denominator.
    fields = ('win_all', 'win_gt', 'win_background', 'text_score',
              'visual_score', 'score_difference', 'selection_visual',
              'text_norm', 'prototype_norm')

    def __init__(self, classes, device):
        self.classes = classes
        self.sums = torch.zeros(len(self.fields), classes, device=device,
                                dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums)
        self.valid = None
        self.pending = None

    def add(self, field, values, counts):
        i = self.fields.index(field)
        self.sums[i] += values.to(self.sums)
        self.counts[i] += counts.to(self.counts)

    @torch.no_grad()
    def features(self, text, prototypes, mappings):
        for b, mapping in enumerate(mappings):
            if self.valid is not None and not self.valid[b]:
                continue
            norms = torch.stack([text['embedded'][b, ids].float().norm(dim=-1).sum()
                                 for _, ids in sorted(mapping.items())])
            counts = norms.new_tensor([len(ids) for _, ids in sorted(mapping.items())])
            self.add('text_norm', norms, counts)
            self.add('prototype_norm', prototypes.float().norm(dim=-1), torch.ones_like(norms))

    @torch.no_grad()
    def selection(self, winners, layout):
        for b, row in enumerate(winners):
            if self.valid is not None and not self.valid[b]:
                continue
            counts = torch.zeros(self.classes, device=row.device)
            wins = torch.zeros_like(counts)
            for c, ids in sorted(layout['names'][b].items()):
                visual = layout['visual'][b][c - 1]
                ids = torch.tensor(ids + [visual], device=row.device)
                counts[c - 1] = (row[:, None] == ids).any(-1).sum()
                wins[c - 1] = (row == visual).sum()
            self.add('selection_visual', wins, counts)

    @torch.no_grad()
    def scores(self, text_logits, visual_logits, fused_logits):
        # Inputs are only final-layer, non-DN queries, detached by the caller.
        text, visual = text_logits.sigmoid(), visual_logits.sigmoid()
        wins = visual_logits > text_logits
        predicted = fused_logits.argmax(-1)
        self.pending = (wins, predicted)
        for b in range(len(text)):
            if self.valid is not None and not self.valid[b]:
                continue
            count = text.new_full((self.classes,), text.shape[1])
            self.add('text_score', text[b].sum(0), count)
            self.add('visual_score', visual[b].sum(0), count)
            self.add('score_difference', (visual[b] - text[b]).sum(0), count)
            labels = predicted[b]
            values = wins[b].gather(1, labels[:, None]).squeeze(1)
            self._wins('win_all', labels, values)

    def _wins(self, field, labels, values):
        counts = torch.bincount(labels, minlength=self.classes)
        sums = torch.bincount(labels[values.bool()], minlength=self.classes)
        self.add(field, sums, counts)

    @torch.no_grad()
    def matches(self, b, assigned):
        if self.valid is not None and not self.valid[b]:
            return
        wins, predicted = self.pending
        positive, background = assigned.gt_inds > 0, assigned.gt_inds == 0
        labels = assigned.labels[positive]
        self._wins('win_gt', labels,
                   wins[b, positive].gather(1, labels[:, None]).squeeze(1))
        labels = predicted[b, background]
        self._wins('win_background', labels,
                   wins[b, background].gather(1, labels[:, None]).squeeze(1))

    def finish_batch(self):
        self.pending = None
        self.valid = None

    def report(self, names):
        sums, counts = self.sums.cpu().tolist(), self.counts.cpu().tolist()
        def entry(i, c=None):
            total = sum(sums[i]) if c is None else sums[i][c]
            count = sum(counts[i]) if c is None else counts[i][c]
            return dict(value=total / count if count else None, count=int(count))
        return dict(overall={f: entry(i) for i, f in enumerate(self.fields)},
                    classes={name: {f: entry(i, c) for i, f in enumerate(self.fields)}
                             for c, name in enumerate(names)})
