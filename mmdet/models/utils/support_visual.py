"""Support prototypes and token layout; no trainable parameters or extra loss."""
import copy

import torch
from torch.nn import functional as F
from torchvision.ops import roi_align


def pool_instances(feature, samples):
    """Pool resized xyxy GT boxes from Swin-B norm1, before the neck."""
    if feature.ndim != 4 or feature.shape[1] != 256:
        raise ValueError('Support requires Swin-B Stage 1: stride 8, 256 channels.')
    rois, labels = [], []
    for i, sample in enumerate(samples):
        boxes = sample.gt_instances.bboxes.to(feature.device, torch.float32)
        target = sample.gt_instances.labels.to(feature.device)
        if (not torch.isfinite(boxes).all() or
                (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError('Support GT boxes must be finite, nonempty xyxy boxes.')
        rois.append(torch.cat([boxes.new_full((len(boxes), 1), i), boxes], 1))
        labels.append(target)
    rois = torch.cat(rois)
    labels = torch.cat(labels)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        features = roi_align(feature.float(), rois, (7, 7), spatial_scale=1 / 8,
                             sampling_ratio=2, aligned=True).mean((-2, -1))
    return features, labels


def insert_visual_tokens(text, prototypes, mappings, max_length):
    """Insert one token before each class delimiter and transfer all masks."""
    batch, old_length, channels = text['embedded'].shape
    classes = len(prototypes)
    if prototypes.shape != (classes, channels) or len(mappings) != batch:
        raise ValueError('Support prototype and text dimensions do not match.')
    lengths = text['text_token_mask'].sum(-1).tolist()
    length = max(lengths) + classes
    if length > max_length:
        raise ValueError('All-class prompt plus visual tokens exceeds max_text_len.')
    embedded, masks, positions, valid = [], [], [], []
    name_maps, visual_indices = [], []
    selection = torch.zeros(batch, length, dtype=torch.bool,
                            device=prototypes.device)
    for b, mapping in enumerate(mappings):
        if set(mapping) != set(range(1, classes + 1)):
            raise ValueError('Visual tokens require every class in canonical order.')
        names = [list(mapping[c + 1]) for c in range(classes)]
        if any(not ids or ids != list(range(ids[0], ids[-1] + 1)) for ids in names):
            raise ValueError('Each class needs a nonempty contiguous name span.')
        ends = [ids[-1] + 1 for ids in names]
        if ends != sorted(set(ends)) or ends[-1] >= lengths[b] - 1:
            raise ValueError('Class names must precede their period and [SEP].')
        # old_index also supplies the class delimiter's existing attention block
        # to the inserted token. Original special-token connectivity is retained.
        old_index, proto_index = [], []
        by_end = {end: c for c, end in enumerate(ends)}
        for i in range(int(lengths[b])):
            if i in by_end:
                old_index.append(i)
                proto_index.append(by_end[i])
            old_index.append(i)
            proto_index.append(-1)
        n = len(old_index)
        index = torch.tensor(old_index, device=prototypes.device)
        pindex = torch.tensor(proto_index, device=prototypes.device)
        values = text['embedded'][b, index]
        values = torch.where((pindex >= 0)[:, None],
                             prototypes[pindex.clamp_min(0)].to(values.dtype), values)
        embedded.append(F.pad(values, (0, 0, 0, length - n)))
        attention = torch.eye(length, dtype=torch.bool, device=index.device)
        attention[:n, :n] = text['masks'][b][index[:, None], index[None, :]]
        masks.append(attention)
        position = torch.zeros(length, dtype=torch.long, device=index.device)
        position[:n] = text['position_ids'][b, index]
        shifted_names, vis = {}, []
        for c, ids in enumerate(names):
            shifted = [i + c for i in ids]
            v = ends[c] + c
            shifted_names[c + 1] = shifted
            vis.append(v)
            position[v] = text['position_ids'][b, ends[c]]
            position[v + 1] += 1
            selection[b, shifted] = True
            selection[b, v] = True
        positions.append(position)
        valid.append(torch.arange(length, device=index.device) < n)
        name_maps.append(shifted_names)
        visual_indices.append(vis)
    result = dict(text)
    result.update(embedded=torch.stack(embedded), masks=torch.stack(masks),
                  position_ids=torch.stack(positions),
                  text_token_mask=torch.stack(valid))
    # hidden is a BERT-only artifact, not an input to FE or the detection head.
    result.pop('hidden', None)
    layout = dict(names=name_maps, visual=visual_indices, selection=selection)
    return result, layout


def fuse_token_logits(logits, layout, width, stats=None, dn_count=0):
    """Return logits whose sigmoid is max(mean(sigmoid(text)), sigmoid(visual)).

    Compute log(mean(p) / mean(1-p)) directly, without a sigmoid/logit round
    trip or probability clipping. Strict > assigns exact ties to text.
    """
    rows, text_rows, visual_rows = [], [], []
    with torch.autocast(device_type=logits.device.type, enabled=False):
        for b, mapping in enumerate(layout['names']):
            classes, texts, visuals = [], [], []
            for c, ids in sorted(mapping.items()):
                z = logits[b, :, ids].float()
                text = (torch.logsumexp(F.logsigmoid(z), -1) -
                        torch.logsumexp(F.logsigmoid(-z), -1))
                visual = logits[b, :, layout['visual'][b][c - 1]].float()
                classes.append(torch.where(visual > text, visual, text))
                if stats is not None:
                    texts.append(text.detach()[dn_count:])
                    visuals.append(visual.detach()[dn_count:])
            fused = torch.stack(classes, -1)
            rows.append(F.pad(fused, (0, width - len(classes)), value=float('-inf')))
            if stats is not None:
                text_rows.append(torch.stack(texts, -1))
                visual_rows.append(torch.stack(visuals, -1))
    result = torch.stack(rows)
    if stats is not None:
        stats.scores(torch.stack(text_rows), torch.stack(visual_rows),
                     result.detach()[:, dn_count:, :len(layout['names'][0])])
    return result


class SupportVisualMixin:
    """Lazy, rank-local full-support loading for train and standalone predict."""

    @property
    def visual_enabled(self):
        return self.support_visual_cfg is not None

    def clear_support_cache(self):
        self._support_prototypes = None

    def train(self, mode=True):
        self.clear_support_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_support_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _get_support_loader(self):
        if getattr(self, '_support_loader', None) is None:
            from mmengine.dataset import pseudo_collate
            from mmengine.fileio import load
            from torch.utils.data import DataLoader
            from mmdet.registry import DATASETS
            cfg = copy.deepcopy(self.support_visual_cfg)
            dataset = DATASETS.build(cfg['dataset'])
            dataset.full_init()
            expected = tuple(cfg['class_names'])
            if tuple(dataset.metainfo['classes']) != expected:
                raise ValueError('Support class metadata does not match the query classes.')
            # CocoDataset.load_data_list() deletes its temporary COCO API.
            # Read category metadata using the dataset's resolved annotation path
            # and file backend, and preserve its persistent category/label order.
            annotation = load(dataset.ann_file, backend_args=dataset.backend_args)
            categories = {category['id']: category['name']
                          for category in annotation['categories']}
            actual = tuple(categories[cat_id] for cat_id in dataset.cat_ids)
            if actual != expected:
                raise ValueError('Support COCO category order must match class/label order.')
            self._support_loader = DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=pseudo_collate, generator=torch.Generator().manual_seed(42))
        return self._support_loader

    @torch.no_grad()
    def build_support_prototypes(self):
        loader = self._get_support_loader()
        classes = self.bbox_head.num_classes
        device = next(self.backbone.parameters()).device
        sums = torch.zeros(classes, 256, device=device, dtype=torch.float32)
        counts = torch.zeros(classes, device=device, dtype=torch.long)
        modes = [(module, module.training) for module in self.backbone.modules()]
        try:
            self.backbone.eval()
            for batch in loader:
                batch = self.data_preprocessor(batch, training=False)
                # Use the raw norm1 return, never extract_feat() / ChannelMapper.
                feature = self.backbone(batch['inputs'])[0]
                values, labels = pool_instances(feature, batch['data_samples'])
                if ((labels < 0) | (labels >= classes)).any():
                    raise ValueError('Support labels must be in 0..C-1.')
                sums.index_add_(0, labels, values)
                counts.index_add_(0, labels, torch.ones_like(labels))
        finally:
            for module, mode in modes:
                module.training = mode
        if (counts == 0).any():
            missing = (counts == 0).nonzero().flatten().tolist()
            raise ValueError(f'Support set is missing GT instances for classes {missing}.')
        return sums / counts[:, None]

    def support_prototypes(self):
        if self.training:
            return self.build_support_prototypes()
        if getattr(self, '_support_prototypes', None) is None:
            self._support_prototypes = self.build_support_prototypes()
        return self._support_prototypes
