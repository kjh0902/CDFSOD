"""Object-wise support memory; no trainable parameters or extra loss."""
import copy

import torch
from torchvision.ops import roi_align


def pool_instances(feature, samples):
    """Pool resized xyxy GT boxes from Swin-B norm1, before the neck."""
    if feature.ndim != 4 or feature.shape[1] != 256:
        raise ValueError('Support requires Swin-B Stage 1: stride 8, 256 channels.')
    rois, labels = [], []
    for i, sample in enumerate(samples):
        boxes = sample.gt_instances.bboxes.to(feature.device, torch.float32)
        target = sample.gt_instances.labels.to(feature.device)
        if (len(boxes) != len(target) or not torch.isfinite(boxes).all() or
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


def support_attention_mask(text, memory, mappings):
    """Extend the existing text allow-mask with class-local memory columns.

    No support padding is introduced: V is the actual number of GT objects.
    Text padding keeps its original diagonal, avoiding fully masked query rows.
    Valid text queries cannot attend text padding or another class's memory.
    """
    batch, length, channels = text['embedded'].shape
    tokens, labels, counts = (memory[k] for k in ('tokens', 'class_ids', 'counts'))
    classes = len(counts)
    if (tokens.ndim != 2 or tokens.shape[1] != channels or
            labels.shape != (len(tokens),) or len(mappings) != batch or
            labels.dtype != torch.long or (counts <= 0).any() or
            ((labels < 0) | (labels >= classes)).any()):
        raise ValueError('Support memory dimensions, classes or counts are invalid.')
    if not torch.equal(torch.bincount(labels, minlength=classes), counts):
        raise ValueError('Support memory counts do not match object labels.')
    allowed = []
    for b, mapping in enumerate(mappings):
        if set(mapping) != set(range(1, classes + 1)):
            raise ValueError('Support memory requires every class in canonical order.')
        names = [list(mapping[c + 1]) for c in range(classes)]
        if any(not ids or ids != list(range(ids[0], ids[-1] + 1)) for ids in names):
            raise ValueError('Each class needs a nonempty contiguous name span.')
        ends = [ids[-1] + 1 for ids in names]
        if (ends != sorted(set(ends)) or names[0][0] < 1 or
                ends[-1] >= length or
                not text['text_token_mask'][b, ends].all()):
            raise ValueError('Class names must precede valid class delimiters.')
        # Inherit the delimiter's existing block, including its period query.
        columns = torch.tensor(ends, device=labels.device)[labels]
        visual = text['masks'][b, :, columns] & text['text_token_mask'][b, :, None]
        allowed.append(torch.cat([text['masks'][b], visual], dim=-1))
    return torch.stack(allowed)


class SupportVisualMixin:
    """Lazy, rank-local full-support loading for train and standalone predict."""

    @property
    def visual_enabled(self):
        return self.support_visual_cfg is not None

    def clear_support_cache(self):
        self._support_memory = None

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
            # CocoDataset deletes its temporary COCO API during initialization.
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
    def build_support_memory(self):
        loader = self._get_support_loader()
        classes = self.bbox_head.num_classes
        device = next(self.backbone.parameters()).device
        values_by_class = [[] for _ in range(classes)]
        modes = [(module, module.training) for module in self.backbone.modules()]
        try:
            self.backbone.eval()
            for batch in loader:
                batch = self.data_preprocessor(batch, training=False)
                # Raw norm1 return, never extract_feat() / ChannelMapper.
                feature = self.backbone(batch['inputs'])[0]
                values, labels = pool_instances(feature, batch['data_samples'])
                if labels.dtype != torch.long or ((labels < 0) | (labels >= classes)).any():
                    raise ValueError('Support labels must be integer labels in 0..C-1.')
                for c in range(classes):
                    selected = values[labels == c]
                    if len(selected):
                        values_by_class[c].append(selected)
        finally:
            for module, mode in modes:
                module.training = mode
        counts = torch.tensor([sum(len(v) for v in rows) for rows in values_by_class],
                              device=device, dtype=torch.long)
        if (counts == 0).any():
            missing = (counts == 0).nonzero().flatten().tolist()
            names = self.support_visual_cfg['class_names']
            raise ValueError('Support set is missing GT instances for classes ' +
                             str([(c, names[c]) for c in missing]))
        tokens = torch.cat([torch.cat(rows) for rows in values_by_class])
        class_ids = torch.arange(classes, device=device).repeat_interleave(counts)
        return dict(tokens=tokens, class_ids=class_ids, counts=counts)

    def support_memory(self):
        if self.training:
            return self.build_support_memory()
        if getattr(self, '_support_memory', None) is None:
            self._support_memory = self.build_support_memory()
        return self._support_memory
