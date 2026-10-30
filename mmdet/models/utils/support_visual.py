"""Fixed, full-support local descriptor extraction and cache validation."""
import copy
import hashlib
import json
import os
from pathlib import Path

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
                             sampling_ratio=2, aligned=True).flatten(2).transpose(1, 2)
    return features, labels


def support_text_classes(text, memory, mappings):
    """Return B*T class IDs (-1 for unrelated queries), never a dense mask."""
    batch, length, channels = text['embedded'].shape
    tokens, labels, counts = (memory[k] for k in ('tokens', 'class_ids', 'counts'))
    classes = len(counts)
    if (tokens.ndim != 2 or tokens.shape[1] != channels or
            labels.shape != (len(tokens),) or len(mappings) != batch or
            labels.dtype != torch.long or counts.dtype != torch.long or
            (counts <= 0).any() or ((labels < 0) | (labels >= classes)).any()):
        raise ValueError('Support memory dimensions, classes or counts are invalid.')
    expected = torch.arange(classes, device=labels.device).repeat_interleave(counts)
    if not torch.equal(labels, expected):
        raise ValueError('Support memory counts/order do not match descriptor labels.')
    result = torch.full((batch, length), -1, dtype=torch.long,
                        device=text['embedded'].device)
    for b, mapping in enumerate(mappings):
        if set(mapping) != set(range(1, classes + 1)):
            raise ValueError('Support memory requires every class in canonical order.')
        names = [list(mapping[c + 1]) for c in range(classes)]
        if any(not ids or ids != list(range(ids[0], ids[-1] + 1)) for ids in names):
            raise ValueError('Each class needs a nonempty contiguous name span.')
        ends = [ids[-1] + 1 for ids in names]
        if (ends != sorted(set(ends)) or names[0][0] < 1 or
                ends[-1] >= length or
                any(names[c][0] <= ends[c-1] for c in range(1, classes)) or
                not text['text_token_mask'][b, ends].all()):
            raise ValueError('Class names must precede valid class delimiters.')
        for c, (ids, end) in enumerate(zip(names, ends)):
            indices = ids + [end]
            if not text['text_token_mask'][b, indices].all():
                raise ValueError('Class name contains padding tokens.')
            result[b, indices] = c
    return result


class SupportVisualMixin:
    """Lazy, rank-local full-support loading for train and standalone predict."""

    @property
    def visual_enabled(self):
        return self.support_visual_cfg is not None

    def clear_support_cache(self):
        self._support_memory = None

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
                        values_by_class[c].append(selected.flatten(0, 1))
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
        if getattr(self, '_support_memory', None) is None:
            self._support_memory = self.build_support_memory()
        device = next(self.backbone.parameters()).device
        if self._support_memory['tokens'].device != device:
            self._support_memory = {k: v.to(device) for k, v in self._support_memory.items()}
        return self._support_memory

    def support_cache_metadata(self, source_checkpoint, source_identity=None):
        from mmengine.fileio import get, join_path
        cfg = self.support_visual_cfg
        dataset = cfg['dataset']
        annotation = join_path(dataset.get('data_root', ''), dataset['ann_file'])
        backend = dataset.get('backend_args')
        def digest(path):
            if path is None:
                raise ValueError('A pretrained load_from checkpoint is required to extract Support.')
            if Path(path).is_file():
                hasher = hashlib.sha256()
                with Path(path).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(chunk)
                return hasher.hexdigest()
            return hashlib.sha256(get(str(path), backend_args=backend)).hexdigest()
        return dict(version=1, class_names=list(cfg['class_names']),
                    annotation_sha256=digest(annotation),
                    source_checkpoint_sha256=(source_identity['source_checkpoint_sha256']
                        if source_identity else digest(source_checkpoint)),
                    source_checkpoint=(source_identity['source_checkpoint']
                        if source_identity else str(source_checkpoint)),
                    dataset=json.loads(json.dumps(dataset, sort_keys=True)),
                    preprocessor=dict(
                        mean=self.data_preprocessor.mean.detach().cpu().flatten().tolist(),
                        std=self.data_preprocessor.std.detach().cpu().flatten().tolist(),
                        bgr_to_rgb=self.data_preprocessor._channel_conversion,
                        pad_size_divisor=self.data_preprocessor.pad_size_divisor,
                        pad_value=self.data_preprocessor.pad_value),
                    extraction=dict(backbone='Swin-B.norm1', stride=8, channels=256,
                                    roi_size=7, aligned=True, sampling_ratio=2,
                                    layout='object-row-column', dtype='float32'))

    def prepare_support_cache(self, path, metadata, require_existing=False):
        """Called after checkpoint loading, before any optimizer/query forward."""
        path = Path(path)
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            # A relocated source checkpoint is allowed if its contents match.
            identity = lambda m: {k: v for k, v in m.items() if k != 'source_checkpoint'}
            if identity(saved['metadata']) != identity(metadata):
                raise ValueError(f'Support cache metadata mismatch: {path}')
            memory = saved['memory']
            tokens, counts, labels = (memory[k] for k in ('tokens', 'counts', 'class_ids'))
            classes = len(metadata['class_names'])
            if (tokens.ndim != 2 or tokens.shape[1] != 256 or
                    tokens.dtype != torch.float32 or not torch.isfinite(tokens).all() or
                    counts.shape != (classes,) or counts.dtype != torch.long or
                    (counts <= 0).any() or (counts % 49 != 0).any() or
                    labels.dtype != torch.long or not torch.equal(
                        labels, torch.arange(classes).repeat_interleave(counts)) or
                    len(tokens) != int(counts.sum())):
                raise ValueError(f'Invalid local descriptor cache: {path}')
            self._support_memory = {k: v.detach() for k, v in memory.items()}
        elif require_existing:
            raise FileNotFoundError(
                f'Resume/Test requires the original Support cache: {path}. '
                'Copy support_local_memory.pt alongside the experiment checkpoint.')
        else:
            memory = self.support_memory()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            try:
                torch.save(dict(metadata=metadata, memory={
                    k: v.detach().cpu() for k, v in memory.items()}), temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        self._support_cache_metadata = metadata
        return self.support_memory()
