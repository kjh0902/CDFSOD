"""Numpy-only diagnostic format, atomic storage and offline reader.

No MMDetection, torch, dataset, tokenizer or checkpoint is needed for reading.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np


SCHEMA_VERSION = 1


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_bytes(value):
    return (json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def pack_mask(mask):
    return np.packbits(np.asarray(mask, dtype=bool).ravel(), bitorder='little')


def unpack_mask(packed, shape):
    return np.unpackbits(packed, count=int(np.prod(shape)),
                         bitorder='little').reshape(shape).astype(bool)


def reconstruct_proposals(spatial_shapes, valid_hw, padding_mask=None):
    """Version 1: same grids/bounds as gen_encoder_output_proposals.

    Returns normalized cxcywh anchors and their validity. Invalid inverse-
    sigmoid anchors are +inf in the model and thus normalize to ones.
    """
    anchors = []
    for level, ((height, width), (valid_h, valid_w)) in enumerate(
            zip(spatial_shapes, valid_hw)):
        y, x = np.meshgrid(np.arange(height, dtype=np.float32),
                           np.arange(width, dtype=np.float32), indexing='ij')
        with np.errstate(divide='ignore', invalid='ignore'):
            xy = (np.stack([x, y], -1) + np.float32(.5)) / np.array(
                [valid_w, valid_h], dtype=np.float32)
        wh = np.full_like(xy, np.float32(.05 * 2. ** level))
        anchors.append(np.concatenate([xy, wh], -1).reshape(-1, 4))
    anchors = np.concatenate(anchors)
    valid = ((anchors > .01) & (anchors < .99)).all(-1)
    if padding_mask is not None:
        valid &= ~np.asarray(padding_mask, dtype=bool)
    anchors[~valid] = 1.
    return anchors, valid


def classwise_ap(precision, iou_thrs, category_ids):
    """Extract unrounded COCO AP at area=all, maxDets=last.

    Missing thresholds and categories with no evaluable GT remain explicit.
    """
    rows = []
    for column, category_id in enumerate(category_ids):
        row = {'category_id': int(category_id), 'reasons': {}}
        for name, threshold in [('AP', None), ('AP50', .5), ('AP75', .75)]:
            values = precision[:, :, column, 0, -1]
            if threshold is not None:
                matching = np.flatnonzero(np.isclose(iou_thrs, threshold))
                if not len(matching):
                    row[name] = None
                    row['reasons'][name] = 'iou_threshold_unavailable'
                    continue
                values = values[matching]
            values = values[values > -1]
            row[name] = float(values.mean()) if values.size else None
            if not values.size:
                row['reasons'][name] = 'no_evaluable_ground_truth'
        rows.append(row)
    return rows


class DiagnosticWriter:
    """Synchronous, bounded-batch storage. All failures propagate."""

    def __init__(self, root, manifest, dataset, reserve_gib=10.,
                 capacity_margin=1.2, estimated_bytes=0):
        self.root = Path(root)
        if not math.isfinite(reserve_gib) or reserve_gib < 0:
            raise ValueError('reserve_gib must be finite and nonnegative')
        if not math.isfinite(capacity_margin) or capacity_margin < 1:
            raise ValueError('capacity_margin must be finite and >= 1')
        self.reserve = int(reserve_gib * 1024 ** 3)
        self.margin = capacity_margin
        self.root.mkdir(parents=True, exist_ok=False)
        self.estimated_bytes = int(estimated_bytes)
        self.written_bytes = 0
        self.check_space(self.estimated_bytes)
        self.manifest = dict(manifest, schema_version=SCHEMA_VERSION,
                             status='running', reserve_gib=reserve_gib,
                             capacity_margin=capacity_margin,
                             estimated_bytes=self.estimated_bytes)
        self._shared = {}
        self._prompts = set()
        self.atomic_json('manifest.json', self.manifest)
        self.atomic_json('dataset.json', dataset)

    def check_space(self, remaining=0, next_bytes=0):
        free = shutil.disk_usage(self.root).free
        needed = self.reserve + max(int(remaining * self.margin),
                                   int(next_bytes * 1.02) + 65536)
        if free < needed:
            raise OSError(f'ACL diagnostics: insufficient disk space at {self.root}: '
                          f'{free / 1024**3:.2f} GiB free, '
                          f'{needed / 1024**3:.2f} GiB required '
                          '(remaining data estimate plus safety reserve).')

    def atomic_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.partial')
        payload = json_bytes(value)
        self.check_space(next_bytes=len(payload))
        with temporary.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def append(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_bytes(value)
        self.check_space(next_bytes=len(payload))
        with path.open('ab') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def save_arrays(self, relative, arrays):
        if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
            raise TypeError('Object arrays are forbidden in ACL diagnostics')
        raw_bytes = sum(a.nbytes for a in arrays.values())
        self.check_space(next_bytes=raw_bytes + 1024 * len(arrays))
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        size = path.stat().st_size
        self.written_bytes += size
        return size

    def shared_text(self, epoch, arrays, metadata):
        # Hash the actual stored bytes and metadata, never assume equal prompts
        # imply equal embeddings or reuse representations across epochs.
        digest = hashlib.sha256(json_bytes(metadata))
        for key, value in sorted(arrays.items()):
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        key = (epoch, digest.hexdigest())
        if key not in self._shared:
            prompt_hash = hashlib.sha256(json_bytes(metadata)).hexdigest()
            prompt_ref = f'prompts/{prompt_hash}.json'
            if prompt_hash not in self._prompts:
                self.atomic_json(prompt_ref, metadata)
                self._prompts.add(prompt_hash)
            relative = f'epoch_{epoch:03d}/text/{key[1]}.npz'
            self.save_arrays(relative, arrays)
            self.atomic_json(relative + '.json', dict(prompt_ref=prompt_ref))
            self._shared[key] = relative
        return self._shared[key]

    def finish(self):
        self.manifest.update(status='complete', tensor_bytes=self.written_bytes)
        self.atomic_json('manifest.json', self.manifest)


class DiagnosticReader:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.read_json('manifest.json')
        if self.manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported diagnostic schema')
        self.dataset = self.read_json('dataset.json')
        self._text_metadata = {}

    def read_json(self, relative):
        return json.loads((self.root / relative).read_text(encoding='utf-8'))

    def records(self, relative):
        with (self.root / relative).open(encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)

    def observations(self):
        for path in sorted(self.root.glob('epoch_*/index.jsonl')):
            for record in self.records(path.relative_to(self.root)):
                text_ref = record['text_ref']
                if text_ref not in self._text_metadata:
                    prompt_ref = self.read_json(text_ref + '.json')['prompt_ref']
                    self._text_metadata[text_ref] = self.read_json(prompt_ref)
                yield dict(record, **self._text_metadata[text_ref])

    def load(self, record):
        with np.load(self.root / record['shard'], allow_pickle=False) as archive:
            prefix = record['prefix']
            arrays = {key[len(prefix):]: archive[key] for key in archive.files
                      if key.startswith(prefix)}
        with np.load(self.root / record['text_ref'], allow_pickle=False) as archive:
            arrays.update({key: archive[key] for key in archive.files})
        return arrays

    def proposals(self, arrays):
        count = int(np.prod(arrays['spatial_shapes'], axis=-1).sum())
        padding = unpack_mask(arrays['padding_mask'], (count,))
        return reconstruct_proposals(arrays['spatial_shapes'], arrays['valid_hw'], padding)

    def verify(self):
        if list(self.root.rglob('*.partial')):
            raise ValueError('Incomplete .partial files found')
        count = 0
        images = {}
        observation_counts = {}
        known_ids = {row['id'] for row in self.dataset['images']}
        for record in self.observations():
            if record['img_id'] not in known_ids:
                raise ValueError('Observation image is absent from the GT snapshot')
            arrays = self.load(record)
            anchors, valid = self.proposals(arrays)
            np.testing.assert_array_equal(valid, unpack_mask(
                arrays['proposal_valid'], valid.shape))
            n, q = len(anchors), arrays['decoder_boxes'].shape[1]
            if arrays['encoder_boxes'].shape != (n, 4) or arrays['encoder_class_scores'].shape[0] != n:
                raise ValueError('Encoder candidate shape mismatch')
            selected = arrays['selected_indices']
            if np.any(selected >= n) or np.any(selected < 0) or len(selected) != q:
                raise ValueError('Invalid encoder-to-query indices')
            query, cls = arrays['det_query_index'], arrays['det_class_index_local']
            classes = arrays['decoder_class_scores'].shape[-1]
            if np.any(query < 0) or np.any(query >= q):
                raise ValueError('Invalid detection query indices')
            if record['class_mapping'] is not None:
                if np.any(cls < 0) or np.any(cls >= classes):
                    raise ValueError('Invalid detection class indices')
                np.testing.assert_array_equal(arrays['det_flat_topk_index'], query * classes + cls)
            np.testing.assert_array_equal(arrays['det_class_index'], cls + record['class_offset'])
            if len(query) != len(arrays['det_boxes']):
                raise ValueError('Detection provenance length mismatch')
            images.setdefault(record['epoch'], set()).add(record['img_id'])
            observation_counts[record['epoch']] = observation_counts.get(record['epoch'], 0) + 1
            count += 1
        for path in sorted(self.root.glob('epoch_*/complete.json')):
            completed = self.read_json(path.relative_to(self.root))
            if len(images.get(completed['epoch'], set())) != completed['unique_images']:
                raise ValueError('Completed epoch image count mismatch')
            if observation_counts.get(completed['epoch'], 0) != completed['observations']:
                raise ValueError('Completed epoch observation count mismatch')
        if self.manifest['status'] != 'complete':
            raise ValueError('Diagnostic run is incomplete (manifest is not complete)')
        if set(images) != set(range(1, self.manifest['max_epochs'] + 1)):
            raise ValueError('Diagnostic run is missing validation epochs')
        if len(list(self.root.glob('epoch_*/complete.json'))) != len(images):
            raise ValueError('Diagnostic run is missing epoch completion markers')
        metric_epochs = [row['epoch'] for row in self.records('metrics.jsonl')]
        if sorted(metric_epochs) != sorted(images):
            raise ValueError('Diagnostic run has missing or duplicate epoch metrics')
        return {'observations': count, 'epochs': len(images),
                'status': self.manifest['status']}
