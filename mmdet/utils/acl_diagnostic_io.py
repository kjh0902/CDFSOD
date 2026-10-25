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


SCHEMA_VERSION = 2


def snapshot_reasons(epoch, max_epochs, interval=5, stage_changed=False):
    reasons = []
    if epoch == 1:
        reasons.append('initial')
    if epoch % interval == 0:
        reasons.append('interval')
    if stage_changed:
        reasons.append('stage_transition')
    if epoch == max_epochs:
        reasons.append('final')
    return reasons


def select_token_probes(dataset, limit=32):
    """Fixed, RNG-free image sample: cover classes/empty GT, then hash order."""
    ids = {row['id'] for row in dataset['images']}
    classes = {row['category_id'] for row in dataset['categories']}
    labels = {key: set() for key in ids}
    for row in dataset['annotations']:
        if (row['image_id'] in labels and row['category_id'] in classes and
                not row.get('ignore') and not row.get('iscrowd')):
            labels[row['image_id']].add(row['category_id'])
    ordered = sorted(ids, key=lambda key: hashlib.sha256(str(key).encode()).digest())
    selected, covered = [], set()
    empty = next((key for key in ordered if not labels[key]), None)
    if empty is not None and limit:
        selected.append(empty)
    while len(selected) < min(limit, len(ids)):
        remaining = [key for key in ordered if key not in selected]
        best = max(remaining, key=lambda key: len(labels[key] - covered))
        selected.append(best)
        covered |= labels[best]
    return selected


def compact_uint(values):
    values = np.asarray(values)
    maximum = int(values.max()) if values.size else 0
    if values.size and int(values.min()) < 0:
        raise ValueError('Diagnostic indices must be nonnegative')
    dtype = np.uint8 if maximum <= 255 else np.uint16 if maximum <= 65535 else np.uint32
    return values.astype(dtype)


def rectangular_padding(spatial_shapes, valid_hw):
    return np.concatenate([~((np.arange(h)[:, None] < vh) &
                              (np.arange(w)[None, :] < vw)).ravel()
                           for (h, w), (vh, vw) in zip(spatial_shapes, valid_hw)])


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
            prompt_ref = self.shared_prompt(metadata)
            relative = f'epoch_{epoch:03d}/text/{key[1]}.npz'
            self.save_arrays(relative, arrays)
            self.atomic_json(relative + '.json', dict(prompt_ref=prompt_ref))
            self._shared[key] = relative
        return self._shared[key]

    def shared_prompt(self, metadata):
        prompt_hash = hashlib.sha256(json_bytes(metadata)).hexdigest()
        relative = f'prompts/{prompt_hash}.json'
        if prompt_hash not in self._prompts:
            self.atomic_json(relative, metadata)
            self._prompts.add(prompt_hash)
        return relative

    def write_batch(self, epoch, batch, records, run_id, step, rank=0):
        """The production write path, also used by the storage benchmark."""
        start_bytes = self.written_bytes
        arrays, index = {}, []
        shard = f'epoch_{epoch:03d}/batch_{batch:06d}.npz'
        for position, row in enumerate(records):
            prompt = dict(caption=row['caption'], class_mapping=row['class_mapping'])
            prompt_ref = self.shared_prompt(prompt)
            text_ref = None
            if row['text']:
                text_ref = self.shared_text(epoch, row['text'], dict(
                    prompt, tokens=row['tokens']))
            prefix = f'obs_{position:04d}__'
            arrays.update({prefix + key: value for key, value in row['arrays'].items()})
            index.append(dict(
                run_id=run_id, epoch=epoch, step=step, rank=rank, batch=batch,
                batch_position=position, image_batch_position=row['image_batch_position'],
                img_id=row['metadata']['img_id'], chunk=row['chunk'],
                class_offset=row['class_offset'], prompt_ref=prompt_ref, text_ref=text_ref,
                image_meta={key: value for key, value in row['metadata'].items() if key != 'text'},
                shard=shard, prefix=prefix, capture_level=row['capture_level'],
                token_probe=row['token_probe'], num_queries=row['num_queries'],
                original_logit_width=row['original_logit_width']))
        self.save_arrays(shard, arrays)
        for record in index:
            self.append(f'epoch_{epoch:03d}/index.jsonl', record)
        self.last_batch_bytes = self.written_bytes - start_bytes
        return index

    def finish(self):
        self.manifest.update(status='complete', tensor_bytes=self.written_bytes)
        self.atomic_json('manifest.json', self.manifest)


class DiagnosticReader:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = self.read_json('manifest.json')
        if self.manifest['schema_version'] not in (1, SCHEMA_VERSION):
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
                metadata_ref = record.get('prompt_ref')
                if metadata_ref is None:
                    metadata_ref = self.read_json(text_ref + '.json')['prompt_ref']
                if metadata_ref not in self._text_metadata:
                    self._text_metadata[metadata_ref] = self.read_json(metadata_ref)
                yield dict(record, **self._text_metadata[metadata_ref])

    def load(self, record):
        with np.load(self.root / record['shard'], allow_pickle=False) as archive:
            prefix = record['prefix']
            arrays = {key[len(prefix):]: archive[key] for key in archive.files
                      if key.startswith(prefix)}
        if record['text_ref'] is not None:
            with np.load(self.root / record['text_ref'], allow_pickle=False) as archive:
                arrays.update({key: archive[key] for key in archive.files})
        if 'det_class_index_local' not in arrays:
            arrays['det_class_index_local'] = arrays['det_class_index'].astype(np.int64) - record['class_offset']
            classes = (len(record['class_mapping']) if record['class_mapping'] is not None
                       else record['original_logit_width'])
            query = arrays['det_query_index'].astype(np.int64)
            arrays['det_flat_topk_index'] = (query * classes + arrays['det_class_index_local']
                                             if record['class_mapping'] is not None else query.copy())
        return arrays

    def proposals(self, arrays):
        count = int(np.prod(arrays['spatial_shapes'], axis=-1).sum())
        padding = (unpack_mask(arrays['padding_mask'], (count,)) if 'padding_mask' in arrays
                   else rectangular_padding(arrays['spatial_shapes'], arrays['valid_hw']))
        return reconstruct_proposals(arrays['spatial_shapes'], arrays['valid_hw'], padding)

    def verify(self):
        if list(self.root.rglob('*.partial')):
            raise ValueError('Incomplete .partial files found')
        count = 0
        images = {}
        observation_counts = {}
        capture_levels = {}
        known_ids = {row['id'] for row in self.dataset['images']}
        for record in self.observations():
            if record['img_id'] not in known_ids:
                raise ValueError('Observation image is absent from the GT snapshot')
            arrays = self.load(record)
            internal = record.get('capture_level', 'full') != 'detections'
            capture_levels.setdefault(record['epoch'], set()).add(record.get('capture_level', 'full'))
            if internal:
                anchors, valid = self.proposals(arrays)
                if 'proposal_valid' in arrays:
                    np.testing.assert_array_equal(valid, unpack_mask(arrays['proposal_valid'], valid.shape))
                n, q = len(anchors), arrays['decoder_boxes'].shape[1]
                retained = arrays.get('encoder_candidate_indices', np.arange(n))
                if (len(np.unique(retained)) != len(retained) or np.any(retained >= n) or
                        arrays['encoder_boxes'].shape != (len(retained), 4) or
                        arrays['encoder_class_scores'].shape[0] != len(retained)):
                    raise ValueError('Encoder candidate shape mismatch')
                selected = arrays['selected_indices']
                if (np.any(selected >= n) or np.any(selected < 0) or len(selected) != q or
                        not np.isin(selected, retained).all()):
                    raise ValueError('Invalid encoder-to-query indices')
                classes = arrays['decoder_class_scores'].shape[-1]
                if record.get('capture_level') == 'internal':
                    required = {'fe_class_prototypes', 'bert_before_map', 'bert_after_map',
                                'gt_annotation_ids', 'gt_candidate_iou_max', 'gt_selected_iou_max'}
                    if not required <= arrays.keys():
                        raise ValueError('Missing internal snapshot arrays')
                    if record['token_probe'] and not {
                            'decoder_final_token_logits', 'fe_final_text'} <= arrays.keys():
                        raise ValueError('Missing token probe arrays')
            else:
                q = record['num_queries']
                classes = (len(record['class_mapping']) if record['class_mapping'] is not None
                           else record['original_logit_width'])
            query, cls = arrays['det_query_index'], arrays['det_class_index_local']
            if np.any(query < 0) or np.any(query >= q):
                raise ValueError('Invalid detection query indices')
            if record['class_mapping'] is not None:
                if np.any(cls < 0) or np.any(cls >= classes):
                    raise ValueError('Invalid detection class indices')
                np.testing.assert_array_equal(arrays['det_flat_topk_index'], query.astype(np.int64) * classes + cls)
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
            if 'capture_level' in completed and capture_levels[completed['epoch']] != {completed['capture_level']}:
                raise ValueError('Epoch capture policy does not match its observations')
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

    @staticmethod
    def selected_candidate_rows(arrays):
        """Map native (full encoder) query indices to retained candidate rows."""
        retained = arrays.get('encoder_candidate_indices')
        selected = arrays['selected_indices']
        if retained is None:
            return selected.astype(np.int64)
        rows = np.searchsorted(retained, selected)
        if np.any(rows >= len(retained)) or not np.array_equal(retained[rows], selected):
            raise ValueError('Selected candidate missing from sparse encoder arrays')
        return rows
