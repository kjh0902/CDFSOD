#!/usr/bin/env python3
"""Read/verify ACL diagnostics with only Python and Numpy installed."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    'acl_diagnostic_io', Path(__file__).resolve().parents[2] /
    'mmdet/utils/acl_diagnostic_io.py')
_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_io)
DiagnosticReader = _io.DiagnosticReader


def box_iou(boxes, gt):
    intersection = np.maximum(0, np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                              np.maximum(boxes[:, None, :2], gt[None, :, :2])).prod(-1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(-1)
    gt_area = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(-1)
    return intersection / np.maximum(area[:, None] + gt_area[None] - intersection, 1e-12)


def original_boxes(boxes, meta):
    boxes = boxes.astype(np.float32)
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2], -1)
    h, w = meta['img_shape'][:2]
    xyxy *= np.array([w, h, w, h], dtype=np.float32)
    return xyxy / np.tile(np.asarray(meta['scale_factor'], dtype=np.float32), 2)


def analyze(reader):
    """Examples, not replacement COCO metrics: per-query diagnostics by epoch.

    Class confusion uses each final query's best IoU>=.5 GT, not one-to-one
    detection assignment. Objectness is explicitly a max-class-score proxy.
    """
    annotations = {}
    cat_to_class = {row['category_id']: row['class_index']
                    for row in reader.dataset['categories']}
    for row in reader.dataset['annotations']:
        if (not row.get('ignore', False) and not row.get('iscrowd', False)
                and row['category_id'] in cat_to_class):
            annotations.setdefault(row['image_id'], []).append(row)
    epochs = {}
    for record in reader.observations():
        arrays = reader.load(record)
        epoch = epochs.setdefault(record['epoch'], dict(
            observations=0, gt=0, candidate_hits=0, selected_hits=0,
            final_query_hits=0, query_class_confusion={},
            objectness_sum=0., queries=0, fe_cosine_sum=0., tokens=0))
        epoch['observations'] += 1
        scores = arrays['decoder_class_scores'][-1].astype(np.float32)
        offset, classes = record['class_offset'], scores.shape[-1]
        gt_rows = [row for row in annotations.get(record['img_id'], [])
                   if offset <= cat_to_class[row['category_id']] < offset + classes]
        if gt_rows:
            gt = np.array([row['bbox'] for row in gt_rows], dtype=np.float32)
            gt[:, 2:] += gt[:, :2]
            _, valid = reader.proposals(arrays)
            candidate = original_boxes(arrays['encoder_boxes'], record['image_meta'])
            selected = arrays['selected_indices']
            final = original_boxes(arrays['decoder_boxes'][-1], record['image_meta'])
            epoch['gt'] += len(gt)
            for name, boxes in [('candidate', candidate[valid]),
                                ('selected', candidate[selected][valid[selected]]),
                                ('final_query', final)]:
                iou = box_iou(boxes, gt)
                epoch[name + '_hits'] += int((iou.max(0) >= .5).sum()) if len(boxes) else 0
            iou = box_iou(final, gt)
            best = iou.argmax(-1)
            labels = scores.argmax(-1) + offset
            for query in np.flatnonzero(iou.max(-1) >= .5):
                truth = cat_to_class[gt_rows[best[query]]['category_id']]
                key = f'{truth}->{labels[query]}'
                epoch['query_class_confusion'][key] = epoch['query_class_confusion'].get(key, 0) + 1
        epoch['objectness_sum'] += float(scores.max(-1).sum())
        epoch['queries'] += len(scores)
        before = arrays['bert_after_map'].astype(np.float32)
        after = arrays['fe_text'][-1].astype(np.float32)
        cosine = (before * after).sum(-1) / np.maximum(
            np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1), 1e-12)
        epoch['fe_cosine_sum'] += float(cosine.sum())
        epoch['tokens'] += len(cosine)
    for epoch in epochs.values():
        for name in ('candidate', 'selected', 'final_query'):
            epoch[name + '_gt_coverage_at_0_5'] = (
                epoch.pop(name + '_hits') / epoch['gt'] if epoch['gt'] else None)
        epoch['mean_objectness_proxy'] = epoch.pop('objectness_sum') / max(1, epoch.pop('queries'))
        epoch['mean_bert_to_fe_token_cosine'] = epoch.pop('fe_cosine_sum') / max(1, epoch.pop('tokens'))
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    reader = DiagnosticReader(args.directory)
    result = {'manifest': reader.manifest}
    if args.verify:
        result['verification'] = reader.verify()
    if args.analyze:
        result['analysis'] = analyze(reader)
    print(json.dumps(_io.json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
