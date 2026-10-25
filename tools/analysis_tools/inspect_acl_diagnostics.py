#!/usr/bin/env python3
"""Read/verify ACL diagnostics with only Python and Numpy installed."""
import argparse
import importlib.util
import itertools
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


def detection_errors(epoch, boxes, scores, labels, gt_rows, cat_to_class, threshold):
    """Score-filtered, class-agnostic greedy IoU matching; not COCO evaluation."""
    keep = np.flatnonzero(scores >= threshold)
    keep = keep[np.argsort(-scores[keep], kind='stable')]
    gt = np.array([row['bbox'] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
    gt[:, 2:] += gt[:, :2]
    iou = box_iou(boxes[keep], gt)
    used = set()
    epoch['detection_gt'] += len(gt)
    epoch['detections_above_threshold'] += len(keep)
    for row, index in enumerate(keep):
        candidates = [j for j in range(len(gt)) if j not in used and iou[row, j] >= .5]
        if candidates:
            matched = max(candidates, key=lambda j: iou[row, j])
            used.add(matched)
            truth = cat_to_class[gt_rows[matched]['category_id']]
            key = f'{truth}->{int(labels[index])}'
            epoch['detection_class_confusion'][key] = epoch['detection_class_confusion'].get(key, 0) + 1
        else:
            best = float(iou[row].max()) if len(gt) else 0.
            reason = 'duplicate' if best >= .5 else 'localization' if best >= .1 else 'background'
            epoch['unmatched_detection_examples'][reason] += 1
    epoch['missed_gt_at_iou_0_5'] += len(gt) - len(used)


def analyze(reader, score_threshold=.3):
    """Every-epoch final detections plus explicitly sparse internal snapshots."""
    annotations = {}
    cat_to_class = {row['category_id']: row['class_index']
                    for row in reader.dataset['categories']}
    for row in reader.dataset['annotations']:
        if (not row.get('ignore', False) and not row.get('iscrowd', False)
                and row['category_id'] in cat_to_class):
            annotations.setdefault(row['image_id'], []).append(row)
    epochs = {}
    key = lambda row: (row['epoch'], row['batch'], row.get('image_batch_position', row['batch_position']))
    for _, group in itertools.groupby(reader.observations(), key=key):
        observations = [(record, reader.load(record)) for record in group]
        first = observations[0][0]
        epoch = epochs.setdefault(first['epoch'], dict(
            observations=0, images=0, internal_observations=0, gt=0,
            candidate_hits=0, selected_hits=0, final_query_hits=0,
            query_class_confusion={}, objectness_sum=0., queries=0,
            fe_cosine_sum=0., tokens=0, fe_class_cosine_sum=0., feature_classes=0,
            detection_score_threshold=score_threshold, detection_gt=0,
            detections_above_threshold=0, detection_class_confusion={},
            unmatched_detection_examples=dict(background=0, localization=0, duplicate=0),
            missed_gt_at_iou_0_5=0))
        epoch['images'] += 1
        all_gt = annotations.get(first['img_id'], [])
        detection_errors(epoch,
            np.concatenate([arrays['det_boxes'] for _, arrays in observations]),
            np.concatenate([arrays['det_scores'] for _, arrays in observations]),
            np.concatenate([arrays['det_class_index'] for _, arrays in observations]),
            all_gt, cat_to_class, score_threshold)
        for record, arrays in observations:
            epoch['observations'] += 1
            if 'decoder_class_scores' not in arrays:
                continue
            epoch['internal_observations'] += 1
            scores = arrays['decoder_class_scores'][-1].astype(np.float32)
            offset, classes = record['class_offset'], scores.shape[-1]
            gt_rows = [row for row in all_gt
                       if offset <= cat_to_class[row['category_id']] < offset + classes]
            if gt_rows:
                gt = np.array([row['bbox'] for row in gt_rows], dtype=np.float32)
                gt[:, 2:] += gt[:, :2]
                final = original_boxes(arrays['decoder_boxes'][-1], record['image_meta'])
                epoch['gt'] += len(gt)
                if 'gt_candidate_iou_max' in arrays:
                    epoch['candidate_hits'] += int((arrays['gt_candidate_iou_max'] >= .5).sum())
                    epoch['selected_hits'] += int((arrays['gt_selected_iou_max'] >= .5).sum())
                else:
                    _, valid = reader.proposals(arrays)
                    candidate = original_boxes(arrays['encoder_boxes'], record['image_meta'])
                    selected = arrays['selected_indices']
                    for name, boxes in [('candidate', candidate[valid]),
                                        ('selected', candidate[selected][valid[selected]])]:
                        iou = box_iou(boxes, gt)
                        epoch[name + '_hits'] += int((iou.max(0) >= .5).sum()) if len(boxes) else 0
                iou = box_iou(final, gt)
                epoch['final_query_hits'] += int((iou.max(0) >= .5).sum())
                best = iou.argmax(-1)
                labels = scores.argmax(-1) + offset
                for query in np.flatnonzero(iou.max(-1) >= .5):
                    truth = cat_to_class[gt_rows[best[query]]['category_id']]
                    pair = f'{truth}->{labels[query]}'
                    epoch['query_class_confusion'][pair] = epoch['query_class_confusion'].get(pair, 0) + 1
            epoch['objectness_sum'] += float(scores.max(-1).sum())
            epoch['queries'] += len(scores)
            before = arrays['bert_after_map'].astype(np.float32)
            if 'fe_class_prototypes' in arrays:
                before = np.stack([before[tokens].mean(0) for _, tokens in sorted(
                    record['class_mapping'].items(), key=lambda item: int(item[0]))])
                after = arrays['fe_class_prototypes'][-1].astype(np.float32)
                prefix, count = 'fe_class_cosine_sum', 'feature_classes'
            else:
                after = arrays['fe_text'][-1].astype(np.float32)
                prefix, count = 'fe_cosine_sum', 'tokens'
            cosine = (before * after).sum(-1) / np.maximum(
                np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1), 1e-12)
            epoch[prefix] += float(cosine.sum())
            epoch[count] += len(cosine)
    for epoch in epochs.values():
        for name in ('candidate', 'selected', 'final_query'):
            hits = epoch.pop(name + '_hits')
            epoch[name + '_gt_coverage_at_0_5'] = hits / epoch['gt'] if epoch['gt'] else None
        total, count = epoch.pop('objectness_sum'), epoch.pop('queries')
        epoch['mean_objectness_proxy'] = total / count if count else None
        for total_key, count_key, result_key in [
                ('fe_cosine_sum', 'tokens', 'mean_bert_to_fe_token_cosine'),
                ('fe_class_cosine_sum', 'feature_classes', 'mean_bert_to_fe_class_cosine')]:
            total, count = epoch.pop(total_key), epoch.pop(count_key)
            epoch[result_key] = total / count if count else None
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--score-threshold', type=float, default=.3)
    args = parser.parse_args()
    reader = DiagnosticReader(args.directory)
    result = {'manifest': reader.manifest}
    if args.verify:
        result['verification'] = reader.verify()
    if args.analyze:
        result['analysis'] = analyze(reader, args.score_threshold)
    print(json.dumps(_io.json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
