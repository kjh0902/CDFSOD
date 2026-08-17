#!/usr/bin/env python
"""Select one attribute description per class with pretrained Grounding DINO.

Example:
    python tools/select_attribute_descriptions.py \
        configs/mm_grounding_dino/CDFSOD/GroundingDINO-few-shot-SwinB.py \
        https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/\
groundingdino_swinb_cogcoor_mmdet-55949c9c.pth \
        --dataset-root /path/to/datasets/NEU-DET \
        --ann-file annotations/1_shot.json \
        --img-prefix train \
        --candidate-file annotations/1_shot_caption_candidates.json \
        --output annotations/1_shot_captions.json
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import torch
from mmcv.transforms import Compose
from mmengine.config import Config

from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from generate_instance_captions import ATTRIBUTE_SPECS


LOG_LABELS = {
    'shape': 'Shape',
    'texture': 'Texture',
    'boundary': 'Boundary',
    'internal_structure': 'Internal Structure',
    'color_intensity_material': 'Color / Intensity / Material',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Select fixed attribute descriptions before fine-tuning.')
    parser.add_argument('config', help='CDFSOD Grounding DINO config file.')
    parser.add_argument(
        'checkpoint', help='Pretrained Grounding DINO checkpoint or URL.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-prefix', default='train')
    parser.add_argument('--candidate-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--log-file',
        help='Human-readable selection log. Defaults beside --output.')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def resolve_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def build_selection_pipeline(cfg):
    """Build a deterministic image-only pipeline for support images."""
    pipeline_cfg = copy.deepcopy(get_test_pipeline_cfg(cfg))
    pipeline_cfg = [
        transform for transform in pipeline_cfg
        if transform.get('type') != 'LoadAnnotations'
    ]
    for transform in pipeline_cfg:
        if transform.get('type') == 'PackDetInputs':
            transform['meta_keys'] = (
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'scale_factor')
    return Compose(pipeline_cfg)


def prepare_image(model, pipeline, image_path, image_id):
    packed = pipeline(dict(img_path=str(image_path), img_id=image_id))
    data = dict(
        inputs=[packed['inputs']], data_samples=[packed['data_samples']])
    return model.data_preprocessor(data, training=False)


def normalize_coco_bboxes(bboxes, data_sample, device):
    """Convert COCO xywh boxes to normalized resized-image xyxy boxes."""
    boxes = torch.tensor(bboxes, dtype=torch.float32, device=device)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    scale_factor = data_sample.scale_factor
    if len(scale_factor) == 2:
        width_scale, height_scale = scale_factor
        scale_factor = (width_scale, height_scale, width_scale, height_scale)
    boxes *= boxes.new_tensor(scale_factor)
    image_height, image_width = data_sample.img_shape[:2]
    boxes[:, 0::2] /= image_width
    boxes[:, 1::2] /= image_height
    return boxes


def count_queries_in_bboxes(query_centers, argmax_text_tokens,
                             target_class_idx, normalized_bboxes):
    """Count target-class selected query centers inside each GT bbox."""
    if query_centers.ndim == 3:
        query_centers = query_centers[0]
    if argmax_text_tokens.ndim == 2:
        argmax_text_tokens = argmax_text_tokens[0]
    target_queries = argmax_text_tokens == target_class_idx
    centers = query_centers[target_queries]
    counts = []
    for x1, y1, x2, y2 in normalized_bboxes:
        inside = ((centers[:, 0] >= x1) & (centers[:, 0] <= x2) &
                  (centers[:, 1] >= y1) & (centers[:, 1] <= y2))
        counts.append(int(inside.sum().item()))
    return counts


def choose_top_attribute(scores):
    """Choose the first maximum in the fixed attribute order."""
    attribute_keys = [key for key, _, _ in ATTRIBUTE_SPECS]
    return max(attribute_keys, key=lambda key: scores[key])


def load_inputs(dataset_root, ann_file, candidate_file):
    with resolve_path(dataset_root, ann_file).open(
            'r', encoding='utf-8') as file:
        coco = json.load(file)
    with resolve_path(dataset_root, candidate_file).open(
            'r', encoding='utf-8') as file:
        candidates = json.load(file)
    entries = candidates.get('captions', [])
    if not entries:
        raise ValueError('Candidate file does not contain any captions.')
    return coco, candidates, entries


def validate_candidates(entries, coco):
    required = [key for key, _, _ in ATTRIBUTE_SPECS]
    category_names = {
        category['id']: category.get('name')
        for category in coco.get('categories', [])
    }
    seen_names = set()
    for entry in entries:
        class_name = entry.get('category_name')
        if not class_name or class_name in seen_names:
            raise ValueError(f'Invalid or duplicate class: {class_name!r}')
        seen_names.add(class_name)
        category_id = entry.get('category_id')
        if category_names.get(category_id) != class_name:
            raise ValueError(
                f'Candidate class {class_name!r} does not match category '
                f'{category_id!r} in the support annotation.')
        candidate_map = entry.get('candidates', {})
        missing = [key for key in required if not candidate_map.get(key)]
        if missing:
            raise ValueError(
                f'{class_name} is missing candidates: {", ".join(missing)}')


def build_support_index(coco):
    images = {image['id']: image for image in coco['images']}
    boxes_by_category_image = defaultdict(lambda: defaultdict(list))
    for annotation in coco['annotations']:
        boxes_by_category_image[annotation['category_id']][
            annotation['image_id']].append(annotation['bbox'])
    return images, boxes_by_category_image


def configure_selection_model(config_path, entries):
    cfg = Config.fromfile(config_path)
    class_names = [entry['category_name'] for entry in entries]
    cfg.model.use_bn_style_prompt = False
    cfg.model.use_class_name_token_prototypes = False
    cfg.model.support_caption_file = None
    cfg.model.support_class_names = class_names
    cfg.model.bbox_head.num_classes = len(class_names)
    cfg.test_dataloader.dataset.metainfo = dict(classes=tuple(class_names))
    return cfg


def format_log_entry(class_name, scores, selected_attribute,
                     selected_description):
    lines = [class_name]
    for key, _, _ in ATTRIBUTE_SPECS:
        lines.append(f'- {LOG_LABELS[key]}: {scores[key]:.6f}')
    lines.extend([
        f'- Selected: {LOG_LABELS[selected_attribute]}',
        f'- Selected description: {selected_description}',
    ])
    return '\n'.join(lines)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = resolve_path(dataset_root, args.output)
    if args.log_file:
        log_path = resolve_path(dataset_root, args.log_file)
    else:
        log_stem = output_path.stem
        if log_stem.endswith('_captions'):
            log_stem = log_stem[:-len('_captions')] + '_caption_selection'
        else:
            log_stem += '_selection'
        log_path = output_path.with_name(log_stem + '.log')

    coco, candidate_data, entries = load_inputs(
        dataset_root, args.ann_file, args.candidate_file)
    validate_candidates(entries, coco)
    images, boxes_by_category_image = build_support_index(coco)
    cfg = configure_selection_model(args.config, entries)
    model = init_detector(
        cfg, args.checkpoint, device=args.device, palette='none')
    model.eval()
    if model.num_queries != 900:
        raise RuntimeError(
            f'Expected Grounding DINO num_queries=900, got '
            f'{model.num_queries}.')
    pipeline = build_selection_pipeline(cfg)
    device = next(model.parameters()).device

    selected_entries = copy.deepcopy(entries)
    log_entries = []
    for target_class_idx, (source_entry, output_entry) in enumerate(
            zip(entries, selected_entries)):
        category_id = source_entry['category_id']
        support_images = boxes_by_category_image.get(category_id, {})
        if not support_images:
            raise ValueError(
                f'No support GT objects for {source_entry["category_name"]}.')

        descriptions = source_entry['candidates']
        with torch.no_grad():
            text_dicts = {
                key: model.build_description_selection_text_dict(
                    target_class_idx, descriptions[key], device)
                for key, _, _ in ATTRIBUTE_SPECS
            }
        object_scores = {
            key: [] for key, _, _ in ATTRIBUTE_SPECS
        }
        for image_id, bboxes in support_images.items():
            image_info = images[image_id]
            image_path = Path(image_info['file_name'])
            if not image_path.is_absolute():
                image_path = resolve_path(
                    dataset_root, Path(args.img_prefix) / image_path)
            data = prepare_image(model, pipeline, image_path, image_id)
            batch_inputs = data['inputs']
            data_samples = data['data_samples']
            normalized_bboxes = normalize_coco_bboxes(
                bboxes, data_samples[0], device)
            with torch.no_grad():
                image_features = model.extract_feat(batch_inputs)
                for key, _, _ in ATTRIBUTE_SPECS:
                    centers, argmax_tokens = \
                        model.get_language_guided_query_selection(
                            image_features, text_dicts[key], data_samples)
                    object_scores[key].extend(count_queries_in_bboxes(
                        centers, argmax_tokens, target_class_idx,
                        normalized_bboxes))

        scores = {
            key: sum(values) / len(values)
            for key, values in object_scores.items()
        }
        selected_attribute = choose_top_attribute(scores)
        selected_description = descriptions[selected_attribute]
        output_entry['scores'] = scores
        output_entry['selected_attribute'] = selected_attribute
        output_entry['selected_attribute_label'] = \
            LOG_LABELS[selected_attribute]
        output_entry['selected_description'] = selected_description
        output_entry['caption'] = selected_description
        log_entry = format_log_entry(
            source_entry['category_name'], scores, selected_attribute,
            selected_description)
        log_entries.append(log_entry)
        print(log_entry)

    result = copy.deepcopy(candidate_data)
    result['selection'] = {
        'config': str(args.config),
        'checkpoint': str(args.checkpoint),
        'num_queries': 900,
        'query_position': 'proposal_center',
        'score_reduction': 'mean_over_support_gt_objects',
    }
    result['captions'] = selected_entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as file:
        file.write('\n\n'.join(log_entries) + '\n')
    print(f'Wrote selected descriptions to {output_path}')
    print(f'Wrote selection log to {log_path}')


if __name__ == '__main__':
    main()
