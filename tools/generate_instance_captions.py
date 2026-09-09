#!/usr/bin/env python
"""Generate one Qwen3-VL common description per support-set class.

Example:
    python tools/generate_instance_captions.py \
        --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
        --ann-file annotations/1_shot.json \
        --img-prefix train \
        --output annotations/1_shot_captions.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image


COMMON_DESCRIPTION_INSTRUCTION = (
    'Describe only the common visual characteristics across all images.\n\n'
    'Output exactly one concise sentence.\n'
    'Start directly with the visual attributes.\n'
    'Do not mention the images, regions, or class name.\n'
    'Do not infer causes or meanings; describe only visible appearance.')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Describe the common GT bbox appearance for each support class.'))
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-prefix', default='train')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--model-name',
        default='Qwen/Qwen3-VL-8B-Instruct',
        help='Hugging Face Qwen3-VL model name or local path.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of classes to process in one inference batch.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def crop_bbox(image, bbox):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError(f'Invalid bbox: {bbox}')
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image.width, int(round(x + width)))
    bottom = min(image.height, int(round(y + height)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert('RGB')


def build_common_description_prompt(class_name):
    return f'Class name: {class_name}\n\n{COMMON_DESCRIPTION_INSTRUCTION}'


def build_class_groups(coco, images, categories, img_root):
    groups = {}
    image_cache = {}

    for ann in coco['annotations']:
        image_info = images[ann['image_id']]
        file_name = image_info['file_name']
        image_path = Path(file_name)
        if not image_path.is_absolute():
            image_path = img_root / file_name

        if image_path not in image_cache:
            with Image.open(image_path) as image:
                image_cache[image_path] = image.convert('RGB')
        crop = crop_bbox(image_cache[image_path], ann['bbox'])
        if crop is None:
            raise ValueError(
                f'Invalid bbox for annotation {ann["id"]}: {ann["bbox"]}')

        category_id = ann['category_id']
        if category_id not in categories:
            raise ValueError(f'Unknown category for annotation {ann["id"]}')
        if category_id not in groups:
            groups[category_id] = {
                'category_id': category_id,
                'category_name': categories.get(category_id, ''),
                'instances': [],
            }
        groups[category_id]['instances'].append({
            'ann_id': ann['id'],
            'image_id': ann['image_id'],
            'bbox': ann['bbox'],
            'file_name': file_name,
            'image': crop,
        })

    return list(groups.values())


def flush_batch(batch, processor, model, device, max_new_tokens, captions):
    if not batch:
        return

    conversations = []
    for class_group in batch:
        content = [
            {
                'type': 'image',
                'image': instance['image'],
            } for instance in class_group['instances']
        ]
        content.append({
            'type': 'text',
            'text': build_common_description_prompt(
                class_group['category_name']),
        })
        conversations.append([{
            'role': 'user',
            'content': content,
        }])

    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
        padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False)

    if len(decoded) != len(batch):
        raise ValueError('Qwen output count does not match class batch size.')

    for class_group, caption in zip(batch, decoded):
        if not caption.strip().rstrip('.').strip():
            raise ValueError(f'Empty description for {class_group["category_name"]}')
        instances = class_group['instances']
        captions.append({
            'category_id': class_group['category_id'],
            'category_name': class_group['category_name'],
            'ann_ids': [instance['ann_id'] for instance in instances],
            'image_ids': [instance['image_id'] for instance in instances],
            'bboxes': [instance['bbox'] for instance in instances],
            'file_names': [instance['file_name'] for instance in instances],
            'caption': caption.strip(),
        })
    batch.clear()


def main():
    args = parse_args()
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    dataset_root = Path(args.dataset_root)
    ann_path = resolve_path(dataset_root, args.ann_file)
    img_root = resolve_path(dataset_root, args.img_prefix)
    output_path = resolve_path(dataset_root, args.output)

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = args.device

    with ann_path.open('r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {image['id']: image for image in coco['images']}
    categories = {
        category['id']: category.get('name', str(category['id']))
        for category in coco.get('categories', [])
    }

    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive.')
    class_groups = build_class_groups(coco, images, categories, img_root)
    missing = set(categories) - {group['category_id'] for group in class_groups}
    if missing or not class_groups:
        raise ValueError(f'Classes without support crops: {sorted(missing)}')

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, dtype='auto')
    model.to(device)
    model.eval()

    captions = []
    batch = []
    for class_group in class_groups:
        batch.append(class_group)

        if len(batch) >= args.batch_size:
            flush_batch(batch, processor, model, device, args.max_new_tokens,
                        captions)

    flush_batch(batch, processor, model, device, args.max_new_tokens, captions)

    output = {
        'ann_file': str(args.ann_file),
        'img_prefix': str(args.img_prefix),
        'model_name': args.model_name,
        'captions': captions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'Wrote {len(captions)} class-level descriptions to {output_path}')


if __name__ == '__main__':
    main()
