#!/usr/bin/env python
"""Generate five Qwen3-VL attribute descriptions per support-set class.

Example:
    python tools/generate_instance_captions.py \
        --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
        --ann-file annotations/1_shot.json \
        --img-prefix train \
        --output annotations/1_shot_caption_candidates.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image


ATTRIBUTE_SPECS = (
    ('shape', 'Shape / Geometry',
     'overall form, length, orientation, curvature, thickness, and geometric '
     'arrangement'),
    ('texture', 'Texture / Surface Pattern',
     'roughness, smoothness, repetition, density, granularity, and local '
     'surface patterns'),
    ('boundary', 'Boundary / Edge',
     'edge sharpness, irregularity, continuity, thickness, contrast, and '
     'contour patterns'),
    ('internal_structure', 'Internal Structure / Parts',
     'constituent parts and their arrangement, connectivity, branching, '
     'repetition, and local structural relationships'),
    ('color_intensity_material', 'Color / Intensity / Material',
     'brightness, darkness, color distribution, local contrast, reflectance, '
     'and material appearance'),
)

COMMON_DESCRIPTION_INSTRUCTION = (
    'Describe only visual characteristics shared across all provided crops.\n'
    'Focus only on {focus}.\n\n'
    'Output exactly one concise sentence.\n'
    'Start directly with the visual attributes.\n'
    'Do not mention the images, crops, regions, or class name.\n'
    'Do not infer causes, functions, or meanings; describe only visible '
    'appearance.')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate five attribute-wise descriptions from all GT bbox '
            'crops of each support class.'))
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
        help='Number of class-attribute prompts per inference batch.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def crop_bbox(image, bbox):
    x, y, width, height = bbox
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image.width, int(round(x + width)))
    bottom = min(image.height, int(round(y + height)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert('RGB')


def build_attribute_description_prompt(class_name, attribute_key):
    specs = {key: (label, focus) for key, label, focus in ATTRIBUTE_SPECS}
    if attribute_key not in specs:
        raise KeyError(f'Unknown attribute: {attribute_key}')
    label, focus = specs[attribute_key]
    instruction = COMMON_DESCRIPTION_INSTRUCTION.format(focus=focus)
    return (f'Class name: {class_name}\n'
            f'Attribute perspective: {label}\n\n{instruction}')


def _class_name_pattern(class_name):
    words = [word for word in re.split(r'[_\-\s]+', class_name.strip())
             if word]
    return r'(?<!\w)' + r'[\s_\-]+'.join(map(re.escape, words)) + r'(?!\w)'


def normalize_and_validate_description(description, class_name):
    """Normalize a generated sentence and enforce the prompt contract."""
    description = ' '.join(description.strip().strip('"\'').split())
    description = re.sub(
        r'^(shape|texture|boundary|internal structure|'
        r'color\s*/\s*intensity\s*/\s*material)\s*:\s*',
        '', description, flags=re.IGNORECASE)
    if not description:
        raise ValueError(
            f'Qwen returned an empty description for {class_name}')
    if re.search(_class_name_pattern(class_name), description,
                 flags=re.IGNORECASE):
        raise ValueError(
            f'Description directly mentions class name {class_name!r}: '
            f'{description!r}')
    if description[-1] not in '.!?':
        description += '.'
    sentence_ends = re.findall(r'[.!?]+(?=\s|$)', description)
    if len(sentence_ends) != 1:
        raise ValueError(
            f'Description must contain exactly one sentence for {class_name}: '
            f'{description!r}')
    return description


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
            image_cache[image_path] = Image.open(image_path).convert('RGB')
        crop = crop_bbox(image_cache[image_path], ann['bbox'])
        if crop is None:
            raise ValueError(
                f'Invalid bbox for annotation {ann["id"]}: {ann["bbox"]}')

        category_id = ann['category_id']
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


def flush_batch(batch, processor, model, device, max_new_tokens,
                captions_by_category):
    if not batch:
        return

    conversations = []
    for task in batch:
        class_group = task['class_group']
        content = [
            {
                'type': 'image',
                'image': instance['image'],
            } for instance in class_group['instances']
        ]
        content.append({
            'type': 'text',
            'text': build_attribute_description_prompt(
                class_group['category_name'], task['attribute_key']),
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

    for task, caption in zip(batch, decoded):
        class_group = task['class_group']
        caption = normalize_and_validate_description(
            caption, class_group['category_name'])
        captions_by_category[class_group['category_id']]['candidates'][
            task['attribute_key']] = caption
    batch.clear()


def main():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    args = parse_args()
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

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, dtype='auto')
    model.to(device)
    model.eval()

    captions_by_category = {}
    batch = []
    class_groups = build_class_groups(coco, images, categories, img_root)

    for class_group in class_groups:
        instances = class_group['instances']
        captions_by_category[class_group['category_id']] = {
            'category_id': class_group['category_id'],
            'category_name': class_group['category_name'],
            'ann_ids': [instance['ann_id'] for instance in instances],
            'image_ids': [instance['image_id'] for instance in instances],
            'bboxes': [instance['bbox'] for instance in instances],
            'file_names': [instance['file_name'] for instance in instances],
            'candidates': {},
        }
        for attribute_key, _, _ in ATTRIBUTE_SPECS:
            batch.append({
                'class_group': class_group,
                'attribute_key': attribute_key,
            })

            if len(batch) >= args.batch_size:
                flush_batch(batch, processor, model, device,
                            args.max_new_tokens, captions_by_category)

    flush_batch(batch, processor, model, device, args.max_new_tokens,
                captions_by_category)

    captions = [
        captions_by_category[class_group['category_id']]
        for class_group in class_groups
    ]

    output = {
        'ann_file': str(args.ann_file),
        'img_prefix': str(args.img_prefix),
        'model_name': args.model_name,
        'attributes': [
            {'key': key, 'label': label} for key, label, _ in ATTRIBUTE_SPECS
        ],
        'captions': captions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'Wrote {len(captions)} class-level candidate sets to {output_path}')


if __name__ == '__main__':
    main()
