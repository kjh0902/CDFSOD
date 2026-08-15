#!/usr/bin/env python
"""Generate Qwen3-VL descriptions for COCO support-set instances.

Example:
    python tools/generate_instance_captions.py \
        --dataset-root /home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET \
        --ann-file annotations/1_shot.json \
        --img-prefix train \
        --output annotations/1_shot_captions.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image


DESCRIPTION_PROMPT = """Class name: {class_name}
Bounding box: {bbox} in [x, y, width, height] format.

Describe only the visual characteristics inside the bounding box.
Do not mention the class name or bounding box in the output."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Describe each GT bbox from the full image with Qwen3-VL.'))
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-prefix', default='train')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--model-name',
        default='Qwen/Qwen3-VL-8B-Instruct',
        help='Hugging Face Qwen3-VL model name or local path.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def build_description_prompt(class_name, bbox):
    return DESCRIPTION_PROMPT.format(
        class_name=class_name,
        bbox=json.dumps(bbox, ensure_ascii=False))


def flush_batch(batch, processor, model, device, max_new_tokens, captions):
    if not batch:
        return

    conversations = []
    for item in batch:
        conversations.append([{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': item['image'],
                },
                {
                    'type': 'text',
                    'text': build_description_prompt(
                        item['meta']['category_name'], item['meta']['bbox']),
                },
            ],
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

    for item, caption in zip(batch, decoded):
        meta = item['meta']
        meta['caption'] = caption.strip()
        captions.append(meta)
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

    captions = []
    batch = []
    image_cache = {}

    for ann in coco['annotations']:
        image_info = images[ann['image_id']]
        file_name = image_info['file_name']
        image_path = Path(file_name)
        if not image_path.is_absolute():
            image_path = img_root / file_name

        if image_path not in image_cache:
            image_cache[image_path] = Image.open(image_path).convert('RGB')

        batch.append({
            'image': image_cache[image_path],
            'meta': {
                'ann_id': ann['id'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'category_name': categories.get(ann['category_id'], ''),
                'bbox': ann['bbox'],
                'file_name': file_name,
            }
        })

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

    print(f'Wrote {len(captions)} visual descriptions to {output_path}')


if __name__ == '__main__':
    main()
