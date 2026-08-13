#!/usr/bin/env python
"""Generate BLIP captions for COCO support-set object crops.

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
from transformers import BlipForConditionalGeneration, BlipProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Caption each GT bbox crop in a COCO annotation file.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--img-prefix', default='train')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--model-name',
        default='Salesforce/blip-image-captioning-base',
        help='Hugging Face BLIP model name or local path.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=30)
    return parser.parse_args()


def resolve_path(root: Path, path: str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def crop_bbox(image: Image.Image, bbox):
    x, y, w, h = bbox
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image.width, int(round(x + w)))
    bottom = min(image.height, int(round(y + h)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert('RGB')


def flush_batch(batch, processor, model, device, max_new_tokens, captions):
    if not batch:
        return

    crops = [item['crop'] for item in batch]
    inputs = processor(images=crops, return_tensors='pt', padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)

    for item, caption in zip(batch, decoded):
        meta = item['meta']
        meta['caption'] = caption.strip()
        captions.append(meta)
    batch.clear()


def main():
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

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
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
        crop = crop_bbox(image_cache[image_path], ann['bbox'])
        if crop is None:
            continue

        batch.append({
            'crop': crop,
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

    print(f'Wrote {len(captions)} captions to {output_path}')


if __name__ == '__main__':
    main()
