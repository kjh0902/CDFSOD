import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_IMAGE_DIR = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/'
    'CDFSOD/datasets/NEU-DET/train'
)
DEFAULT_ANN_FILE = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/'
    'CDFSOD/datasets/NEU-DET/annotations/train.json'
)
DEFAULT_MODEL_ID = 'Salesforce/blip-image-captioning-base'
DEFAULT_OUTPUT_DIR = 'caption_outputs/neu_det_blip_objects'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Caption random NEU-DET object crops with BLIP.')
    parser.add_argument(
        '--image-dir',
        default=DEFAULT_IMAGE_DIR,
        help='Directory containing training images.')
    parser.add_argument(
        '--ann-file',
        default=DEFAULT_ANN_FILE,
        help='COCO annotation file containing object bounding boxes.')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random object crops to caption.')
    parser.add_argument(
        '--model-id',
        default=DEFAULT_MODEL_ID,
        help='Hugging Face model id to load.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional random seed for repeatable sampling.')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=50,
        help='Maximum number of caption tokens to generate.')
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Directory where annotated images and captions are saved.')
    parser.add_argument(
        '--bbox-width',
        type=int,
        default=4,
        help='Line width for the red ground-truth bbox.')
    return parser.parse_args()


def load_coco_annotations(ann_file: Path) -> tuple[dict, dict, list[dict]]:
    with ann_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    images_by_id = {image['id']: image for image in data.get('images', [])}
    categories_by_id = {
        category['id']: category.get('name', str(category['id']))
        for category in data.get('categories', [])
    }

    annotations = []
    for annotation in data.get('annotations', []):
        bbox = annotation.get('bbox')
        image_id = annotation.get('image_id')
        if image_id not in images_by_id:
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        annotations.append(annotation)

    return images_by_id, categories_by_id, annotations


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    image_path = image_dir / file_name
    if image_path.is_file():
        return image_path

    fallback_path = image_dir / Path(file_name).name
    if fallback_path.is_file():
        return fallback_path

    return image_path


def sample_annotations(annotations: list[dict], num_samples: int,
                       seed: int | None) -> list[dict]:
    if num_samples <= 0:
        raise ValueError('--num-samples must be positive.')

    rng = random.Random(seed)
    if len(annotations) <= num_samples:
        return list(annotations)
    return rng.sample(annotations, num_samples)


def clamp_bbox(image: Image.Image, bbox: list[float]) -> tuple[int, int, int, int]:
    image_width, image_height = image.size
    x, y, width, height = bbox

    left = max(0, int(x))
    top = max(0, int(y))
    right = min(image_width, int(x + width))
    bottom = min(image_height, int(y + height))

    if right <= left or bottom <= top:
        raise ValueError(f'Invalid crop after clamping: {bbox}')

    return left, top, right, bottom


def crop_object(image: Image.Image, bbox: list[float]) -> Image.Image:
    return image.crop(clamp_bbox(image, bbox))


def draw_gt_bbox(
    image: Image.Image,
    bbox: list[float],
    category_name: str,
    bbox_width: int,
) -> Image.Image:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    left, top, right, bottom = clamp_bbox(annotated_image, bbox)
    draw.rectangle(
        (left, top, right, bottom),
        outline='red',
        width=max(1, bbox_width),
    )

    label = f'GT: {category_name}'
    text_bbox = draw.textbbox((left, top), label)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label_top = max(0, top - text_height - 4)
    draw.rectangle(
        (left, label_top, left + text_width + 6, label_top + text_height + 4),
        fill='red',
    )
    draw.text((left + 3, label_top + 2), label, fill='white')
    return annotated_image


def make_sample_dir(output_dir: Path, index: int, annotation: dict) -> Path:
    annotation_id = annotation.get('id', 'unknown')
    sample_dir = output_dir / f'sample_{index:03d}_ann_{annotation_id}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def save_sample_outputs(
    sample_dir: Path,
    image: Image.Image,
    crop: Image.Image,
    bbox: list[float],
    caption: str,
    image_path: Path,
    annotation: dict,
    category_name: str,
    bbox_width: int,
) -> None:
    annotated_image = draw_gt_bbox(image, bbox, category_name, bbox_width)
    annotated_image.save(sample_dir / 'image_with_gt_bbox.jpg', quality=95)
    crop.save(sample_dir / 'object_crop.jpg', quality=95)

    with (sample_dir / 'caption.txt').open('w', encoding='utf-8') as f:
        f.write(f'image_path: {image_path}\n')
        f.write(f'annotation_id: {annotation.get("id", "unknown")}\n')
        f.write(f'image_id: {annotation.get("image_id", "unknown")}\n')
        f.write(f'category: {category_name}\n')
        f.write(f'bbox_xywh: {bbox}\n')
        f.write(f'caption: {caption}\n')


def caption_crop(
    crop: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = processor(crop, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    ann_file = Path(args.ann_file)
    output_dir = Path(args.output_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if not ann_file.is_file():
        raise FileNotFoundError(f'Annotation file not found: {ann_file}')

    images_by_id, categories_by_id, annotations = load_coco_annotations(ann_file)
    if not annotations:
        raise ValueError(f'No valid bbox annotations found in: {ann_file}')

    selected_annotations = sample_annotations(
        annotations,
        args.num_samples,
        args.seed,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = BlipProcessor.from_pretrained(args.model_id)
    model = BlipForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()

    print(f'Model: {args.model_id}')
    print(f'Device: {device}')
    print(f'Image directory: {image_dir}')
    print(f'Annotation file: {ann_file}')
    print(f'Output directory: {output_dir}')
    print(f'Captioning {len(selected_annotations)} object crop(s)')
    print()

    for index, annotation in enumerate(selected_annotations, start=1):
        image_info = images_by_id[annotation['image_id']]
        image_path = resolve_image_path(image_dir, image_info['file_name'])
        if not image_path.is_file():
            raise FileNotFoundError(f'Image not found: {image_path}')

        image = Image.open(image_path).convert('RGB')
        crop = crop_object(image, annotation['bbox'])
        caption = caption_crop(
            crop=crop,
            processor=processor,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        category_name = categories_by_id.get(annotation.get('category_id'), 'unknown')
        sample_dir = make_sample_dir(output_dir, index, annotation)
        save_sample_outputs(
            sample_dir=sample_dir,
            image=image,
            crop=crop,
            bbox=annotation['bbox'],
            caption=caption,
            image_path=image_path,
            annotation=annotation,
            category_name=category_name,
            bbox_width=args.bbox_width,
        )

        print(f'[{index}] {image_path}')
        print(f'annotation_id: {annotation.get("id", "unknown")}')
        print(f'category: {category_name}')
        print(f'bbox_xywh: {annotation["bbox"]}')
        print(f'caption: {caption}')
        print(f'saved_dir: {sample_dir}')
        print()


if __name__ == '__main__':
    main()
