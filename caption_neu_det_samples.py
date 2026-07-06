import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_IMAGE_DIR = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/'
    'CDFSOD/datasets/NEU-DET/train'
)
DEFAULT_MODEL_ID = 'Salesforce/blip-image-captioning-base'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Caption random NEU-DET training images with BLIP.')
    parser.add_argument(
        '--image-dir',
        default=DEFAULT_IMAGE_DIR,
        help='Directory containing training images.')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random images to caption.')
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
    return parser.parse_args()


def find_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_images(images: list[Path], num_samples: int,
                  seed: int | None) -> list[Path]:
    if num_samples <= 0:
        raise ValueError('--num-samples must be positive.')

    rng = random.Random(seed)
    if len(images) <= num_samples:
        return list(images)
    return rng.sample(images, num_samples)


def caption_image(
    image_path: Path,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')

    images = find_images(image_dir)
    if not images:
        raise FileNotFoundError(f'No image files found in: {image_dir}')

    selected_images = sample_images(images, args.num_samples, args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = BlipProcessor.from_pretrained(args.model_id)
    model = BlipForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()

    print(f'Model: {args.model_id}')
    print(f'Device: {device}')
    print(f'Image directory: {image_dir}')
    print(f'Captioning {len(selected_images)} image(s)')
    print()

    for index, image_path in enumerate(selected_images, start=1):
        caption = caption_image(
            image_path=image_path,
            processor=processor,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        print(f'[{index}] {image_path}')
        print(f'caption: {caption}')
        print()


if __name__ == '__main__':
    main()
