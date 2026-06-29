import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def normalize_image(image: dict) -> dict:
    normalized = dict(image)
    normalized['file_name'] = Path(str(image['file_name'])).name
    return normalized


def make_kshot(data: dict, k_shot: int, seed: int) -> dict:
    rng = random.Random(seed)
    images_by_id = {image['id']: normalize_image(image) for image in data['images']}
    annotations_by_image = defaultdict(list)
    image_ids_by_category = defaultdict(set)

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        annotations_by_image[image_id].append(annotation)
        image_ids_by_category[annotation['category_id']].add(image_id)

    selected_image_ids = set()
    for category in data['categories']:
        category_id = category['id']
        candidates = sorted(image_ids_by_category.get(category_id, ()))
        if len(candidates) <= k_shot:
            selected_image_ids.update(candidates)
        else:
            selected_image_ids.update(rng.sample(candidates, k_shot))

    selected_images = [
        images_by_id[image_id]
        for image_id in sorted(selected_image_ids)
        if image_id in images_by_id
    ]
    selected_annotations = [
        annotation
        for image_id in sorted(selected_image_ids)
        for annotation in annotations_by_image.get(image_id, [])
    ]

    return {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': data['categories'],
    }


def write_kshot_files(
    input_file: Path,
    output_dir: Path,
    shots: list[int],
    seed: int,
) -> None:
    with input_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        output_file = output_dir / f'{shot}_shot.json'
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(make_kshot(data, shot, seed), f, indent=2)
        print(f'{shot}-shot annotations saved to {output_file}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create k-shot COCO annotation files from NEU-DET train.json.')
    parser.add_argument('--dataset-root', default='datasets/NEU-DET')
    parser.add_argument('--input', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    annotation_dir = dataset_root / 'annotations'
    input_file = Path(args.input) if args.input else annotation_dir / 'train.json'
    output_dir = Path(args.output_dir) if args.output_dir else annotation_dir
    write_kshot_files(input_file, output_dir, args.shots, args.seed)


if __name__ == '__main__':
    main()
