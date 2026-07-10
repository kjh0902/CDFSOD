import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_INPUT_NAMES = (
    'data.json',
    'instances.json',
    'instances_default.json',
    'annotations.json',
    'trainval.json',
)


def find_default_input(annotation_dir: Path) -> Path:
    for name in DEFAULT_INPUT_NAMES:
        candidate = annotation_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'No default annotation file found in {annotation_dir}. '
        'Pass one explicitly with --input.')


def collect_image_names(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        return set()
    return {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def normalize_image(image: dict) -> dict:
    normalized = dict(image)
    normalized['file_name'] = Path(str(image['file_name'])).name
    return normalized


def subset_coco(data: dict, image_ids: set) -> dict:
    images = [
        normalize_image(image)
        for image in data['images']
        if image['id'] in image_ids
    ]
    annotations = [
        annotation
        for annotation in data['annotations']
        if annotation['image_id'] in image_ids
    ]
    return {
        'images': images,
        'annotations': annotations,
        'categories': data['categories'],
    }


def split_dataset(
    input_file: Path,
    train_output_file: Path,
    test_output_file: Path,
    train_dir: Path,
    test_dir: Path,
    train_size: float,
    seed: int,
) -> None:
    with input_file.open('r', encoding='utf-8') as f:
        data = json.load(f)

    train_names = collect_image_names(train_dir)
    test_names = collect_image_names(test_dir)
    image_by_name = {
        Path(str(image['file_name'])).name: image
        for image in data['images']
    }

    if train_names or test_names:
        train_ids = {
            image_by_name[name]['id']
            for name in train_names
            if name in image_by_name
        }
        test_ids = {
            image_by_name[name]['id']
            for name in test_names
            if name in image_by_name
        }
    else:
        rng = random.Random(seed)
        images = list(data['images'])
        rng.shuffle(images)
        split_index = int(len(images) * train_size)
        train_ids = {image['id'] for image in images[:split_index]}
        test_ids = {image['id'] for image in images[split_index:]}

    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f'{len(overlap)} images appear in both splits.')

    train_output_file.parent.mkdir(parents=True, exist_ok=True)
    test_output_file.parent.mkdir(parents=True, exist_ok=True)
    with train_output_file.open('w', encoding='utf-8') as f:
        json.dump(subset_coco(data, train_ids), f, indent=2)
    with test_output_file.open('w', encoding='utf-8') as f:
        json.dump(subset_coco(data, test_ids), f, indent=2)

    print(f'Train annotations saved to {train_output_file}')
    print(f'Test annotations saved to {test_output_file}')
    print(f'Train images: {len(train_ids)}')
    print(f'Test images: {len(test_ids)}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create NEU-DET train/test COCO annotation files.')
    parser.add_argument(
        '--dataset-root',
        default='/home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET')
    parser.add_argument('--input', default=None)
    parser.add_argument('--train-output', default=None)
    parser.add_argument('--test-output', default=None)
    parser.add_argument('--train-size', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    annotation_dir = dataset_root / 'annotations'
    input_file = (
        Path(args.input) if args.input else find_default_input(annotation_dir))
    train_output = (
        Path(args.train_output) if args.train_output
        else annotation_dir / 'train.json')
    test_output = (
        Path(args.test_output) if args.test_output
        else annotation_dir / 'test.json')

    split_dataset(
        input_file=input_file,
        train_output_file=train_output,
        test_output_file=test_output,
        train_dir=dataset_root / 'train',
        test_dir=dataset_root / 'test',
        train_size=args.train_size,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
