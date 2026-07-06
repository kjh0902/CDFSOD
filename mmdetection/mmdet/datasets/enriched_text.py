# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os.path as osp
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

DEFAULT_BLIP_MODEL_ID = 'Salesforce/blip-image-captioning-base'
DEFAULT_NEU_DET_DOMAIN_ATTRIBUTE = (
    'gray-scale hot-rolled steel surface with metallic texture, low color '
    'variation, and subtle industrial defect patterns')

_BLIP_CAPTIONERS = {}
_PROMPT_CACHE = {}


def build_enriched_prompt(class_name: str, object_caption: str,
                          domain_attribute: str) -> str:
    """Build the exact enriched text template used by GroundingDINO."""
    return f'{class_name}, {object_caption}, {domain_attribute}.'


def _join_path(data_root: Optional[str], path: str) -> str:
    if osp.isabs(path):
        return path
    return osp.join(data_root or '', path)


def _resolve_image_path(image_root: str, file_name: str) -> str:
    image_path = osp.join(image_root, file_name)
    if osp.exists(image_path):
        return image_path

    basename_path = osp.join(image_root, osp.basename(file_name))
    if osp.exists(basename_path):
        return basename_path

    return image_path


def _clamp_coco_bbox(bbox: Sequence[float],
                     image_size: Tuple[int, int]) -> Optional[Tuple[int, int,
                                                                    int, int]]:
    width, height = image_size
    x, y, w, h = bbox
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(width, math.ceil(x + w))
    bottom = min(height, math.ceil(y + h))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def extract_support_object_crops(
        ann_file: str,
        image_root: str,
        class_names: Sequence[str],
        cat_ids: Optional[Sequence[int]] = None,
        log_progress: bool = True) -> Dict[str, Image.Image]:
    """Extract one GT object crop per class from a COCO support annotation."""
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_by_id = {image['id']: image for image in coco.get('images', [])}
    categories = coco.get('categories', [])
    name_to_cat_id = {category['name']: category['id'] for category in categories}

    if cat_ids is None:
        target_cat_ids = [
            name_to_cat_id[class_name] for class_name in class_names
            if class_name in name_to_cat_id
        ]
    else:
        target_cat_ids = list(cat_ids)

    cat_id_to_class_name = dict(zip(target_cat_ids, class_names))
    crops: Dict[str, Image.Image] = {}

    for ann in coco.get('annotations', []):
        cat_id = ann.get('category_id')
        class_name = cat_id_to_class_name.get(cat_id)
        if class_name is None or class_name in crops:
            continue

        image_info = images_by_id.get(ann.get('image_id'))
        if image_info is None:
            continue

        image_path = _resolve_image_path(image_root, image_info['file_name'])
        if not osp.exists(image_path):
            continue

        with Image.open(image_path) as image:
            image = image.convert('RGB')
            crop_box = _clamp_coco_bbox(ann['bbox'], image.size)
            if crop_box is None:
                continue
            crops[class_name] = image.crop(crop_box).copy()

        if len(crops) == len(class_names):
            break

    if log_progress:
        per_class = ', '.join(
            f'{class_name}: {1 if class_name in crops else 0}'
            for class_name in class_names)
        print(
            f'[EnrichedText] Extracted {len(crops)} support crops '
            f'({per_class}).',
            flush=True)

    return crops


class BlipObjectCaptioner:
    """Small cached BLIP wrapper for object crop captioning."""

    def __init__(self,
                 model_id: str = DEFAULT_BLIP_MODEL_ID,
                 device: str = 'auto',
                 max_new_tokens: int = 30) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.torch = torch
        print(
            f'[EnrichedText] Loading BLIP processor from {model_id}...',
            flush=True)
        self.processor = BlipProcessor.from_pretrained(model_id)
        print(
            f'[EnrichedText] Loading BLIP model from {model_id}...',
            flush=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        print(
            f'[EnrichedText] Moving BLIP model to {self.device}...',
            flush=True)
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(
            f'[EnrichedText] Loaded BLIP captioner ({model_id}) on '
            f'{self.device}.',
            flush=True)

    def caption(self, crop: Image.Image) -> str:
        inputs = self.processor(
            crop.convert('RGB'), return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


def _get_blip_captioner(model_id: str, device: str,
                        max_new_tokens: int) -> BlipObjectCaptioner:
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    key = (model_id, device, max_new_tokens)
    if key not in _BLIP_CAPTIONERS:
        _BLIP_CAPTIONERS[key] = BlipObjectCaptioner(
            model_id=model_id, device=device, max_new_tokens=max_new_tokens)
    return _BLIP_CAPTIONERS[key]


def generate_object_captions(
        crops: Dict[str, Image.Image],
        class_names: Iterable[str],
        model_id: str = DEFAULT_BLIP_MODEL_ID,
        device: str = 'auto',
        max_new_tokens: int = 30,
        fallback_caption: str = 'a close-up object crop',
        log_progress: bool = True) -> Dict[str, str]:
    """Generate one BLIP caption per class support crop."""
    captioner = _get_blip_captioner(model_id, device, max_new_tokens)
    captions = {}
    class_names = list(class_names)
    total_crops = sum(1 for class_name in class_names if class_name in crops)
    processed_crops = 0
    if log_progress:
        print(
            f'[EnrichedText] Generating BLIP captions for {total_crops} '
            'class support crops.',
            flush=True)

    for class_name in class_names:
        crop = crops.get(class_name)
        if crop is None:
            captions[class_name] = fallback_caption
            continue

        processed_crops += 1
        if log_progress:
            print(
                f'[EnrichedText] Captioning class crop '
                f'{processed_crops}/{total_crops} ({class_name}).',
                flush=True)
        captions[class_name] = captioner.caption(crop)
    return captions


def build_enriched_class_prompts(
        data_root: Optional[str],
        ann_file: str,
        image_prefix: str,
        class_names: Sequence[str],
        cat_ids: Optional[Sequence[int]] = None,
        domain_attribute: str = DEFAULT_NEU_DET_DOMAIN_ATTRIBUTE,
        model_id: str = DEFAULT_BLIP_MODEL_ID,
        device: str = 'auto',
        max_new_tokens: int = 30,
        fallback_caption: str = 'a close-up object crop',
        log_progress: bool = True,
        enabled: bool = True) -> List[str]:
    """Build class-ordered enriched prompts from one crop per class."""
    if not enabled:
        return list(class_names)

    resolved_ann_file = _join_path(data_root, ann_file)
    image_root = _join_path(data_root, image_prefix)
    cache_key = (resolved_ann_file, image_root, tuple(class_names),
                 tuple(cat_ids or []), domain_attribute, model_id, device,
                 max_new_tokens, fallback_caption)
    if cache_key in _PROMPT_CACHE:
        return list(_PROMPT_CACHE[cache_key])

    if log_progress:
        print(
            f'[EnrichedText] Support annotation: {resolved_ann_file}',
            flush=True)
        print(f'[EnrichedText] Support image root: {image_root}', flush=True)
        print('[EnrichedText] Extracting one support crop per class...',
              flush=True)

    crops = extract_support_object_crops(
        resolved_ann_file,
        image_root,
        class_names=class_names,
        cat_ids=cat_ids,
        log_progress=log_progress)
    captions = generate_object_captions(
        crops,
        class_names,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        fallback_caption=fallback_caption,
        log_progress=log_progress)

    prompts = [
        build_enriched_prompt(class_name, captions[class_name],
                              domain_attribute)
        for class_name in class_names
    ]
    _PROMPT_CACHE[cache_key] = tuple(prompts)
    return prompts
