# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os
import os.path as osp
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

DEFAULT_BLIP_MODEL_ID = 'Salesforce/blip-image-captioning-base'
DEFAULT_NEU_DET_DOMAIN_ATTRIBUTE = (
    'gray-scale hot-rolled steel surface with metallic texture, low color '
    'variation, and subtle industrial defect patterns')

_BLIP_CAPTIONERS = {}
_CAPTION_LIST_CACHE = {}


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
        cat_ids: Optional[Sequence[int]] = None
) -> Dict[str, List[Image.Image]]:
    """Extract all GT object crops per class from a COCO support annotation."""
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
    crops: Dict[str, List[Image.Image]] = {
        class_name: []
        for class_name in class_names
    }

    for ann in coco.get('annotations', []):
        cat_id = ann.get('category_id')
        class_name = cat_id_to_class_name.get(cat_id)
        if class_name is None:
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
            crops[class_name].append(image.crop(crop_box).copy())

    return {
        class_name: class_crops
        for class_name, class_crops in crops.items() if class_crops
    }


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


def _safe_cache_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')


def _default_caption_cache_file(ann_file: str, model_id: str,
                                max_new_tokens: int) -> str:
    ann_dir = osp.dirname(ann_file)
    ann_name = osp.splitext(osp.basename(ann_file))[0]
    model_name = _safe_cache_name(model_id)
    cache_name = f'{ann_name}_{model_name}_max{max_new_tokens}.json'
    return osp.join(ann_dir, 'caption_cache', cache_name)


def _caption_cache_metadata(ann_file: str, image_root: str,
                            class_names: Sequence[str], model_id: str,
                            max_new_tokens: int,
                            fallback_caption: str) -> dict:
    return dict(
        ann_file=osp.abspath(ann_file),
        ann_mtime=osp.getmtime(ann_file),
        image_root=osp.abspath(image_root),
        class_names=list(class_names),
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        fallback_caption=fallback_caption)


def _load_caption_cache(cache_file: str, metadata: dict,
                        class_names: Sequence[str]) -> Optional[Dict[str,
                                                                     List[str]]]:
    if not osp.exists(cache_file):
        return None

    with open(cache_file, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if payload.get('version') != 1 or payload.get('metadata') != metadata:
        return None

    captions = payload.get('captions', {})
    if not all(isinstance(captions.get(name), list) for name in class_names):
        return None

    print(f'[EnrichedText] Loaded caption cache: {cache_file}', flush=True)
    return {name: list(captions[name]) for name in class_names}


def _save_caption_cache(cache_file: str, metadata: dict,
                        captions: Dict[str, List[str]]) -> None:
    cache_dir = osp.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(
            dict(version=1, metadata=metadata, captions=captions),
            f,
            ensure_ascii=False,
            indent=2)
    print(f'[EnrichedText] Saved caption cache: {cache_file}', flush=True)


def generate_object_captions(
        crops: Dict[str, List[Image.Image]],
        class_names: Iterable[str],
        model_id: str = DEFAULT_BLIP_MODEL_ID,
        device: str = 'auto',
        max_new_tokens: int = 30,
        fallback_caption: str = 'a close-up object crop',
        log_progress: bool = True) -> Dict[str, List[str]]:
    """Generate BLIP captions for support object crops."""
    captioner = _get_blip_captioner(model_id, device, max_new_tokens)
    captions = {}
    class_names = list(class_names)
    total_crops = sum(
        len(crops.get(class_name, [])) for class_name in class_names)
    processed_crops = 0
    if log_progress:
        print(
            f'[EnrichedText] Generating BLIP captions for {total_crops} '
            'support object crops.',
            flush=True)

    for class_name in class_names:
        class_crops = crops.get(class_name, [])
        if class_crops:
            captions[class_name] = []
            for crop in class_crops:
                processed_crops += 1
                if log_progress:
                    print(
                        f'[EnrichedText] Captioning crop '
                        f'{processed_crops}/{total_crops} '
                        f'({class_name}).',
                        flush=True)
                captions[class_name].append(captioner.caption(crop))
        else:
            captions[class_name] = [fallback_caption]
    return captions


def build_enriched_class_caption_lists(
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
        caption_cache_file: Optional[str] = None,
        log_progress: bool = True,
        enabled: bool = True) -> Dict[str, List[str]]:
    """Build class-wise support object caption lists."""
    if not enabled:
        return {class_name: [class_name] for class_name in class_names}

    resolved_ann_file = _join_path(data_root, ann_file)
    image_root = _join_path(data_root, image_prefix)
    if caption_cache_file is None:
        caption_cache_file = _default_caption_cache_file(
            resolved_ann_file, model_id, max_new_tokens)
    else:
        caption_cache_file = _join_path(data_root, caption_cache_file)

    metadata = _caption_cache_metadata(
        resolved_ann_file,
        image_root,
        class_names,
        model_id,
        max_new_tokens,
        fallback_caption)

    if log_progress:
        print(
            f'[EnrichedText] Support annotation: {resolved_ann_file}',
            flush=True)
        print(f'[EnrichedText] Support image root: {image_root}', flush=True)
        print(f'[EnrichedText] Caption cache: {caption_cache_file}', flush=True)

    cache_key = (resolved_ann_file, image_root, tuple(class_names),
                 tuple(cat_ids or []), domain_attribute, model_id, device,
                 max_new_tokens, fallback_caption, caption_cache_file)
    if cache_key in _CAPTION_LIST_CACHE:
        return {
            class_name: list(captions)
            for class_name, captions in _CAPTION_LIST_CACHE[cache_key].items()
        }

    cached_captions = _load_caption_cache(caption_cache_file, metadata,
                                          class_names)
    if cached_captions is not None:
        _CAPTION_LIST_CACHE[cache_key] = {
            class_name: tuple(caption_list)
            for class_name, caption_list in cached_captions.items()
        }
        return cached_captions

    if log_progress:
        print('[EnrichedText] Caption cache miss. Extracting support crops...',
              flush=True)
    crops = extract_support_object_crops(
        resolved_ann_file,
        image_root,
        class_names=class_names,
        cat_ids=cat_ids)
    if log_progress:
        total_crops = sum(len(class_crops) for class_crops in crops.values())
        per_class = ', '.join(
            f'{class_name}: {len(crops.get(class_name, []))}'
            for class_name in class_names)
        print(
            f'[EnrichedText] Extracted {total_crops} support crops '
            f'({per_class}).',
            flush=True)
    captions = generate_object_captions(
        crops,
        class_names,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        fallback_caption=fallback_caption,
        log_progress=log_progress)

    _CAPTION_LIST_CACHE[cache_key] = {
        class_name: tuple(caption_list)
        for class_name, caption_list in captions.items()
    }
    _save_caption_cache(caption_cache_file, metadata, captions)
    return captions


def select_enriched_class_prompts(
        class_names: Sequence[str],
        caption_lists: Dict[str, List[str]],
        domain_attribute: str = DEFAULT_NEU_DET_DOMAIN_ATTRIBUTE,
        selection: str = 'first') -> List[str]:
    """Select one caption per class and build class-ordered prompts."""
    if selection not in ('first', 'random'):
        raise ValueError(
            "selection must be either 'first' or 'random', "
            f'but got {selection!r}')

    prompts = []
    for class_name in class_names:
        captions = caption_lists.get(class_name) or ['a close-up object crop']
        caption = (
            random.choice(captions) if selection == 'random' else captions[0])
        prompts.append(
            build_enriched_prompt(class_name, caption, domain_attribute))
    return prompts


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
        caption_cache_file: Optional[str] = None,
        log_progress: bool = True,
        enabled: bool = True,
        selection: str = 'first') -> List[str]:
    """Build active class prompts from cached class-wise caption lists."""
    if not enabled:
        return list(class_names)

    caption_lists = build_enriched_class_caption_lists(
        data_root=data_root,
        ann_file=ann_file,
        image_prefix=image_prefix,
        class_names=class_names,
        cat_ids=cat_ids,
        domain_attribute=domain_attribute,
        model_id=model_id,
        device=device,
        max_new_tokens=max_new_tokens,
        fallback_caption=fallback_caption,
        caption_cache_file=caption_cache_file,
        log_progress=log_progress,
        enabled=enabled)
    return select_enriched_class_prompts(
        class_names=class_names,
        caption_lists=caption_lists,
        domain_attribute=domain_attribute,
        selection=selection)
