#!/usr/bin/env python
"""Compare class text prototype cosine similarity from two checkpoints.

This is a disposable analysis script for the CDFSOD GroundingDINO prototype
experiment. It loads two checkpoints, rebuilds the support-prompt class text
prototypes from the same support caption JSON, and prints cosine similarities.

Default example:
    python mmdetection/tools/compare_text_prototype_similarity.py
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_state_dict


THIS_FILE = Path(__file__).resolve()
MMDET_DIR = THIS_FILE.parents[1]
REPO_ROOT = MMDET_DIR.parent
if str(MMDET_DIR) not in sys.path:
    sys.path.insert(0, str(MMDET_DIR))

from mmdet.registry import MODELS  # noqa: E402


DEFAULT_CONFIG = (
    MMDET_DIR / 'configs/mm_grounding_dino/CDFSOD/'
    'GroundingDINO-few-shot-SwinB.py')
DEFAULT_CLASS_NAME_CKPT = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/CDFSOD/'
    'mmdetection/work_dirs/NEU-DET_1shot_class_name/epoch_30.pth')
DEFAULT_CLASS_PROTO_CKPT = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/CDFSOD/'
    'mmdetection/work_dirs/NEU-DET_1shot_class_prototype/epoch_30.pth')
DEFAULT_CAPTION_FILE = (
    '/home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET/'
    'annotations/1_shot_captions.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare class text prototype cosine similarities.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--class-name-ckpt', default=DEFAULT_CLASS_NAME_CKPT)
    parser.add_argument('--class-prototype-ckpt',
                        default=DEFAULT_CLASS_PROTO_CKPT)
    parser.add_argument('--caption-file', default=DEFAULT_CAPTION_FILE)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--print-matrix',
        action='store_true',
        help='Print full class-by-class cosine matrices.')
    return parser.parse_args()


def build_model(config_path, checkpoint_path, caption_file, device):
    cfg = Config.fromfile(config_path)
    init_default_scope('mmdet')

    cfg.model.use_class_text_prototypes = True
    cfg.model.support_caption_file = caption_file
    cfg.model.debug_text_prototype = False
    cfg.model.train_cfg = None

    model = MODELS.build(cfg.model)
    load_checkpoint_state_dict(model, checkpoint_path)
    model.to(device)
    model.eval()
    return model


def load_checkpoint_state_dict(model, checkpoint_path):
    """Load trusted local MMEngine checkpoints with PyTorch >= 2.6."""
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)


def get_raw_and_projected_prototypes(model):
    """Return BERT CLS prototypes and detector-projected prototypes."""
    with torch.no_grad():
        raw = model.compute_class_text_prototypes()
        projected = model.text_feat_map(raw) if model.text_feat_map else raw
    return raw.detach(), projected.detach()


def cosine_matrix(features):
    features = F.normalize(features, dim=-1)
    return features @ features.t()


def offdiag_values(matrix):
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def summarize_class_separation(name, class_names, features, print_matrix):
    matrix = cosine_matrix(features)
    values = offdiag_values(matrix)
    norms = features.norm(dim=-1)
    print(f'\n[{name}]')
    print(f'prototype shape: {tuple(features.shape)}')
    print(f'prototype norm min/max: {norms.min().item():.6f} / '
          f'{norms.max().item():.6f}')
    print(f'off-diagonal mean cosine: {values.mean().item():.6f}')
    print(f'off-diagonal std cosine:  {values.std().item():.6f}')
    print(f'off-diagonal min cosine:  {values.min().item():.6f}')
    print(f'off-diagonal max cosine:  {values.max().item():.6f}')
    upper = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    pair_scores = matrix[upper]
    pair_indices = upper.nonzero(as_tuple=False)
    best_idx = pair_scores.argmax()
    worst_idx = pair_scores.argmin()
    best_pair = pair_indices[best_idx]
    worst_pair = pair_indices[worst_idx]
    print('most similar pair: '
          f'{class_names[best_pair[0]]} - {class_names[best_pair[1]]} '
          f'({pair_scores[best_idx].item():.6f})')
    print('least similar pair: '
          f'{class_names[worst_pair[0]]} - {class_names[worst_pair[1]]} '
          f'({pair_scores[worst_idx].item():.6f})')

    if print_matrix:
        print('\nclass-by-class cosine matrix:')
        header = 'class'.ljust(18) + ''.join(
            cls[:10].rjust(12) for cls in class_names)
        print(header)
        for cls, row in zip(class_names, matrix):
            values_text = ''.join(f'{value.item():12.4f}' for value in row)
            print(cls[:17].ljust(18) + values_text)
    return matrix


def compare_same_class_between_checkpoints(name, class_names, left, right):
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    sims = (left * right).sum(dim=-1)

    print(f'\n[{name}] same-class cosine between checkpoints')
    for cls, sim in zip(class_names, sims):
        print(f'{cls:18s}: {sim.item():.6f}')
    print(f'mean same-class cosine: {sims.mean().item():.6f}')


def main():
    args = parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        device = 'cpu'

    for path_name, path in [
            ('config', args.config),
            ('class-name checkpoint', args.class_name_ckpt),
            ('class-prototype checkpoint', args.class_prototype_ckpt),
            ('caption file', args.caption_file)]:
        if not osp.exists(path):
            raise FileNotFoundError(f'{path_name} not found: {path}')

    print('config:', args.config)
    print('caption file:', args.caption_file)
    print('class-name checkpoint:', args.class_name_ckpt)
    print('class-prototype checkpoint:', args.class_prototype_ckpt)
    print('device:', device)

    class_name_model = build_model(
        args.config, args.class_name_ckpt, args.caption_file, device)
    class_proto_model = build_model(
        args.config, args.class_prototype_ckpt, args.caption_file, device)
    class_names = class_name_model.support_class_names

    name_raw, name_proj = get_raw_and_projected_prototypes(class_name_model)
    proto_raw, proto_proj = get_raw_and_projected_prototypes(class_proto_model)

    summarize_class_separation(
        'class_name checkpoint / raw BERT CLS prototypes',
        class_names, name_raw, args.print_matrix)
    summarize_class_separation(
        'class_prototype checkpoint / raw BERT CLS prototypes',
        class_names, proto_raw, args.print_matrix)
    summarize_class_separation(
        'class_name checkpoint / projected detector prototypes',
        class_names, name_proj, args.print_matrix)
    summarize_class_separation(
        'class_prototype checkpoint / projected detector prototypes',
        class_names, proto_proj, args.print_matrix)

    compare_same_class_between_checkpoints(
        'raw BERT CLS prototypes', class_names, name_raw, proto_raw)
    compare_same_class_between_checkpoints(
        'projected detector prototypes', class_names, name_proj, proto_proj)

    print('\nInterpretation hint:')
    print('- Lower off-diagonal mean cosine means class prototypes are more '
          'separated inside that checkpoint.')
    print('- Compare class_name vs class_prototype off-diagonal means to test '
          'your hypothesis.')


if __name__ == '__main__':
    main()
