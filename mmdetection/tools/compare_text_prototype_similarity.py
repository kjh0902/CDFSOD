#!/usr/bin/env python
"""Compare selected class-name token cosine similarity from two checkpoints.

This disposable analysis script loads two checkpoints, encodes the same
support enriched prompts, selects only the BERT output tokens whose offsets
overlap the class-name span, and prints class-to-class token similarities.

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
DEFAULT_ENRICHED_CLASS_TOKEN_CKPT = (
    '/home/aislab5090/CDFSOD/junhyung/grounding_dino_idea/CDFSOD/'
    'mmdetection/work_dirs/NEU-DET_1shot_enriched_class_tokens/epoch_30.pth')
DEFAULT_CAPTION_FILE = (
    '/home/aislab5090/CDFSOD/junhyung/datasets/NEU-DET/'
    'annotations/1_shot_captions.json')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare selected class-name token cosine similarities.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--class-name-ckpt', default=DEFAULT_CLASS_NAME_CKPT)
    parser.add_argument(
        '--enriched-class-token-ckpt',
        '--class-prototype-ckpt',
        dest='enriched_class_token_ckpt',
        default=DEFAULT_ENRICHED_CLASS_TOKEN_CKPT)
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

    cfg.model.use_enriched_class_tokens = True
    cfg.model.support_caption_file = caption_file
    cfg.model.debug_text_tokens = False
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


def get_raw_and_projected_class_tokens(model):
    """Return selected class-name token features before/after text_feat_map."""
    with torch.no_grad():
        raw, labels = model.compute_support_class_token_features()
        projected = model.text_feat_map(raw) if model.text_feat_map else raw
    return raw.detach(), projected.detach(), labels.detach()


def cosine_matrix(features):
    features = F.normalize(features, dim=-1)
    return features @ features.t()


def offdiag_values(matrix):
    mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def class_pair_cosine_matrix(features, labels, num_classes):
    token_matrix = cosine_matrix(features)
    class_matrix = features.new_zeros((num_classes, num_classes))
    for i in range(num_classes):
        row_mask = labels == i
        for j in range(num_classes):
            col_mask = labels == j
            class_matrix[i, j] = token_matrix[row_mask][:, col_mask].mean()
    return class_matrix, token_matrix


def summarize_class_separation(name, class_names, features, labels,
                               print_matrix):
    matrix, _ = class_pair_cosine_matrix(
        features, labels, len(class_names))
    values = offdiag_values(matrix)
    norms = features.norm(dim=-1)
    print(f'\n[{name}]')
    print(f'selected class-name token feature shape: {tuple(features.shape)}')
    print('selected token counts per class:')
    for class_idx, class_name in enumerate(class_names):
        print(f'  - {class_name}: {(labels == class_idx).sum().item()}')
    print(f'token feature norm min/max: {norms.min().item():.6f} / '
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
        print('\nclass-by-class mean pairwise token cosine matrix:')
        header = 'class'.ljust(18) + ''.join(
            cls[:10].rjust(12) for cls in class_names)
        print(header)
        for cls, row in zip(class_names, matrix):
            values_text = ''.join(f'{value.item():12.4f}' for value in row)
            print(cls[:17].ljust(18) + values_text)
    return matrix


def compare_same_class_between_checkpoints(name, class_names, left, right,
                                           labels):
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    token_sims = (left * right).sum(dim=-1)

    print(f'\n[{name}] same selected-token cosine between checkpoints')
    for class_idx, cls in enumerate(class_names):
        sims = token_sims[labels == class_idx]
        print(f'{cls:18s}: mean={sims.mean().item():.6f}, '
              f'min={sims.min().item():.6f}, max={sims.max().item():.6f}')
    print(f'overall mean selected-token cosine: {token_sims.mean().item():.6f}')


def main():
    args = parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        device = 'cpu'

    for path_name, path in [
            ('config', args.config),
            ('class-name checkpoint', args.class_name_ckpt),
            ('enriched-class-token checkpoint',
             args.enriched_class_token_ckpt),
            ('caption file', args.caption_file)]:
        if not osp.exists(path):
            raise FileNotFoundError(f'{path_name} not found: {path}')

    print('config:', args.config)
    print('caption file:', args.caption_file)
    print('class-name checkpoint:', args.class_name_ckpt)
    print('enriched-class-token checkpoint:',
          args.enriched_class_token_ckpt)
    print('device:', device)

    class_name_model = build_model(
        args.config, args.class_name_ckpt, args.caption_file, device)
    class_token_model = build_model(
        args.config, args.enriched_class_token_ckpt, args.caption_file,
        device)
    class_names = class_name_model.support_class_names

    name_raw, name_proj, labels = get_raw_and_projected_class_tokens(
        class_name_model)
    token_raw, token_proj, token_labels = get_raw_and_projected_class_tokens(
        class_token_model)
    if not torch.equal(labels.cpu(), token_labels.cpu()):
        raise RuntimeError('Selected class-token labels differ between '
                           'checkpoints. Check caption/config alignment.')

    summarize_class_separation(
        'class_name checkpoint / raw BERT class-name token features',
        class_names, name_raw, labels, args.print_matrix)
    summarize_class_separation(
        'enriched-class-token checkpoint / raw BERT class-name token features',
        class_names, token_raw, labels, args.print_matrix)
    summarize_class_separation(
        'class_name checkpoint / projected detector class-name tokens',
        class_names, name_proj, labels, args.print_matrix)
    summarize_class_separation(
        'enriched-class-token checkpoint / projected detector class-name '
        'tokens',
        class_names, token_proj, labels, args.print_matrix)

    compare_same_class_between_checkpoints(
        'raw BERT class-name token features', class_names, name_raw,
        token_raw, labels)
    compare_same_class_between_checkpoints(
        'projected detector class-name tokens', class_names, name_proj,
        token_proj, labels)

    print('\nInterpretation hint:')
    print('- Lower off-diagonal mean cosine means selected class-name token '
          'features are more separated inside that checkpoint.')
    print('- The script does not average token features for '
          'training; it only averages pairwise token similarities for this '
          'summary table.')


if __name__ == '__main__':
    main()
