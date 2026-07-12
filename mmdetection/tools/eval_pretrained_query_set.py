#!/usr/bin/env python
"""Evaluate pretrained GroundingDINO directly on the query set.

This disposable script does not use the K-shot support set or support
captions. It loads the natural-image pretrained GroundingDINO weight below,
disables the class-name-token prototype path, and evaluates the dataset
query/test split with ordinary class-name prompts.

Example:
    python mmdetection/tools/eval_pretrained_query_set.py \
      --dataset NEU-DET
"""

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

THIS_FILE = Path(__file__).resolve()
MMDET_DIR = THIS_FILE.parents[1]
if str(MMDET_DIR) not in sys.path:
    sys.path.insert(0, str(MMDET_DIR))

from mmdet.registry import RUNNERS  # noqa: E402
from mmdet.utils import setup_cache_size_limit_of_dynamo  # noqa: E402


DEFAULT_CONFIG = (
    MMDET_DIR / 'configs/mm_grounding_dino/CDFSOD/'
    'GroundingDINO-few-shot-SwinB.py')
DEFAULT_DATA_ROOT = '/home/aislab5090/CDFSOD/junhyung/datasets'
PRETRAINED_GROUNDING_DINO = (
    'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/'
    'groundingdino_swinb_cogcoor_mmdet-55949c9c.pth')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a pretrained model on the CDFSOD query set.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--dataset', default='NEU-DET')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--query-ann', default='annotations/test.json')
    parser.add_argument('--query-img-prefix', default='test/')
    parser.add_argument('--work-dir', default=None)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. key=value.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def set_dataset_env(args) -> None:
    os.environ['CDFSOD_DATA_ROOT'] = args.data_root
    os.environ['CDFSOD_DATASET'] = args.dataset
    os.environ['CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES'] = '0'


def disable_support_prototypes(cfg: Config) -> None:
    cfg.model.use_class_name_token_prototypes = False
    cfg.model.use_enriched_class_tokens = False
    cfg.model.use_class_text_prototypes = False
    cfg.model.support_caption_file = None
    cfg.model.debug_text_tokens = False


def set_query_split(cfg: Config, args) -> None:
    data_root = osp.join(args.data_root, args.dataset) + '/'
    cfg.test_dataloader.dataset.data_root = data_root
    cfg.test_dataloader.dataset.ann_file = args.query_ann
    cfg.test_dataloader.dataset.data_prefix = dict(img=args.query_img_prefix)
    cfg.test_evaluator.ann_file = data_root + args.query_ann


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    set_dataset_env(args)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = PRETRAINED_GROUNDING_DINO
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join(
            'work_dirs',
            f'{args.dataset}_pretrained_query_eval')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    disable_support_prototypes(cfg)
    set_query_split(cfg, args)

    print('dataset:', args.dataset)
    print('pretrained load_from:', cfg.load_from)
    print('query ann:', cfg.test_dataloader.dataset.ann_file)
    print('query img prefix:', cfg.test_dataloader.dataset.data_prefix['img'])
    print('support prototypes:', cfg.model.use_class_name_token_prototypes)
    print('work dir:', cfg.work_dir)

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.test()


if __name__ == '__main__':
    main()
