"""Load pure helpers / actual repository methods without requiring MMCV."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS
import warnings

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visual = load('mmdet/models/utils/support_visual.py', 'support_visual')
stats_module = load('mmdet/utils/visual_token_stats.py', 'visual_stats')
compat = load('mmdet/utils/randomness_compat.py', 'randomness_compat')


def source(rel, baseline=False):
    if baseline:
        return subprocess.check_output(['git', 'show', '8926970:' + rel],
                                       cwd=ROOT, encoding='utf8')
    return (ROOT / rel).read_text(encoding='utf-8-sig')


def node(text, name):
    return next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def execute(nodes, env):
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<repository methods>', 'exec'), env)


ENV = dict(torch=torch, nn=nn, math=math, re=re, warnings=warnings, copy=copy,
           support_text_classes=visual.support_text_classes)
execute([node(source('mmdet/models/layers/transformer/utils.py'), 'inverse_sigmoid'),
         node(source('mmdet/models/layers/transformer/utils.py'), 'coordinate_to_encoding')], ENV)
execute([node(source('mmdet/models/detectors/glip.py'), 'create_positive_map'),
         node(source('mmdet/models/detectors/glip.py'), 'create_positive_map_label_to_token')], ENV)
execute([node(source('mmdet/models/dense_heads/grounding_dino_head.py'), 'ContrastiveEmbed')], ENV)


class Sample(NS):
    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Instances(NS):
    def __len__(self):
        return len(next(iter(vars(self).values())))


def sample(boxes, labels):
    return Sample(gt_instances=Instances(
        bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor(labels, dtype=torch.long)),
        metainfo=dict(img_shape=(32, 32), ori_shape=(32, 32), scale_factor=(1., 1.)))


def tokenizer_masks(ids):
    env = dict(torch=torch)
    execute([node(source('mmdet/models/language_models/bert.py'),
                  'generate_masks_with_special_tokens_and_transfer_map')], env)
    return env['generate_masks_with_special_tokens_and_transfer_map'](
        dict(input_ids=ids), [101, 102, 119])


def text_fixture(batch=1, channels=256):
    # [CLS] red fox . blue . [SEP] [PAD] [PAD]
    ids = torch.tensor([[101, 10, 11, 119, 12, 119, 102, 0, 0]] * batch)
    mask, positions = tokenizer_masks(ids)
    text = dict(embedded=torch.randn(batch, 9, channels, requires_grad=True),
                masks=mask, position_ids=positions, text_token_mask=ids != 0)
    return text, [{1: [1, 2], 2: [4]} for _ in range(batch)]
