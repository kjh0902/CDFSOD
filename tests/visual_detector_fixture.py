"""Integration regressions without MMCV/CUDA or downloaded BERT weights.

Execute repository detector/proposal/serial-decoder/head-forward methods using
small deterministic encoder/attention fixtures. Compare against the independent
serial GroundingDINO source pinned in grounding_dino_acl, not another copy of
our modified path. Full CUDA training is outside this CPU harness.
"""
import ast
import copy
import math
import importlib.util
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import warnings

import torch
from torch import nn
from visual_test_support import ROOT, sample, visual, tokenizer_masks
from support_transformer_fixture import Encoder as MemoryEncoder

BASE = '8926970ebff1a549088b0a4c87c272e1a70fe0dd'
DETECTOR = 'mmdet/models/detectors/grounding_dino.py'
LAYERS = 'mmdet/models/layers/transformer/grounding_dino_layers.py'
HEAD = 'mmdet/models/dense_heads/grounding_dino_head.py'


def source(path, baseline=False):
    if baseline:
        return subprocess.check_output(['git', 'show', f'{BASE}:{path}'],
                                       cwd=ROOT, encoding='utf8')
    return (ROOT / path).read_text(encoding='utf-8-sig')


def node(text, name):
    return next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def execute(nodes, env):
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<repository methods>', 'exec'), env)


ENV = dict(torch=torch, nn=nn, math=math, re=re, warnings=warnings, copy=copy,
           support_text_classes=visual.support_text_classes,
           SupportVisualMixin=visual.SupportVisualMixin)
utils = source('mmdet/models/layers/transformer/utils.py')
execute([node(utils, 'inverse_sigmoid'), node(utils, 'coordinate_to_encoding')], ENV)
glip = source('mmdet/models/detectors/glip.py')
execute([node(glip, 'create_positive_map'),
         node(glip, 'create_positive_map_label_to_token')], ENV)
execute([node(source(HEAD), 'ContrastiveEmbed')], ENV)


def detector_class(baseline=False):
    env = dict(ENV)
    cls = node(source(DETECTOR, baseline), 'GroundingDINO')
    cls.decorator_list = []
    cls.bases = ([ast.Name(id='SupportVisualMixin', ctx=ast.Load())] if not baseline else []) + [
        ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())]
    execute([node(source(DETECTOR), 'clean_label_name'), cls], env)
    detector = env['GroundingDINO']
    for filename, method in [('mmdet/models/detectors/dino.py', 'forward_decoder'),
                             ('mmdet/models/detectors/deformable_detr.py',
                              'gen_encoder_output_proposals')]:
        execute([node(source(filename), method)], env)
        setattr(detector, method, env[method])
    return detector, env


class Sample(NS):
    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Tokens(dict):
    def __init__(self, text):
        self.spans = [m.span() for m in re.finditer(r'\w+|\.', text)]
        ids = [101] + [119 if text[a:b] == '.' else 10+i for i,(a,b) in enumerate(self.spans)] + [102]
        super().__init__(attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                         input_ids=torch.tensor([ids]))

    def char_to_token(self, char):
        return next((i + 1 for i, (a, b) in enumerate(self.spans) if a <= char < b), None)


class Tokenizer:
    all_special_ids = [0, 101, 102]
    def __call__(self, captions, **kwargs):
        return Tokens(captions[0])
    def convert_tokens_to_ids(self, token):
        return 119 if token == '.' else -1


class Language(nn.Module):
    pad_to_max = False
    max_tokens = 32

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.tokenizer = Tokenizer()

    def tokenizer(self, captions, **kwargs):
        return Tokens(captions[0])

    def forward(self, captions):
        lengths = [len(Tokens(c)['attention_mask'][0]) for c in captions]
        ids = torch.arange(max(lengths))[None].expand(len(captions), -1)
        mask = ids < torch.tensor(lengths)[:, None]
        token_ids = torch.zeros_like(ids)
        for i, caption in enumerate(captions):
            token_ids[i, :lengths[i]] = Tokens(caption)['input_ids'][0]
        attention, positions = tokenizer_masks(token_ids)
        return dict(embedded=self.embedding(ids), text_token_mask=mask,
                    position_ids=positions, masks=attention)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(8, 8)
        self.text = nn.Linear(8, 8)

    def forward(self, query, memory_text, **kwargs):
        return (self.visual(query) + memory_text.mean(1, keepdim=True),
                self.text(memory_text) + query.mean(1, keepdim=True))


class AttentionFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.1)
        self.inputs = []
        self.outputs = []

    def forward(self, query, value, memory_text, **kwargs):
        self.inputs.append(query.clone())
        output = self.dropout(self.linear(query + value.mean(1, keepdim=True)
                                          + memory_text.mean(1, keepdim=True)))
        self.outputs.append(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 3
        self.layers = nn.ModuleList([AttentionFixture() for _ in range(3)])
        self.ref_point_head = nn.Linear(512, 8)
        self.norm = nn.LayerNorm(8)
        self.return_intermediate = True


dec_env = dict(ENV)
execute([node(source('mmdet/models/layers/transformer/dino_layers.py'), 'forward')], dec_env)
Decoder.forward = dec_env['forward']


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.max_text_len = 32
        self.cls_branches = nn.ModuleList([ENV['ContrastiveEmbed'](32) for _ in range(4)])
        self.reg_branches = nn.ModuleList([nn.Linear(8, 4) for _ in range(4)])
        self.inputs = None

    def loss(self, batch_data_samples, **inputs):
        # Record every actual detection-head input. Matching/loss implementations
        # are unchanged and therefore receive bitwise identical tensors.
        self.inputs = inputs
        scores, boxes = self.forward(inputs['hidden_states'], inputs['references'],
                                     inputs['memory_text'], inputs['text_token_mask'],
                                     )
        self.outputs = (scores, boxes)
        return dict(loss_cls=scores[torch.isfinite(scores)].square().mean(),
                    loss_bbox=boxes.square().mean(),
                    loss_encoder=inputs['enc_outputs_coord'].square().mean())


head_env = dict(ENV)
execute([node(ast.unparse(node(source(HEAD), 'GroundingDINOHead')), 'forward')], head_env)
Head.forward = head_env['forward']


class DN(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embedding = nn.Embedding(3, 8)
        self.calls = 0

    def forward(self, samples):
        self.calls += 1
        batch = len(samples)
        return (self.label_embedding.weight[None, :1].expand(batch, -1, -1)
                + torch.randn(batch, 1, 8) * .01,
                torch.randn(batch, 1, 4), torch.zeros(4, 4, dtype=torch.bool),
                dict(num_denoising_queries=1, num_denoising_groups=1))


def build(baseline=False, weight=0, real_encoder=False):
    cls, env = detector_class(baseline)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.support_visual_cfg = None
    model._visual_stats = None
    model.language_model = Language()
    model._special_tokens = '. '
    model.use_autocast = False
    model.text_feat_map = nn.Linear(8, 8)
    model.encoder = MemoryEncoder() if real_encoder else Encoder()
    model.decoder = Decoder()
    model.bbox_head = Head()
    model.num_queries = 3
    model.query_embedding = nn.Embedding(3, 8)
    model.memory_trans_fc = nn.Linear(8, 8)
    model.memory_trans_norm = nn.LayerNorm(8)
    model.dn_query_generator = DN()
    model.extract_feat = lambda inputs: inputs

    def pre_transformer(features, samples):
        shapes = torch.tensor([[4, 4], [2, 2]])
        starts = torch.tensor([0, 16])
        mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        mask[0, 15] = True
        ratios = torch.ones(len(features), 2, 2)
        return (dict(feat=features, feat_mask=mask, feat_pos=torch.zeros_like(features),
                     spatial_shapes=shapes, level_start_index=starts, valid_ratios=ratios),
                dict(memory_mask=mask, spatial_shapes=shapes,
                     level_start_index=starts, valid_ratios=ratios))
    model.pre_transformer = pre_transformer
    return model, env


def samples(mode='shared', empty=False):
    items = [sample([[0, 0, 4, 4], [3, 2, 7, 8]], [0, 0]),
             sample([[1, 0, 6, 7]], [1])]
    result = [Sample(**vars(s), text=('red fox', 'blue', 'green')) for s in items]
    if mode == 'different':
        result[1].text = ('red fox', 'green', 'blue')
    if mode == 'explicit':
        for s in result:
            s.text = 'red fox. blue. green. '
            s.tokens_positive = [[[0, 7]], [[9, 13]], [[15, 20]]]
    if empty:
        for s in result:
            s.gt_instances = sample([], []).gt_instances
    return result


def assert_same(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            assert_same(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_same(test, x, y)
    else:
        test.assertEqual(a, b)
