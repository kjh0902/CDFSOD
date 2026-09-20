"""ACL regression tests with production methods and lightweight dependencies.

No downloaded weights or MMCV extensions: AST loading replaces only the heavy
base class/imports; encoder loop, prompt mapping, detector forward and HED head
forward run the repository's actual code. This is not a full training test.
"""
import ast
import copy
import math
import random
import re
import subprocess
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from test_second_order_etf_loss import ROOT, second

BASE = '8926970ebff1a549088b0a4c87c272e1a70fe0dd'
DETECTOR = 'mmdet/models/detectors/grounding_dino_HED.py'
HEAD = 'mmdet/models/dense_heads/grounding_dino_head_HED.py'
LAYERS = 'mmdet/models/layers/transformer/grounding_dino_layers_HED.py'
NAMES = ('crazing', 'inclusion', 'patches', 'pitted_surface',
         'rolled-in_scale', 'scratches')


def source(path, base=False):
    if base:
        return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode()
    return (ROOT / path).read_text(encoding='utf-8')


def execute(nodes, **extra):
    env = dict(torch=torch, nn=nn, math=math, copy=copy, re=re,
               random=random, warnings=warnings,
               second_order_etf_loss=second.second_order_etf_loss)
    env.update(extra)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<production>', 'exec'), env)
    return env


def method(path, cls_name, name, **env):
    cls = next(n for n in ast.parse(source(path)).body
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return execute([node], **env)[name]


def load_detector(base=False):
    nodes = [n for n in ast.parse(source(DETECTOR, base)).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef))
    cls.bases = [ast.Name(id='Module', ctx=ast.Load())]
    cls.decorator_list = []
    glip = [n for n in ast.parse(source('mmdet/models/detectors/glip.py')).body
            if isinstance(n, ast.FunctionDef) and n.name in
            {'create_positive_map', 'create_positive_map_label_to_token'}]
    return execute(glip + nodes, Module=nn.Module)[cls.name]


Detector = load_detector()
BaseDetector = load_detector(base=True)


class Sample(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Tokens(dict):
    def __init__(self, ids, mask, spans):
        super().__init__(input_ids=ids, attention_mask=mask)
        self.spans = spans

    def char_to_token(self, char):
        return next((i for i, (start, end) in enumerate(self.spans[0])
                     if start <= char < end), None)


class Tokenizer:
    def __call__(self, prompts, padding='longest', max_length=None, **kw):
        all_spans, all_ids = [], []
        for prompt in prompts:
            spans = [(m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', prompt)]
            # Emulate a multi-WordPiece class without external tokenizer files.
            spans = [piece for s, e in spans for piece in
                     ([(s, s + 3), (s + 3, e)] if prompt[s:e] == 'scratches' else [(s, e)])]
            if kw.get('truncation'):
                spans = spans[:max_length - 2]
            ids = [101] + [1 + sum(map(ord, prompt[s:e])) % 99 for s, e in spans] + [102]
            all_spans.append([(0, 0)] + spans + [(0, 0)])
            all_ids.append(ids)
        length = max(map(len, all_ids))
        if padding == 'max_length':
            length = max(length, max_length or 128)
        masks = [[1] * len(row) + [0] * (length - len(row)) for row in all_ids]
        ids = [row + [0] * (length - len(row)) for row in all_ids]
        return Tokens(torch.tensor(ids), torch.tensor(masks), all_spans)


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.embedding = nn.Embedding(128, 4)
        self.pad_to_max = False
        self.max_tokens = 64

    def forward(self, prompts):
        tokens = self.tokenizer(prompts, truncation=True, max_length=self.max_tokens,
                                padding='max_length' if self.pad_to_max else 'longest')
        x = self.embedding(tokens['input_ids'])
        self.output = x
        batch, length = x.shape[:2]
        return dict(embedded=x, text_token_mask=tokens['attention_mask'].bool(),
                    position_ids=torch.arange(length).expand(batch, -1),
                    masks=torch.ones(batch, length, length, dtype=torch.bool))


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, visual_feature, lang_feature, **kw):
        return (visual_feature + lang_feature.mean(1, keepdim=True) * 0.05,
                lang_feature + self.projection(visual_feature.mean(1, keepdim=True)) * 0.1)


class TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_cfg = SimpleNamespace(num_heads=1)
        self.attn = nn.MultiheadAttention(4, 1, batch_first=True)

    def forward(self, query, query_pos, attn_mask, **kw):
        self.output = query + self.attn(query + query_pos, query + query_pos,
                                        query, attn_mask=attn_mask)[0] * 0.1
        return self.output


class VisualLayer(nn.Module):
    def forward(self, query, **kw):
        return query


class Encoder(nn.Module):
    forward = method(LAYERS, 'GroundingDinoTransformerEncoder', 'forward',
                     get_text_sine_pos_embed=lambda x, **kw: x.expand(-1, -1, 4).float() * 0.01)
    get_encoder_reference_points = staticmethod(lambda *a, **kw: None)

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([VisualLayer() for _ in range(6)])
        self.fusion_layers = nn.ModuleList([Fusion() for _ in range(6)])
        self.text_layers = nn.ModuleList([TextLayer() for _ in range(6)])


class Decoder(nn.Module):
    num_layers = 6

    def forward(self, query, reference_points, **kw):
        self.seen = dict(query=query, reference_points=reference_points, **kw)
        return query.unsqueeze(0).expand(6, -1, -1, -1), [reference_points] * 7


class Classifier(nn.Module):
    max_text_len = 64
    forward_impl = method(HEAD, 'ContrastiveEmbed', 'forward')
    log_scale = None
    bias = None

    def forward(self, visual_feat, text_feat, text_token_mask):
        self.seen = text_feat
        return self.forward_impl(visual_feat, text_feat, text_token_mask)


class Head(nn.Module):
    num_classes = 6
    forward = method(HEAD, 'GroundingDINOHead_ParallelDecoder_DN', 'forward',
                     inverse_sigmoid=lambda x: torch.logit(x.clamp(1e-5, 1 - 1e-5)))

    def __init__(self):
        super().__init__()
        self.cls_branches = nn.ModuleList([Classifier() for _ in range(7)])
        self.reg_branches = nn.ModuleList([nn.Linear(4, 4) for _ in range(7)])

    def loss(self, batch_data_samples, **kw):
        self.seen = kw
        self.scores, self.boxes = self.forward(kw['hidden_states'], kw['references'],
                                              kw['memory_text'], kw['text_token_mask'])
        finite = self.scores[torch.isfinite(self.scores)]
        return dict(loss_cls=finite.square().mean(), loss_bbox=self.boxes.mean())

    def predict(self, **kw):
        self.seen = kw
        return [Prediction() for _ in kw['batch_data_samples']]


class Prediction:
    labels = torch.tensor([0])

    def __len__(self):
        return 1


def fixture(base=False, weight=0.1):
    torch.manual_seed(29)
    cls = BaseDetector if base else Detector
    model = cls(language_model={})
    if not base:
        model.second_order_etf_loss_weight = weight
    model.language_model = Language()
    model.text_feat_map = nn.Linear(4, 4)
    model.encoder = Encoder()
    model.decoder = Decoder()
    model.bbox_head = Head()
    model.query_embedding = nn.Embedding(3, 4)
    model.num_queries = 3
    model.test_cfg = {}
    features = torch.randn(2, 5, 4)
    model.extract_feat = lambda _: (features,)
    model.pre_transformer = lambda *a: (dict(
        feat=features, feat_mask=torch.zeros(2, 5, dtype=torch.bool),
        feat_pos=torch.zeros_like(features), spatial_shapes=torch.tensor([[1, 5]]),
        level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2)),
        dict(memory_mask=torch.zeros(2, 5, dtype=torch.bool),
             spatial_shapes=torch.tensor([[1, 5]]), level_start_index=torch.tensor([0]),
             valid_ratios=torch.ones(2, 1, 2)))
    model.gen_encoder_output_proposals = lambda memory, *a: (memory, torch.zeros_like(memory))
    model.dn_query_generator = lambda _: (torch.randn(2, 1, 4), torch.randn(2, 1, 4),
                                           None, dict(num_denoising_queries=1))
    return model


def samples():
    return [Sample(text=NAMES, gt_instances=Sample(labels=torch.tensor(labels, dtype=torch.long)))
            for labels in [[0, 2], []]]


def run(model, data):
    torch.manual_seed(73)
    random.seed(73)
    losses = model.loss(torch.zeros(2, 3, 4, 4), data)
    head = model.bbox_head
    snapshot = dict(losses=losses, head=head.seen, decoder=model.decoder.seen,
                    scores=head.scores, boxes=head.boxes,
                    positive_maps=[s.gt_instances.positive_maps for s in data],
                    gt_masks=[s.gt_instances.text_token_mask for s in data])
    return losses, snapshot


class SecondOrderIntegrationTests(unittest.TestCase):
    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, nn.Module):
            self.assert_nested_equal(a.state_dict(), b.state_dict())
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_base_detection_and_hed_inputs_unchanged_with_loss_enabled_or_disabled(self):
        base = fixture(base=True)
        _, expected = run(base, samples())
        for weight in [0., 0.1]:
            model = fixture(weight=weight)
            losses, actual = run(model, samples())
            self.assertEqual(base.state_dict().keys(), model.state_dict().keys())
            if weight:
                actual['losses'] = dict(losses)
                auxiliary = actual['losses'].pop('loss_second_order_etf')
                tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                mapping = model._get_second_order_class_token_map(tokenized, spans)
                final = model.encoder.text_layers[-1].output
                torch.testing.assert_close(auxiliary, weight * second.second_order_etf_loss(
                    final, [mapping] * 2, model.bbox_head.seen['text_token_mask']))
                for value in [model.decoder.seen['memory_text'], model.bbox_head.seen['memory_text'],
                              *[branch.seen for branch in model.bbox_head.cls_branches]]:
                    self.assertIs(value, final)
            self.assert_nested_equal(actual, expected)

    def test_auxiliary_gradient_reaches_absent_classes_and_final_enhancer(self):
        model = fixture()
        losses, _ = run(model, samples())
        final = model.encoder.text_layers[-1].output
        final.retain_grad()
        losses['loss_second_order_etf'].backward()
        tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        mapping = model._get_second_order_class_token_map(tokenized, spans)
        self.assertEqual(len(mapping), 6)
        for indices in mapping.values():
            self.assertTrue((final.grad[:, indices].norm(dim=-1) > 0).all())
        for p in [model.encoder.text_layers[-1].attn.in_proj_weight,
                  model.encoder.fusion_layers[-1].projection.weight,
                  model.text_feat_map.weight, model.language_model.embedding.weight]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertIsNone(model.query_embedding.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.bbox_head.parameters()))

    def test_shared_different_and_explicit_prompt_mapping(self):
        for mode in ['shared', 'different', 'explicit_list', 'explicit_dict']:
            with self.subTest(mode=mode):
                model = fixture()
                data = samples()
                if mode == 'different':
                    data[1].text = NAMES[::-1]
                if mode.startswith('explicit'):
                    _, caption, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                    for sample in data:
                        sample.text = caption
                        sample.tokens_positive = (dict(enumerate(spans)) if mode.endswith('dict') else spans)
                with patch.object(model, '_get_second_order_class_token_map',
                                  wraps=model._get_second_order_class_token_map) as mapping:
                    losses, _ = run(model, data)
                self.assertEqual(mapping.call_count, 1 if mode == 'shared' else 2)
                self.assertTrue(torch.isfinite(losses['loss_second_order_etf']))
                final = model.bbox_head.seen['memory_text']
                _, caption, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                first = model.get_positive_map(model.language_model.tokenizer([caption]), spans)[0]
                if mode == 'different':
                    tok, _, spans, _ = model.get_tokens_and_prompts(NAMES[::-1], True)
                    last = model.get_positive_map(tok, spans)[0]
                else:
                    last = first
                torch.testing.assert_close(losses['loss_second_order_etf'],
                    0.1 * second.second_order_etf_loss(final, [first, last], model.bbox_head.seen['text_token_mask']))

    def test_disabled_and_inference_never_build_auxiliary_mapping_or_solve(self):
        model = fixture(weight=0.)
        with patch.object(model, '_get_second_order_class_token_map', side_effect=AssertionError), \
                patch.dict(Detector.loss.__globals__, second_order_etf_loss=lambda *a: self.fail('ETF called')):
            losses, _ = run(model, samples())
            self.assertNotIn('loss_second_order_etf', losses)
            model.second_order_etf_loss_weight = 0.1
            model.eval()
            result = model.predict(torch.zeros(2, 3, 4, 4), samples())
            self.assertEqual(result[0].pred_instances.label_names, ['crazing'])

    def test_missing_classes_truncation_empty_spans_and_padding(self):
        model = fixture()
        tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        for invalid in [spans[:2], {0: spans[0], 5: spans[5]},
                        [[], *spans[1:]], -1]:
            with self.assertRaises(ValueError):
                model._get_second_order_class_token_map(tokenized, invalid)
        model.language_model.max_tokens = 5
        with self.assertRaisesRegex(ValueError, 'untruncated'):
            model._get_second_order_class_token_map(tokenized, spans)
        model.language_model.max_tokens = 64
        model.language_model.pad_to_max = True
        losses, _ = run(model, samples())
        self.assertTrue(torch.isfinite(losses['loss_second_order_etf']))

    def test_constructor_weight_validation(self):
        self.assertEqual(Detector(language_model={}).second_order_etf_loss_weight, 0.)
        for weight in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                Detector(language_model={}, second_order_etf_loss_weight=weight)

    def test_architecture_and_all_configs_preserve_base(self):
        def methods(text):
            cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef))
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(source(DETECTOR, True)), methods(source(DETECTOR))
        self.assertEqual(after.keys() - before.keys(), {'_get_second_order_class_token_map'})
        for name in before.keys() - {'__init__', 'loss'}:
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        for path in [HEAD, LAYERS, 'mmdet/models/detectors/grounding_dino.py',
                     'mmdet/datasets/coco.py', 'mmdet/engine/hooks/stage_lr_hook.py']:
            self.assertEqual(source(path), source(path, True), path)
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(configs), 18)
        for path in configs:
            rel = path.relative_to(ROOT).as_posix()
            new = source(rel)
            self.assertEqual(new.count('second_order_etf_loss_weight=0.1,'), 1)
            self.assertEqual(new.replace('    second_order_etf_loss_weight=0.1,\n', ''), source(rel, True))


if __name__ == '__main__':
    unittest.main()
