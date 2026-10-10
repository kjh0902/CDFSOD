"""ACL regression tests with production methods and lightweight dependencies.

No downloaded weights or MMCV extensions: AST loading replaces only the heavy
base class/imports; encoder/serial ACL decoder loops, prompt mapping, detector and detection head
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

from test_raw_mean_etf_loss import ROOT, raw_mean

BASE = '8bd52df24303a470287d552c85e242ad46efaf2a'
PREVIOUS = '0a2dcb061bb8aed1c6d3b579b539d30c95cf2957'
DETECTOR = 'mmdet/models/detectors/grounding_dino_HED.py'
HEAD = 'mmdet/models/dense_heads/grounding_dino_head_HED.py'
LAYERS = 'mmdet/models/layers/transformer/grounding_dino_layers_HED.py'
NAMES = ('crazing', 'inclusion', 'patches', 'pitted_surface',
         'rolled-in_scale', 'scratches')
FEATURE_DIM = 8  # Must be >= num_classes - 1 for raw mean ETF.


def source(path, base=False):
    if base:
        return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode()
    return (ROOT / path).read_text(encoding='utf-8')


def execute(nodes, **extra):
    env = dict(torch=torch, nn=nn, math=math, copy=copy, re=re,
               random=random, warnings=warnings,
               raw_mean_etf_loss=raw_mean.raw_mean_etf_loss,
               raw_mean_geometry_losses=raw_mean.raw_mean_geometry_losses)
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
        self.embedding = nn.Embedding(128, FEATURE_DIM)
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


class Projection(nn.Linear):
    def forward(self, x):
        self.output = super().forward(x)
        return self.output


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, FEATURE_DIM)

    def forward(self, visual_feature, lang_feature, **kw):
        return (visual_feature + lang_feature.mean(1, keepdim=True) * 0.05,
                lang_feature + self.projection(visual_feature.mean(1, keepdim=True)) * 0.1)


class TextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_cfg = SimpleNamespace(num_heads=1)
        self.attn = nn.MultiheadAttention(FEATURE_DIM, 1, batch_first=True)

    def forward(self, query, query_pos, attn_mask, **kw):
        self.output = query + self.attn(query + query_pos, query + query_pos,
                                        query, attn_mask=attn_mask)[0] * 0.1
        return self.output


class VisualLayer(nn.Module):
    def forward(self, query, **kw):
        return query


class Encoder(nn.Module):
    forward = method(LAYERS, 'GroundingDinoTransformerEncoder', 'forward',
                     get_text_sine_pos_embed=lambda x, **kw: x.expand(-1, -1, FEATURE_DIM).float() * 0.01)
    get_encoder_reference_points = staticmethod(lambda *a, **kw: None)

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([VisualLayer() for _ in range(6)])
        self.fusion_layers = nn.ModuleList([Fusion() for _ in range(6)])
        self.text_layers = nn.ModuleList([TextLayer() for _ in range(6)])


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, FEATURE_DIM)

    def forward(self, query, value, memory_text, **kw):
        self.input_query = query
        self.input_references = kw['reference_points']
        return (query + self.projection(query).tanh() * 0.1
                + value.mean(1, keepdim=True) * 0.05
                + memory_text.mean(1, keepdim=True) * 0.05)


class Decoder(nn.Module):
    num_layers = 6
    forward_impl = method('mmdet/models/layers/transformer/dino_layers.py',
                          'DinoTransformerDecoder', 'forward',
                          coordinate_to_encoding=lambda x: x,
                          inverse_sigmoid=lambda x, eps: torch.logit(x.clamp(eps, 1 - eps)))

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(6)])
        self.ref_point_head = nn.Linear(4, FEATURE_DIM)
        self.norm = nn.Identity()
        self.return_intermediate = True

    def forward(self, query, reference_points, **kw):
        self.seen = dict(query=query, reference_points=reference_points, **kw)
        return self.forward_impl(query=query, reference_points=reference_points, **kw)


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
        self.reg_branches = nn.ModuleList([nn.Linear(FEATURE_DIM, 4) for _ in range(7)])

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


def fixture(weight=1.0, detector_class=Detector):
    torch.manual_seed(29)
    model = detector_class(language_model={})
    model.bert_etf_loss_weight = weight
    model.language_model = Language()
    model.text_feat_map = Projection(FEATURE_DIM, FEATURE_DIM)
    model.encoder = Encoder()
    model.decoder = Decoder()
    model.bbox_head = Head()
    model.query_embedding = nn.Embedding(3, FEATURE_DIM)
    model.num_queries = 3
    model.test_cfg = {}
    features = torch.randn(2, 5, FEATURE_DIM)
    model.backbone = nn.Linear(FEATURE_DIM, FEATURE_DIM)
    model.extract_feat = lambda _: (model.backbone(features),)
    model.pre_transformer = lambda feats, *a: (dict(
        feat=feats[0], feat_mask=torch.zeros(2, 5, dtype=torch.bool),
        feat_pos=torch.zeros_like(features), spatial_shapes=torch.tensor([[1, 5]]),
        level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2)),
        dict(memory_mask=torch.zeros(2, 5, dtype=torch.bool),
             spatial_shapes=torch.tensor([[1, 5]]), level_start_index=torch.tensor([0]),
             valid_ratios=torch.ones(2, 1, 2)))
    model.gen_encoder_output_proposals = lambda memory, *a: (memory, memory.new_zeros(*memory.shape[:2], 4))
    model.dn_query_generator = lambda _: (torch.randn(2, 1, FEATURE_DIM), torch.randn(2, 1, 4),
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


class RawMeanIntegrationTests(unittest.TestCase):
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

    def test_serial_detection_matches_acl_reference_with_loss_enabled_or_disabled(self):
        base = fixture(weight=0., detector_class=load_detector(base=True))
        _, expected = run(base, samples())
        for weight in [0., 0.1, 1., 2.5]:
            model = fixture(weight=weight)
            losses, actual = run(model, samples())
            self.assertEqual(base.state_dict().keys(), model.state_dict().keys())
            if weight:
                actual['losses'] = dict(losses)
                auxiliary = actual['losses'].pop('loss_bert_etf')
                tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                mapping = model._get_raw_mean_class_token_map(tokenized, spans)
                final = model.encoder.text_layers[-1].output
                torch.testing.assert_close(auxiliary, weight * raw_mean.raw_mean_etf_loss(
                    model.text_feat_map.output, [mapping] * 2, model.bbox_head.seen['text_token_mask']))
                for value in [model.decoder.seen['memory_text'], model.bbox_head.seen['memory_text'],
                              *[branch.seen for branch in model.bbox_head.cls_branches]]:
                    self.assertIs(value, final)
            self.assert_nested_equal(actual, expected)

    def test_nearest_etf_receives_raw_means_before_enhancer(self):
        model = fixture()
        def check_before_enhancer(*args, **kwargs):
            self.assertFalse(hasattr(model.encoder.text_layers[-1], 'output'))
            return original_nearest(*args, **kwargs)
        original_nearest = raw_mean.nearest_etf_loss
        with patch.object(raw_mean, 'nearest_etf_loss',
                          side_effect=check_before_enhancer) as nearest:
            losses, _ = run(model, samples())
        nearest.assert_called_once()
        tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        mapping = model._get_raw_mean_class_token_map(tokenized, spans)
        final = model.encoder.text_layers[-1].output
        expected = torch.stack([
            torch.stack([row[mapping[c]].mean(0) for c in range(1, 7)])
            for row in model.text_feat_map.output])
        actual = nearest.call_args.args[0]
        self.assertEqual(actual.shape, (2, 6, FEATURE_DIM))
        torch.testing.assert_close(actual, expected)
        final_means = torch.stack([
            torch.stack([row[mapping[c]].mean(0) for c in range(1, 7)])
            for row in final])
        self.assertFalse(torch.allclose(actual, final_means))
        torch.testing.assert_close(losses['loss_bert_etf'],
                                   model.bert_etf_loss_weight * raw_mean.nearest_etf_loss(expected))

    def test_auxiliary_gradient_reaches_only_projected_bert_tokens(self):
        model = fixture()
        losses, _ = run(model, samples())
        final = model.encoder.text_layers[-1].output
        final.retain_grad()
        projected = model.text_feat_map.output
        projected.retain_grad()
        losses['loss_bert_etf'].backward()
        tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        mapping = model._get_raw_mean_class_token_map(tokenized, spans)
        self.assertEqual(len(mapping), 6)
        for indices in mapping.values():
            self.assertTrue((projected.grad[:, indices].norm(dim=-1) > 0).all())
        selected = {i for indices in mapping.values() for i in indices}
        unselected = [i for i in range(projected.size(1)) if i not in selected]
        torch.testing.assert_close(projected.grad[:, unselected],
                                   torch.zeros_like(projected.grad[:, unselected]))
        for p in [model.text_feat_map.weight, model.language_model.embedding.weight]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        self.assertIsNone(model.query_embedding.weight.grad)
        self.assertIsNone(final.grad)
        for module in [model.encoder, model.backbone, model.decoder, model.bbox_head]:
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_detection_gradients_unchanged_by_auxiliary_branch(self):
        gradients = []
        for detector_class, weight in [(load_detector(base=True), 0.),
                                       (Detector, 0.), (Detector, 1.)]:
            model = fixture(weight=weight, detector_class=detector_class)
            losses, _ = run(model, samples())
            (losses['loss_cls'] + losses['loss_bbox']).backward()
            gradients.append({name: p.grad for name, p in model.named_parameters()})
            for module in [model.encoder, model.backbone, model.text_feat_map,
                           model.language_model, model.bbox_head, *model.decoder.layers]:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in module.parameters()))
        self.assert_nested_equal(gradients[0], gradients[1])
        self.assert_nested_equal(gradients[0], gradients[2])

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
                with patch.object(model, '_get_raw_mean_class_token_map',
                                  wraps=model._get_raw_mean_class_token_map) as mapping:
                    losses, _ = run(model, data)
                self.assertEqual(mapping.call_count, 1 if mode == 'shared' else 2)
                self.assertTrue(torch.isfinite(losses['loss_bert_etf']))
                projected = model.text_feat_map.output
                _, caption, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                first = model.get_positive_map(model.language_model.tokenizer([caption]), spans)[0]
                if mode == 'different':
                    tok, _, spans, _ = model.get_tokens_and_prompts(NAMES[::-1], True)
                    last = model.get_positive_map(tok, spans)[0]
                else:
                    last = first
                torch.testing.assert_close(losses['loss_bert_etf'],
                    model.bert_etf_loss_weight * raw_mean.raw_mean_etf_loss(
                        projected, [first, last], model.bbox_head.seen['text_token_mask']))

    def test_disabled_and_inference_never_build_auxiliary_mapping_or_solve(self):
        model = fixture(weight=0.)
        with patch.object(model, '_get_raw_mean_class_token_map', side_effect=AssertionError), \
                patch.dict(Detector.loss.__globals__, raw_mean_geometry_losses=lambda *a, **kw: self.fail('ETF called')):
            losses, _ = run(model, samples())
            self.assertNotIn('loss_bert_etf', losses)
            model.bert_etf_loss_weight = 0.1
            model.eval()
            result = model.predict(torch.zeros(2, 3, 4, 4), samples())
            self.assertEqual(result[0].pred_instances.label_names, ['crazing'])

    def test_missing_classes_truncation_empty_spans_and_padding(self):
        model = fixture()
        tokenized, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        for invalid in [spans[:2], {0: spans[0], 5: spans[5]},
                        [[], *spans[1:]], -1]:
            with self.assertRaises(ValueError):
                model._get_raw_mean_class_token_map(tokenized, invalid)
        model.language_model.max_tokens = 5
        with self.assertRaisesRegex(ValueError, 'untruncated'):
            model._get_raw_mean_class_token_map(tokenized, spans)
        model.language_model.max_tokens = 64
        model.language_model.pad_to_max = True
        losses, _ = run(model, samples())
        self.assertTrue(torch.isfinite(losses['loss_bert_etf']))

    def test_constructor_weight_validation(self):
        model = Detector(language_model={})
        self.assertEqual(model.bert_etf_loss_weight, 1.)
        self.assertFalse(hasattr(model, "rand_dnquery_rate"))
        for legacy_rate in [0., .5, 1.]:
            compatible = Detector(language_model={}, rand_dnquery_rate=legacy_rate)
            self.assertFalse(hasattr(compatible, "rand_dnquery_rate"))
        for name, default in [('bert_etf_loss_weight', 1.),
                              ('bert_orth_loss_weight', 0.),
                              ('fe_etf_loss_weight', 0.), ('fe_orth_loss_weight', 0.)]:
            self.assertEqual(getattr(model, name), default)
            for weight in [-1, float('nan'), float('inf')]:
                with self.assertRaisesRegex(ValueError, name):
                    Detector(language_model={}, **{name: weight})

    def test_experiments_a_b_and_combined_preserve_detection_and_mapping(self):
        _, expected = run(fixture(weight=0., detector_class=load_detector(base=True)), samples())
        for weights in [(1., .3, 0., 0.), (0., 0., .7, .4), (1., .3, .7, .4),
                        (0., .3, 0., 0.), (0., 0., 0., .4)]:
            model = fixture(weight=weights[0])
            model.bert_orth_loss_weight, model.fe_etf_loss_weight, model.fe_orth_loss_weight = weights[1:]
            losses, actual = run(model, samples())
            tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
            mapping = model._get_raw_mean_class_token_map(tok, spans)
            mask = model.bbox_head.seen['text_token_mask']
            actual['losses'] = dict(losses)
            for location, features, etf_weight, orth_weight in [
                    ('bert', model.text_feat_map.output, *weights[:2]),
                    ('fe', model.encoder.text_layers[-1].output, *weights[2:])]:
                geometry = raw_mean.raw_mean_geometry_losses(features, [mapping] * 2, mask,
                                      etf_weight=etf_weight, orth_weight=orth_weight)
                for name, value in geometry.items():
                    torch.testing.assert_close(actual['losses'].pop(f'loss_{location}_{name}'), value)
                self.assertIs(model.bbox_head.seen['memory_text'], model.encoder.text_layers[-1].output)
            self.assert_nested_equal(actual, expected)

    def test_both_losses_share_prototypes_at_the_requested_locations(self):
        model = fixture()
        model.bert_orth_loss_weight = .5
        model.fe_etf_loss_weight = 1.
        model.fe_orth_loss_weight = .5
        with patch.object(raw_mean, '_raw_mean_prototypes', wraps=raw_mean._raw_mean_prototypes) as pool, \
             patch.object(raw_mean, 'nearest_etf_loss', wraps=raw_mean.nearest_etf_loss) as etf, \
             patch.object(raw_mean, 'mu_orthogonality_loss', wraps=raw_mean.mu_orthogonality_loss) as orth:
            run(model, samples())
        self.assertEqual(pool.call_count, 2)
        self.assertIs(pool.call_args_list[0].args[0], model.text_feat_map.output)
        self.assertIs(pool.call_args_list[1].args[0], model.encoder.text_layers[-1].output)
        for i in range(2):
            self.assertIs(etf.call_args_list[i].args[0], orth.call_args_list[i].args[0])
            self.assertEqual(etf.call_args_list[i].args[0].shape, (2, 6, FEATURE_DIM))

    def test_fe_gradients_reach_enhancer_bert_and_backbone_without_decoder(self):
        for name in ['loss_fe_etf', 'loss_fe_orth']:
            model = fixture(weight=0.)
            model.fe_etf_loss_weight = .7
            model.fe_orth_loss_weight = .4
            losses, _ = run(model, samples())
            losses[name].backward()
            for module in [model.encoder, model.backbone, model.language_model, model.text_feat_map]:
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)
            for module in [model.decoder, model.bbox_head, model.query_embedding]:
                self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_bert_orth_gradients_reach_all_class_tokens_including_empty_gt(self):
        model = fixture(weight=0.)
        model.bert_orth_loss_weight = 1.
        losses, _ = run(model, samples())
        projected = model.text_feat_map.output
        projected.retain_grad()
        losses['loss_bert_orth'].backward()
        tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        mapping = model._get_raw_mean_class_token_map(tok, spans)
        for indices in mapping.values():
            self.assertTrue((projected.grad[:, indices].norm(dim=-1) > 0).all())
        for module in [model.encoder, model.backbone, model.decoder, model.bbox_head]:
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_fe_loss_keeps_stage1_zero_lr_policy_and_upstream_gradients(self):
        model = fixture(weight=0.)
        model.fe_etf_loss_weight = model.fe_orth_loss_weight = 1.
        optimizer = torch.optim.SGD([
            dict(params=list(model.encoder.parameters()), lr=0.),
            dict(params=list(model.text_feat_map.parameters()), lr=.01),
            dict(params=list(model.language_model.parameters()), lr=.01),
            dict(params=list(model.backbone.parameters()), lr=.01)])
        before = copy.deepcopy(model.encoder.state_dict())
        projection_before = model.text_feat_map.weight.detach().clone()
        losses, _ = run(model, samples())
        (losses['loss_fe_etf'] + losses['loss_fe_orth']).backward()
        optimizer.step()
        self.assert_nested_equal(before, model.encoder.state_dict())
        self.assertFalse(torch.equal(projection_before, model.text_feat_map.weight))

    def test_all_losses_skip_inference_even_with_positive_weights(self):
        model = fixture()
        model.bert_orth_loss_weight = model.fe_etf_loss_weight = model.fe_orth_loss_weight = 1.
        model.eval()
        with patch.dict(Detector.predict.__globals__,
                        raw_mean_geometry_losses=lambda *a, **kw: self.fail('geometry called')):
            result = model.predict(torch.zeros(2, 3, 4, 4), samples())
        self.assertEqual(result[0].pred_instances.label_names, ['crazing'])

    def test_single_class_fish_all_four_losses_are_finite_zero_without_svd(self):
        model = fixture()
        model.bert_orth_loss_weight = model.fe_etf_loss_weight = model.fe_orth_loss_weight = 1.
        model.bbox_head.num_classes = 1
        data = [Sample(text=('fish',), gt_instances=Sample(
            labels=torch.tensor(labels, dtype=torch.long))) for labels in [[0], []]]
        with patch.object(raw_mean, 'nearest_etf_loss', side_effect=AssertionError('SVD called')):
            losses, _ = run(model, data)
        auxiliary = [losses[f'loss_{location}_{name}']
                     for location in ['bert', 'fe'] for name in ['etf', 'orth']]
        self.assertTrue(all(value.item() == 0. for value in auxiliary))
        sum(auxiliary).backward()
        for module in [model.text_feat_map, model.language_model, model.encoder, model.backbone]:
            for p in module.parameters():
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_single_class_fish_keeps_detection_and_returns_zero_auxiliary_loss(self):
        snapshots = []
        for detector_class in [load_detector(base=True), Detector]:
            model = fixture(detector_class=detector_class)
            model.bbox_head.num_classes = 1
            data = [Sample(text=('fish',), gt_instances=Sample(
                labels=torch.tensor(labels, dtype=torch.long))) for labels in [[0], []]]
            losses, snapshot = run(model, data)
            snapshot['losses'] = dict(losses)
            if detector_class is Detector:
                auxiliary = snapshot['losses'].pop('loss_bert_etf')
                self.assertEqual(auxiliary.item(), 0.)
                auxiliary.backward()
                torch.testing.assert_close(model.text_feat_map.weight.grad,
                                           torch.zeros_like(model.text_feat_map.weight))
            snapshots.append(snapshot)
        self.assert_nested_equal(snapshots[0], snapshots[1])

    def test_architecture_matches_acl_reference_and_preserves_existing_configs(self):
        def methods(text):
            cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef))
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(source(DETECTOR, True)), methods(source(DETECTOR))
        self.assertEqual(after.keys(), before.keys())
        for name in before.keys() - {'__init__', 'loss', '_get_raw_mean_class_token_map'}:
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        # The reference head and retired HED decoder are used verbatim.
        for path in [HEAD, LAYERS]:
            self.assertEqual(source(path), source(path, True), path)
        paths = ['mmdet/models/layers/transformer/dino_layers.py',
                 'mmdet/models/layers/transformer/grounding_dino_layers.py',
                 'mmdet/models/detectors/grounding_dino.py',
                 'mmdet/datasets/coco.py', 'mmdet/engine/hooks/stage_lr_hook.py',
                 'tools/train.py', 'tools/test.py', 'run_cdfsod.sh',
                 'tools/dist_train.sh', 'tools/dist_test.sh', 'src_path.py',
                 'environment.yml', 'requirements.txt', 'scripts/install_rtx3090.sh',
                 'tools/verify_environment.py',
                 'mmdet/models/losses/nearest_etf_loss.py',
                 'mmdet/models/losses/raw_mean_etf_loss.py',
                 'mmdet/models/losses/mu_orthogonality_loss.py']
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(configs), 18)
        paths += [p.relative_to(ROOT).as_posix()
                  for p in (ROOT / 'configs_cdfsod').rglob('*.py')]
        for path in paths:
            previous = subprocess.check_output(['git', 'show', f'{PREVIOUS}:{path}'],
                                               cwd=ROOT).decode('utf-8')
            self.assertEqual(source(path), previous, path)

    def test_decoder_refines_all_queries_and_references_sequentially_with_one_dn_batch(self):
        model = fixture()
        with patch.object(model, 'dn_query_generator', wraps=model.dn_query_generator) as dn:
            _, snapshot = run(model, samples())
            dn.assert_called_once()
        self.assertNotIn('additional_dn_items', snapshot['decoder'])
        self.assertEqual(snapshot['head']['dn_meta']['num_denoising_queries'], 1)
        layers = model.decoder.layers
        for i, layer in enumerate(layers):
            if i:
                torch.testing.assert_close(layer.input_query,
                                           snapshot['head']['hidden_states'][i - 1])
                self.assertFalse(torch.allclose(layer.input_query, layers[i - 1].input_query))
            torch.testing.assert_close(layer.input_references[:, :, 0],
                                       snapshot['head']['references'][i].detach())
        model.eval()
        with patch.object(model, 'dn_query_generator', side_effect=AssertionError('DN called in inference')):
            model.predict(torch.zeros(2, 3, 4, 4), samples())
        self.assertNotIn('additional_dn_items', model.decoder.seen)

    def test_progressive_fine_tuning_group_lrs_and_stage_transition_match_base(self):
        node = next(n for n in ast.parse(source('mmdet/engine/hooks/stage_lr_hook.py')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'BBoxHeadFirstHook6')
        node.decorator_list = []
        Hook = execute([node], Hook=object,
                       is_model_wrapper=lambda model: False)['BBoxHeadFirstHook6']
        stages = []
        for detector_class in [load_detector(base=True), Detector]:
            model = fixture(detector_class=detector_class)
            named = list(model.named_parameters())
            optimizer = torch.optim.SGD([
                dict(params=[p], lr=0.02 if name.startswith(('language_model', 'backbone')) else 0.1)
                for name, p in named])
            scheduler = SimpleNamespace(patience=5)
            runner = SimpleNamespace(model=model, epoch=0, scheduler=scheduler,
                optim_wrapper=SimpleNamespace(optimizer=optimizer),
                logger=SimpleNamespace(info=lambda message: None, warning=lambda message: None))
            hook = Hook(adjust_scheduler_patience=True, patience_frozen=3, patience_unfrozen=8)
            hook.before_train(runner)
            stage1 = {name: group['lr'] for (name, _), group in zip(named, optimizer.param_groups)}
            self.assertEqual(scheduler.patience, 3)
            self.assertFalse(hook._should_start_stage2(runner))
            self.assertEqual(stage1['encoder.fusion_layers.0.projection.weight'], 0.)
            self.assertEqual(stage1['language_model.embedding.weight'], 0.02)
            self.assertEqual(stage1['text_feat_map.weight'], 0.1)
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
            hook.before_train_epoch(runner)
            self.assertTrue(hook._stage2_started)
            self.assertEqual(scheduler.patience, 8)
            stage2 = {name: group['lr'] for (name, _), group in zip(named, optimizer.param_groups)}
            self.assertEqual(stage2['encoder.fusion_layers.0.projection.weight'], 0.05)
            self.assertEqual(stage2['language_model.embedding.weight'], 0.01)
            stages.append((stage1, stage2))
        self.assertEqual(stages[0], stages[1])


if __name__ == '__main__':
    unittest.main()
