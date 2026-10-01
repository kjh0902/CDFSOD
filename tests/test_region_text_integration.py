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
from test_region_text_loss import ROOT, region, sample

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
           region_text_loss=region.region_text_loss)
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
    cls.bases = [ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()),
                               attr='Module', ctx=ast.Load())]
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
        super().__init__(attention_mask=torch.ones(1, len(self.spans) + 2,
                                                   dtype=torch.long))

    def char_to_token(self, char):
        return next((i + 1 for i, (a, b) in enumerate(self.spans) if a <= char < b), None)


class Language(nn.Module):
    pad_to_max = False
    max_tokens = 32

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def tokenizer(self, captions, **kwargs):
        return Tokens(captions[0])

    def forward(self, captions):
        lengths = [len(Tokens(c)['attention_mask'][0]) for c in captions]
        ids = torch.arange(max(lengths))[None].expand(len(captions), -1)
        mask = ids < torch.tensor(lengths)[:, None]
        return dict(embedded=self.embedding(ids), text_token_mask=mask,
                    position_ids=ids, masks=mask[:, :, None] & mask[:, None, :])


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
        self.cls_branches = nn.ModuleList([ENV['ContrastiveEmbed'](32) for _ in range(4)])
        self.reg_branches = nn.ModuleList([nn.Linear(8, 4) for _ in range(4)])
        self.inputs = None

    def loss(self, batch_data_samples, **inputs):
        # Record every actual detection-head input. Matching/loss implementations
        # are unchanged and therefore receive bitwise identical tensors.
        self.inputs = inputs
        scores, boxes = self.forward(inputs['hidden_states'], inputs['references'],
                                     inputs['memory_text'], inputs['text_token_mask'])
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


def build(baseline=False, weight=0):
    cls, env = detector_class(baseline)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.lambda_region_text = weight
    model.region_text_roi_size = 3
    model.region_text_featmap_strides = (2, 4)
    model.language_model = Language()
    model._special_tokens = '. '
    model.use_autocast = False
    model.text_feat_map = nn.Linear(8, 8)
    model.encoder = Encoder()
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


class IntegrationTest(unittest.TestCase):
    def test_zero_weight_bitwise_detection_regression(self):
        for mode in ('shared', 'different', 'explicit'):
            for empty in (False, True):
                with self.subTest(mode=mode, empty=empty):
                    torch.manual_seed(81)
                    base, _ = build(baseline=True)
                    current, env = build(weight=0)
                    current.load_state_dict(base.state_dict(), strict=True)
                    inputs = torch.randn(2, 20, 8)
                    initial_rng = torch.get_rng_state()
                    baseline_loss = base.loss(inputs, samples(mode, empty))
                    after_rng = torch.get_rng_state()
                    torch.set_rng_state(initial_rng)
                    with patch.dict(env, region_text_loss=lambda **kw: self.fail('Disabled loss ran')), \
                            patch.object(current, '_get_class_token_map', side_effect=AssertionError):
                        actual_loss = current.loss(inputs, samples(mode, empty))
                    assert_same(self, baseline_loss, actual_loss)
                    assert_same(self, base.bbox_head.inputs, current.bbox_head.inputs)
                    assert_same(self, base.bbox_head.outputs, current.bbox_head.outputs)
                    self.assertTrue(torch.equal(after_rng, torch.get_rng_state()))
                    self.assertEqual(current.dn_query_generator.calls, 1)

    def test_enabled_uses_exact_query_selection_memory_and_final_text(self):
        for mode in ('shared', 'different', 'explicit'):
            model, env = build(weight=.01)
            captured = {}
            real_proposals = model.gen_encoder_output_proposals
            def proposals(*args):
                outputs = real_proposals(*args)
                captured['output_memory'] = outputs[0]
                captured['raw_memory'] = args[0]
                return outputs
            model.gen_encoder_output_proposals = proposals
            def auxiliary(**kwargs):
                self.assertIs(kwargs['output_memory'], captured['output_memory'])
                self.assertIsNot(kwargs['output_memory'], captured['raw_memory'])
                self.assertIs(kwargs['memory_text'], model.bbox_head.inputs['memory_text'])
                self.assertEqual(kwargs['class_token_maps'][0], {1: [1, 2], 2: [4], 3: [6]})
                return region.region_text_loss(**kwargs)
            with patch.dict(env, region_text_loss=auxiliary):
                losses = model.loss(torch.randn(2, 20, 8), samples(mode))
            self.assertIn('loss_region_text', losses)
            losses['loss_region_text'].backward()
            for param in (model.encoder.visual.weight, model.encoder.text.weight,
                          model.memory_trans_fc.weight, model.memory_trans_norm.weight,
                          model.language_model.embedding.weight):
                self.assertIsNotNone(param.grad)
                self.assertGreater(param.grad.abs().sum(), 0)

    def test_enabled_detection_and_rng_are_unchanged(self):
        torch.manual_seed(2)
        base, _ = build(baseline=True)
        model, _ = build(weight=.01)
        model.load_state_dict(base.state_dict(), strict=True)
        inputs = torch.randn(2, 20, 8)
        rng = torch.get_rng_state()
        a = base.loss(inputs, samples())
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        b = model.loss(inputs, samples())
        b.pop('loss_region_text')
        assert_same(self, a, b)
        assert_same(self, base.bbox_head.outputs, model.bbox_head.outputs)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_inference_has_no_auxiliary_payload(self):
        model, _ = build(weight=1)
        model.eval()
        text = model.language_model(['red fox. blue. green. '] * 2)
        output = model.forward_transformer(torch.randn(2, 20, 8), text, samples())
        self.assertFalse(any(k.startswith('region_text') for k in output))
        self.assertEqual(model.dn_query_generator.calls, 0)

    def test_full_regression_runner_on_fixture(self):
        spec = importlib.util.spec_from_file_location(
            'full_regression', ROOT / 'tools/analysis_tools/check_region_text_baseline.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, _ = build(weight=.01)
        keys = module.check_zero_weight(model, torch.randn(2, 20, 8), samples())
        self.assertEqual(keys, ['loss_bbox', 'loss_cls', 'loss_encoder'])
        self.assertEqual(model.lambda_region_text, .01)

    def test_invalid_lambda_and_incomplete_prompts(self):
        cls, _ = detector_class()
        for value in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                cls(language_model={}, lambda_region_text=value)
        model, _ = build(weight=1)
        tokenized = Tokens('red fox. blue. green.')
        for spans in ([[[0, 7]]], {0: [[0, 7]]}, [[[0, 7]], [[9, 13]], [[100, 110]]]):
            with self.assertRaises(ValueError):
                model._get_class_token_map(tokenized, spans)
        model.language_model.max_tokens = 4
        with self.assertRaises(ValueError):
            model._get_class_token_map(tokenized, [[[0, 7]], [[9, 13]], [[15, 20]]])

    def test_serial_decoder_chains_queries(self):
        model, _ = build()
        model.eval()
        text = model.language_model(['red fox. blue. green. '] * 2)
        model.forward_transformer(torch.randn(2, 20, 8), text, samples())
        for previous, current in zip(model.decoder.layers, model.decoder.layers[1:]):
            torch.testing.assert_close(previous.outputs[0], current.inputs[0], rtol=0, atol=0)

    def test_progressive_ft_all_trainable_and_transition(self):
        for hook_name in ('BBoxHeadFirstHook6', 'StageWiseFreezeHook'):
            with self.subTest(hook=hook_name):
                hook_env = dict(nn=nn, is_model_wrapper=lambda _: False)
                hook = node(source('mmdet/engine/hooks/stage_lr_hook.py'), hook_name)
                hook.bases = []
                hook.decorator_list = []
                execute([hook], hook_env)
                model, _ = build(weight=.01)
                optimizer = torch.optim.SGD([
                    dict(params=[p], lr=.0002 if n.startswith('language_model')
                         else .001) for n, p in model.named_parameters()])
                original_lrs = [g['lr'] for g in optimizer.param_groups]
                runner = NS(model=model, optim_wrapper=NS(optimizer=optimizer), epoch=0,
                            logger=NS(info=lambda *args: None, warning=lambda *args: None),
                            param_schedulers=[NS(patience=5)])
                progressive = hook_name == 'BBoxHeadFirstHook6'
                policy = (hook_env[hook_name](True, 3, 8) if progressive
                          else hook_env[hook_name](stage1_epochs=2))
                policy.before_train(runner)
                self.assertEqual([g['lr'] for g in optimizer.param_groups], original_lrs)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertTrue(policy._other_groups)
                policy.before_train_epoch(runner)
                self.assertFalse(policy._stage2_started)
                if progressive:
                    self.assertEqual(runner.param_schedulers[0].patience, 3)
                # A real auxiliary backward/SGD step updates FE and projection in Stage 1.
                params = [model.encoder.visual.weight, model.encoder.text.weight,
                          model.memory_trans_fc.weight, model.memory_trans_norm.weight]
                before = [p.detach().clone() for p in params]
                model.loss(torch.randn(2, 20, 8), samples())['loss_region_text'].backward()
                optimizer.step()
                for old, param in zip(before, params):
                    self.assertGreater(param.grad.abs().sum(), 0)
                    self.assertFalse(torch.equal(old, param))
                # Every formerly frozen parameter can update, including those outside
                # the auxiliary path. No weight decay can mask a zero learning rate.
                optimizer.zero_grad()
                before = [p.detach().clone() for p in model.parameters()]
                sum(p.sum() for p in model.parameters()).backward()
                optimizer.step()
                for old, param in zip(before, model.parameters()):
                    self.assertFalse(torch.equal(old, param))
                # Mimic the existing validation-plateau LR reduction.
                for group in optimizer.param_groups:
                    group['lr'] *= .5
                runner.epoch = 2
                policy.before_train_epoch(runner)
                self.assertTrue(policy._stage2_started)
                if progressive:
                    self.assertEqual(runner.param_schedulers[0].patience, 8)
                for i in policy._other_groups:
                    self.assertEqual(optimizer.param_groups[i]['lr'], .0005)
                for i in policy._lang_model_groups:
                    self.assertEqual(optimizer.param_groups[i]['lr'], .0001)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                # Stage 2 setup is one-shot; later scheduler changes survive.
                for group in optimizer.param_groups:
                    group['lr'] *= .5
                lrs = [g['lr'] for g in optimizer.param_groups]
                policy.before_train_epoch(runner)
                self.assertEqual([g['lr'] for g in optimizer.param_groups], lrs)


class ScopeTest(unittest.TestCase):
    def test_original_serial_layers_and_detection_head_unchanged(self):
        for name in ('GroundingDinoTransformerEncoder', 'GroundingDinoTransformerDecoder',
                     'GroundingDinoTransformerDecoderLayer'):
            self.assertEqual(ast.dump(node(source(LAYERS), name)),
                             ast.dump(node(source(LAYERS, True), name)))
        self.assertEqual(source(HEAD), source(HEAD, True))
        self.assertNotIn('parallel', source(LAYERS).lower())
        self.assertFalse(list((ROOT / 'mmdet/models').rglob('*HED*')))

    def test_all_configs_only_change_model_experiment_fields(self):
        paths = sorted((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            before = ast.parse(source(rel, True))
            after = ast.parse(source(rel))
            def without_model(tree):
                tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))]
                return ast.dump(tree)
            self.assertEqual(without_model(before), without_model(after), rel)
            self.assertIn('lambda_region_text=0.01', source(rel))
            self.assertIn('region_text_roi_size=3', source(rel))
            self.assertNotIn('ParallelDecoder', source(rel))



if __name__ == '__main__':
    unittest.main()
