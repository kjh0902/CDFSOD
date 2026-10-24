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
from test_batch_shared_mu_orth_token_etf_loss import unified

BASE = 'd560d786363a7f5a699bfe6b5738f27f6b5062ac'
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
               batch_shared_mu_orth_token_etf_loss=unified.batch_shared_mu_orth_token_etf_loss,
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
    if hasattr(model, 'unified_etf_loss_weight'):
        model.unified_etf_loss_weight = weight
    else:
        for name in ('bert_etf_loss_weight', 'bert_orth_loss_weight',
                     'fe_etf_loss_weight', 'fe_orth_loss_weight'):
            setattr(model, name, 0.)
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


class UnifiedETFIntegrationTests(unittest.TestCase):
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

    def test_detection_and_state_dict_match_starting_branch(self):
        base = fixture(weight=0., detector_class=load_detector(base=True))
        _, expected = run(base, samples())
        for weight in [0., .1, 10., 2.5]:
            model = fixture(weight=weight)
            losses, actual = run(model, samples())
            self.assertEqual(base.state_dict().keys(), model.state_dict().keys())
            actual['losses'] = dict(losses)
            if weight:
                auxiliary = actual['losses'].pop('loss_unified_etf')
                tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                mapping = model._get_raw_mean_class_token_map(tok, spans)
                final = model.encoder.text_layers[-1].output
                torch.testing.assert_close(auxiliary, weight * unified.batch_shared_mu_orth_token_etf_loss(
                    final, [mapping] * 2, model.bbox_head.seen['text_token_mask']))
                for value in [model.decoder.seen['memory_text'], model.bbox_head.seen['memory_text'],
                              *[branch.seen for branch in model.bbox_head.cls_branches]]:
                    self.assertIs(value, final)
            self.assert_nested_equal(actual, expected)

    def test_one_solver_call_receives_final_fe_not_projected_bert(self):
        model = fixture()
        calls = []
        def capture(features, *args):
            self.assertIs(features, model.encoder.text_layers[-1].output)
            self.assertIsNot(features, model.text_feat_map.output)
            calls.append(features)
            return unified.batch_shared_mu_orth_token_etf_loss(features, *args)
        with patch.dict(Detector.loss.__globals__, batch_shared_mu_orth_token_etf_loss=capture), \
             patch.object(unified, 'nearest_mu_orthogonal_simplex_etf',
                          wraps=unified.nearest_mu_orthogonal_simplex_etf) as solve:
            losses, _ = run(model, samples())
        self.assertEqual(len(calls), 1)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.args[0].shape, (6, FEATURE_DIM))
        self.assertEqual(set(losses), {'loss_cls', 'loss_bbox', 'loss_unified_etf'})

    def test_all_class_token_gradients_and_upstream_gradients_including_empty_gt(self):
        model = fixture()
        losses, _ = run(model, samples())
        final = model.encoder.text_layers[-1].output
        final.retain_grad()
        losses['loss_unified_etf'].backward()
        tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        mapping = model._get_raw_mean_class_token_map(tok, spans)
        for indices in mapping.values():
            self.assertTrue((final.grad[:, indices].norm(dim=-1) > 0).all())
        selected = {i for indices in mapping.values() for i in indices}
        others = [i for i in range(final.size(1)) if i not in selected]
        torch.testing.assert_close(final.grad[:, others], torch.zeros_like(final.grad[:, others]))
        for module in [model.encoder, model.backbone, model.language_model, model.text_feat_map]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        for module in [model.decoder, model.bbox_head, model.query_embedding]:
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_detection_gradients_unchanged(self):
        gradients = []
        for cls, weight in [(load_detector(base=True), 0.), (Detector, 0.), (Detector, 10.)]:
            model = fixture(weight=weight, detector_class=cls)
            losses, _ = run(model, samples())
            (losses['loss_cls'] + losses['loss_bbox']).backward()
            gradients.append({name: p.grad for name, p in model.named_parameters()})
        self.assert_nested_equal(gradients[0], gradients[1])
        self.assert_nested_equal(gradients[0], gradients[2])

    def test_shared_different_and_explicit_prompt_mapping(self):
        for mode in ['shared', 'different', 'explicit_list', 'explicit_dict']:
            with self.subTest(mode=mode):
                model = fixture()
                data = samples()
                second_names = ('Crazing', *NAMES[1:])
                if mode == 'different':
                    data[1].text = second_names
                if mode.startswith('explicit'):
                    _, caption, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                    for sample in data:
                        sample.text = caption
                        sample.tokens_positive = dict(enumerate(spans)) if mode.endswith('dict') else spans
                with patch.object(model, '_get_raw_mean_class_token_map',
                                  wraps=model._get_raw_mean_class_token_map) as mapping:
                    losses, _ = run(model, data)
                self.assertEqual(mapping.call_count, 1 if mode == 'shared' else 2)
                tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                first = model._get_raw_mean_class_token_map(tok, spans)
                if mode == 'different':
                    tok, _, spans, _ = model.get_tokens_and_prompts(second_names, True)
                    last = model._get_raw_mean_class_token_map(tok, spans)
                else:
                    last = first
                torch.testing.assert_close(losses['loss_unified_etf'],
                    unified.batch_shared_mu_orth_token_etf_loss(model.encoder.text_layers[-1].output,
                        [first, last], model.bbox_head.seen['text_token_mask']))

    def test_disabled_and_inference_skip_mapping_and_solver(self):
        model = fixture(weight=0.)
        with patch.object(model, '_get_raw_mean_class_token_map', side_effect=AssertionError), \
             patch.dict(Detector.loss.__globals__, batch_shared_mu_orth_token_etf_loss=
                        lambda *a, **kw: self.fail('ETF called')):
            losses, _ = run(model, samples())
            self.assertNotIn('loss_unified_etf', losses)
            model.unified_etf_loss_weight = 10.
            model.eval()
            result = model.predict(torch.zeros(2, 3, 4, 4), samples())
            self.assertEqual(result[0].pred_instances.label_names, ['crazing'])

    def test_constructor_defaults_validation_and_retired_options(self):
        model = Detector(language_model={})
        self.assertEqual(model.unified_etf_loss_weight, 10.)
        for weight in [-1., float('nan'), float('inf')]:
            with self.assertRaisesRegex(ValueError, 'unified_etf_loss_weight'):
                Detector(language_model={}, unified_etf_loss_weight=weight)
        self.assertEqual(Detector(language_model={}, unified_etf_loss_weight=0).unified_etf_loss_weight, 0.)
        for old in ['bert_etf_loss_weight', 'bert_orth_loss_weight', 'fe_etf_loss_weight',
                    'fe_orth_loss_weight', 'raw_mean_etf_loss_weight']:
            self.assertFalse(hasattr(model, old))
            with self.assertRaisesRegex(ValueError, 'Replace retired'):
                Detector(language_model={}, **{old: 0.})
        for rate in [0., .5, 1.]:
            self.assertFalse(hasattr(Detector(language_model={}, rand_dnquery_rate=rate), 'rand_dnquery_rate'))

    def test_missing_classes_truncation_and_padding(self):
        model = fixture()
        tok, _, spans, _ = model.get_tokens_and_prompts(NAMES, True)
        for invalid in [spans[:2], {0: spans[0], 5: spans[5]}, [[], *spans[1:]], -1]:
            with self.assertRaises(ValueError):
                model._get_raw_mean_class_token_map(tok, invalid)
        model.language_model.max_tokens = 5
        with self.assertRaisesRegex(ValueError, 'untruncated'):
            model._get_raw_mean_class_token_map(tok, spans)
        model.language_model.max_tokens = 64
        model.language_model.pad_to_max = True
        losses, _ = run(model, samples())
        self.assertTrue(torch.isfinite(losses['loss_unified_etf']))

    def test_stage_one_zero_lr_preserves_fe_but_allows_upstream_training(self):
        model = fixture()
        optimizer = torch.optim.SGD([
            dict(params=list(model.encoder.parameters()), lr=0.),
            dict(params=list(model.text_feat_map.parameters()), lr=.01),
            dict(params=list(model.language_model.parameters()), lr=.01),
            dict(params=list(model.backbone.parameters()), lr=.01)])
        before = copy.deepcopy(model.encoder.state_dict())
        projection_before = model.text_feat_map.weight.detach().clone()
        losses, _ = run(model, samples())
        losses['loss_unified_etf'].backward()
        optimizer.step()
        self.assert_nested_equal(before, model.encoder.state_dict())
        self.assertFalse(torch.equal(projection_before, model.text_feat_map.weight))

    def test_single_class_zero_preserves_detection_without_solver(self):
        snapshots = []
        for cls in [load_detector(base=True), Detector]:
            model = fixture(detector_class=cls)
            model.bbox_head.num_classes = 1
            data = [Sample(text=('fish',), gt_instances=Sample(labels=torch.tensor(labels, dtype=torch.long)))
                    for labels in [[0], []]]
            with patch.object(unified, 'nearest_mu_orthogonal_simplex_etf', side_effect=AssertionError):
                losses, snapshot = run(model, data)
            snapshot['losses'] = dict(losses)
            if cls is Detector:
                auxiliary = snapshot['losses'].pop('loss_unified_etf')
                self.assertEqual(auxiliary.item(), 0.)
                auxiliary.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))
            snapshots.append(snapshot)
        self.assert_nested_equal(snapshots[0], snapshots[1])

    def test_architecture_and_all_non_loss_settings_preserved(self):
        def methods(text):
            cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef))
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(source(DETECTOR, True)), methods(source(DETECTOR))
        self.assertEqual(before.keys(), after.keys())
        for name in before.keys() - {'__init__', 'loss', '_get_raw_mean_class_token_map'}:
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        unchanged = [HEAD, LAYERS, 'mmdet/models/layers/transformer/dino_layers.py',
                     'mmdet/models/layers/transformer/grounding_dino_layers.py',
                     'mmdet/models/detectors/grounding_dino.py', 'mmdet/datasets/coco.py',
                     'mmdet/engine/hooks/stage_lr_hook.py', 'tools/train.py', 'tools/test.py',
                     'tools/dist_train.sh', 'tools/dist_test.sh', 'src_path.py', 'environment.yml',
                     'requirements.txt', 'scripts/install_rtx3090.sh', 'tools/verify_environment.py',
                     'mmdet/models/losses/nearest_etf_loss.py',
                     'mmdet/models/losses/raw_mean_etf_loss.py',
                     'mmdet/models/losses/mu_orthogonality_loss.py']
        for path in unchanged:
            self.assertEqual(source(path), source(path, True), path)
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(configs), 18)
        retired = {'bert_etf_loss_weight', 'bert_orth_loss_weight',
                   'fe_etf_loss_weight', 'fe_orth_loss_weight'}
        for path in (ROOT / 'configs_cdfsod').rglob('*.py'):
            relative = path.relative_to(ROOT).as_posix()
            before = ast.parse(source(relative, True))
            after = ast.parse(source(relative))
            if path in configs:
                for tree, removed in [(before, retired), (after, {'unified_etf_loss_weight'})]:
                    model = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
                    if tree is after:
                        self.assertEqual(next(ast.literal_eval(k.value) for k in model.keywords
                                              if k.arg == 'unified_etf_loss_weight'), 10.)
                    model.keywords = [k for k in model.keywords if k.arg not in removed]
            self.assertEqual(ast.dump(before), ast.dump(after), relative)

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
