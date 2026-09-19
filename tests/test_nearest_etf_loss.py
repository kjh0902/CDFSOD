"""Exact geometry and detector integration without MMDetection extensions."""
import ast
import copy
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import test_qwen_offline_sanity as sanity
from test_prototype_ddp import fusion_layer
from test_serial_decoder import method

etf = sanity.etf


def normalize(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class NearestETFGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_etf(self, target):
        batch, classes, _ = target.shape
        h = torch.eye(classes, dtype=target.dtype, device=target.device)
        h = (h - 1 / classes) / (classes - 1)
        torch.testing.assert_close(target @ target.transpose(-2, -1),
                                   h.expand(batch, -1, -1))
        torch.testing.assert_close(target.sum(1), torch.zeros_like(target[:, 0]))
        torch.testing.assert_close(target.square().sum((-2, -1)),
                                   target.new_ones(batch))
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)

    def test_target_geometry_exact_objective_and_official_parameterization(self):
        for classes, dimensions in [(2, 3), (5, 4), (5, 8), (7, 256)]:
            with self.subTest(classes=classes, dimensions=dimensions):
                z = normalize(torch.randn(3, classes, dimensions, dtype=torch.float64))
                target = etf.nearest_simplex_etf(z.requires_grad_())
                self.assert_etf(target)
                actual = (z - target).square().sum((-2, -1))
                # Centering gives one zero singular value when D >= C.
                singular_values = torch.linalg.svdvals(z.detach())
                optimum = z.square().sum((-2, -1)) + 1 - (
                    2 * singular_values.sum(-1) / (classes - 1)**0.5)
                torch.testing.assert_close(actual, optimum)
                if dimensions >= classes:
                    h = torch.eye(classes, dtype=z.dtype) - 1 / classes
                    u, _, vh = torch.linalg.svd(z.detach().transpose(-2, -1) @ h,
                                               full_matrices=False)
                    official_target = ((u @ vh) @ h / (classes - 1)**0.5).transpose(-2, -1)
                    torch.testing.assert_close(actual,
                                               (z - official_target).square().sum((-2, -1)))

    def test_rotated_etf_has_zero_loss(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        target = etf.nearest_simplex_etf(normalize(x))
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=x.dtype))
        self.assertLess(etf.nearest_etf_loss(target @ rotation).item(), 1e-25)

    def test_invariance_batch_reduction_and_whole_matrix_normalization(self):
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        base = etf.nearest_etf_loss(x)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=x.dtype))
        shifted = x + torch.randn(3, 1, 8, dtype=x.dtype)
        scaled = x * torch.tensor([0.2, 3., 12.], dtype=x.dtype)[:, None, None]
        for transformed in [shifted, scaled, x @ rotation, x[:, [3, 0, 4, 1, 2]]]:
            torch.testing.assert_close(base, etf.nearest_etf_loss(transformed))
        individual = torch.stack([etf.nearest_etf_loss(row[None]) for row in x])
        torch.testing.assert_close(base, individual.mean())
        with patch.object(etf, 'nearest_simplex_etf', wraps=etf.nearest_simplex_etf) as solve:
            etf.nearest_etf_loss(x)
        z = solve.call_args.args[0]
        torch.testing.assert_close(z, normalize(x))
        self.assertGreater(torch.linalg.vector_norm(z, dim=-1).std().item(), 0.01)

    def test_only_normalized_branch_has_gradient(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        z = normalize(x)
        target = etf.nearest_simplex_etf(z)
        self.assert_etf(target)
        expected = torch.autograd.grad((z - target).square().sum((-2, -1)).mean(), x)[0]
        actual = torch.autograd.grad(etf.nearest_etf_loss(x), x)[0]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(actual.abs().sum().item(), 0)
        self.assertTrue(torch.autograd.gradcheck(etf.nearest_etf_loss, (x,)))

    def test_rank_deficient_duplicate_and_collapsed_inputs(self):
        rank_one = torch.arange(5, dtype=torch.float64)[None, :, None].expand(2, 5, 8).clone()
        duplicate = torch.randn(2, 5, 8, dtype=torch.float64)
        duplicate[:, 1] = duplicate[:, 0]
        for x in [rank_one, duplicate, torch.ones_like(rank_one),
                  torch.zeros_like(rank_one), rank_one * 1e-10]:
            x.requires_grad_()
            target = etf.nearest_simplex_etf(normalize(x))
            self.assert_etf(target)
            loss = etf.nearest_etf_loss(x)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(etf.nearest_etf_loss(torch.ones_like(rank_one)).item(), 1.)

    def test_input_validation(self):
        for shape in [(5, 8), (0, 5, 8), (2, 1, 8), (2, 5, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                etf.nearest_etf_loss(torch.zeros(shape))
        with self.assertRaises(ValueError):
            etf.nearest_etf_loss(torch.zeros(2, 3, 4, dtype=torch.long))
        for eps in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                etf.nearest_etf_loss(torch.randn(2, 3, 4), eps=eps)

    def check_low_precision(self, device, dtype):
        x = torch.randn(2, 5, 8, device=device, dtype=dtype, requires_grad=True)
        expected = etf.nearest_etf_loss(x.float())
        with torch.autocast(device_type=device, dtype=dtype):
            actual = etf.nearest_etf_loss(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cpu_low_precision_and_autocast(self):
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.check_low_precision('cpu', dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_autocast(self):
        self.check_low_precision('cuda', torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_low_precision('cuda', torch.bfloat16)


class NearestETFIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.fixture = sanity.QwenSanity()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.model = self.fixture.model(
            entries={'pitted_surface': 'small uneven holes',
                     'other': 'dark round shape', 'beetles': 'shiny wings'},
            names=['pitted_surface', 'other', 'beetles'])
        self.model.encoder = sanity.recording_encoder()
        self.model.encoder.fusion_layers = nn.ModuleList([fusion_layer() for _ in range(6)])

    def test_etf_alone_reaches_bert_projection_and_final_enhancer(self):
        model = self.model
        text = model.build_prototype_text_dict(2, 'cpu')
        text['embedded'].retain_grad()
        output = model.forward_encoder(
            feat=torch.randn(2, 4, 3), feat_mask=torch.zeros(2, 4, dtype=torch.bool),
            feat_pos=torch.zeros(2, 4, 3), spatial_shapes=torch.tensor([[2, 2]]),
            level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2),
            text_dict=text)
        final = output['memory_text']
        final.retain_grad()
        loss = etf.nearest_etf_loss(final)
        self.assertGreater(loss.item(), 1e-6)
        loss.backward()
        for tensor in [final, text['embedded'], model.text_feat_map.weight,
                       model.language_model.language_backbone.body.model.embedding.weight,
                       model.encoder.text_layers[-1].attn.in_proj_weight]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        fusion_grads = [p.grad for p in model.encoder.fusion_layers[-1].parameters()
                        if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in fusion_grads))
        self.assertGreater(sum(g.abs().sum().item() for g in fusion_grads), 0)

    def test_loss_addition_preserves_memory_consumers_and_detection_outputs(self):
        model = self.model
        model.num_queries = 2
        model.query_embedding = nn.Embedding(2, 3)
        model.extract_feat = lambda x: (x,)
        features = torch.randn(2, 4, 3)
        model.pre_transformer = lambda *a: (dict(
            feat=features, feat_mask=torch.zeros(2, 4, dtype=torch.bool),
            feat_pos=torch.zeros_like(features), spatial_shapes=torch.tensor([[2, 2]]),
            level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2)), {})
        model.gen_encoder_output_proposals = lambda memory, *a: (memory, memory.new_zeros(2, 4, 4))
        model.dn_query_generator = lambda _: (features.new_zeros(2, 0, 3),
                                             features.new_zeros(2, 0, 4), None, {})
        seen = {}

        class Classifier:
            max_text_len = 8

            def __init__(self, stage):
                self.stage = stage

            def __call__(self, memory, memory_text, mask):
                seen[self.stage] = memory_text
                scores = memory @ memory_text.transpose(1, 2)
                return torch.cat([scores, scores.new_full((*scores.shape[:2], 5), -float('inf'))], -1)

        model.bbox_head.cls_branches = [Classifier('classification'), Classifier('selection')]
        model.bbox_head.reg_branches = [lambda x: x.new_zeros(*x.shape[:2], 4)] * 2

        def decoder(**kw):
            seen['decoder'] = kw['memory_text']
            seen['references'] = kw['reference_points']
            return dict(hidden_states=kw['memory'][:, :2].unsqueeze(0),
                        references=[kw['reference_points']])

        model.forward_decoder = decoder
        head_forward = method('mmdet/models/dense_heads/grounding_dino_head_HED.py',
                              'forward', 'GroundingDINOHead_ParallelDecoder_DN',
                              inverse_sigmoid=lambda x: torch.logit(x.clamp(1e-5, 1 - 1e-5)))

        def head_loss(**kw):
            seen['head'] = kw['memory_text']
            scores, boxes = head_forward(model.bbox_head, kw['hidden_states'], kw['references'],
                                         kw['memory_text'], kw['text_token_mask'])
            seen['scores'], seen['boxes'] = scores, boxes
            return dict(loss_cls=scores[..., :3].square().mean(), loss_bbox=boxes.mean())

        model.bbox_head.loss = head_loss
        samples = [SimpleNamespace(text=tuple(model.support_class_names),
                   gt_instances=SimpleNamespace(labels=torch.tensor([0, 2]))) for _ in range(2)]
        images = torch.zeros(2, 3, 4, 4)
        base = model.loss(images, samples)
        baseline = {key: value.detach().clone() for key, value in seen.items()}
        state_keys = model.state_dict().keys()
        model.nearest_etf_loss_weight = 0.1
        updated = model.loss(images, samples)
        self.assertEqual(set(updated), set(base) | {'loss_nearest_etf'})
        for key in base:
            torch.testing.assert_close(updated[key], base[key], rtol=0, atol=0)
        final = model.encoder.text_layers[-1].output
        for stage in ['selection', 'decoder', 'head', 'classification']:
            self.assertIs(seen[stage], final)
        for key in baseline:
            torch.testing.assert_close(seen[key], baseline[key], rtol=0, atol=0)
        torch.testing.assert_close(updated['loss_nearest_etf'], 0.1 * etf.nearest_etf_loss(final))
        self.assertEqual(state_keys, model.state_dict().keys())

        loss_globals = sanity.Detector.loss.__globals__
        with patch.dict(loss_globals, nearest_etf_loss=lambda *_: self.fail('Unexpected ETF solve')):
            model.nearest_etf_loss_weight = 0
            self.assertNotIn('loss_nearest_etf', model.loss(images, samples))
            model.nearest_etf_loss_weight = 0.1
            model.eval()

            class Prediction:
                labels = torch.tensor([0])

                def __len__(self):
                    return 1

            model.bbox_head.predict = lambda **kw: [Prediction() for _ in samples]
            model.predict(images, samples)

    def test_existing_architecture_and_nonprototype_loss_are_unchanged(self):
        base = 'bb6dcd23fabb26cdc8e03587a723888368ccf0ab'
        path = sanity.DETECTOR
        old = ast.parse(subprocess.check_output(['git', 'show', f'{base}:{path}'],
                                               cwd=sanity.ROOT).decode())
        new = ast.parse((sanity.ROOT / path).read_text(encoding='utf-8'))
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(old), methods(new)
        self.assertEqual(before.keys(), after.keys())
        for name in before.keys() - {'__init__', 'loss'}:
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        for methods_dict in [before, after]:
            loss = copy.deepcopy(methods_dict['loss'])
            loss.body = [n for n in loss.body if not (
                isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                and n.test.attr == 'use_class_name_token_prototypes')]
            methods_dict['loss'] = loss
        self.assertEqual(ast.dump(before['loss']), ast.dump(after['loss']))
        for path in ['mmdet/models/dense_heads/grounding_dino_head_HED.py',
                     'mmdet/models/layers/transformer/grounding_dino_layers_HED.py']:
            original = subprocess.check_output(['git', 'show', f'{base}:{path}'],
                                               cwd=sanity.ROOT).decode()
            self.assertEqual((sanity.ROOT / path).read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
