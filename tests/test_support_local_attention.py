import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from visual_test_support import execute, node, source, stats_module, text_fixture, visual
from support_transformer_fixture import Encoder, SupportAttention, run
from test_support_visual import SupportModel, memory_fixture


class LocalAttentionTests(unittest.TestCase):
    def test_grouped_matches_dense_class_mask_output_and_gradients(self):
        torch.manual_seed(13)
        layer = SupportAttention(8, 2, gamma_init=.2).train()
        dense = copy.deepcopy(layer)
        query = torch.randn(2, 7, 8, requires_grad=True)
        q2 = query.detach().clone().requires_grad_()
        support = torch.randn(11, 8, requires_grad=True)
        classes = torch.tensor([[-1, 0, 0, 1, 1, -1, -1],
                                [-1, 0, 1, 1, 1, 0, -1]])
        output = layer(query, support, classes, (4, 7))
        labels = torch.tensor([0]*4 + [1]*7)
        mask = classes[:, :, None] != labels
        mask[classes < 0] = False  # safe placeholder for excluded query rows
        normalized_support = dense.support_norm(support.detach())[None].expand(2, -1, -1)
        delta = dense.attn(dense.text_norm(q2), normalized_support,
                           normalized_support,
                           attn_mask=mask.repeat_interleave(2, 0), need_weights=False)[0]
        expected = q2 + delta * dense.gamma * (classes >= 0)[..., None]
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(query.grad, q2.grad, rtol=1e-5, atol=1e-6)
        for p, q in zip(layer.parameters(), dense.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)

    def test_support_gradient_detached_and_special_queries_exactly_unchanged(self):
        layer = SupportAttention(8, 2)
        query = torch.randn(2, 4, 8, requires_grad=True)
        support = torch.randn(5, 8, requires_grad=True)
        classes = torch.tensor([[-1, 0, 1, -1]]*2)
        output = layer(query, support, classes, (3, 2))
        torch.testing.assert_close(output[:, [0, 3]], query[:, [0, 3]], rtol=0, atol=0)
        output.square().sum().backward()
        self.assertIsNone(support.grad)
        self.assertGreater(layer.gamma.grad.abs().sum().item(), 0)

    def test_zero_gamma_recovers_original_encoder_and_layer_order(self):
        encoder = Encoder(layers=2).eval()
        for layer in encoder.support_layers:
            layer.gamma.data.zero_()
        text, maps = text_fixture(channels=8)
        memory = memory_fixture(channels=8)
        classes = visual.support_text_classes(text, memory, maps)
        image = torch.randn(1, 4, 8)
        baseline = run(encoder, text, image=image)
        order, handles = [], []
        for i in range(2):
            for label, module in [('fusion', encoder.fusion_layers[i]),
                    ('support', encoder.support_layers[i]), ('text', encoder.text_layers[i]),
                    ('image', encoder.layers[i])]:
                handles.append(module.register_forward_pre_hook(
                    lambda m, a, label=label, i=i: order.append((i, label))))
        current = run(encoder, text, memory, classes, image)
        for a, b in zip(baseline, current):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(order, [(i, label) for i in range(2)
                                for label in ('fusion', 'support', 'text', 'image')])
        for h in handles:
            h.remove()

    @unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine required')
    def test_checkpoint_keys_only_add_support_and_disabled_init_matches(self):
        from support_transformer_fixture import env, TextLayer
        from mmengine.model import ModuleList
        class Image(nn.Module):
            embed_dims = 8
            def __init__(self, **kwargs):
                super().__init__()
                self.linear = nn.Linear(8, 8)
        namespace = dict(env, ModuleList=ModuleList, SupportTextCrossAttention=SupportAttention,
                         DeformableDetrTransformerEncoderLayer=Image,
                         DetrTransformerEncoderLayer=lambda **kw: TextLayer())
        method = next(n for n in node(source(
            'mmdet/models/layers/transformer/grounding_dino_layers.py'),
            'GroundingDinoTransformerEncoder').body if getattr(n, 'name', '') == '_init_layers')
        execute([method], namespace)
        def build(enabled):
            obj = nn.Module()
            obj.num_layers, obj.num_cp = 2, 0
            obj.layer_cfg = {}
            obj.text_layer_cfg = {'self_attn_cfg': {'num_heads': 2}}
            obj.fusion_layer_cfg = dict(v_dim=8, l_dim=8, embed_dim=16, num_heads=2)
            obj.support_cross_attn = enabled
            namespace['_init_layers'](obj)
            return obj
        torch.manual_seed(42)
        disabled = build(False)
        torch.manual_seed(42)
        enabled = build(True)
        result = enabled.load_state_dict(disabled.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(all(k.startswith('support_layers.') for k in result.missing_keys))
        self.assertEqual(len(disabled.support_layers), 0)
        self.assertFalse(any('support' in k for k in disabled.state_dict()))
        for key, value in disabled.state_dict().items():
            torch.testing.assert_close(value, enabled.state_dict()[key], rtol=0, atol=0)
        for layer in enabled.support_layers:
            torch.testing.assert_close(layer.gamma, torch.full((8,), 1e-4))


class DiagnosticTests(unittest.TestCase):
    def test_streamed_entropy_matches_actual_attention_and_extremes(self):
        torch.manual_seed(2)
        attn = nn.MultiheadAttention(8, 2, dropout=0., batch_first=True)
        query, support = torch.randn(5, 8), torch.randn(31, 8)
        probs = attn(query[None], support[None], support[None],
                     need_weights=True, average_attn_weights=False)[1][0]
        expected = -(probs * probs.clamp_min(1e-30).log()).sum(-1) / torch.log(torch.tensor(31.))
        stats = stats_module.VisualTokenStats
        torch.testing.assert_close(stats.attention_entropy(attn, query, support, 7), expected)
        with torch.no_grad():
            attn.in_proj_weight.zero_()
            attn.in_proj_bias.zero_()
        torch.testing.assert_close(stats.attention_entropy(attn, query, support, 7),
                                   torch.ones(2, 5))
        torch.testing.assert_close(stats.attention_entropy(attn, query, support[:1]),
                                   torch.zeros(2, 5))
        with torch.no_grad():
            attn.in_proj_weight[:8].copy_(torch.eye(8))
            attn.in_proj_weight[8:16].copy_(torch.eye(8))
        concentrated = torch.zeros(31, 8)
        concentrated[0] = 100.
        self.assertLess(stats.attention_entropy(attn, torch.ones(5, 8), concentrated).max(), 1e-5)

    def test_sampling_padding_contribution_gamma_reduction_and_size(self):
        stats = stats_module.VisualTokenStats(2, 'cpu', num_layers=2)
        layer = SupportAttention(8, 2)
        stats.valid = [True, False]
        stats.record_gamma(0, layer.gamma)
        query = torch.ones(40, 8)
        residual = query * .25
        # Half are sampler padding; only the first eight valid queries get entropy.
        batch = torch.tensor([0]*20 + [1]*20)
        stats.cross_attention(0, layer.attn, query, torch.ones(51, 8), residual, batch)
        row = stats.report(('a', 'b'))['support_cross_attention']['0']
        self.assertEqual(row['normalized_attention_entropy']['count'], 16)
        self.assertEqual(row['residual_contribution'], dict(value=.25, count=20))
        self.assertEqual(row['gamma']['std'], 0.)
        original = copy.deepcopy(row)
        stats.layer_sums.mul_(3)
        stats.layer_counts.mul_(3)
        stats.head_sums.mul_(3)
        row = stats.report(('a', 'b'))['support_cross_attention']['0']
        self.assertAlmostEqual(row['normalized_attention_entropy']['value'],
                               original['normalized_attention_entropy']['value'])
        self.assertEqual(row['residual_contribution']['value'], .25)
        self.assertEqual(row['gamma'], original['gamma'])
        self.assertLess(len(json.dumps(stats.report(('a', 'b')))), 4096)
        self.assertTrue(all(t.grad_fn is None for t in (stats.layer_sums, stats.layer_counts, stats.gamma)))


class SupportRefreshTests(unittest.TestCase):
    def hook(self, enabled=False):
        namespace = dict(torch=torch, dist=torch.distributed, Path=Path, json=json,
                         Hook=object, is_model_wrapper=lambda m: False,
                         VisualTokenStats=stats_module.VisualTokenStats)
        cls = node(source('mmdet/engine/hooks/visual_token_hook.py'), 'VisualTokenHook')
        cls.decorator_list = []
        execute([cls], namespace)
        return namespace['VisualTokenHook']('toy', 1, enabled=enabled)

    def test_every_iteration_extracts_entire_set_with_current_weights_without_grad(self):
        model = SupportModel().train()
        # Preserve a deliberately mixed mode, including on extraction failure.
        model.backbone.dropout.eval()
        modes = [m.training for m in model.backbone.modules()]
        observed = []
        handle = model.backbone.register_forward_pre_hook(lambda module, args:
            observed.append((torch.is_grad_enabled(), [m.training for m in module.modules()])))
        expected = torch.tensor([2., 2., 8., 2., 8.]).repeat_interleave(49)
        rng = torch.get_rng_state().clone()
        optimizer = torch.optim.SGD(model.backbone.parameters(), lr=.1)
        previous = None
        for iteration in range(3):
            weight = model.backbone.weight.detach().clone()
            before = model.backbone.calls
            memory = model.support_memory()
            self.assertEqual(model.backbone.calls - before, len(model._support_loader))
            torch.testing.assert_close(memory['tokens'][:, 0], expected * weight)
            self.assertEqual(memory['tokens'].shape, (245, 256))
            self.assertEqual(memory['counts'].tolist(), [147, 98])
            self.assertFalse(memory['tokens'].requires_grad)
            self.assertIsNone(memory['tokens'].grad_fn)
            self.assertIsNone(model.backbone.weight.grad)
            self.assertEqual(modes, [m.training for m in model.backbone.modules()])
            if previous is not None:
                self.assertFalse(torch.equal(previous, memory['tokens']))
            previous = memory['tokens'].clone()
            # Query backbone stays trainable; the next support extraction sees its update.
            query = model.backbone(torch.ones(1, 3, 64, 64))[0]
            query.mean().backward()
            self.assertGreater(model.backbone.weight.grad.abs().item(), 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        handle.remove()
        extraction_calls = [entry for i, entry in enumerate(observed) if i % 3 != 2]
        self.assertEqual(len(extraction_calls), 6)
        self.assertTrue(all(not grad and not any(states) for grad, states in extraction_calls))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_validation_epoch_eager_refresh_then_reuse_with_logging_disabled(self):
        model = SupportModel().train()
        runner = NS(model=model)
        hook = self.hook()  # descriptor lifecycle must not depend on logging
        for weight in (1., 2.):
            model.train()
            model.backbone.weight.data.fill_(weight)
            before = model.backbone.calls
            hook.before_val_epoch(runner)  # MMEngine invokes this before model.eval()
            self.assertEqual(model.backbone.calls - before, 2)
            self.assertTrue(model.training)
            fresh = model._support_memory
            model.eval()
            for _ in range(3):
                self.assertIs(model.support_memory(), fresh)
            self.assertEqual(model.backbone.calls - before, 2)
            torch.testing.assert_close(fresh['tokens'][0], torch.full((256,), 2. * weight))
            self.assertFalse(fresh['tokens'].requires_grad)

    def test_best_checkpoint_test_and_resume_use_loaded_backbone(self):
        model = SupportModel().eval()
        stale = model.support_memory()['tokens'].clone()
        best = copy.deepcopy(model.state_dict())
        best['backbone.weight'].fill_(3.)
        model.load_state_dict(best)
        self.assertIsNone(model._support_memory)
        before = model.backbone.calls
        self.hook().before_test_epoch(NS(model=model))
        self.assertEqual(model.backbone.calls - before, 2)
        model.eval()
        memory = model.support_memory()
        torch.testing.assert_close(memory['tokens'], stale * 3.)
        self.assertIs(model.support_memory(), memory)
        self.assertEqual(model.backbone.calls - before, 2)
        # Resume starts with a different loaded weight and refreshes every iteration.
        resumed = copy.deepcopy(best)
        resumed['backbone.weight'].fill_(4.)
        model.load_state_dict(resumed)
        model.train()
        self.assertIsNone(model._support_memory)
        before = model.backbone.calls
        for weight in (4., 5.):
            model.backbone.weight.data.fill_(weight)
            torch.testing.assert_close(model.support_memory()['tokens'], stale * weight)
            self.assertIsNone(getattr(model, '_support_memory', None))
        self.assertEqual(model.backbone.calls - before, 4)

    def test_failed_refresh_discards_stale_memory_and_restores_modes(self):
        model = SupportModel().eval()
        model.support_memory()
        model.backbone.train()  # retain a stale snapshot and deliberately mixed modes
        model.backbone.dropout.eval()
        modes = [m.training for m in model.backbone.modules()]
        with patch.object(model.backbone, 'forward', side_effect=RuntimeError('extraction failed')):
            with self.assertRaisesRegex(RuntimeError, 'extraction failed'):
                self.hook().before_val_epoch(NS(model=model))
        self.assertIsNone(model._support_memory)
        self.assertEqual(modes, [m.training for m in model.backbone.modules()])

    def test_ddp_writer_reduces_only_sums_counts_and_skips_other_ranks(self):
        namespace = dict(torch=torch, dist=torch.distributed, Path=Path, json=json,
                         Hook=object, is_model_wrapper=lambda m: False,
                         VisualTokenStats=stats_module.VisualTokenStats)
        cls = node(source('mmdet/engine/hooks/visual_token_hook.py'), 'VisualTokenHook')
        cls.decorator_list = []
        execute([cls], namespace)
        with tempfile.TemporaryDirectory() as tmp:
            model = SupportModel()
            runner = NS(model=model, work_dir=tmp, epoch=0, iter=0, logger=Mock())
            hook = namespace['VisualTokenHook']('toy', 1)
            hook.stats = stats_module.VisualTokenStats(2, 'cpu', 1)
            stats = hook.stats
            stats.record_gamma(0, torch.tensor([-.2, .1]))
            gamma = stats.gamma.clone()
            with patch.object(torch.distributed, 'is_initialized', return_value=True), \
                    patch.object(torch.distributed, 'get_rank', return_value=0), \
                    patch.object(torch.distributed, 'all_reduce', side_effect=lambda x: x.mul_(2)) as reduce:
                hook._write(runner, 'val')
                self.assertEqual(reduce.call_count, 5)
            torch.testing.assert_close(stats.gamma, gamma, rtol=0, atol=0)
            row = json.loads(Path(tmp, 'visual_token_stats.jsonl').read_text())
            self.assertAlmostEqual(row['support_cross_attention']['0']['gamma']['max_absolute'], .2)
            hook.stats = stats_module.VisualTokenStats(2, 'cpu', 1)
            with patch.object(torch.distributed, 'is_initialized', return_value=True), \
                    patch.object(torch.distributed, 'get_rank', return_value=1), \
                    patch.object(torch.distributed, 'all_reduce'):
                hook._write(runner, 'val')
            self.assertEqual(len(Path(tmp, 'visual_token_stats.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
