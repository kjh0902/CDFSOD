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
        delta = dense.attn(q2, support[None].expand(2, -1, -1),
                           support[None].expand(2, -1, -1),
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
        row = stats.report(('a', 'b'))['support_cross_attention']['0']
        self.assertAlmostEqual(row['normalized_attention_entropy']['value'],
                               original['normalized_attention_entropy']['value'])
        self.assertEqual(row['residual_contribution']['value'], .25)
        self.assertEqual(row['gamma'], original['gamma'])
        self.assertLess(len(json.dumps(stats.report(('a', 'b')))), 2500)
        self.assertTrue(all(t.grad_fn is None for t in (stats.layer_sums, stats.layer_counts, stats.gamma)))


class CacheTests(unittest.TestCase):
    def metadata(self):
        return dict(version=1, class_names=['a', 'b'], source_checkpoint='initial.pth',
                    source_checkpoint_sha256='abc', annotation_sha256='def',
                    preprocessor={}, extraction={}, dataset={})

    def test_disk_reuse_across_weights_modes_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'support_local_memory.pt'
            model = SupportModel()
            memory = model.prepare_support_cache(path, self.metadata())
            saved = memory['tokens'].clone()
            self.assertEqual(model.backbone.calls, 2)
            model.backbone.weight.data.fill_(9.)
            model.eval().train()
            torch.testing.assert_close(model.support_memory()['tokens'], saved, rtol=0, atol=0)
            other = SupportModel()
            other.backbone.weight.data.fill_(12.)
            restored = other.prepare_support_cache(path, self.metadata(), require_existing=True)
            self.assertEqual(other.backbone.calls, 0)
            torch.testing.assert_close(restored['tokens'], saved, rtol=0, atol=0)
            other.load_state_dict(other.state_dict())
            other.prepare_support_cache(path, self.metadata(), require_existing=True)
            self.assertEqual(other.backbone.calls, 0)
            self.assertFalse(other.support_memory()['tokens'].requires_grad)
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_missing_mismatch_and_invalid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'support_local_memory.pt'
            model = SupportModel()
            with self.assertRaisesRegex(FileNotFoundError, 'original Support cache'):
                model.prepare_support_cache(path, self.metadata(), require_existing=True)
            model.prepare_support_cache(path, self.metadata())
            for key in ('class_names', 'source_checkpoint_sha256', 'annotation_sha256', 'preprocessor'):
                altered = self.metadata()
                altered[key] = ['different']
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                    model.prepare_support_cache(path, altered)
            artifact = torch.load(path, weights_only=True)
            artifact['memory']['tokens'][0, 0] = float('nan')
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, 'Invalid local descriptor cache'):
                model.prepare_support_cache(path, self.metadata())

    @unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine required')
    def test_metadata_hashes_annotation_weights_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'support.json').write_text('{}', encoding='utf-8')
            (root / 'initial.pth').write_bytes(b'initial weights')
            model = SupportModel()
            model.support_visual_cfg['dataset'] = dict(data_root=tmp, ann_file='support.json',
                                                       pipeline=[dict(type='FixScaleResize')])
            model.data_preprocessor = NS(mean=torch.ones(3), std=torch.ones(3),
                _channel_conversion=True, pad_size_divisor=1, pad_value=0)
            initial = model.support_cache_metadata(root / 'initial.pth')
            # Resume/Test can validate the source identity without the initial weight file.
            restored = model.support_cache_metadata(None, initial)
            self.assertEqual(initial, restored)
            model.data_preprocessor.mean[0] = 2
            self.assertNotEqual(initial, model.support_cache_metadata(None, initial))
            model.data_preprocessor.mean[0] = 1
            (root / 'support.json').write_text('{"modified": true}', encoding='utf-8')
            self.assertNotEqual(initial, model.support_cache_metadata(None, initial))
            (root / 'support.json').write_text('{}', encoding='utf-8')
            (root / 'initial.pth').write_bytes(b'other weights')
            self.assertNotEqual(initial, model.support_cache_metadata(root / 'initial.pth'))

    def test_hook_training_resume_test_and_logging_disabled_use_same_cache(self):
        namespace = dict(torch=torch, dist=torch.distributed, Path=Path, json=json,
                         Hook=object, is_model_wrapper=lambda m: False,
                         VisualTokenStats=stats_module.VisualTokenStats)
        cls = node(source('mmdet/engine/hooks/visual_token_hook.py'), 'VisualTokenHook')
        cls.decorator_list = []
        execute([cls], namespace)
        Hook = namespace['VisualTokenHook']
        with tempfile.TemporaryDirectory() as tmp:
            def runner():
                model = SupportModel()
                model.support_cache_metadata = Mock(return_value=self.metadata())
                return NS(model=model, cfg=dict(load_from='initial.pth'), work_dir=tmp,
                          _resume=False)
            first = runner()
            hook = Hook('toy', 1, enabled=False)
            hook.before_train(first)
            saved = first.model.support_memory()['tokens'].clone()
            checkpoint = {}
            hook.before_save_checkpoint(first, checkpoint)
            self.assertEqual(checkpoint['meta']['support_local_memory'], self.metadata())
            for phase in ('resume', 'test'):
                other = runner()
                other.cfg['load_from'] = 'finetuned.pth'
                other._resume = phase == 'resume'
                hook = Hook('toy', 1, enabled=False)
                hook.after_load_checkpoint(other, checkpoint)
                if phase == 'resume':
                    hook.before_train(other)
                else:
                    hook.before_test_epoch(other)
                self.assertEqual(other.model.backbone.calls, 0)
                torch.testing.assert_close(other.model.support_memory()['tokens'], saved, rtol=0, atol=0)
                other.model.eval().train()
                self.assertEqual(other.model.backbone.calls, 0)
            Path(tmp, 'support_local_memory.pt').unlink()
            other = runner()
            hook = Hook('toy', 1, enabled=False)
            with self.assertRaisesRegex(RuntimeError, 'original Support cache'):
                hook.before_test_epoch(other)

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
                self.assertEqual(reduce.call_count, 4)
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
