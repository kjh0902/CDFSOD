"""Support-only LN / gamma experiment invariants and checkpoint compatibility."""
import copy
import subprocess
import unittest

import torch
from torch import nn

from visual_test_support import ROOT, execute, node, visual, stats_module
from support_transformer_fixture import Encoder, SupportAttention


BASE = '944d0d13f15ddc46ab7b210594148c54125d9265'


class Model(visual.SupportVisualMixin, nn.Module):
    def __init__(self, support=True):
        super().__init__()
        self.support_visual_cfg = {} if support else None
        self.encoder = Encoder(layers=2, support=support)
        self._support_memory = None


class SupportLNGammaTests(unittest.TestCase):
    def test_projection_initialization_and_rng_match_base_including_dino_xavier(self):
        original = subprocess.check_output(
            ['git', 'show', BASE + ':mmdet/models/utils/support_cross_attention.py'],
            cwd=ROOT, encoding='utf-8')
        namespace = dict(torch=torch, nn=nn)
        execute([node(original, 'SupportTextCrossAttention')], namespace)
        for dino_init in (False, True):
            torch.manual_seed(42)
            old = namespace['SupportTextCrossAttention'](256, 4)
            if dino_init:
                for p in old.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            old_rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            new = SupportAttention(256, 4)
            if dino_init:
                for p in new.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            for key, value in old.attn.state_dict().items():
                torch.testing.assert_close(new.attn.state_dict()[key], value, rtol=0, atol=0)
            self.assertIsInstance(new.attn, nn.MultiheadAttention)
            self.assertEqual((new.attn.embed_dim, new.attn.num_heads, new.attn.dropout),
                             (256, 4, 0.))
            torch.testing.assert_close(new.gamma, torch.full((256,), .01), rtol=0, atol=0)
            torch.testing.assert_close(new.support_norm.weight, torch.ones(256), rtol=0, atol=0)
            torch.testing.assert_close(new.support_norm.bias, torch.zeros(256), rtol=0, atol=0)
            self.assertEqual(new.support_norm.eps, 1e-5)
            self.assertFalse(hasattr(new, 'text_norm'))

    def test_each_layer_has_independent_trainable_support_norm_and_gamma(self):
        encoder = Encoder(layers=6)
        parameters = [p for layer in encoder.support_layers
                      for p in (layer.gamma, layer.support_norm.weight, layer.support_norm.bias)]
        self.assertEqual(len({p.data_ptr() for p in parameters}), 18)
        self.assertTrue(all(p.requires_grad for p in parameters))
        before = copy.deepcopy(encoder.state_dict())
        with torch.no_grad():
            encoder.support_layers[0].support_norm.weight.add_(1)
            encoder.support_layers[0].gamma.mul_(2)
        for key, value in encoder.state_dict().items():
            if not key.startswith('support_layers.0.'):
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_only_kv_normalized_with_gradients_after_detach_fp32_and_amp(self):
        torch.manual_seed(19)
        layer = SupportAttention(256, 4)
        text = torch.randn(2, 4, 256, requires_grad=True) * 3 + 2
        support = (torch.randn(13, 256) * 5 + 7).requires_grad_()
        original = support.detach().clone()
        classes = torch.tensor([[-1, 0, 1, -1]] * 2)
        calls = []
        handle = layer.attn.register_forward_pre_hook(
            lambda module, args: calls.append(tuple(x.detach().clone() for x in args)))
        for amp in (False, True):
            calls.clear()
            layer.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                output = layer(text, support, classes, (6, 7))
                loss = (output * torch.randn_like(output)).sum()
            loss.backward(retain_graph=True)
            for class_id, (start, end) in enumerate(((0, 6), (6, 13))):
                batch, token = (classes == class_id).nonzero(as_tuple=True)
                q, k, v = calls[class_id]
                torch.testing.assert_close(q[0], text[batch, token], rtol=0, atol=0)
                normalized = layer.support_norm(original[start:end]).detach()
                torch.testing.assert_close(k[0], normalized, rtol=0, atol=0)
                torch.testing.assert_close(k, v, rtol=0, atol=0)
            for name in ('gamma', 'support_norm.weight', 'support_norm.bias'):
                grad = dict(layer.named_parameters())[name].grad
                self.assertIsNotNone(grad, name)
                self.assertTrue(torch.isfinite(grad).all(), name)
                self.assertGreater(grad.abs().sum().item(), 0, name)
            weights = layer.attn.in_proj_weight.grad.chunk(3)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in weights))
            self.assertGreater(layer.attn.out_proj.weight.grad.abs().sum().item(), 0)
            self.assertIsNone(support.grad)
            torch.testing.assert_close(support, original, rtol=0, atol=0)
            torch.testing.assert_close(output[:, [0, 3]], text[:, [0, 3]], rtol=0, atol=0)
        handle.remove()
        layer.gamma.data.zero_()
        torch.testing.assert_close(layer(text, support, classes, (6, 7)), text, rtol=0, atol=0)

    def test_statistics_use_normalized_keys_and_original_text_denominator(self):
        torch.manual_seed(21)
        layer = SupportAttention(8, 2)
        layer.support_norm.weight.data.copy_(torch.linspace(.2, 2, 8))
        layer.support_norm.bias.data.copy_(torch.linspace(-.3, .5, 8))
        text = torch.randn(1, 12, 8) * 4 + 3
        support = torch.randn(31, 8) * 7 + 11
        normalized = layer.support_norm(support)
        update, probs = layer.attn(text, normalized[None], normalized[None],
                                   need_weights=True, average_attn_weights=False)
        expected_entropy = -(probs[0, :, :8] * probs[0, :, :8].clamp_min(1e-30).log()).sum(-1)
        expected_entropy = expected_entropy.mean() / torch.tensor(31.).log()
        stats = stats_module.VisualTokenStats(1, 'cpu', 1)
        output = layer(text, support, torch.zeros(1, 12, dtype=torch.long), (31,), stats)
        row = stats.report(('a',))['support_cross_attention']['0']
        self.assertAlmostEqual(row['normalized_attention_entropy']['value'],
                               float(expected_entropy.detach()), places=6)
        self.assertEqual(row['normalized_attention_entropy']['count'], 16)
        self.assertAlmostEqual(row['residual_contribution']['value'],
                               float((update * layer.gamma).norm().detach() / text.norm()), places=7)
        torch.testing.assert_close(output, text + update * layer.gamma, rtol=1e-5, atol=1e-6)

    def test_base_checkpoint_keeps_random_support_and_default_ln_gamma(self):
        from mmengine.runner.checkpoint import load_state_dict
        base = Model(False).state_dict()
        for loader in (lambda m: m.load_state_dict(base, strict=False),
                       lambda m: load_state_dict(m, base, strict=False)):
            model = Model()
            before = copy.deepcopy(model.encoder.support_layers.state_dict())
            loader(model)
            for key, value in model.encoder.support_layers.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            for key, value in base.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_new_checkpoint_roundtrip_preserves_ln_gamma_and_clears_cache(self):
        from mmengine.runner.checkpoint import load_state_dict
        trained = Model()
        for layer in trained.encoder.support_layers:
            layer.support_norm.weight.data.fill_(1.7)
            layer.support_norm.bias.data.fill_(-.4)
            layer.gamma.data.fill_(.23)
        state = copy.deepcopy(trained.state_dict())
        for loader in (lambda m: m.load_state_dict(state, strict=True),
                       lambda m: load_state_dict(m, state, strict=True)):
            model = Model()
            model._support_memory = {'tokens': torch.ones(1)}
            loader(model)
            self.assertIsNone(model._support_memory)
            for key, value in state.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_legacy_partial_wrong_shape_rejected_by_both_loaders(self):
        from mmengine.runner.checkpoint import load_state_dict
        state = Model().state_dict()
        legacy = {k:v for k,v in state.items() if '.support_norm.' not in k}
        partial = dict(state); del partial['encoder.support_layers.0.support_norm.bias']
        wrong = dict(state); wrong['encoder.support_layers.0.attn.in_proj_weight'] = torch.ones(12, 8)
        for bad in (legacy, partial, wrong):
            for loader in (lambda m: m.load_state_dict(bad, strict=False),
                           lambda m: load_state_dict(m, bad, strict=False)):
                with self.assertRaisesRegex(RuntimeError, 'Incompatible support checkpoint'):
                    loader(Model())

    def test_nested_prefix_loading_and_disabled_base_path(self):
        from mmengine.runner.checkpoint import load_state_dict
        state = Model().state_dict()
        nested = {'detector.' + k:v for k,v in state.items()}
        wrapper = nn.Module(); wrapper.detector = Model()
        for loader in (lambda: wrapper.load_state_dict(nested),
                       lambda: load_state_dict(wrapper, nested, strict=True)):
            loader()
            for key, value in state.items():
                torch.testing.assert_close(wrapper.detector.state_dict()[key], value, rtol=0, atol=0)
        base = Model(False)
        base.load_state_dict(base.state_dict(), strict=True)
        self.assertFalse(any('support' in key for key in base.state_dict()))


if __name__ == '__main__':
    unittest.main()
