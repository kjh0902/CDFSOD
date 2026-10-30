"""CPU validation with the actual MMCV Attention/FFN/Norm modules."""
import copy
import importlib.util
import unittest

import torch

from visual_test_support import ENV, execute, node, source, text_fixture, visual
from test_support_visual import memory_fixture
from support_transformer_fixture import Encoder, run


@unittest.skipUnless(importlib.util.find_spec('mmcv') and importlib.util.find_spec('mmengine'),
                     'MMCV (or mmcv-lite) and MMEngine required')
class ActualMMCVTests(unittest.TestCase):
    def layers(self):
        from mmcv.cnn import build_norm_layer
        from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
        from mmengine.model import BaseModule, ModuleList
        from mmengine.config import ConfigDict
        env = dict(ENV, BaseModule=BaseModule, ModuleList=ModuleList,
                   MultiheadAttention=MultiheadAttention, FFN=FFN,
                   build_norm_layer=build_norm_layer)
        rel = 'mmdet/models/layers/transformer/detr_layers.py'
        config = dict(self_attn_cfg=ConfigDict(embed_dims=256, num_heads=4, dropout=0.),
                      ffn_cfg=ConfigDict(embed_dims=256, feedforward_channels=1024,
                                        ffn_drop=0., act_cfg=dict(type='ReLU', inplace=True)))
        execute([node(source(rel, baseline=True), 'DetrTransformerEncoderLayer')], env)
        old = env['DetrTransformerEncoderLayer'](**copy.deepcopy(config))
        execute([node(source(rel), 'DetrTransformerEncoderLayer')], env)
        new = env['DetrTransformerEncoderLayer'](**copy.deepcopy(config))
        new.load_state_dict(old.state_dict(), strict=True)
        return old, new

    def test_original_parameter_keys_and_disabled_forward_backward(self):
        old, new = self.layers()
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        self.assertEqual(sum(p.numel() for p in old.parameters()),
                         sum(p.numel() for p in new.parameters()))
        query = torch.randn(2, 9, 256)
        position = torch.randn_like(query)
        mask = ~torch.eye(9, dtype=torch.bool).repeat(8, 1, 1)
        outputs = [layer(query, position, None, attn_mask=mask) for layer in (old, new)]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for output in outputs:
            output[..., 0].sum().backward()
        for p, q in zip(old.parameters(), new.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_real_mmcv_rectangular_attention_fp32_amp_and_backward(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                _, layer = self.layers()
                encoder = Encoder(dim=256, heads=4, layers=1, fusion=False)
                encoder.text_layers[0] = layer
                text, maps = text_fixture(2)
                memory = memory_fixture(counts=(5, 2), requires_grad=True)
                original = memory['tokens'].detach().clone()
                allowed = visual.support_text_classes(text, memory, maps)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    image, output = run(encoder, text, memory, allowed)
                    loss = output[..., 0].square().mean() + image.square().mean()
                loss.backward()
                self.assertEqual(output.shape, (2, 9, 256))
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNone(memory['tokens'].grad)
                torch.testing.assert_close(memory['tokens'], original, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(text['embedded'].grad).all())
                gradient = layer.self_attn.attn.in_proj_weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
