"""CPU regression tests for ACL encoder routing (requires only PyTorch).

Run: python -m unittest discover -s tests -p test_parallel_encoder.py -v

Load the production encoder class via AST to avoid importing CUDA/MMCV.
Attention modules are deterministic differentiable stand-ins; these tests do
not substitute for an integration test with real deformable attention.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


LAYERS = (Path(__file__).resolve().parents[1] / 'mmdet' / 'models' /
          'layers' / 'transformer')


class BaseEncoder(nn.Module):
    def __init__(self, num_layers=6, num_cp=0, layer_cfg=None):
        super().__init__()
        self.num_layers = num_layers
        self.num_cp = num_cp
        self.layer_cfg = layer_cfg or {}
        self._init_layers()

    @staticmethod
    def get_encoder_reference_points(spatial_shapes, valid_ratios, device):
        return torch.zeros(1, 2, 1, 2, device=device)


class Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.25))
        self.embed_dims = 4
        self.self_attn_cfg = SimpleNamespace(num_heads=2)
        self.inputs = []
        self.outputs = []
        self.arguments = []

    def forward(self, query, **kwargs):
        self.inputs.append(query)
        self.arguments.append(kwargs)
        output = query * self.weight + 0.1
        self.outputs.append(output)
        return output


class Fusion(Attention):
    def forward(self, visual_feature, lang_feature, **kwargs):
        self.inputs.append((visual_feature, lang_feature))
        self.arguments.append(kwargs)
        output = (visual_feature * self.weight + lang_feature.mean(),
                  lang_feature * self.weight + visual_feature.mean())
        self.outputs.append(output)
        return output


def load_encoder(filename):
    tree = ast.parse((LAYERS / filename).read_text(encoding='utf-8'))
    encoder = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'GroundingDinoTransformerEncoder')
    namespace = dict(
        torch=torch, Tensor=torch.Tensor, ConfigType=dict,
        ModuleList=nn.ModuleList, DeformableDetrTransformerEncoder=BaseEncoder,
        DeformableDetrTransformerEncoderLayer=Attention,
        DetrTransformerEncoderLayer=Attention,
        SingleScaleBiAttentionBlock=Fusion)
    exec(compile(ast.Module(body=[encoder], type_ignores=[]), filename, 'exec'),
         namespace)
    return namespace['GroundingDinoTransformerEncoder']


ParallelEncoder = load_encoder('grounding_dino_layers_HED.py')
SequentialEncoder = load_encoder('grounding_dino_layers.py')


def make_encoder(cls=ParallelEncoder, num_layers=6):
    encoder = cls(text_layer_cfg={}, fusion_layer_cfg={}, num_layers=num_layers)
    with torch.no_grad():
        for index, modules in enumerate(zip(encoder.fusion_layers,
                                            encoder.text_layers,
                                            encoder.layers)):
            for module in modules:
                module.weight.fill_(1.0 + index / 10)
    return encoder


def make_inputs():
    return dict(
        query=torch.ones(1, 2, 4, requires_grad=True),
        query_pos=torch.zeros(1, 2, 4),
        key_padding_mask=torch.tensor([[False, True]]),
        spatial_shapes=torch.tensor([[1, 2]]),
        level_start_index=torch.tensor([0]),
        valid_ratios=torch.ones(1, 1, 2),
        memory_text=torch.full((1, 3, 4), 2.0, requires_grad=True),
        text_attention_mask=torch.tensor([[False, False, True]]),
        pos_text=torch.zeros(1, 3, 4),
        text_self_attention_masks=torch.eye(3, dtype=torch.bool).unsqueeze(0))


class TestParallelEncoder(unittest.TestCase):
    def test_shared_inputs_and_means(self):
        for num_layers in (2, 4, 6):
            with self.subTest(num_layers=num_layers):
                encoder = make_encoder(num_layers=num_layers)
                inputs = make_inputs()
                visual, text = encoder(**inputs)
                e1_visual = encoder.layers[0].outputs[0]
                e1_text = encoder.text_layers[0].outputs[0]
                for index in range(num_layers):
                    fusion = encoder.fusion_layers[index]
                    self.assertEqual(len(fusion.inputs), 1)
                    self.assertEqual(len(encoder.layers[index].inputs), 1)
                    self.assertEqual(len(encoder.text_layers[index].inputs), 1)
                    expected_v = e1_visual if index else inputs['query']
                    expected_t = e1_text if index else inputs['memory_text']
                    self.assertIs(fusion.inputs[0][0], expected_v)
                    self.assertIs(fusion.inputs[0][1], expected_t)
                    self.assertIs(encoder.layers[index].inputs[0],
                                  fusion.outputs[0][0])
                    self.assertIs(encoder.text_layers[index].inputs[0],
                                  fusion.outputs[0][1])
                    self.assertIs(fusion.arguments[0]['attention_mask_l'],
                                  inputs['text_attention_mask'])
                    self.assertIs(encoder.layers[index].arguments[0]['query_pos'],
                                  inputs['query_pos'])
                    torch.testing.assert_close(
                        encoder.text_layers[index].arguments[0]['attn_mask'],
                        ~inputs['text_self_attention_masks'].repeat(2, 1, 1))
                expected_v = sum(layer.outputs[0] for layer in
                                 encoder.layers[1:]) / (num_layers - 1)
                expected_t = sum(layer.outputs[0] for layer in
                                 encoder.text_layers[1:]) / (num_layers - 1)
                torch.testing.assert_close(visual, expected_v)
                torch.testing.assert_close(text, expected_t)

    def test_branch_isolation(self):
        original, changed = make_encoder(), make_encoder()
        with torch.no_grad():
            changed.fusion_layers[1].weight.add_(1)
        before = original(**make_inputs())
        after = changed(**make_inputs())
        for index in (0, 2, 3, 4, 5):
            for left, right in zip(original.fusion_layers[index].inputs[0],
                                   changed.fusion_layers[index].inputs[0]):
                torch.testing.assert_close(left, right)
            torch.testing.assert_close(original.layers[index].outputs[0],
                                       changed.layers[index].outputs[0])
            torch.testing.assert_close(original.text_layers[index].outputs[0],
                                       changed.text_layers[index].outputs[0])
        self.assertFalse(torch.equal(before[0], after[0]))
        self.assertFalse(torch.equal(before[1], after[1]))

    def test_gradients_reach_all_stages(self):
        encoder, inputs = make_encoder(), make_inputs()
        visual, text = encoder(**inputs)
        for layer in list(encoder.layers[1:]) + list(encoder.text_layers[1:]):
            layer.outputs[0].retain_grad()
        (visual.square().mean() + text.square().mean()).backward()
        for layers, mean in ((encoder.layers, visual),
                             (encoder.text_layers, text)):
            expected = 2 * mean.detach() / mean.numel() / (len(layers) - 1)
            for layer in layers[1:]:
                torch.testing.assert_close(layer.outputs[0].grad, expected)
        for name, parameter in encoder.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for name in ('query', 'memory_text'):
            self.assertGreater(inputs[name].grad.abs().sum().item(), 0)

    def test_single_layer_matches_sequential(self):
        sequential = make_encoder(SequentialEncoder, num_layers=1)
        parallel = make_encoder(num_layers=1)
        for expected, actual in zip(sequential(**make_inputs()),
                                    parallel(**make_inputs())):
            torch.testing.assert_close(actual, expected)

    def test_state_dict_compatibility(self):
        sequential, parallel = make_encoder(SequentialEncoder), make_encoder()
        state = sequential.state_dict()
        result = parallel.load_state_dict(state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(list(state), list(parallel.state_dict()))
        for name, value in parallel.state_dict().items():
            torch.testing.assert_close(value, state[name])
        self.assertEqual(sum(p.numel() for p in sequential.parameters()),
                         sum(p.numel() for p in parallel.parameters()))


if __name__ == '__main__':
    unittest.main()
