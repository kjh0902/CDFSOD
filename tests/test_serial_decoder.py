"""CPU regression checks for the serial ACL decoder without MMCV extensions."""
import ast
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BASE = '7dd4f84'
DETECTOR = 'mmdet/models/detectors/grounding_dino_HED.py'
HEAD = 'mmdet/models/dense_heads/grounding_dino_head_HED.py'
LAYERS = 'mmdet/models/layers/transformer/'


def source(path, base=False):
    if base:
        return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode()
    return (ROOT / path).read_text(encoding='utf-8')


def cls_node(path, name=None, base=False):
    return next(n for n in ast.parse(source(path, base)).body
                if isinstance(n, ast.ClassDef) and (name is None or n.name == name))


def method(path, name, cls=None, **env):
    node = next(n for n in cls_node(path, cls).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    env.update(torch=torch, nn=nn)
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), env)
    return env[name]


class SerialDecoderTests(unittest.TestCase):
    def test_non_decoder_methods_and_training_head_unchanged(self):
        for path, cls, allowed in [
            (DETECTOR, None, {'__init__', '_init_layers', 'pre_decoder', 'forward_decoder'}),
            (HEAD, 'GroundingDINOHead_ParallelDecoder_DN', {'predict_by_feat'}),
            (LAYERS + 'grounding_dino_layers_HED.py', 'GroundingDinoTransformerEncoder', set()),
        ]:
            before = {n.name: n for n in cls_node(path, cls, True).body
                      if isinstance(n, ast.FunctionDef)}
            after = {n.name: n for n in cls_node(path, cls).body
                     if isinstance(n, ast.FunctionDef)}
            self.assertEqual(before.keys(), after.keys())
            for name in before.keys() - allowed:
                self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        self.assertNotIn('additional_dn_items', source(DETECTOR))
        self.assertIn('self.decoder = GroundingDinoTransformerDecoder(', source(DETECTOR))

    def test_decoder_parameter_structure_unchanged(self):
        old = cls_node(LAYERS + 'grounding_dino_layers_HED.py',
                       'GroundingDinoTransformerDecoder_parallel_15_DNQueryRand', True)
        new = cls_node(LAYERS + 'grounding_dino_layers.py', 'GroundingDinoTransformerDecoder')
        init = lambda n: next(x for x in n.body if isinstance(x, ast.FunctionDef)
                              and x.name == '_init_layers')
        self.assertEqual(ast.dump(init(old)), ast.dump(init(new)))
        old_layer = cls_node(LAYERS + 'grounding_dino_layers_HED.py',
                             'GroundingDinoTransformerDecoderLayer', True)
        new_layer = cls_node(LAYERS + 'grounding_dino_layers.py',
                             'GroundingDinoTransformerDecoderLayer')
        self.assertEqual(ast.dump(old_layer), ast.dump(new_layer))

    def test_dn_generated_once_in_training_and_never_in_inference(self):
        pre_decoder = method(DETECTOR, 'pre_decoder')
        for training, dn_count in [(True, 4), (True, 0), (False, 0)]:
            with self.subTest(training=training, dn_count=dn_count):
                cls_branch = Mock(return_value=torch.ones(2, 5, 3))
                cls_branch.max_text_len = 3
                dn_mask = torch.zeros(dn_count + 2, dn_count + 2, dtype=torch.bool)
                meta = dict(num_denoising_queries=dn_count, num_denoising_groups=1)
                dn = Mock(return_value=(torch.ones(2, dn_count, 4),
                                        torch.zeros(2, dn_count, 4), dn_mask, meta))
                model = SimpleNamespace(training=training, num_queries=2,
                    decoder=SimpleNamespace(num_layers=6), query_embedding=nn.Embedding(2, 4),
                    bbox_head=SimpleNamespace(cls_branches=[cls_branch] * 7,
                        reg_branches=[lambda x: torch.zeros_like(x)] * 7),
                    dn_query_generator=dn,
                    gen_encoder_output_proposals=lambda memory, *args: (memory, torch.zeros_like(memory)))
                decoder, head = pre_decoder(model, torch.zeros(2, 5, 4), None,
                    torch.tensor([[1, 5]]), torch.zeros(2, 3, 4),
                    torch.ones(2, 3, dtype=torch.bool), [])
                self.assertEqual(dn.call_count, int(training))
                self.assertEqual(decoder['query'].shape, (2, 2 + dn_count, 4))
                self.assertNotIn('additional_dn_items', decoder)
                if training:
                    self.assertIs(decoder['dn_mask'], dn_mask)
                    self.assertIs(head['dn_meta'], meta)
                else:
                    self.assertIsNone(decoder['dn_mask'])
                    self.assertNotIn('dn_meta', head)

    def test_all_queries_and_references_refine_sequentially(self):
        forward = method(LAYERS + 'dino_layers.py', 'forward', 'DinoTransformerDecoder',
                         coordinate_to_encoding=lambda x: x,
                         inverse_sigmoid=lambda x, eps: torch.logit(x.clamp(eps, 1 - eps)))
        for dn_count in (0, 4):
            calls = []
            class Layer(nn.Module):
                def __init__(self, index):
                    super().__init__()
                    self.delta = nn.Parameter(torch.tensor(float(index + 1)))
                def forward(self, query, **kwargs):
                    calls.append((query.detach().clone(), kwargs))
                    return query + self.delta
            layers = nn.ModuleList([Layer(i) for i in range(6)])
            model = SimpleNamespace(layers=layers, return_intermediate=True,
                                    ref_point_head=nn.Identity(), norm=nn.Identity())
            query = torch.zeros(2, 2 + dn_count, 4, requires_grad=True)
            refs = torch.full_like(query, 0.5)
            mask = torch.zeros(2 + dn_count, 2 + dn_count, dtype=torch.bool) if dn_count else None
            states, references = forward(model, query, None, None, mask, refs,
                torch.tensor([[1, 2]]), torch.tensor([0]), torch.ones(2, 1, 2),
                [lambda q: q * 0.01 for _ in range(6)])
            self.assertEqual(states.shape, (6, 2, 2 + dn_count, 4))
            self.assertEqual(references.shape, (7, 2, 2 + dn_count, 4))
            for i, (incoming, kwargs) in enumerate(calls):
                torch.testing.assert_close(incoming, torch.full_like(incoming, i * (i + 1) / 2))
                torch.testing.assert_close(kwargs['reference_points'][:, :, 0], references[i].detach())
                self.assertFalse(kwargs['reference_points'].requires_grad)
                self.assertIs(kwargs['self_attn_mask'], mask)
            states[-1].sum().backward()
            for layer in layers:
                self.assertGreater(layer.delta.grad.item(), 0)

    def test_inference_uses_only_final_layer_for_both_prompt_modes(self):
        convert = Mock(side_effect=lambda logits, positive_maps: logits[..., :1])
        predict = method(HEAD, 'predict_by_feat', 'GroundingDINOHead_ParallelDecoder_DN',
                         convert_grounding_to_cls_scores=convert)
        for positive_map in (None, {1: [0]}):
            model = SimpleNamespace(_predict_by_feat_single=Mock(return_value='result'))
            scores = torch.randn(6, 2, 3, 4)
            boxes = torch.rand(6, 2, 3, 4)
            result = predict(model, scores, boxes, [{}, {}], [positive_map] * 2)
            self.assertEqual(result, ['result', 'result'])
            for i, call in enumerate(model._predict_by_feat_single.call_args_list):
                expected = scores[-1, i].sigmoid()
                if positive_map is not None:
                    expected = expected[..., :1]
                torch.testing.assert_close(call.args[0], expected)
                torch.testing.assert_close(call.args[1], boxes[-1, i])


if __name__ == '__main__':
    unittest.main()
