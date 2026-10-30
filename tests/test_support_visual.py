import copy
import unittest

import torch
from torch import nn

from visual_test_support import (visual, sample, text_fixture, tokenizer_masks,
                                 execute, node, source)
from support_transformer_fixture import Encoder, TextLayer, run


def memory_fixture(counts=(3, 2), channels=256, requires_grad=False):
    counts = torch.tensor(counts, dtype=torch.long)
    return dict(tokens=torch.randn(int(counts.sum()), channels, requires_grad=requires_grad),
                class_ids=torch.arange(len(counts)).repeat_interleave(counts), counts=counts)


class SupportModel(visual.SupportVisualMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.support_visual_cfg = dict(class_names=('a', 'b'))
        self.bbox_head = type('Head', (), dict(num_classes=2))()
        self.backbone = Backbone()
        self.data_preprocessor = lambda batch, training: batch
        self._support_loader = [
            dict(inputs=torch.full((1, 3, 64, 64), 2.),
                 data_samples=[sample([[8, 8, 40, 40]] * 3, [0, 0, 1])]),
            dict(inputs=torch.full((1, 3, 64, 64), 8.),
                 data_samples=[sample([[8, 8, 40, 40]] * 2, [0, 1])])]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(.5)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [(self.dropout(x[:, :1, ::8, ::8] * self.weight)).expand(-1, 256, -1, -1)]


class MemoryTests(unittest.TestCase):
    def test_real_roi_stride_coordinates_and_all_spatial_bins(self):
        y, x = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing='ij')
        features = (x + 10 * y)[None, None].expand(1, 256, -1, -1)
        pooled, labels = visual.pool_instances(features, [sample([[8, 8, 40, 48]], [1])])
        from torchvision.ops import roi_align
        expected = roi_align(features, torch.tensor([[0., 8., 8., 40., 48.]]),
                             (7, 7), spatial_scale=1/8, sampling_ratio=2, aligned=True)
        torch.testing.assert_close(pooled, expected.flatten(2).transpose(1, 2))
        self.assertEqual(pooled.shape, (1, 49, 256))
        self.assertGreater(pooled[0, :, 0].std().item(), 0)
        self.assertEqual(labels.tolist(), [1])

    def test_all_objects_preserved_class_concatenation_no_grad_and_rng(self):
        model = SupportModel().train()
        model.backbone.dropout.eval()
        states = [m.training for m in model.backbone.modules()]
        rng = torch.get_rng_state().clone()
        output = model.support_memory()
        torch.testing.assert_close(output['tokens'][:, 0], torch.tensor([2., 2., 8., 2., 8.]).repeat_interleave(49))
        self.assertEqual(output['counts'].tolist(), [147, 98])
        self.assertEqual(output['class_ids'].tolist(), [0] * 147 + [1] * 98)
        self.assertFalse(output['tokens'].requires_grad)
        self.assertEqual(states, [m.training for m in model.backbone.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.backbone(torch.ones(1, 3, 64, 64))[0].sum().backward()
        self.assertIsNotNone(model.backbone.weight.grad)

    def test_actual_counts_for_1_5_10_shot_not_fixed_per_class(self):
        for shot in (1, 5, 10):
            model = SupportModel()
            model._support_loader = model._support_loader * shot
            result = model.support_memory()
            self.assertEqual(result['tokens'].shape, (5 * shot * 49, 256))
            self.assertEqual(result['counts'].tolist(), [3 * shot * 49, 2 * shot * 49])

    def test_fixed_cache_modes_weights_and_state_dict(self):
        model = SupportModel().train()
        first = model.support_memory()
        with torch.no_grad():
            model.backbone.weight.fill_(2.)
        self.assertIs(first, model.support_memory())
        model.eval()
        self.assertIs(first, model.support_memory())
        model.train()
        self.assertIs(first, model.support_memory())
        self.assertEqual(model.backbone.calls, 2)
        state = copy.deepcopy(model.state_dict())
        self.assertFalse(any('support' in key or 'stats' in key for key in state))
        model.load_state_dict(state)
        self.assertIsNone(model._support_memory)

    def test_missing_class_invalid_label_boxes_and_mode_restoration(self):
        for failure in ('missing', 'labels', 'boxes'):
            model = SupportModel().train()
            model._support_loader = model._support_loader[:1]
            instances = model._support_loader[0]['data_samples'][0].gt_instances
            if failure == 'missing':
                instances.labels[:] = 0
            elif failure == 'labels':
                instances.labels[0] = 2
            else:
                instances.bboxes[0, 2] = 0
            with self.assertRaisesRegex(ValueError, failure):
                model.support_memory()
            self.assertTrue(model.backbone.training)

    def test_empty_images_add_no_tokens(self):
        model = SupportModel()
        model._support_loader.insert(0, dict(inputs=torch.ones(1, 3, 64, 64),
                                             data_samples=[sample([], [])]))
        result = model.support_memory()
        self.assertEqual(result['counts'].tolist(), [147, 98])
        torch.testing.assert_close(result['tokens'][:, 0], torch.tensor([2., 2., 8., 2., 8.]).repeat_interleave(49))


class AttentionTests(unittest.TestCase):
    def test_text_mapping_positions_and_specials_unchanged(self):
        text, maps = text_fixture(channels=8)
        original = copy.deepcopy({k: v.detach().clone() for k, v in text.items()})
        memory = memory_fixture(channels=8)
        allowed = visual.support_text_classes(text, memory, maps)
        self.assertEqual(allowed.shape, (1, 9))
        self.assertEqual(allowed.tolist(), [[-1, 0, 0, 0, 1, 1, -1, -1, -1]])
        for key in text:
            torch.testing.assert_close(text[key], original[key])
        self.assertEqual(maps, [{1: [1, 2], 2: [4]}])

    def test_invalid_layout_and_counts_fail(self):
        text, _ = text_fixture(channels=8)
        for maps in ([{1: [1]}], [{1: [], 2: [4]}], [{1: [1, 3], 2: [4]}]):
            with self.assertRaises(ValueError):
                visual.support_text_classes(text, memory_fixture(channels=8), maps)
        memory = memory_fixture(channels=8)
        memory['counts'][0] += 1
        with self.assertRaisesRegex(ValueError, 'counts'):
            visual.support_text_classes(text, memory, [{1: [1, 2], 2: [4]}])

    def test_class_isolation_and_special_padding_outputs(self):
        torch.manual_seed(5)
        text, maps = text_fixture(channels=8)
        memory = memory_fixture(channels=8)
        encoder = Encoder(layers=1, fusion=False).eval()
        allowed = visual.support_text_classes(text, memory, maps)
        image = torch.randn(1, 4, 8)
        a = run(encoder, text, memory, allowed, image)[1]
        other = memory['tokens'].clone()
        other[:3] = torch.randn_like(other[:3]) * 20
        b = run(encoder, text, dict(memory, tokens=other), allowed, image)[1]
        self.assertGreater((a[:, 1:4] - b[:, 1:4]).abs().max().item(), 1e-4)
        torch.testing.assert_close(a[:, [0, 4, 5, 6, 7, 8]], b[:, [0, 4, 5, 6, 7, 8]],
                                   rtol=0, atol=0)
        changed_text = dict(text, embedded=text['embedded'].detach().clone())
        changed_text['embedded'][:, 7:] = 10000
        c = run(encoder, changed_text, memory, allowed, image)[1]
        torch.testing.assert_close(a[:, :7], c[:, :7], rtol=0, atol=0)

    def test_unequal_batch_lengths_and_head_mask_order(self):
        torch.manual_seed(7)
        text, maps = text_fixture(2, channels=8)
        ids = torch.tensor([[101, 10, 11, 119, 12, 119, 102, 0, 0],
                            [101, 10, 119, 12, 119, 102, 0, 0, 0]])
        text['masks'], text['position_ids'] = tokenizer_masks(ids)
        text['text_token_mask'] = ids != 0
        maps[1] = {1: [1], 2: [3]}
        memory = memory_fixture(channels=8)
        allowed = visual.support_text_classes(text, memory, maps)
        encoder = Encoder(layers=1, heads=2, fusion=False).eval()
        image = torch.randn(2, 4, 8)
        batch = run(encoder, text, memory, allowed, image)[1]
        for i in range(2):
            single = {key: value[i:i + 1] for key, value in text.items()}
            output = run(encoder, single, memory, allowed[i:i + 1], image[i:i + 1])[1]
            torch.testing.assert_close(batch[i:i + 1], output, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(batch).all())
        self.assertTrue(torch.equal(encoder.text_layers[0].self_attn.calls[0]['mask'],
                                   ~text['masks'].repeat_interleave(2, dim=0)))

    def test_memory_immutable_gradients_and_text_only_fusion(self):
        torch.manual_seed(8)
        text, maps = text_fixture(2, channels=8)
        memory = memory_fixture(channels=8, requires_grad=True)
        original = memory['tokens'].detach().clone()
        encoder = Encoder().train()
        fusion_lengths = []
        handles = [layer.register_forward_pre_hook(
            lambda module, args, kwargs: fusion_lengths.append(kwargs['lang_feature'].shape[1]),
            with_kwargs=True) for layer in encoder.fusion_layers]
        allowed = visual.support_text_classes(text, memory, maps)
        image, output = run(encoder, text, memory, allowed)
        (output.square().mean() + image.square().mean() + output[..., 0].mean()).backward()
        self.assertIsNone(memory['tokens'].grad)
        self.assertTrue(torch.isfinite(text['embedded'].grad).all())
        self.assertGreater(encoder.text_layers[0].self_attn.attn.in_proj_weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(memory['tokens'], original, rtol=0, atol=0)
        self.assertEqual(fusion_lengths, [9, 9])
        for layer in encoder.text_layers:
            call = layer.self_attn.calls[0]
            self.assertEqual(call['query'].shape, (2, 9, 8))
            self.assertEqual(call['key'].shape, (2, 9, 8))
            torch.testing.assert_close(call['key'], call['query'])
            torch.testing.assert_close(call['value'], call['query'])
        for handle in handles:
            handle.remove()

    def test_memory_permutation_has_no_positional_effect(self):
        text, maps = text_fixture(channels=8)
        memory = memory_fixture(channels=8)
        encoder = Encoder().eval()
        image = torch.randn(1, 4, 8)
        allowed = visual.support_text_classes(text, memory, maps)
        a = run(encoder, text, memory, allowed, image)
        order = torch.tensor([2, 0, 1, 4, 3])
        permuted = dict(memory, tokens=memory['tokens'][order], class_ids=memory['class_ids'][order])
        b = run(encoder, text, permuted,
                visual.support_text_classes(text, permuted, maps), image)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)

    def test_disabled_text_layer_reuses_original_weights_structure_and_forward(self):
        env = dict(torch=torch)
        cls = node(source('mmdet/models/layers/transformer/detr_layers.py', baseline=True),
                   'DetrTransformerEncoderLayer')
        execute([next(n for n in cls.body if getattr(n, 'name', None) == 'forward')], env)
        base = TextLayer()
        current = copy.deepcopy(base)
        before = set(current.state_dict())
        query = torch.randn(2, 5, 8)
        position = torch.randn_like(query)
        mask = torch.eye(5, dtype=torch.bool).logical_not().repeat(4, 1, 1)
        a = env['forward'](base, query, position, None, attn_mask=mask)
        b = current(query, position, None, attn_mask=mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a[..., 0].sum().backward()
        b[..., 0].sum().backward()
        self.assertEqual(before, set(current.state_dict()))
        for p, q in zip(base.parameters(), current.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_disabled_encoder_matches_original_forward_with_unequal_batch_masks(self):
        import subprocess
        from visual_test_support import ROOT
        from support_transformer_fixture import env
        original = subprocess.check_output(
            ['git', 'show', '72bced2:mmdet/models/layers/transformer/grounding_dino_layers.py'],
            cwd=ROOT, encoding='utf8')
        cls = node(original, 'GroundingDinoTransformerEncoder')
        baseline = next(n for n in cls.body if getattr(n, 'name', None) == 'forward')
        namespace = dict(env)
        execute([baseline], namespace)
        encoder = Encoder(dim=256, heads=4, layers=1).eval()
        old = copy.deepcopy(encoder)
        old.forward = namespace['forward'].__get__(old)
        text, _ = text_fixture(2)
        ids = torch.tensor([[101, 10, 11, 119, 12, 119, 102, 0, 0],
                            [101, 10, 119, 12, 119, 102, 0, 0, 0]])
        text['masks'], text['position_ids'] = tokenizer_masks(ids)
        text['text_token_mask'] = ids != 0
        image = torch.randn(2, 4, 256)
        kwargs = dict(query=image, query_pos=torch.zeros_like(image),
                      key_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
                      spatial_shapes=torch.tensor([[2, 2]]), level_start_index=torch.tensor([0]),
                      valid_ratios=torch.ones(2, 1, 2), memory_text=text['embedded'],
                      text_attention_mask=~text['text_token_mask'],
                      position_ids=text['position_ids'], text_self_attention_masks=text['masks'])
        for a, b in zip(old(**kwargs), encoder(**kwargs)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
