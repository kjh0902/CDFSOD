import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from visual_test_support import visual, stats_module, sample, text_fixture


class SupportModel(visual.SupportVisualMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.support_visual_cfg = dict(class_names=('a', 'b'))
        self.bbox_head = NS(num_classes=2)
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
        out = self.dropout(x[:, :1, ::8, ::8] * self.weight)
        return [out.expand(-1, 256, -1, -1)]


class PrototypeTests(unittest.TestCase):
    def test_real_roi_stride_coordinates_and_gap(self):
        y, x = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing='ij')
        features = (x + 10 * y)[None, None].expand(1, 256, -1, -1)
        pooled, labels = visual.pool_instances(features, [sample([[8, 8, 40, 48]], [1])])
        torch.testing.assert_close(pooled, torch.full((1, 256), 32.5))
        self.assertEqual(labels.tolist(), [1])

    def test_instance_mean_full_support_and_no_grad(self):
        model = SupportModel().train()
        model.backbone.dropout.eval()  # Preserve heterogeneous submodule modes.
        states = [m.training for m in model.backbone.modules()]
        rng = torch.get_rng_state().clone()
        output = model.support_prototypes()
        torch.testing.assert_close(output[:, 0], torch.tensor([4., 5.]))
        self.assertFalse(output.requires_grad)
        self.assertEqual(model.backbone.calls, 2)
        self.assertEqual(states, [m.training for m in model.backbone.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        query = model.backbone(torch.ones(1, 3, 64, 64))[0].sum()
        query.backward()
        self.assertIsNotNone(model.backbone.weight.grad)

    def test_every_iteration_recomputes_eval_caches_and_load_invalidates(self):
        model = SupportModel().train()
        first = model.support_prototypes()
        with torch.no_grad():
            model.backbone.weight.fill_(2.)
        torch.testing.assert_close(model.support_prototypes(), first * 2)
        self.assertEqual(model.backbone.calls, 4)
        model.eval()
        first = model.support_prototypes()
        self.assertIs(first, model.support_prototypes())
        self.assertEqual(model.backbone.calls, 6)
        state = copy.deepcopy(model.state_dict())
        self.assertFalse(any('prototype' in key or 'stats' in key for key in state))
        model.load_state_dict(state)
        model.support_prototypes()
        self.assertEqual(model.backbone.calls, 8)

    def test_missing_class_and_mode_restoration_on_exception(self):
        model = SupportModel().train()
        model._support_loader = model._support_loader[:1]
        model._support_loader[0]['data_samples'][0].gt_instances.labels[:] = 0
        with self.assertRaisesRegex(ValueError, 'missing'):
            model.support_prototypes()
        self.assertTrue(model.backbone.training)
        model._support_loader[0]['data_samples'][0].gt_instances.bboxes[0, 2] = 0
        with self.assertRaisesRegex(ValueError, 'boxes'):
            model.support_prototypes()
        self.assertTrue(model.backbone.training)

    def test_empty_gt_does_not_bias_class_average(self):
        model = SupportModel()
        model._support_loader.insert(0, dict(inputs=torch.ones(1, 3, 64, 64),
                                             data_samples=[sample([], [])]))
        torch.testing.assert_close(model.support_prototypes()[:, 0], torch.tensor([4., 5.]))


class TokenFusionTests(unittest.TestCase):
    def test_positions_masks_mapping_specials_and_query_gradient(self):
        text, maps = text_fixture()
        prototypes = torch.randn(2, 256)
        new, layout = visual.insert_visual_tokens(text, prototypes, maps, 32)
        self.assertEqual(layout['names'], [{1: [1, 2], 2: [5]}])
        self.assertEqual(layout['visual'], [[3, 6]])
        self.assertEqual(new['position_ids'].tolist(), [[0, 0, 1, 2, 3, 0, 1, 2, 0]])
        torch.testing.assert_close(new['embedded'][0, 3], prototypes[0])
        torch.testing.assert_close(new['embedded'][0, 6], prototypes[1])
        self.assertTrue(new['masks'][0, 1:5, 1:5].all())
        self.assertTrue(new['masks'][0, 5:8, 5:8].all())
        self.assertFalse(new['masks'][0, 1:5, 5:8].any())
        self.assertEqual(layout['selection'].nonzero()[:, 1].tolist(), [1, 2, 3, 5, 6])
        new['embedded'].sum().backward()
        self.assertIsNotNone(text['embedded'].grad)
        self.assertFalse(prototypes.requires_grad)

    def test_unequal_lengths_and_padding_remain_masked(self):
        text, maps = text_fixture(2)
        text['text_token_mask'][1, 6:] = False
        maps[1] = {1: [1], 2: [3]}
        new, layout = visual.insert_visual_tokens(text, torch.randn(2, 256), maps, 32)
        self.assertEqual(new['text_token_mask'][1].sum().item(), 8)
        self.assertFalse(new['text_token_mask'][1, -1])
        self.assertFalse(layout['selection'][1, -1])

    def test_invalid_mapping_and_overflow_fail_instead_of_truncation(self):
        text, maps = text_fixture()
        for bad in ([{1: [1]}], [{1: [], 2: [4]}], [{1: [1, 3], 2: [4]}]):
            with self.assertRaises(ValueError):
                visual.insert_visual_tokens(text, torch.randn(2, 256), bad, 32)
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            visual.insert_visual_tokens(text, torch.randn(2, 256), maps, 8)

    def test_exact_score_formula_differs_from_mean_logit_and_max_token(self):
        logits = torch.tensor([[[100., -4., 2., -1., 80., .1, .2, -90.]]], requires_grad=True)
        layout = dict(names=[{1: [1, 2], 2: [5]}], visual=[[3, 6]])
        fused = visual.fuse_token_logits(logits, layout, 8)
        expected = torch.stack([torch.maximum(logits[0, :, [1, 2]].sigmoid().mean(-1),
                                              logits[0, :, 3].sigmoid()),
                                torch.maximum(logits[0, :, 5].sigmoid(), logits[0, :, 6].sigmoid())], -1)
        torch.testing.assert_close(fused[0, :, :2].sigmoid(), expected)
        self.assertFalse(torch.allclose(expected[:, 0], logits[0, :, [1, 2]].mean(-1).sigmoid()))
        self.assertTrue(torch.isneginf(fused[..., 2:]).all())
        F.binary_cross_entropy_with_logits(fused[..., :2], torch.ones_like(fused[..., :2])).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 0, 0].item(), 0.)

    def test_extreme_logits_amp_and_tie_go_to_text(self):
        layout = dict(names=[{1: [0, 1]}], visual=[[2]])
        for values in ([-100., -80., -95.], [80., 100., 85.], [0., 0., 0.]):
            logits = torch.tensor([[values]], requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                fused = visual.fuse_token_logits(logits, layout, 1)
            self.assertTrue(torch.isfinite(fused).all())
            fused.sum().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            if values == [0., 0., 0.]:
                self.assertEqual(logits.grad[0, 0, 2].item(), 0.)
                torch.testing.assert_close(logits.grad[0, 0, :2], torch.tensor([.5, .5]))

    def test_logging_preserves_fusion_gradients_rng_and_optimizer_updates(self):
        torch.manual_seed(18)
        layout = dict(names=[{1: [0, 1], 2: [3]}], visual=[[2, 4]])
        initial = torch.randn(1, 6, 5)
        outputs = []
        for enabled in (False, True):
            parameter = nn.Parameter(initial.clone())
            optimizer = torch.optim.AdamW([parameter], lr=.01)
            stats = stats_module.VisualTokenStats(2, 'cpu') if enabled else None
            rng = torch.get_rng_state().clone()
            result = visual.fuse_token_logits(parameter, layout, 5, stats, dn_count=2)
            loss = F.binary_cross_entropy_with_logits(result[..., :2], torch.ones_like(result[..., :2]))
            loss.backward()
            grad = parameter.grad.clone()
            optimizer.step()
            outputs.append((result.detach(), loss.detach(), grad, parameter.detach().clone()))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            if enabled:
                self.assertEqual(stats.pending[0].shape, (1, 4, 2))
                stats.finish_batch()
                self.assertIsNone(stats.pending)
                self.assertEqual(stats.sums.shape, (9, 2))
                json.dumps(stats.report(('a', 'b')), allow_nan=False)
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
