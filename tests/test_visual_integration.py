import copy
import unittest
from unittest.mock import Mock

import torch

from visual_test_support import Instances, stats_module, visual
import visual_detector_fixture as fixture


def enable(model):
    model.support_visual_cfg = dict(class_names=('red fox', 'blue', 'green'))
    model.support_prototypes = Mock(return_value=torch.randn(3, 8))


class DetectorIntegrationTests(unittest.TestCase):
    def test_disabled_baseline_bitwise_loss_outputs_gradient_rng_and_serial_queries(self):
        for mode in ('shared', 'different', 'explicit'):
            for empty in (False, True):
                with self.subTest(mode=mode, empty=empty):
                    torch.manual_seed(42)
                    base, _ = fixture.build(baseline=True)
                    current, _ = fixture.build()
                    current.load_state_dict(base.state_dict())
                    inputs = torch.randn(2, 20, 8)
                    rng = torch.get_rng_state()
                    a = base.loss(inputs, fixture.samples(mode, empty))
                    after = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    b = current.loss(inputs, fixture.samples(mode, empty))
                    fixture.assert_same(self, a, b)
                    fixture.assert_same(self, base.bbox_head.outputs, current.bbox_head.outputs)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    sum(a.values()).backward()
                    sum(b.values()).backward()
                    for (_, p), (_, q) in zip(base.named_parameters(), current.named_parameters()):
                        if p.grad is None:
                            self.assertIsNone(q.grad)
                        else:
                            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                    for m in (base, current):
                        self.assertEqual(m.dn_query_generator.calls, 1)
                        for i in range(1, 3):
                            torch.testing.assert_close(m.decoder.layers[i].inputs[0],
                                                       m.decoder.layers[i - 1].outputs[0])

    def test_visual_training_full_classes_targets_and_backprop(self):
        for mode in ('shared', 'explicit'):
            for empty in (False, True):
                with self.subTest(mode=mode, empty=empty):
                    model, _ = fixture.build()
                    enable(model)
                    samples = fixture.samples(mode, empty)
                    losses = model.loss(torch.randn(2, 20, 8), samples)
                    model.support_prototypes.assert_called_once()
                    self.assertFalse(any('region' in k or 'visual' in k for k in losses))
                    for sample in samples:
                        targets = sample.gt_instances.positive_maps
                        self.assertEqual(targets.shape, (len(sample.gt_instances.labels), 32))
                        if targets.numel():
                            self.assertTrue((targets.sum(-1) == 1).all())
                            self.assertEqual(targets.argmax(-1).tolist(), sample.gt_instances.labels.tolist())
                        self.assertEqual(sample.gt_instances.text_token_mask.shape[-1], 3)
                    scores = model.bbox_head.outputs[0]
                    self.assertTrue(torch.isneginf(scores[..., 3:]).all())
                    self.assertTrue(torch.isfinite(scores[..., :3]).all())
                    sum(losses.values()).backward()
                    self.assertIsNotNone(model.encoder.visual.weight.grad)
                    self.assertIsNotNone(model.language_model.embedding.weight.grad)
                    self.assertEqual(model.dn_query_generator.calls, 1)

    def test_logging_full_detector_optimizer_rng_and_forward_counts(self):
        torch.manual_seed(17)
        original, _ = fixture.build()
        model, _ = fixture.build()
        model.load_state_dict(original.state_dict())
        prototype = torch.randn(3, 8)
        inputs = torch.randn(2, 20, 8)
        rng = torch.get_rng_state().clone()
        results = []
        for m, logging in ((original, False), (model, True)):
            enable(m)
            m.support_prototypes.return_value = prototype
            m.language_model.forward = Mock(wraps=m.language_model.forward)
            m.encoder.forward = Mock(wraps=m.encoder.forward)
            if logging:
                stats = stats_module.VisualTokenStats(3, 'cpu')
                m._visual_stats = m.bbox_head._visual_stats = stats
            torch.set_rng_state(rng)
            optimizer = torch.optim.SGD(m.parameters(), lr=.01)
            losses = m.loss(inputs, fixture.samples())
            sum(losses.values()).backward()
            optimizer.step()
            results.append((copy.deepcopy(m.state_dict()), torch.get_rng_state().clone(), losses))
            self.assertEqual(m.language_model.forward.call_count, 1)
            self.assertEqual(m.encoder.forward.call_count, 1)
            self.assertEqual(m.dn_query_generator.calls, 1)
            m.support_prototypes.assert_called_once()
            if logging:
                self.assertEqual(stats.report(('a', 'b', 'c'))['overall']['win_all']['count'], 6)
                stats.finish_batch()
        fixture.assert_same(self, results[0], results[1])

    def test_class_order_and_overlong_prompt_fail(self):
        model, _ = fixture.build()
        enable(model)
        with self.assertRaisesRegex(ValueError, 'order'):
            model.loss(torch.randn(2, 20, 8), fixture.samples('different'))
        model.language_model.max_tokens = 8
        with self.assertRaisesRegex(ValueError, 'limit'):
            model.loss(torch.randn(2, 20, 8), fixture.samples())

    def test_encoder_selects_raw_token_max_not_special_or_fused_max(self):
        model, _ = fixture.build()
        model.eval()
        model.num_queries = 1
        # Candidate 0 wins only via CLS; candidate 1 wins via a name subtoken.
        # Its averaged text probability loses to candidate 2's visual score.
        raw = torch.tensor([[[100., -5., -5., -5., -5.],
                             [-2., 8., -8., -4., -5.],
                             [-2., -3., -3., 4., -5.]]])
        class Branch(torch.nn.Module):
            max_text_len = 5
            def forward(self, *args):
                return raw
        model.bbox_head.cls_branches[-1] = Branch()
        model.gen_encoder_output_proposals = lambda *args: (
            torch.randn(1, 3, 8), torch.tensor([[[0., 0., 0., 0.],
                                              [1., 1., 1., 1.], [2., 2., 2., 2.]]]))
        torch.nn.init.zeros_(model.bbox_head.reg_branches[-1].weight)
        torch.nn.init.zeros_(model.bbox_head.reg_branches[-1].bias)
        layout = dict(names=[{1: [1, 2]}], visual=[[3]],
                      selection=torch.tensor([[False, True, True, True, False]]))
        stats = stats_module.VisualTokenStats(1, 'cpu')
        model._visual_stats = stats
        decoder, _ = model.pre_decoder(torch.randn(1, 3, 8), None,
                                       torch.tensor([[1, 3]]), torch.randn(1, 5, 8),
                                       torch.ones(1, 5, dtype=torch.bool), visual_layout=layout)
        torch.testing.assert_close(decoder['reference_points'], torch.ones(1, 1, 4).sigmoid())
        report = stats.report(('a',))['overall']['selection_visual']
        self.assertEqual(report, dict(value=0., count=1))

    def test_actual_predict_uses_same_fused_class_scores(self):
        env = dict(fixture.ENV, InstanceData=Instances)
        for rel, names in [
            ('mmdet/structures/bbox/transforms.py', ['bbox_cxcywh_to_xyxy']),
            ('mmdet/models/dense_heads/atss_vlfusion_head.py', ['convert_grounding_to_cls_scores'])]:
            fixture.execute([fixture.node(fixture.source(rel), name) for name in names], env)
        # Bind actual postprocessing only for this fixture instance's class.
        class PredictionHead(fixture.Head):
            pass
        for name in ('predict', 'predict_by_feat', '_predict_by_feat_single'):
            fixture.execute([fixture.node(fixture.ast.unparse(fixture.node(
                fixture.source(fixture.HEAD), 'GroundingDINOHead')), name)], env)
            setattr(PredictionHead, name, env[name])
        head = PredictionHead()
        head.test_cfg = dict(max_per_img=3)
        head.max_text_len = 32
        model, _ = fixture.build()
        model.bbox_head = head
        model.test_cfg = dict(max_per_img=3)
        enable(model)
        model.eval()
        samples = fixture.samples()
        for s in samples:
            s.custom_entities = True
        result = model.predict(torch.randn(2, 20, 8), samples)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(len(s.pred_instances.labels) == 3 for s in result))
        self.assertTrue(all((s.pred_instances.labels < 3).all() for s in result))
        model.support_prototypes.assert_called_once()


if __name__ == '__main__':
    unittest.main()
