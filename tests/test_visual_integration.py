import copy
import unittest
from unittest.mock import Mock

import torch

from visual_test_support import Instances, stats_module
from test_support_visual import memory_fixture
import visual_detector_fixture as fixture


def enable(model, memory=None):
    model.support_visual_cfg = dict(class_names=('red fox', 'blue', 'green'))
    model.support_memory = Mock(return_value=memory or memory_fixture((3, 1, 2), channels=8))


class DetectorIntegrationTests(unittest.TestCase):
    def test_memory_through_actual_detection_losses_and_backward(self):
        import visual_loss_fixture as loss_fixture
        model, _ = fixture.build(real_encoder=True)
        model.bbox_head = loss_fixture.head(classes=3)
        memory = memory_fixture((3, 1, 2), channels=8, requires_grad=True)
        enable(model, memory)

        def dn(samples):
            batch = len(samples)
            return (model.dn_query_generator.label_embedding.weight[None, :1].expand(batch, 4, -1),
                    torch.randn(batch, 4, 4), torch.zeros(7, 7, dtype=torch.bool),
                    dict(num_denoising_queries=4, num_denoising_groups=1))

        model.dn_query_generator.forward = dn
        model.bbox_head.assigner.assign = Mock(wraps=model.bbox_head.assigner.assign)
        image = torch.randn(2, 20, 8, requires_grad=True)
        losses = model.loss(image, fixture.samples())
        self.assertEqual(len(losses), 21)
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = {n: p.detach().clone() for n, p in model.encoder.support_layers.named_parameters()}
        sum(losses.values()).backward()
        for parameter in model.encoder.support_layers.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        optimizer.step()
        for n, p in model.encoder.support_layers.named_parameters():
            self.assertFalse(torch.equal(before[n], p), n)
        self.assertIsNone(memory['tokens'].grad)
        for gradient in (image.grad, model.language_model.embedding.weight.grad,
                         model.encoder.text_layers[0].self_attn.attn.in_proj_weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)
        self.assertEqual(model.bbox_head.assigner.assign.call_count, 8)

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


    def test_memory_training_keeps_native_text_targets_and_token_logits(self):
        for mode in ('shared', 'explicit'):
            for empty in (False, True):
                with self.subTest(mode=mode, empty=empty):
                    model, _ = fixture.build(real_encoder=True)
                    enable(model)
                    samples = fixture.samples(mode, empty)
                    losses = model.loss(torch.randn(2, 20, 8), samples)
                    model.support_memory.assert_called_once()
                    for item in samples:
                        target = item.gt_instances.positive_maps
                        self.assertEqual(target.shape, (len(item.gt_instances), 32))
                        if target.numel():
                            expected = [1 if c == 0 else 4 for c in item.gt_instances.labels.tolist()]
                            self.assertEqual(target.argmax(-1).tolist(), expected)
                            if item.gt_instances.labels[0] == 0:
                                self.assertEqual(target[0, 2].item(), 1.)
                        self.assertEqual(item.gt_instances.text_token_mask.shape[-1], 9)
                    scores = model.bbox_head.outputs[0]
                    self.assertTrue(torch.isneginf(scores[..., 9:]).all())
                    self.assertTrue(torch.isfinite(scores[..., :9]).all())
                    self.assertEqual(model.bbox_head.inputs['memory_text'].shape, (2, 9, 8))
                    self.assertFalse(any('support' in k or 'visual' in k for k in model.bbox_head.inputs))
                    sum(losses.values()).backward()
                    self.assertIsNotNone(model.encoder.text_layers[0].self_attn.attn.in_proj_weight.grad)
                    self.assertIsNotNone(model.language_model.embedding.weight.grad)
                    self.assertEqual(model.dn_query_generator.calls, 1)

    def test_logging_preserves_detector_loss_gradients_optimizer_rng_and_forward_counts(self):
        torch.manual_seed(17)
        original, _ = fixture.build(real_encoder=True)
        model, _ = fixture.build(real_encoder=True)
        model.load_state_dict(original.state_dict())
        memory = memory_fixture((3, 1, 2), channels=8)
        inputs = torch.randn(2, 20, 8)
        rng = torch.get_rng_state().clone()
        results = []
        for m, logging in ((original, False), (model, True)):
            enable(m, memory)
            m.language_model.forward = Mock(wraps=m.language_model.forward)
            m.encoder.forward = Mock(wraps=m.encoder.forward)
            if logging:
                stats = stats_module.VisualTokenStats(3, 'cpu')
                m._visual_stats = stats
            torch.set_rng_state(rng)
            optimizer = torch.optim.SGD(m.parameters(), lr=.01)
            losses = m.loss(inputs, fixture.samples())
            sum(losses.values()).backward()
            grads = [p.grad.detach().clone() if p.grad is not None else None for p in m.parameters()]
            optimizer.step()
            results.append((copy.deepcopy(m.state_dict()), torch.get_rng_state().clone(), losses, grads))
            self.assertEqual(m.language_model.forward.call_count, 1)
            self.assertEqual(m.encoder.forward.call_count, 1)
            self.assertEqual(m.dn_query_generator.calls, 1)
            m.support_memory.assert_called_once()
            if logging:
                report = stats.report(('a', 'b', 'c'))
                self.assertEqual(report['overall']['support_token_count'], 6)
                self.assertEqual(report['overall']['text_norm']['count'], 8)
                self.assertEqual(report['overall']['enhanced_text_norm']['count'], 8)
                stats.finish_batch()
        fixture.assert_same(self, results[0], results[1])

    def test_class_order_text_overflow_and_memory_not_consuming_text_limit(self):
        model, _ = fixture.build(real_encoder=True)
        enable(model, memory_fixture((10, 20, 30), channels=8))
        model.language_model.max_tokens = 9  # Exact text length, even with 60 support objects.
        model.loss(torch.randn(2, 20, 8), fixture.samples())
        with self.assertRaisesRegex(ValueError, 'order'):
            model.loss(torch.randn(2, 20, 8), fixture.samples('different'))
        model.language_model.max_tokens = 8
        with self.assertRaisesRegex(ValueError, 'limit'):
            model.loss(torch.randn(2, 20, 8), fixture.samples())

    def test_encoder_selection_restores_acl_text_token_max(self):
        model, _ = fixture.build()
        model.eval()
        model.num_queries = 1
        raw = torch.tensor([[[100., -5., -5., -5., -5.],
                             [-2., 8., -8., -4., -5.], [-2., -3., -3., 4., -5.]]])
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
        decoder, _ = model.pre_decoder(torch.randn(1, 3, 8), None,
                                       torch.tensor([[1, 3]]), torch.randn(1, 5, 8),
                                       torch.ones(1, 5, dtype=torch.bool))
        torch.testing.assert_close(decoder['reference_points'], torch.zeros(1, 1, 4).sigmoid())
        self.assertEqual(decoder['memory_text'].shape[1], 5)

    def test_predict_uses_original_name_token_maps_and_text_only_decoder(self):
        env = dict(fixture.ENV, InstanceData=Instances)
        for rel, names in [
            ('mmdet/structures/bbox/transforms.py', ['bbox_cxcywh_to_xyxy']),
            ('mmdet/models/dense_heads/atss_vlfusion_head.py', ['convert_grounding_to_cls_scores'])]:
            fixture.execute([fixture.node(fixture.source(rel), name) for name in names], env)
        class PredictionHead(fixture.Head):
            pass
        for name in ('predict', 'predict_by_feat', '_predict_by_feat_single'):
            fixture.execute([fixture.node(fixture.ast.unparse(fixture.node(
                fixture.source(fixture.HEAD), 'GroundingDINOHead')), name)], env)
            setattr(PredictionHead, name, env[name])
        head = PredictionHead()
        head.test_cfg = dict(max_per_img=3)
        head.max_text_len = 32
        model, _ = fixture.build(real_encoder=True)
        model.bbox_head = head
        model.test_cfg = dict(max_per_img=3)
        enable(model)
        model.eval()
        samples = fixture.samples()
        for item in samples:
            item.custom_entities = True
        lengths = []
        handles = [layer.register_forward_pre_hook(
            lambda module, args, kwargs: lengths.append(kwargs['memory_text'].shape[1]),
            with_kwargs=True) for layer in model.decoder.layers]
        result = model.predict(torch.randn(2, 20, 8), samples)
        self.assertEqual(lengths, [9, 9, 9])
        self.assertEqual(result[0].token_positive_map, {1: [1, 2], 2: [4], 3: [6]})
        self.assertEqual(len(result), 2)
        self.assertTrue(all(len(item.pred_instances.labels) == 3 for item in result))
        self.assertTrue(all((item.pred_instances.labels < 3).all() for item in result))
        model.support_memory.assert_called_once()
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
