import copy
import unittest
from unittest.mock import Mock

import torch

from visual_test_support import stats_module
import visual_loss_fixture as fixture


class DetectionLossTests(unittest.TestCase):
    def test_all_decoder_dn_encoder_losses_and_logging_are_identical(self):
        for classes in (1, 2):
            for empty in (False, True):
                with self.subTest(classes=classes, empty=empty):
                    torch.manual_seed(9)
                    base = fixture.head(classes)
                    model = copy.deepcopy(base)
                    hidden = torch.randn(3, 2, 6, 8)
                    text = torch.randn(2, 9, 8)
                    references = [torch.rand(2, 6, 4) for _ in range(4)]
                    encoder = torch.randn(2, 4, 16)
                    encoder[..., classes:] = float('-inf')
                    coords = torch.rand(2, 4, 4)
                    samples, layout = fixture.data(classes, empty)
                    dn_meta = dict(num_denoising_queries=2, num_denoising_groups=1)
                    outputs = []
                    for h, enabled in ((base, False), (model, True)):
                        stats = stats_module.VisualTokenStats(classes, 'cpu') if enabled else None
                        h._visual_stats = stats
                        h.assigner.assign = Mock(wraps=h.assigner.assign)
                        x = hidden.clone().requires_grad_()
                        t = text.clone().requires_grad_()
                        e = encoder.clone().requires_grad_()
                        losses = h.loss(x, references, t, torch.ones(2, 9, dtype=torch.bool),
                                        e, coords, copy.deepcopy(samples), dn_meta, layout)
                        self.assertEqual(len(losses), 21)  # 3 decoder + encoder + 3 DN, each cls/L1/GIoU.
                        self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                        sum(losses.values()).backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        self.assertTrue(torch.isfinite(t.grad).all())
                        self.assertTrue(torch.isfinite(e.grad).all())
                        self.assertEqual(h.assigner.assign.call_count, 8)  # 3 layers + encoder, two images.
                        outputs.append((losses, x.grad, t.grad, e.grad))
                        if enabled:
                            self.assertIsNone(stats.pending)
                            report = stats.report(tuple(str(c) for c in range(classes)))['overall']
                            self.assertEqual(report['win_all']['count'], 8)  # no DN, no auxiliary layers.
                            self.assertEqual(report['win_gt']['count'], 0 if empty else 1)
                            self.assertEqual(report['win_background']['count'], 8 if empty else 7)
                    for key in outputs[0][0]:
                        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=0, atol=0)
                    for a, b in zip(outputs[0][1:], outputs[1][1:]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_validation_matching_uses_original_pixels_and_does_not_mutate_gt(self):
        model = fixture.head()
        samples, _ = fixture.data()
        samples[0].metainfo.update(ori_shape=(64, 96), img_shape=(32, 48))
        original = copy.deepcopy(samples[0].gt_instances.bboxes)
        logits = torch.randn(1, 2, 4, 16)
        boxes = torch.rand(1, 2, 4, 4)
        stats = stats_module.VisualTokenStats(2, 'cpu')
        stats.scores(torch.zeros(2, 4, 2), torch.ones(2, 4, 2), torch.ones(2, 4, 2))
        model._visual_stats = stats
        model.assigner.assign = Mock(wraps=model.assigner.assign)
        model._visual_validation_matches((logits, boxes), samples)
        self.assertEqual(model.assigner.assign.call_count, 2)
        first = model.assigner.assign.call_args_list[0].kwargs
        self.assertEqual(first['img_meta']['img_shape'], (64, 96))
        expected = fixture.env['bbox_cxcywh_to_xyxy'](boxes[-1, 0]) * boxes.new_tensor([96, 64, 96, 64])
        torch.testing.assert_close(first['pred_instances'].bboxes, expected)
        torch.testing.assert_close(samples[0].gt_instances.bboxes, original)
        self.assertIsNone(stats.pending)


if __name__ == '__main__':
    unittest.main()
