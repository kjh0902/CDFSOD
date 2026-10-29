import copy
import unittest
from unittest.mock import Mock

import torch

from visual_test_support import ROOT, source
import visual_loss_fixture as fixture


class DetectionLossTests(unittest.TestCase):
    def test_real_decoder_dn_encoder_token_losses_and_gradients(self):
        for classes in (1, 2):
            for empty in (False, True):
                with self.subTest(classes=classes, empty=empty):
                    torch.manual_seed(9)
                    head = fixture.head(classes)
                    hidden = torch.randn(3, 2, 6, 8, requires_grad=True)
                    text = torch.randn(2, 9, 8, requires_grad=True)
                    references = [torch.rand(2, 6, 4) for _ in range(4)]
                    encoder = torch.randn(2, 4, 16)
                    encoder[..., 9:] = float('-inf')
                    encoder.requires_grad_()
                    samples, _ = fixture.data(classes, empty)
                    original = copy.deepcopy(samples)
                    head.assigner.assign = Mock(wraps=head.assigner.assign)
                    losses = head.loss(hidden, references, text,
                                       torch.ones(2, 9, dtype=torch.bool),
                                       encoder, torch.rand(2, 4, 4), samples,
                                       dict(num_denoising_queries=2, num_denoising_groups=1))
                    self.assertEqual(len(losses), 21)
                    self.assertTrue(all(torch.isfinite(v).all() for v in losses.values()))
                    sum(losses.values()).backward()
                    for value in (hidden, text, encoder):
                        self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertEqual(head.assigner.assign.call_count, 8)
                    for a, b in zip(samples, original):
                        torch.testing.assert_close(a.gt_instances.positive_maps, b.gt_instances.positive_maps)
                    self.assertFalse(any('visual' in key or 'region' in key for key in losses))

    def test_head_is_exact_original_acl_no_fusion_or_extra_eval_matching(self):
        import subprocess
        original = subprocess.check_output(
            ['git', 'show', '449e69a:mmdet/models/dense_heads/grounding_dino_head.py'],
            cwd=ROOT, encoding='utf8')
        self.assertEqual(source('mmdet/models/dense_heads/grounding_dino_head.py'), original)


if __name__ == '__main__':
    unittest.main()
