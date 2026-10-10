"""FP32 formula, shared prototypes, edge cases and non-detached gradients."""
import unittest
from unittest.mock import patch

import torch

from test_raw_mean_etf_loss import orth, raw_mean


def reference(x, eps=1e-6):
    x = x.float()
    values = []
    for row in x:
        mu = row.mean(0)
        for prototype in row:
            residual = prototype - mu
            values.append((mu.dot(residual) / (mu.norm() * residual.norm() + eps)).square())
    return torch.stack(values).mean()


class MuOrthogonalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_formula_batch_reduction_and_full_gradient(self):
        x = torch.randn(3, 6, 8, requires_grad=True)
        actual = orth.mu_orthogonality_loss(x)
        expected = reference(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, x)[0],
                                   torch.autograd.grad(expected, x, retain_graph=True)[0])
        individual = [orth.mu_orthogonality_loss(row[None]) for row in x]
        torch.testing.assert_close(actual, torch.stack(individual).mean())
        # Both mu and residuals must participate in the chain rule.
        mu = x.mean(1, keepdim=True)
        residuals = x - mu
        for m, r in [(mu.detach(), residuals), (mu, residuals.detach())]:
            detached_loss = ((m * r).sum(-1) / (m.norm(dim=-1) * r.norm(dim=-1) + 1e-6)).square().mean()
            self.assertFalse(torch.allclose(torch.autograd.grad(detached_loss, x, retain_graph=True)[0],
                                            torch.autograd.grad(expected, x, retain_graph=True)[0]))

    def test_fp32_under_autocast_and_finite_low_precision_backward(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x = torch.randn(2, 4, 8).to(dtype).requires_grad_()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = orth.mu_orthogonality_loss(x)
            self.assertEqual(loss.dtype, torch.float32)
            torch.testing.assert_close(loss, reference(x))
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_mean_zero_residual_single_class_and_tiny_values(self):
        for x in [torch.zeros(2, 3, 4), torch.ones(2, 3, 4),
                  torch.tensor([[[1., 0.], [-1., 0.]]]), torch.randn(2, 1, 4),
                  torch.randn(2, 3, 4) * 1e-10]:
            x.requires_grad_()
            loss = orth.mu_orthogonality_loss(x)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
            if x.size(1) == 1 or torch.equal(x, torch.ones_like(x)) or x.norm() == 0:
                self.assertEqual(loss.item(), 0.)
                torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_known_orthogonal_and_parallel_geometry(self):
        orthogonal = torch.tensor([[[1., 2.], [1., -2.]]])
        parallel = torch.tensor([[[3., 0.], [1., 0.]]])
        self.assertEqual(orth.mu_orthogonality_loss(orthogonal).item(), 0.)
        self.assertGreater(orth.mu_orthogonality_loss(parallel).item(), .99999)

    def test_etf_and_orth_share_one_raw_mean_tensor(self):
        tokens = torch.randn(2, 9, 8, requires_grad=True)
        maps = [{1: [1, 2], 2: [4], 3: [6, 7]}] * 2
        mask = torch.ones(2, 9, dtype=torch.bool)
        with patch.object(raw_mean, '_raw_mean_prototypes', wraps=raw_mean._raw_mean_prototypes) as pool, \
             patch.object(raw_mean, 'nearest_etf_loss', wraps=raw_mean.nearest_etf_loss) as etf, \
             patch.object(raw_mean, 'mu_orthogonality_loss', wraps=orth.mu_orthogonality_loss) as mu:
            losses = raw_mean.raw_mean_geometry_losses(tokens, maps, mask, etf_weight=.2, orth_weight=.7)
        pool.assert_called_once()
        self.assertIs(etf.call_args.args[0], mu.call_args.args[0])
        torch.testing.assert_close(losses['orth'], .7 * reference(mu.call_args.args[0]))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        torch.testing.assert_close(tokens.grad[:, [0, 3, 5, 8]], torch.zeros(2, 4, 8))

    def test_zero_weights_skip_pooling_and_individual_objectives(self):
        with patch.object(raw_mean, '_raw_mean_prototypes', side_effect=AssertionError('pool called')):
            self.assertEqual(raw_mean.raw_mean_geometry_losses(None, None, None,
                             etf_weight=0, orth_weight=0), {})
        tokens = torch.randn(1, 3, 2)
        maps, mask = [{1: [0], 2: [1, 2]}], torch.ones(1, 3, dtype=torch.bool)
        for etf_weight, orth_weight, disabled in [(0, 1, 'nearest_etf_loss'),
                                                (1, 0, 'mu_orthogonality_loss')]:
            with patch.object(raw_mean, disabled, side_effect=AssertionError('disabled loss called')):
                losses = raw_mean.raw_mean_geometry_losses(tokens, maps, mask,
                            etf_weight=etf_weight, orth_weight=orth_weight)
            self.assertEqual(set(losses), {'orth'} if orth_weight else {'etf'})

    def test_validation(self):
        for value in [torch.ones(2, 3), torch.ones(2, 3, 4, dtype=torch.long),
                      torch.ones(0, 3, 4), torch.ones(2, 0, 4), torch.ones(2, 3, 0)]:
            with self.assertRaises(ValueError):
                orth.mu_orthogonality_loss(value)
        for eps in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                orth.mu_orthogonality_loss(torch.ones(2, 3, 4), eps)


if __name__ == '__main__':
    unittest.main()
