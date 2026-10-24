"""CPU geometry, token weighting and stopped-target gradient regressions."""
import importlib.util
import math
import sys
import unittest
from unittest.mock import patch

import torch

from test_raw_mean_etf_loss import ROOT, package, raw_mean

sys.modules[package.__name__ + '.raw_mean_etf_loss'] = raw_mean
spec = importlib.util.spec_from_file_location(
    package.__name__ + '.batch_shared_mu_orth_token_etf_loss',
    ROOT / 'mmdet/models/losses/batch_shared_mu_orth_token_etf_loss.py')
unified = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified)


def reference(x, maps, target=None, eps=1e-6):
    # Explicit per-image and per-class loops, with no D normalization.
    x = x.float()
    means = torch.stack([torch.stack([x[b, m[c]].mean(0)
                        for c in range(1, len(m) + 1)])
                        for b, m in enumerate(maps)])
    batch_means = means.mean(0)
    mu = batch_means.mean(0)
    s = (batch_means - mu).norm().clamp_min(eps)
    if target is None:
        target = unified.nearest_mu_orthogonal_simplex_etf((batch_means - mu) / s, mu)
    value = sum(sum(sum(((x[b, k] - mu) / s - target[c-1]).square().sum()
                        for k in indices) / len(indices)
                    for c, indices in m.items()) for b, m in enumerate(maps))
    return value / (len(maps) * len(maps[0])), mu, s, target


class ConstrainedSimplexTests(unittest.TestCase):
    def test_geometry_and_optimality_including_rank_deficiency(self):
        torch.manual_seed(8)
        for kind in ['random', 'zero', 'rank_one', 'zero_mu', 'tiny_mu',
                     'subnormal_mu', 'positive_axis', 'negative_axis']:
            with self.subTest(kind=kind):
                c, d = 4, 7
                z = torch.randn(c, d)
                z = z - z.mean(0)
                z = z / z.norm()
                mu = torch.randn(d)
                if kind == 'zero':
                    z.zero_()
                elif kind == 'rank_one':
                    z = torch.arange(c).float()[:, None] * torch.randn(1, d)
                    z = (z - z.mean(0)) / z.norm()
                elif kind == 'zero_mu':
                    mu.zero_()
                elif kind in ['tiny_mu', 'subnormal_mu']:
                    mu *= 1e-30 if kind == 'tiny_mu' else 1e-40
                elif kind.endswith('axis'):
                    mu.zero_()
                    mu[0] = 1 if kind == 'positive_axis' else -1
                z.requires_grad_()
                mu.requires_grad_()
                m = unified.nearest_mu_orthogonal_simplex_etf(z, mu)
                h = (torch.eye(c) - torch.ones(c, c) / c) / (c - 1)
                torch.testing.assert_close(m @ m.T, h, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(m.sum(0), torch.zeros(d), atol=2e-6, rtol=0)
                self.assertAlmostEqual(m.norm().item(), 1., places=5)
                if mu.detach().abs().max().item() != 0:
                    direction = mu.detach().double() / mu.detach().abs().max().double()
                    direction /= direction.norm()
                    torch.testing.assert_close(m.double() @ direction, torch.zeros(c).double(),
                                               atol=2e-6, rtol=0)
                    w = torch.linalg.qr(direction[:, None], mode='complete')[0][:, 1:]
                else:
                    w = torch.eye(d).double()
                # Independent class basis and double-precision optimum oracle.
                h0 = torch.eye(c).double() - torch.ones(c, c).double() / c
                q = torch.linalg.eigh(h0)[1][:, 1:]
                sv = torch.linalg.svdvals(w.T @ z.detach().double().T @ q)
                optimum = z.detach().double().square().sum() + 1 - 2 * sv.sum() / math.sqrt(c-1)
                torch.testing.assert_close((z.detach().double() - m.double()).square().sum(),
                                           optimum, atol=3e-6, rtol=3e-6)
                self.assertFalse(m.requires_grad)
                self.assertIsNone(m.grad_fn)
                self.assertEqual(m.dtype, torch.float32)

    def test_minimal_dimension_and_infeasible_constraint(self):
        z = torch.randn(4, 3)
        m = unified.nearest_mu_orthogonal_simplex_etf(z, torch.zeros(3))
        self.assertEqual(m.shape, (4, 3))
        with self.assertRaisesRegex(ValueError, 'D >= C'):
            unified.nearest_mu_orthogonal_simplex_etf(z, torch.ones(3))
        m = unified.nearest_mu_orthogonal_simplex_etf(torch.randn(4, 4), torch.ones(4))
        torch.testing.assert_close(m @ torch.ones(4), torch.zeros(4), atol=2e-6, rtol=0)

    def test_invalid_geometry_is_rejected(self):
        for z, mu in [(torch.ones(1, 3), torch.ones(3)),
                      (torch.ones(4, 2), torch.ones(2)),
                      (torch.ones(2, 3, 4), torch.ones(4)),
                      (torch.ones(2, 3).long(), torch.ones(3)),
                      (torch.ones(2, 3), torch.ones(1, 3)),
                      (torch.ones(2, 3), torch.ones(3).long()),
                      (torch.full((2, 3), float('nan')), torch.ones(3)),
                      (torch.ones(2, 3), torch.full((3,), float('inf')))]:
            with self.assertRaises(ValueError):
                unified.nearest_mu_orthogonal_simplex_etf(z, mu)


class BatchSharedTokenLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 9, 5, requires_grad=True)
        self.maps = [{1: [1], 2: [2, 3], 3: [5, 6, 7]},
                     {1: [1, 2], 2: [3], 3: [5, 6]}]
        self.mask = torch.ones(2, 9, dtype=torch.bool)

    def test_exact_formula_shared_prototypes_and_detached_target(self):
        original = self.x.detach().clone()
        with patch.object(unified, 'nearest_mu_orthogonal_simplex_etf',
                          wraps=unified.nearest_mu_orthogonal_simplex_etf) as solve:
            value = unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, self.mask)
        solve.assert_called_once()
        expected, mu, s, _ = reference(self.x, self.maps)
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(solve.call_args.args[1], mu)
        self.assertEqual(solve.call_args.args[0].shape, (3, 5))
        self.assertTrue(solve.call_args.args[0].requires_grad)
        torch.testing.assert_close(self.x, original)
        self.assertFalse(torch.allclose(value, torch.stack([
            unified.batch_shared_mu_orth_token_etf_loss(
                self.x[b:b+1], [self.maps[b]], self.mask[b:b+1]) for b in range(2)]).mean()))

    def test_stopped_target_gradient_matches_differentiable_mu_and_scale(self):
        actual = unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, self.mask)
        expected, mu, s, target = reference(self.x, self.maps)
        gradients = torch.autograd.grad(actual, self.x)[0]
        expected_grad = torch.autograd.grad(expected, self.x)[0]
        torch.testing.assert_close(gradients, expected_grad, atol=2e-6, rtol=2e-6)
        frozen_stats = torch.stack([
            ((self.x[b, indices] - mu.detach()) / s.detach() - target[c-1])
            .square().sum(-1).mean() for b, m in enumerate(self.maps)
            for c, indices in m.items()]).mean()
        wrong_grad = torch.autograd.grad(frozen_stats, self.x)[0]
        self.assertFalse(torch.allclose(gradients, wrong_grad))
        for b, mapping in enumerate(self.maps):
            selected = {i for indices in mapping.values() for i in indices}
            self.assertTrue((gradients[b, sorted(selected)].norm(dim=-1) > 0).all())
            other = [i for i in range(self.x.size(1)) if i not in selected]
            torch.testing.assert_close(gradients[b, other], torch.zeros_like(gradients[b, other]))
        self.assertFalse(target.requires_grad)

    def test_within_class_token_spread_penalized_with_unchanged_prototypes(self):
        base = torch.tensor([[[2., 0., 0.], [2., 0., 0.],
                              [0., 1., 0.], [0., 1., 0.],
                              [0., 0., 3.], [0., 0., 3.]]])
        maps = [{1: [0, 1], 2: [2, 3], 3: [4, 5]}]
        mask = torch.ones(1, 6, dtype=torch.bool)
        spread = base.clone()
        delta = torch.tensor([.2, .4, -.3])
        spread[0, 0] += delta
        spread[0, 1] -= delta
        initial, _, s, _ = reference(base, maps)
        changed = unified.batch_shared_mu_orth_token_etf_loss(spread, maps, mask)
        torch.testing.assert_close(changed - initial, delta.square().sum() / (3 * s.square()))

    def test_batch_token_replication_and_class_length_invariance(self):
        initial = unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, self.mask)
        repeated = self.x.repeat(3, 1, 1)
        torch.testing.assert_close(initial, unified.batch_shared_mu_orth_token_etf_loss(
            repeated, self.maps * 3, self.mask.repeat(3, 1)))
        # Replicate one entire class only: it must retain its original weight.
        extra = self.x[:, 2:4]
        # Use identical layouts here, since the other tests cover ragged maps.
        maps = [self.maps[0], self.maps[0]]
        original = unified.batch_shared_mu_orth_token_etf_loss(self.x, maps, self.mask)
        longer = torch.cat([self.x, extra], dim=1)
        longer_maps = [{**m, 2: [2, 3, 9, 10]} for m in maps]
        torch.testing.assert_close(original, unified.batch_shared_mu_orth_token_etf_loss(
            longer, longer_maps, torch.ones(2, 11, dtype=torch.bool)))

    def test_batch_coupling_and_no_target_cache(self):
        targets = []
        solve = unified.nearest_mu_orthogonal_simplex_etf
        def capture(*args):
            result = solve(*args)
            targets.append(result)
            return result
        with patch.object(unified, 'nearest_mu_orthogonal_simplex_etf', side_effect=capture):
            unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, self.mask)
            changed = self.x.detach().clone()
            changed[1, self.maps[1][1]] += torch.tensor([2., -.5, 1., -.4, .3])
            unified.batch_shared_mu_orth_token_etf_loss(changed, self.maps, self.mask)
        self.assertEqual(len(targets), 2)
        self.assertFalse(torch.allclose(targets[0], targets[1]))
        for batch in [1, 4, 2, 7]:
            x = torch.randn(batch, 9, 5, requires_grad=True)
            value = unified.batch_shared_mu_orth_token_etf_loss(
                x, [self.maps[0]] * batch, torch.ones(batch, 9, dtype=torch.bool))
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_collapsed_duplicate_tiny_and_zero_mean_prototypes(self):
        for kind in ['zeros', 'constant', 'tiny', 'duplicates', 'zero_mu']:
            x = torch.randn(2, 6, 4)
            if kind == 'zeros':
                x.zero_()
            elif kind == 'constant':
                x.fill_(1)
            elif kind == 'tiny':
                x *= 1e-30
            elif kind == 'duplicates':
                x[:, 2:4] = x[:, :2]
            else:
                x = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.],
                                   [0., 1., 0., 0.], [0., 1., 0., 0.],
                                   [-1., -1., 0., 0.], [-1., -1., 0., 0.]]]).repeat(2, 1, 1)
            x.requires_grad_()
            maps = [{1: [0, 1], 2: [2, 3], 3: [4, 5]}] * 2
            value = unified.batch_shared_mu_orth_token_etf_loss(
                x, maps, torch.ones(2, 6, dtype=torch.bool))
            value.backward()
            self.assertTrue(torch.isfinite(value), kind)
            self.assertTrue(torch.isfinite(x.grad).all(), kind)

    def test_single_class_zero_without_solver(self):
        with patch.object(unified, 'nearest_mu_orthogonal_simplex_etf', side_effect=AssertionError):
            loss = unified.batch_shared_mu_orth_token_etf_loss(
                self.x, [{1: [1, 2]}] * 2, self.mask)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(self.x.grad, torch.zeros_like(self.x))

    def test_fp32_computation_and_autocast(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x = self.x.detach().to(dtype).requires_grad_()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = unified.batch_shared_mu_orth_token_etf_loss(x, self.maps, self.mask)
            expected = reference(x, self.maps)[0]
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_autocast_keeps_geometry_fp32_and_backpropagates(self):
        x = self.x.detach().cuda().half().requires_grad_()
        mask = self.mask.cuda()
        expected = unified.batch_shared_mu_orth_token_etf_loss(x.float(), self.maps, mask)
        with torch.autocast('cuda', dtype=torch.float16):
            actual = unified.batch_shared_mu_orth_token_etf_loss(x, self.maps, mask)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_maps_masks_inputs_and_epsilon(self):
        for mapping in [{1: [], 2: [2], 3: [3]}, {1: [-1], 2: [2], 3: [3]},
                        {1: [9], 2: [2], 3: [3]}, {1: [1, 1], 2: [2], 3: [3]},
                        {1: [1.5], 2: [2], 3: [3]}, {1: [True], 2: [2], 3: [3]},
                        {0: [1], 1: [2], 2: [3]}, {1: [1], 2: [2]}]:
            with self.assertRaises(ValueError):
                unified.batch_shared_mu_orth_token_etf_loss(self.x, [mapping, self.maps[1]], self.mask)
        padding = self.mask.clone()
        padding[0, 1] = False
        for mask in [padding, self.mask.float(), self.mask[:, :-1]]:
            with self.assertRaises(ValueError):
                unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, mask)
        for eps in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                unified.batch_shared_mu_orth_token_etf_loss(self.x, self.maps, self.mask, eps)
        for x, maps, mask in [(self.x.long(), self.maps, self.mask),
                              (self.x[0], self.maps, self.mask),
                              (self.x[:0], [], self.mask[:0]),
                              (self.x, self.maps[:1], self.mask),
                              (self.x[:, :, :1], self.maps, self.mask)]:
            with self.assertRaises(ValueError):
                unified.batch_shared_mu_orth_token_etf_loss(x, maps, mask)
        x = self.x.detach().clone()
        x[0, 1, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            unified.batch_shared_mu_orth_token_etf_loss(x, self.maps, self.mask)


if __name__ == '__main__':
    unittest.main()
