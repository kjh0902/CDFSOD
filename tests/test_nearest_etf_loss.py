"""Nearest ETF geometry tests; no MMDetection extensions required."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location(
    'nearest_etf_loss', Path(__file__).resolve().parents[1] /
    'mmdet/models/losses/nearest_etf_loss.py')
etf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(etf)


def normalize(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(1e-6)


class NearestETFGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_etf(self, target):
        batch, classes, _ = target.shape
        h = torch.eye(classes, dtype=target.dtype, device=target.device)
        h = (h - 1 / classes) / (classes - 1)
        torch.testing.assert_close(target @ target.transpose(-2, -1),
                                   h.expand(batch, -1, -1))
        torch.testing.assert_close(target.sum(1), torch.zeros_like(target[:, 0]))
        torch.testing.assert_close(target.square().sum((-2, -1)),
                                   target.new_ones(batch))
        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)

    def test_target_geometry_exact_objective_and_official_parameterization(self):
        for classes, dimensions in [(2, 3), (5, 4), (5, 8), (7, 256)]:
            with self.subTest(classes=classes, dimensions=dimensions):
                z = normalize(torch.randn(3, classes, dimensions, dtype=torch.float64))
                target = etf.nearest_simplex_etf(z.requires_grad_())
                self.assert_etf(target)
                actual = (z - target).square().sum((-2, -1))
                # Centering gives one zero singular value when D >= C.
                singular_values = torch.linalg.svdvals(z.detach())
                optimum = z.square().sum((-2, -1)) + 1 - (
                    2 * singular_values.sum(-1) / (classes - 1)**0.5)
                torch.testing.assert_close(actual, optimum)
                if dimensions >= classes:
                    h = torch.eye(classes, dtype=z.dtype) - 1 / classes
                    u, _, vh = torch.linalg.svd(z.detach().transpose(-2, -1) @ h,
                                               full_matrices=False)
                    official_target = ((u @ vh) @ h / (classes - 1)**0.5).transpose(-2, -1)
                    torch.testing.assert_close(actual,
                                               (z - official_target).square().sum((-2, -1)))

    def test_rotated_etf_has_zero_loss(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        target = etf.nearest_simplex_etf(normalize(x))
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=x.dtype))
        self.assertLess(etf.nearest_etf_loss(target @ rotation).item(), 1e-25)

    def test_invariance_batch_reduction_and_whole_matrix_normalization(self):
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        base = etf.nearest_etf_loss(x)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=x.dtype))
        shifted = x + torch.randn(3, 1, 8, dtype=x.dtype)
        scaled = x * torch.tensor([0.2, 3., 12.], dtype=x.dtype)[:, None, None]
        for transformed in [shifted, scaled, x @ rotation, x[:, [3, 0, 4, 1, 2]]]:
            torch.testing.assert_close(base, etf.nearest_etf_loss(transformed))
        individual = torch.stack([etf.nearest_etf_loss(row[None]) for row in x])
        torch.testing.assert_close(base, individual.mean())
        with patch.object(etf, 'nearest_simplex_etf', wraps=etf.nearest_simplex_etf) as solve:
            etf.nearest_etf_loss(x)
        z = solve.call_args.args[0]
        torch.testing.assert_close(z, normalize(x))
        self.assertGreater(torch.linalg.vector_norm(z, dim=-1).std().item(), 0.01)

    def test_only_normalized_branch_has_gradient(self):
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        z = normalize(x)
        target = etf.nearest_simplex_etf(z)
        self.assert_etf(target)
        expected = torch.autograd.grad((z - target).square().sum((-2, -1)).mean(), x)[0]
        actual = torch.autograd.grad(etf.nearest_etf_loss(x), x)[0]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(actual.abs().sum().item(), 0)
        self.assertTrue(torch.autograd.gradcheck(etf.nearest_etf_loss, (x,)))

    def test_rank_deficient_duplicate_and_collapsed_inputs(self):
        rank_one = torch.arange(5, dtype=torch.float64)[None, :, None].expand(2, 5, 8).clone()
        duplicate = torch.randn(2, 5, 8, dtype=torch.float64)
        duplicate[:, 1] = duplicate[:, 0]
        for x in [rank_one, duplicate, torch.ones_like(rank_one),
                  torch.zeros_like(rank_one), rank_one * 1e-10]:
            x.requires_grad_()
            target = etf.nearest_simplex_etf(normalize(x))
            self.assert_etf(target)
            loss = etf.nearest_etf_loss(x)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(etf.nearest_etf_loss(torch.ones_like(rank_one)).item(), 1.)

    def test_input_validation(self):
        for shape in [(5, 8), (0, 5, 8), (2, 1, 8), (2, 5, 3)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                etf.nearest_etf_loss(torch.zeros(shape))
        with self.assertRaises(ValueError):
            etf.nearest_etf_loss(torch.zeros(2, 3, 4, dtype=torch.long))
        for eps in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                etf.nearest_etf_loss(torch.randn(2, 3, 4), eps=eps)

    def check_low_precision(self, device, dtype):
        x = torch.randn(2, 5, 8, device=device, dtype=dtype, requires_grad=True)
        expected = etf.nearest_etf_loss(x.float())
        with torch.autocast(device_type=device, dtype=dtype):
            actual = etf.nearest_etf_loss(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cpu_low_precision_and_autocast(self):
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.check_low_precision('cpu', dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_autocast(self):
        self.check_low_precision('cuda', torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_low_precision('cuda', torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
