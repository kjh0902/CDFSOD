"""Raw mean geometry checks, runnable with CPU PyTorch alone."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_nearest_etf_loss import etf

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('_acl_loss_tests')
package.__path__ = []
sys.modules[package.__name__] = package
sys.modules[package.__name__ + '.nearest_etf_loss'] = etf
spec = importlib.util.spec_from_file_location(
    package.__name__ + '.raw_mean_etf_loss',
    ROOT / 'mmdet/models/losses/raw_mean_etf_loss.py')
raw_mean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw_mean)


class RawMeanGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 9, 4, dtype=torch.float64, requires_grad=True)
        self.maps = [{1: [1], 2: [2, 3], 3: [5, 6, 7]} for _ in range(2)]
        self.mask = torch.ones(2, 9, dtype=torch.bool)

    def test_exact_raw_mean_without_token_or_class_normalization(self):
        actual = raw_mean._raw_mean_prototypes(self.x, self.maps, self.mask)
        expected = []
        for row, mapping in zip(self.x, self.maps):
            classes = []
            for indices in mapping.values():
                tokens = row[indices]
                classes.append(sum(tokens.unbind()) / len(indices))
            expected.append(torch.stack(classes))
        self.assertEqual(actual.shape, (2, 3, 4))
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual, torch.stack(expected))
        self.assertGreater(actual.norm(dim=-1).std().item(), 0.01)
        with patch.object(raw_mean, 'nearest_etf_loss', wraps=etf.nearest_etf_loss) as loss:
            result = raw_mean.raw_mean_etf_loss(self.x, self.maps, self.mask, eps=1e-5)
        loss.assert_called_once()
        self.assertEqual(loss.call_args.kwargs, {'eps': 1e-5})
        torch.testing.assert_close(loss.call_args.args[0], actual)
        torch.testing.assert_close(result, etf.nearest_etf_loss(actual, eps=1e-5))
        reversed_maps = [{c: indices[::-1] for c, indices in m.items()} for m in self.maps]
        torch.testing.assert_close(actual, raw_mean._raw_mean_prototypes(
            self.x, reversed_maps, self.mask))

    def test_opposite_tokens_cancel_and_zero_token_is_finite(self):
        x = torch.tensor([[[1., 0.], [-1., 0.], [0., 0.]]], requires_grad=True)
        maps = [{1: [0, 1], 2: [2]}]
        mask = torch.ones(1, 3, dtype=torch.bool)
        result = raw_mean._raw_mean_prototypes(x, maps, mask)
        torch.testing.assert_close(result, torch.zeros(1, 2, 2))
        loss = raw_mean.raw_mean_etf_loss(x, maps, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_raw_magnitudes_and_per_image_prototypes_are_preserved(self):
        x = torch.tensor([[[6., 0.], [0., 2.], [0., 9.]],
                          [[2., 0.], [0., 8.], [0., 3.]]])
        maps = [{1: [0, 1], 2: [2]} for _ in range(2)]
        mask = torch.ones(2, 3, dtype=torch.bool)
        original = x.clone()
        expected = torch.tensor([[[3., 1.], [0., 9.]],
                                 [[1., 4.], [0., 3.]]])
        actual = raw_mean._raw_mean_prototypes(x, maps, mask)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x, original)
        x[0] *= 5
        changed = raw_mean._raw_mean_prototypes(x, maps, mask)
        torch.testing.assert_close(changed[0], expected[0] * 5)
        torch.testing.assert_close(changed[1], expected[1])

    def test_minimum_dimension_and_second_order_only_dimension(self):
        # C=3 accepts D=2, but C=4 rejects D=2 even though D*D >= C-1.
        x = self.x[:, :, :2]
        self.assertTrue(torch.isfinite(raw_mean.raw_mean_etf_loss(
            x, self.maps, self.mask)))
        maps = [{**mapping, 4: [8]} for mapping in self.maps]
        with self.assertRaisesRegex(ValueError, 'D >= C - 1'):
            raw_mean.raw_mean_etf_loss(x, maps, self.mask)

    def test_batch_independence_and_gradients_only_for_selected_tokens(self):
        loss = raw_mean.raw_mean_etf_loss(self.x, self.maps, self.mask)
        individual = [raw_mean.raw_mean_etf_loss(
            self.x[b:b + 1], [self.maps[b]], self.mask[b:b + 1]) for b in range(2)]
        torch.testing.assert_close(loss, torch.stack(individual).mean())
        grad = torch.autograd.grad(loss, self.x)[0]
        single_grad = torch.autograd.grad(individual[0], self.x)[0]
        torch.testing.assert_close(grad[0], single_grad[0] / 2)
        torch.testing.assert_close(grad[:, [0, 4, 8]], torch.zeros_like(grad[:, [0, 4, 8]]))
        self.assertTrue((grad[:, [1, 2, 3, 5, 6, 7]].norm(dim=-1) > 0).all())
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: raw_mean.raw_mean_etf_loss(x, self.maps, self.mask), (self.x,)))

    def test_low_precision_autocast(self):
        for dtype in [torch.float16, torch.bfloat16]:
            x = self.x.detach().to(dtype).requires_grad_()
            expected = raw_mean.raw_mean_etf_loss(x.float(), self.maps, self.mask)
            with torch.autocast('cpu', dtype=dtype):
                actual = raw_mean.raw_mean_etf_loss(x, self.maps, self.mask)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_duplicate_and_collapsed_classes(self):
        for x in [torch.zeros_like(self.x), torch.ones_like(self.x), self.x.detach() * 1e-10]:
            x.requires_grad_()
            loss = raw_mean.raw_mean_etf_loss(x, self.maps, self.mask)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_maps_and_mask(self):
        for mapping in [{1: [], 2: [2], 3: [3]}, {1: [-1], 2: [2], 3: [3]},
                        {1: [9], 2: [2], 3: [3]}, {1: [1, 1], 2: [2], 3: [3]},
                        {1: [1.5], 2: [2], 3: [3]}, {0: [1], 1: [2], 2: [3]},
                        {1: [1]}, {1: [1], 2: [2]}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                raw_mean.raw_mean_etf_loss(self.x, [mapping, self.maps[1]], self.mask)
        mask = self.mask.clone()
        mask[0, 1] = False
        for invalid in [mask, self.mask.float(), self.mask[:, :-1]]:
            with self.assertRaises(ValueError):
                raw_mean.raw_mean_etf_loss(self.x, self.maps, invalid)
        for eps in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                raw_mean.raw_mean_etf_loss(self.x, self.maps, self.mask, eps)
        with self.assertRaises(ValueError):
            raw_mean.raw_mean_etf_loss(self.x, self.maps[:1], self.mask)
        with self.assertRaises(ValueError):
            raw_mean.raw_mean_etf_loss(self.x[:, :, :1], self.maps, self.mask)


if __name__ == '__main__':
    unittest.main()
