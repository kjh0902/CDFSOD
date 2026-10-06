"""Independent geometry/gradient checks; requires PyTorch, no MMCV or weights."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'background_etf', ROOT / 'mmdet/models/losses/background_anchored_etf_loss.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def reference_deformation(classes, presence, alpha):
    """Independent eigenspace construction, not the production Helmert basis."""
    h = torch.eye(classes, dtype=torch.float64) - torch.ones(
        classes, classes, dtype=torch.float64) / classes
    _, vectors = torch.linalg.eigh(h)
    a = (1 + alpha * presence.double()).unsqueeze(-1) * vectors[:, 1:]
    return a / torch.linalg.vector_norm(a, dim=(-2, -1), keepdim=True)


class BackgroundETFGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_exact_optimum_and_deformed_gram_at_minimal_and_large_dimensions(self):
        for classes, dimensions in [(2, 1), (3, 2), (6, 256)]:
            with self.subTest(C=classes, D=dimensions):
                q = torch.randn(2, classes, dimensions, dtype=torch.float64)
                q /= torch.linalg.vector_norm(q, dim=(-2, -1), keepdim=True)
                q.requires_grad_()
                presence = torch.zeros(2, classes, dtype=torch.bool)
                presence[0, 0] = True
                presence[1, ::2] = True
                a = reference_deformation(classes, presence, 0.5)
                target = geometry.nearest_deformed_etf(q, presence)
                torch.testing.assert_close(target @ target.transpose(-2, -1),
                                           a @ a.transpose(-2, -1))
                torch.testing.assert_close(torch.linalg.vector_norm(
                    target, dim=(-2, -1)), torch.ones(2, dtype=torch.float64))
                sigma = torch.linalg.svdvals(q.transpose(-2, -1) @ a)
                torch.testing.assert_close((q - target).square().sum((-2, -1)),
                                           2 - 2 * sigma.sum(-1))
                # Reconstruct the returned orientation and check the constraint.
                rotation_t = torch.linalg.lstsq(a, target).solution
                torch.testing.assert_close(rotation_t @ rotation_t.transpose(-2, -1),
                                           torch.eye(classes - 1, dtype=torch.float64)
                                           .expand(2, -1, -1))
                self.assertFalse(target.requires_grad)
                self.assertGreater(target[0].mean(0).norm().item(), 0)

    def test_exact_target_has_zero_loss_with_nonzero_class_mean(self):
        presence = torch.tensor([[True, False, False]])
        a = reference_deformation(3, presence, 0.5)
        rotation, _ = torch.linalg.qr(torch.randn(5, 2, dtype=torch.float64))
        target = a @ rotation.T
        bg = torch.randn(1, 5, dtype=torch.float64)
        foreground = target * 7 + bg.unsqueeze(1)
        loss = geometry.background_anchored_prototype_loss(foreground, bg, presence)
        self.assertLess(loss.item(), 1e-25)
        self.assertGreater((foreground - bg.unsqueeze(1)).mean(1).norm().item(), 0)

    def test_presence_counts_uniform_scaling_and_alpha_zero(self):
        duplicate = geometry.class_presence([torch.tensor([0, 0, 2])], 3, 'cpu')
        unique = geometry.class_presence([torch.tensor([0, 2])], 3, 'cpu')
        self.assertTrue(torch.equal(duplicate, unique))
        empty = geometry.class_presence([torch.empty(0, dtype=torch.long)], 3, 'cpu')
        self.assertFalse(empty.any())
        q = torch.randn(1, 3, 5, dtype=torch.float64)
        q /= q.norm()
        absent = geometry.nearest_deformed_etf(q, empty)
        all_present = geometry.nearest_deformed_etf(q, torch.ones_like(empty))
        no_deformation = geometry.nearest_deformed_etf(q, duplicate, alpha=0)
        torch.testing.assert_close(absent, all_present)
        torch.testing.assert_close(absent, no_deformation)
        self.assertFalse(torch.allclose(absent, geometry.nearest_deformed_etf(q, duplicate)))

    def test_rank_deficient_and_collapsed_inputs_are_finite(self):
        for collapsed in (False, True):
            foreground = torch.ones(2, 4, 3, requires_grad=True)
            background = torch.full((2, 3), float(collapsed), requires_grad=True)
            loss = geometry.background_anchored_prototype_loss(
                foreground, background, torch.zeros(2, 4, dtype=torch.bool))
            self.assertTrue(torch.isfinite(loss))
            if collapsed:
                torch.testing.assert_close(loss, torch.tensor(1.0))
            loss.backward()
            self.assertTrue(torch.isfinite(foreground.grad).all())
            self.assertTrue(torch.isfinite(background.grad).all())

    def test_raw_means_and_only_name_tokens_receive_direct_gradients(self):
        features = torch.randn(2, 10, 5, dtype=torch.float64, requires_grad=True)
        maps = [{1: [1, 2], 2: [4], 3: [6, 7]}] * 2
        bg_indices = [[8]] * 2
        mask = torch.ones(2, 10, dtype=torch.bool)
        foreground, bg = geometry.raw_mean_prototypes(features, maps, bg_indices, mask)
        torch.testing.assert_close(foreground[:, 0], features[:, [1, 2]].mean(1))
        torch.testing.assert_close(bg, features[:, 8])
        with patch.object(geometry, 'nearest_deformed_etf',
                          wraps=geometry.nearest_deformed_etf) as solve:
            loss = geometry.background_anchored_etf_loss(
                features, maps, bg_indices, mask,
                [torch.tensor([0, 0]), torch.tensor([], dtype=torch.long)])
        loss.backward()
        self.assertEqual(solve.call_count, 1)
        for index in [1, 2, 4, 6, 7, 8]:
            self.assertGreater(features.grad[:, index].abs().sum().item(), 0)
        for index in [0, 3, 5, 9]:
            torch.testing.assert_close(features.grad[:, index],
                                       torch.zeros_like(features.grad[:, index]))
        torch.testing.assert_close(features.grad[:, 1], features.grad[:, 2])

    def test_feature_side_gradient_and_background_gradient_match_finite_differences(self):
        foreground = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        bg = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
        presence = torch.tensor([[False, True, False]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda fg, anchor: geometry.background_anchored_prototype_loss(
                fg, anchor, presence), (foreground, bg)))
        loss = geometry.background_anchored_prototype_loss(foreground, bg, presence)
        loss.backward()
        self.assertGreater(bg.grad.abs().sum().item(), 0)
        shifted = geometry.background_anchored_prototype_loss(
            foreground.detach() + 13, bg.detach() + 13, presence)
        torch.testing.assert_close(loss, shifted)

    def test_single_class_zero_keeps_both_gradients_and_never_calls_svd(self):
        fg = torch.randn(2, 1, 4, requires_grad=True)
        bg = torch.randn(2, 4, requires_grad=True)
        with patch.object(torch.linalg, 'svd', side_effect=AssertionError('SVD')):
            loss = geometry.background_anchored_prototype_loss(
                fg, bg, torch.tensor([[True], [False]]))
        self.assertTrue(loss.requires_grad)
        loss.backward()
        torch.testing.assert_close(fg.grad, torch.zeros_like(fg))
        torch.testing.assert_close(bg.grad, torch.zeros_like(bg))

    def test_autocast_and_precision(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                fg = torch.randn(2, 3, 5).to(dtype).requires_grad_()
                bg = torch.randn(2, 5).to(dtype).requires_grad_()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    loss = geometry.background_anchored_prototype_loss(
                        fg, bg, torch.zeros(2, 3, dtype=torch.bool))
                self.assertEqual(loss.dtype, torch.float64 if dtype == torch.float64
                                 else torch.float32)
                loss.backward()
                self.assertTrue(torch.isfinite(bg.grad).all())

    def test_invalid_geometry_maps_labels_and_scalars(self):
        fg, bg = torch.randn(1, 3, 4), torch.randn(1, 4)
        presence = torch.zeros(1, 3, dtype=torch.bool)
        for invalid in [-1, float('nan'), float('inf'), True, [0.5]]:
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                geometry.background_anchored_prototype_loss(fg, bg, presence, invalid)
        for invalid in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                geometry.background_anchored_prototype_loss(fg, bg, presence, eps=invalid)
        for labels in [torch.tensor([-1]), torch.tensor([3]), torch.tensor([0.0])]:
            with self.assertRaises(ValueError):
                geometry.class_presence([labels], 3, 'cpu')
        with self.assertRaises(ValueError):
            geometry.background_anchored_prototype_loss(
                torch.randn(1, 4, 2), torch.randn(1, 2), torch.zeros(1, 4, dtype=torch.bool))
        features, mask = torch.randn(1, 5, 4), torch.ones(1, 5, dtype=torch.bool)
        for maps, background in [([{1: []}], [[3]]), ([{2: [1]}], [[3]]),
                                 ([{1: [1, 1]}], [[3]]), ([{1: [5]}], [[3]]),
                                 ([{1: [1]}], [[1]])]:
            with self.assertRaises(ValueError):
                geometry.raw_mean_prototypes(features, maps, background, mask)
        mask[0, 3] = False
        with self.assertRaises(ValueError):
            geometry.raw_mean_prototypes(features, [{1: [1]}], [[3]], mask)


if __name__ == '__main__':
    unittest.main()
