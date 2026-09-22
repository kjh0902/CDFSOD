"""Exact formula and gradient checks, runnable with CPU PyTorch alone."""
import importlib.util
import math
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'class_separation_loss', ROOT / 'mmdet/models/losses/class_separation_loss.py')
separation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(separation)
loss_fn = separation.class_separation_loss


def pairwise_reference(x, maps, labels, temperature):
    """Independent literal token-pair implementation of the requested formula."""
    images = []
    for b, mapping in enumerate(maps):
        anchors = []
        for g in sorted(set(labels[b].tolist())):
            negatives = []
            for j in sorted(mapping):
                if j == g + 1:
                    continue
                pairs = [torch.dot(x[b, a], x[b, t])
                         for a in mapping[g + 1] for t in mapping[j]]
                negatives.append((torch.stack(pairs).mean() / temperature).exp())
            anchors.append(torch.log(1 + torch.stack(negatives).sum())
                           if negatives else x[b, :0].sum())
        images.append(torch.stack(anchors).mean() if anchors else x[b, :0].sum())
    return torch.stack(images).mean()


class ClassSeparationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 9, 4, dtype=torch.float64, requires_grad=True)
        self.maps = [{1: [1], 2: [2, 3], 3: [5, 6, 7]} for _ in range(2)]
        self.mask = torch.ones(2, 9, dtype=torch.bool)
        self.labels = [torch.tensor([0, 0, 2]), torch.tensor([1])]

    def test_exact_pairwise_formula_and_gradient(self):
        for temperature in [0.5, 1., 2.]:
            actual = loss_fn(self.x, self.maps, self.mask, self.labels, temperature)
            expected = pairwise_reference(self.x, self.maps, self.labels, temperature)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, self.x)[0],
                                       torch.autograd.grad(expected, self.x)[0])
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: loss_fn(x, self.maps, self.mask, self.labels, 1.3), (self.x,)))

    def test_unique_anchors_token_order_and_batch_mean(self):
        actual = loss_fn(self.x, self.maps, self.mask, self.labels)
        unique = [torch.tensor([2, 0]), self.labels[1]]
        reversed_maps = [{c: m[c][::-1] for c in reversed(m)} for m in self.maps]
        torch.testing.assert_close(actual, loss_fn(self.x, reversed_maps, self.mask, unique))
        individual = [loss_fn(self.x[b:b + 1], [self.maps[b]],
                              self.mask[b:b + 1], [self.labels[b]]) for b in range(2)]
        torch.testing.assert_close(actual, torch.stack(individual).mean())
        grad = torch.autograd.grad(actual, self.x)[0]
        single_grad = torch.autograd.grad(individual[0], self.x)[0]
        torch.testing.assert_close(grad[0], single_grad[0] / 2)
        torch.testing.assert_close(grad[:, [0, 4, 8]], torch.zeros_like(grad[:, [0, 4, 8]]))
        self.assertTrue((grad[:, [1, 2, 3, 5, 6, 7]].norm(dim=-1) > 0).all())

    def test_other_gt_and_absent_classes_are_negatives_but_self_is_not(self):
        x = torch.tensor([[[2.], [3.], [-1.]]], dtype=torch.float64)
        mapping = [{1: [0], 2: [1], 3: [2]}]
        mask = torch.ones(1, 3, dtype=torch.bool)
        result = loss_fn(x, mapping, mask, [torch.tensor([0, 1])])
        expected = (math.log(1 + math.exp(6) + math.exp(-2))
                    + math.log(1 + math.exp(6) + math.exp(-3))) / 2
        self.assertAlmostEqual(result.item(), expected)
        # Raw dot product scales quadratically with feature magnitude.
        torch.testing.assert_close(loss_fn(2 * x, mapping, mask, [torch.tensor([0])], 4.),
                                   loss_fn(x, mapping, mask, [torch.tensor([0])]))

    def test_hard_negative_has_larger_gradient(self):
        x = torch.tensor([[[1., 0.], [3., 1.], [-3., 1.]]], requires_grad=True)
        result = loss_fn(x, [{1: [0], 2: [1], 3: [2]}],
                         torch.ones(1, 3, dtype=torch.bool), [torch.tensor([0])])
        result.backward()
        self.assertGreater(x.grad[0, 1].norm().item(), 100 * x.grad[0, 2].norm().item())

    def test_empty_gt_included_in_batch_mean_and_all_empty_backward(self):
        empty = torch.empty(0, dtype=torch.long)
        loss = loss_fn(self.x, self.maps, self.mask, [self.labels[0], empty])
        single = loss_fn(self.x[:1], self.maps[:1], self.mask[:1], self.labels[:1])
        torch.testing.assert_close(loss, single / 2)
        grad = torch.autograd.grad(loss, self.x)[0]
        torch.testing.assert_close(grad[1], torch.zeros_like(grad[1]))
        zero = loss_fn(self.x, self.maps, self.mask, [empty, empty])
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        torch.testing.assert_close(self.x.grad, torch.zeros_like(self.x))

    def test_single_class_and_low_feature_dimension(self):
        x = self.x[:, :, :1]
        maps = [{1: [1, 2]}, {1: [3]}]
        loss = loss_fn(x, maps, self.mask, [torch.tensor([0]), torch.tensor([0])])
        self.assertEqual(loss.item(), 0.)
        grad = torch.autograd.grad(loss, self.x)[0]
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        self.assertTrue(torch.isfinite(loss_fn(x, self.maps, self.mask, self.labels)))

    def test_large_logits_zero_and_opposite_tokens(self):
        for scale in [0., 1e-10, 1e3]:
            x = (self.x.detach() * scale).requires_grad_()
            loss = loss_fn(x, self.maps, self.mask, self.labels, 0.1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
            if scale == 0:
                self.assertAlmostEqual(loss.item(), math.log(3))
        x = torch.tensor([[[1.], [-1.], [20.]]], requires_grad=True)
        loss = loss_fn(x, [{1: [0, 1], 2: [2]}],
                       torch.ones(1, 3, dtype=torch.bool), [torch.tensor([0])])
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)

    def check_low_precision(self, device, dtype):
        x = self.x.detach().to(device=device, dtype=dtype).requires_grad_()
        mask = self.mask.to(device)
        labels = [label.to(device) for label in self.labels]
        expected = loss_fn(x.float(), self.maps, mask, labels)
        with torch.autocast(device_type=device, dtype=dtype):
            actual = loss_fn(x, self.maps, mask, labels)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_cpu_low_precision_and_autocast(self):
        for dtype in [torch.float16, torch.bfloat16]:
            self.check_low_precision('cpu', dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
    def test_cuda_autocast(self):
        self.check_low_precision('cuda', torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_low_precision('cuda', torch.bfloat16)

    def test_invalid_maps_masks_labels_and_temperature(self):
        for mapping in [{1: [], 2: [2], 3: [3]}, {1: [-1], 2: [2], 3: [3]},
                        {1: [9], 2: [2], 3: [3]}, {1: [1, 1], 2: [2], 3: [3]},
                        {1: [1.5], 2: [2], 3: [3]}, {0: [1], 1: [2], 2: [3]},
                        {1: [True], 2: [2], 3: [3]}, {1: [1]}, {}, [],
                        {1: [1], 2: [2]}, {1.0: [1], 2: [2], 3: [3]}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                loss_fn(self.x, [mapping, self.maps[1]], self.mask, self.labels)
        mask = self.mask.clone()
        mask[0, 1] = False
        for invalid in [mask, self.mask.float(), self.mask[:, :-1]]:
            with self.assertRaises(ValueError):
                loss_fn(self.x, self.maps, invalid, self.labels)
        for labels in [torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.]),
                       torch.tensor([[0]]), torch.tensor([True])]:
            with self.assertRaises(ValueError):
                loss_fn(self.x, self.maps, self.mask, [labels, self.labels[1]])
        for temperature in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                loss_fn(self.x, self.maps, self.mask, self.labels, temperature)
        for maps, labels in [(self.maps[:1], self.labels), (self.maps, self.labels[:1])]:
            with self.assertRaises(ValueError):
                loss_fn(self.x, maps, self.mask, labels)

    def test_invalid_feature_shapes_and_dtype(self):
        for x in [self.x[0], self.x[:0], self.x[:, :0], self.x[:, :, :0], self.x.long()]:
            with self.assertRaises(ValueError):
                loss_fn(x, self.maps, self.mask, self.labels)


if __name__ == '__main__':
    unittest.main()
