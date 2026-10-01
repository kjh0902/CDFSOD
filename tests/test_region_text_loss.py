"""CPU numerical tests; run: python -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'region_loss', ROOT / 'mmdet/models/losses/region_text_loss.py')
region = importlib.util.module_from_spec(spec)
spec.loader.exec_module(region)


def sample(boxes, labels):
    return NS(gt_instances=NS(
        bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor(labels, dtype=torch.long)))


def fixture():
    torch.manual_seed(17)
    return dict(
        output_memory=torch.randn(2, 20, 4, requires_grad=True),
        memory_text=torch.randn(2, 8, 4, requires_grad=True),
        spatial_shapes=torch.tensor([[4, 4], [2, 2]]),
        level_start_index=torch.tensor([0, 16]),
        text_token_mask=torch.tensor([[True] * 7 + [False]] * 2),
        class_token_maps=[{1: [1, 2], 2: [4], 3: [6]}] * 2,
        batch_data_samples=[sample([[0, 0, 4, 4], [3, 2, 7, 8]], [0, 0]),
                            sample([[1, 0, 6, 7]], [1])],
        featmap_strides=(2, 4), roi_size=3)


class RegionLossTest(unittest.TestCase):
    def test_raw_mean_only_name_tokens(self):
        d = fixture()
        p = region.class_name_prototypes(d['memory_text'], d['class_token_maps'],
                                         d['text_token_mask'])
        torch.testing.assert_close(p[0][0], d['memory_text'][0, [1, 2]].mean(0))
        torch.testing.assert_close(p[1][2], d['memory_text'][1, 6])
        p[0].sum().backward()
        self.assertEqual(d['memory_text'].grad[0, [0, 3, 5, 7]].abs().sum(), 0)

    def test_object_mean_and_all_class_raw_logits(self):
        d = fixture()
        objects, labels, ids = region.pool_gt_regions(**{
            k: d[k] for k in ('output_memory', 'spatial_shapes', 'level_start_index',
                             'batch_data_samples', 'featmap_strides', 'roi_size')})
        p = region.class_name_prototypes(d['memory_text'], d['class_token_maps'],
                                         d['text_token_mask'])
        self.assertEqual(objects.shape, (3, 4))
        self.assertFalse(torch.equal(objects[0], objects[1]))
        logits = torch.stack([objects[i] @ p[ids[i]].T for i in range(3)])
        expected = F.cross_entropy(logits, labels)
        actual = region.region_text_loss(**d)
        torch.testing.assert_close(actual, expected)
        # A class absent from both images is still a negative and gets gradients.
        actual.backward()
        self.assertGreater(d['memory_text'].grad[:, 6].abs().sum(), 0)
        self.assertGreater(d['output_memory'].grad[:, :16].abs().sum(), 0)
        self.assertEqual(d['output_memory'].grad[:, 16:].abs().sum(), 0)
        # This is intentionally not equal-image or equal-class weighting.
        per = F.cross_entropy(logits, labels, reduction='none')
        self.assertFalse(torch.isclose(actual, (per[:2].mean() + per[2]) / 2))

    def test_analytic_roi_coordinates_on_selected_level(self):
        # Linear ramps have exact bilinear means. Padding/image size metadata
        # must not rescale these post-augmentation boxes a second time.
        y, x = torch.meshgrid(torch.arange(6.), torch.arange(8.), indexing='ij')
        first = x + 10 * y
        y, x = torch.meshgrid(torch.arange(3.), torch.arange(4.), indexing='ij')
        second = 100 + x + 10 * y
        memory = torch.cat([first.flatten(), second.flatten()])[None, :, None]
        s = sample([[2, 2, 6, 6], [6, 2, 10, 6]], [0, 0])
        s.img_shape = (7, 11)
        s.batch_input_shape = (12, 16)
        s.scale_factor = (0.5, 0.5)
        pooled, _, _ = region.pool_gt_regions(
            memory, torch.tensor([[6, 8], [3, 4]]), torch.tensor([0, 48]),
            [s], (2, 4))
        # Box centers / stride - 0.5, using aligned=True.
        expected = torch.tensor([[16.5], [18.5]])
        torch.testing.assert_close(pooled, expected)

    def test_fpn_assignment_boundaries_order_and_gradients(self):
        # P3..P6: thresholds 224, 448, 896; clamp both ends.
        sides = [896., 2., 447., 224., 223., 448., 895., 1200.]
        expected_levels = [3, 0, 1, 1, 0, 2, 2, 3]
        shapes = [(160, 160), (80, 80), (40, 40), (20, 20)]
        starts = [0, 25600, 32000, 33600]
        memory = torch.cat([torch.full((2, h * w, 1), float(i + 1))
                            for i, (h, w) in enumerate(shapes)], dim=1)
        memory.requires_grad_()
        boxes = [[0, 0, side, side] for side in sides]
        # Non-square box with the same area as a 224px square must select P4.
        boxes.append([0, 0, 112, 448])
        expected_levels.append(1)
        samples = [sample(boxes[:4], [0, 1, 2, 3]),
                   sample(boxes[4:], [4, 5, 6, 7, 8])]
        with patch.object(region, 'roi_align', wraps=region.roi_align) as align:
            pooled, labels, ids = region.pool_gt_regions(
                memory, torch.tensor(shapes), torch.tensor(starts), samples,
                (8, 16, 32, 64))
        torch.testing.assert_close(pooled[:, 0], torch.tensor(expected_levels).float() + 1)
        self.assertEqual(labels.tolist(), list(range(9)))
        self.assertEqual(ids.tolist(), [0] * 4 + [1] * 5)
        self.assertEqual(align.call_count, 4)
        self.assertEqual(sum(len(c.args[1]) for c in align.call_args_list), 9)
        for level, call in enumerate(align.call_args_list):
            indices = [i for i, target in enumerate(expected_levels) if target == level]
            torch.testing.assert_close(call.args[1][:, 1:], torch.tensor(boxes)[indices])
            self.assertEqual(call.kwargs['output_size'], 3)
            self.assertEqual(call.kwargs['spatial_scale'], 1 / (8 * 2 ** level))
        # Each object's gradient reaches exactly its own level and image.
        for i, level in enumerate(expected_levels):
            grad, = torch.autograd.grad(pooled[i].sum(), memory, retain_graph=True)
            image = ids[i].item()
            for j, ((h, w), start) in enumerate(zip(shapes, starts)):
                magnitude = grad[image, start:start + h * w].abs().sum()
                if j == level:
                    self.assertGreater(magnitude, 0)
                else:
                    self.assertEqual(magnitude, 0)
            self.assertEqual(grad[1 - image].abs().sum(), 0)

    def test_unused_levels_skip_roi_align_and_single_level_clamps(self):
        d = fixture()
        with patch.object(region, 'roi_align', wraps=region.roi_align) as align:
            region.region_text_loss(**d)
        self.assertEqual(align.call_count, 1)
        d['output_memory'] = d['output_memory'][:, :16]
        d['spatial_shapes'] = torch.tensor([[4, 4]])
        d['level_start_index'] = torch.tensor([0])
        d['featmap_strides'] = (2,)
        with patch.object(region, 'roi_align', wraps=region.roi_align) as align:
            self.assertTrue(torch.isfinite(region.region_text_loss(**d)))
        self.assertEqual(align.call_count, 1)

    def test_empty_and_single_class(self):
        for single in (False, True):
            d = fixture()
            if single:
                d['class_token_maps'] = [{1: [1, 2]}] * 2
                for s in d['batch_data_samples']:
                    s.gt_instances.labels.zero_()
            else:
                d['batch_data_samples'] = [sample([], []), sample([], [])]
            loss = region.region_text_loss(**d)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertIsNotNone(d['output_memory'].grad)
            self.assertIsNotNone(d['memory_text'].grad)

    def test_one_empty_image_object_denominator(self):
        d = fixture()
        d['batch_data_samples'][0] = sample([], [])
        loss = region.region_text_loss(**d)
        one = dict(d)
        for k in ('output_memory', 'memory_text', 'text_token_mask'):
            one[k] = d[k][1:]
        one['batch_data_samples'] = d['batch_data_samples'][1:]
        one['class_token_maps'] = d['class_token_maps'][1:]
        torch.testing.assert_close(loss, region.region_text_loss(**one))

    def test_amp_fp32_logits_and_gradients(self):
        d = fixture()
        for k in ('output_memory', 'memory_text'):
            d[k] = d[k].detach().to(torch.bfloat16).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = region.region_text_loss(**d)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(d['output_memory'].grad).all())
        self.assertGreater(d['memory_text'].grad.abs().sum(), 0)

    def test_rejects_missing_padded_and_truncated_class_tokens(self):
        for bad in ({1: []}, {2: [1]}, {1: [7]}, {1: [8]}):
            d = fixture()
            d['class_token_maps'][0] = bad
            with self.assertRaises(ValueError):
                region.region_text_loss(**d)

    def test_rejects_bad_feature_layout_and_boxes(self):
        for key, value in [('level_start_index', torch.tensor([0, 15])),
                           ('featmap_strides', (2,)),
                           ('featmap_strides', (2, 8)),
                           ('featmap_strides', (0, 0)),
                           ('spatial_shapes', torch.tensor([[3, 4], [2, 2]]))]:
            d = fixture()
            d[key] = value
            with self.assertRaises(ValueError):
                region.region_text_loss(**d)
        d = fixture()
        d['batch_data_samples'][0].gt_instances.bboxes[0, 2] = 0
        with self.assertRaises(ValueError):
            region.region_text_loss(**d)

    def test_ddp_global_object_denominator(self):
        d = fixture()
        local = region.region_text_loss(**d)
        # Local 3 objects, other rank 1: DDP subsequently divides gradients by 2.
        with patch.object(region.dist, 'is_initialized', return_value=True), \
                patch.object(region.dist, 'get_world_size', return_value=2), \
                patch.object(region.dist, 'all_reduce', side_effect=lambda t: t.fill_(4)):
            torch.testing.assert_close(region.region_text_loss(**d), local * 1.5)
        # A single global GT must divide by 1/world_size, not clamp to one.
        d['batch_data_samples'][0] = sample([], [])
        local = region.region_text_loss(**d)
        with patch.object(region.dist, 'is_initialized', return_value=True), \
                patch.object(region.dist, 'get_world_size', return_value=2), \
                patch.object(region.dist, 'all_reduce', side_effect=lambda t: t.fill_(1)):
            torch.testing.assert_close(region.region_text_loss(**d), local * 2)


if __name__ == '__main__':
    unittest.main()
