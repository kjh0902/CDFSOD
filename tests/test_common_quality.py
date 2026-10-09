"""Common quality regressions using production methods without MMCV or weights.

Only the heavy base initialization, Hungarian solver and box-loss modules are
test doubles. The actual head, inherited DINO/DETR loss routing, Hungarian target
construction and Python focal/QFL implementations execute unchanged.
"""
import ast
import functools
from numbers import Integral
from unittest.mock import patch
import math
import subprocess
import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from test_second_order_etf_integration import (
    ROOT, HEAD, execute, method, fixture, samples, run, Detector)
from test_second_order_etf_loss import second

BASE = '530e2e8'
torch.set_num_threads(1)


def function(path, name, **env):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return execute([node], **env)[name]


reduce_loss = function('mmdet/models/losses/utils.py', 'reduce_loss', F=F)
weight_reduce_loss = function('mmdet/models/losses/utils.py', 'weight_reduce_loss',
                              F=F, reduce_loss=reduce_loss)
focal = function('mmdet/models/losses/focal_loss.py', 'py_sigmoid_focal_loss',
                 F=F, weight_reduce_loss=weight_reduce_loss)


class Focal(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.last_target = None

    def forward(self, pred, target, weight=None, avg_factor=None):
        self.last_target = target.detach().clone()
        return focal(pred, target, weight=weight, avg_factor=avg_factor)


weighted_loss = function('mmdet/models/losses/utils.py', 'weighted_loss',
                         functools=functools, weight_reduce_loss=weight_reduce_loss)
qfl = function('mmdet/models/losses/gfocal_loss.py', 'quality_focal_loss_tensor_target',
               F=F, weighted_loss=weighted_loss)
qfl_class = next(n for n in ast.parse((ROOT / 'mmdet/models/losses/gfocal_loss.py').read_text(
    encoding='utf-8')).body if isinstance(n, ast.ClassDef) and n.name == 'QualityFocalLoss')
qfl_class.decorator_list = []
Quality = execute([qfl_class], partial=functools.partial,
                  quality_focal_loss=None, quality_focal_loss_with_prob=None,
                  quality_focal_loss_tensor_target=qfl)['QualityFocalLoss']


class RecordingQuality(Quality):
    def __init__(self, **kwargs):
        kwargs.pop('type', None)
        super().__init__(**kwargs)
        self.calls = []

    def forward(self, pred, target, **kwargs):
        self.calls.append((pred.detach().clone(), target.detach().clone(), kwargs))
        return super().forward(pred, target, **kwargs)


def multi_apply(fn, *args, **kwargs):
    return tuple(map(list, zip(*(fn(*items, **kwargs) for items in zip(*args)))))


def cxcywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2,
                      box[..., :2] + box[..., 2:] / 2), dim=-1)


def xyxy_to_cxcywh(box):
    return torch.cat(((box[..., :2] + box[..., 2:]) / 2,
                      box[..., 2:] - box[..., :2]), dim=-1)


fp16_clamp = function('mmdet/structures/bbox/bbox_overlaps.py', 'fp16_clamp')
box_iou = function('mmdet/structures/bbox/bbox_overlaps.py', 'bbox_overlaps',
                   fp16_clamp=fp16_clamp)
quality_nodes = [n for n in ast.parse((ROOT / 'mmdet/models/losses/common_quality_loss.py').read_text(
    encoding='utf-8')).body if isinstance(n, ast.FunctionDef)]
quality_env = execute(quality_nodes, Integral=Integral,
                       bbox_cxcywh_to_xyxy=cxcywh_to_xyxy, bbox_overlaps=box_iou)
quality_targets = quality_env['common_quality_targets']
validate_cfg = quality_env['validate_common_quality_cfg']


class Assigner:
    def __init__(self):
        self.targets = []

    def assign(self, pred_instances, gt_instances, img_meta):
        # Different matched indices for different layers/encoder calls expose
        # accidental reuse of encoder or intermediate-layer targets.
        ids = torch.zeros(len(pred_instances.bboxes), dtype=torch.long)
        if len(gt_instances.bboxes):
            ids[(len(self.targets) // 2) % len(ids)] = 1
        self.targets.append(ids.clone())
        return SimpleNamespace(gt_inds=ids)


class DETRBase(nn.Module):
    get_targets = method('mmdet/models/dense_heads/detr_head.py', 'DETRHead',
                         'get_targets', multi_apply=multi_apply)
    loss_by_feat = method('mmdet/models/dense_heads/detr_head.py', 'DETRHead',
                          'loss_by_feat', multi_apply=multi_apply)


class DeformableBase(DETRBase):
    pass


class DINOBase(DeformableBase):
    loss_by_feat = method('mmdet/models/dense_heads/dino_head.py', 'DINOHead',
                          'loss_by_feat', DeformableDETRHead=DeformableBase)
    split_outputs_impl = staticmethod(method('mmdet/models/dense_heads/dino_head.py',
                                             'DINOHead', 'split_outputs'))
    loss_dn = method('mmdet/models/dense_heads/dino_head.py', 'DINOHead',
                     'loss_dn', multi_apply=multi_apply)
    get_dn_targets = method('mmdet/models/dense_heads/dino_head.py', 'DINOHead',
                            'get_dn_targets', multi_apply=multi_apply)

    @staticmethod
    def split_outputs(scores, boxes, meta):
        # A no-DN fixture has no DN targets. Real DN generation allocates room
        # for GTs; an empty synthetic allocation must not enter the DN targeter.
        if meta['num_denoising_queries'] == 0:
            return scores, boxes, None, None
        return DINOBase.split_outputs_impl(scores, boxes, meta)

    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dims, self.num_reg_fcs = 4, 1
        self.num_pred_layer, self.share_pred_layer = 4, False
        self.bg_cls_weight, self.sync_cls_avg_factor = 0., True
        self.loss_cls = Focal()
        self.loss_bbox = self.loss_iou = lambda p, t, w, avg_factor: (
            (p - t).abs() * w).sum() / avg_factor
        self.assigner = Assigner()
        self._init_layers()


def load_head(base=False):
    text = (subprocess.check_output(['git', 'show', f'{BASE}:{HEAD}'], cwd=ROOT).decode()
            if base else (ROOT / HEAD).read_text(encoding='utf-8'))
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)]
    for node in nodes:
        node.decorator_list = []
    return execute(nodes, DINOHead=DINOBase, Linear=nn.Linear,
                   MODELS=SimpleNamespace(build=lambda cfg: RecordingQuality(**cfg)
                       if cfg['type'] == 'QualityFocalLoss' else Focal(**cfg)),
                   QualityFocalLoss=Quality, common_quality_targets=quality_targets,
                   validate_common_quality_cfg=validate_cfg,
                   InstanceData=SimpleNamespace,
                   bbox_cxcywh_to_xyxy=cxcywh_to_xyxy,
                   bbox_xyxy_to_cxcywh=xyxy_to_cxcywh,
                   convert_grounding_to_cls_scores=function(
                       'mmdet/models/dense_heads/atss_vlfusion_head.py',
                       'convert_grounding_to_cls_scores'),
                   inverse_sigmoid=lambda x: torch.logit(x.clamp(1e-5, 1 - 1e-5)),
                   reduce_mean=lambda x: x)['GroundingDINOHead_ParallelDecoder_DN']


Head, BaseHead = load_head(), load_head(base=True)


def inputs(empty=False, dn=2):
    torch.manual_seed(63)
    states = torch.randn(3, 2, dn + 4, 4, requires_grad=True)
    text = torch.randn(2, 7, 4, requires_grad=True)
    mask = torch.tensor([[True] * 6 + [False]] * 2)
    maps = [{1: [1], 2: [2, 3], 3: [4, 5]}] * 2
    common = second._second_order_representations(text, maps, mask).mean(1)
    data = []
    for populated in [not empty, False]:
        gt = SimpleNamespace(bboxes=torch.tensor([[2., 3., 8., 9.]]) if populated
                             else torch.empty(0, 4),
                             labels=torch.tensor([0], dtype=torch.long) if populated
                             else torch.empty(0, dtype=torch.long),
                             positive_maps=torch.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]])
                             if populated else torch.empty(0, 8))
        data.append(SimpleNamespace(gt_instances=gt, metainfo=dict(img_shape=(12, 16))))
    encoder_features = torch.randn(2, 7, 4, requires_grad=True)
    encoder_boxes = torch.rand(2, 7, 4, requires_grad=True)
    encoder_valid = torch.tensor([[True] * 6 + [False]] * 2)
    return dict(enc_quality_features=encoder_features, enc_quality_boxes=encoder_boxes,
                enc_quality_valid_mask=encoder_valid, hidden_states=states, references=[torch.full((2, dn + 4, 4), .5)] * 4,
                memory_text=text, text_token_mask=mask,
                enc_outputs_class=torch.randn(2, 4, 8),
                enc_outputs_coord=torch.rand(2, 4, 4), batch_data_samples=data,
                dn_meta=dict(num_denoising_queries=dn, num_denoising_groups=1),
                class_common=common)


def make_head(enc=.1, dec=.1, base=False, **kwargs):
    torch.manual_seed(17)
    cls = BaseHead if base else Head
    if not base:
        kwargs.update(enc_mu_quality_loss_weight=enc, dec_mu_quality_loss_weight=dec)
    return cls(contrastive_cfg=dict(max_text_len=8, log_scale='auto', bias=True), **kwargs)


def pixel_boxes(boxes):
    return xyxy_to_cxcywh(torch.tensor(boxes, dtype=torch.float32)) / 10


class QualityAssignmentTests(unittest.TestCase):
    def assign(self, boxes, gt, **kwargs):
        return quality_targets(boxes.unsqueeze(0),
                               [SimpleNamespace(bboxes=gt)],
                               [dict(img_shape=(10, 10))], **kwargs)

    def test_topk_ignore_background_and_padding(self):
        boxes = pixel_boxes([[0, 0, 10, 10], [1, 1, 9, 9], [0, 0, 9, 9],
                             [20, 20, 22, 22], [0, 0, 10, 10]])
        targets, mask = self.assign(boxes, torch.tensor([[0., 0., 10., 10.]]),
                                    topk=2, valid_mask=torch.tensor([[1, 1, 1, 1, 0]],
                                                                  dtype=torch.bool))
        torch.testing.assert_close(targets, torch.tensor([[1., 0., .81, 0., 0.]]))
        self.assertEqual(mask.tolist(), [[True, False, True, True, False]])

    def test_ignore_boundary_and_configurable_threshold(self):
        boxes = pixel_boxes([[0, 0, 10, 10], [0, 0, 10, 5], [0, 0, 10, 4.9]])
        gt = torch.tensor([[0., 0., 10., 10.]])
        _, mask = self.assign(boxes, gt, topk=1)
        self.assertEqual(mask.tolist(), [[True, False, True]])
        _, mask = self.assign(boxes, gt, topk=1, ignore_iou_thr=.4)
        self.assertEqual(mask.tolist(), [[True, False, False]])

    def test_multiple_gt_nominations_use_max_iou_once(self):
        boxes = pixel_boxes([[0, 0, 10, 10], [5, 0, 15, 10], [2, 0, 12, 10]])
        gt = torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.]])
        targets, mask = self.assign(boxes, gt, topk=2)
        torch.testing.assert_close(targets, torch.tensor([[1., 1., 2/3]]))
        self.assertTrue(mask.all())

    def test_fewer_candidates_than_k_and_zero_overlap(self):
        boxes = pixel_boxes([[0, 0, 10, 10], [20, 20, 22, 22]])
        targets, mask = self.assign(boxes, torch.tensor([[0., 0., 10., 10.]]), topk=5)
        torch.testing.assert_close(targets, torch.tensor([[1., 0.]]))
        self.assertTrue(mask.all())
        targets, mask = self.assign(boxes[1:], torch.tensor([[0., 0., 10., 10.]]))
        self.assertEqual(targets.count_nonzero().item(), 0)
        self.assertTrue(mask.all())

    def test_empty_gt_invalid_boxes_and_empty_proposals(self):
        boxes = torch.tensor([[.5, .5, 1., 1.], [.5, .5, 0., .2],
                              [float('nan'), .5, .2, .2]])
        targets, mask = self.assign(boxes, torch.empty(0, 4))
        self.assertEqual(targets.count_nonzero().item(), 0)
        self.assertEqual(mask.tolist(), [[True, False, False]])
        targets, mask = self.assign(torch.empty(0, 4), torch.empty(0, 4))
        self.assertEqual(targets.shape, (1, 0))
        self.assertEqual(mask.shape, (1, 0))

    def test_targets_detached_and_image_coordinates(self):
        boxes = torch.tensor([[[.5, .5, .5, .5]]], requires_grad=True)
        gt = torch.tensor([[5., 2.5, 15., 7.5]], requires_grad=True)
        targets, _ = quality_targets(boxes, [SimpleNamespace(bboxes=gt)],
                                     [dict(img_shape=(10, 20, 3))])
        torch.testing.assert_close(targets, torch.ones(1, 1))
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(boxes.grad)
        self.assertIsNone(gt.grad)

    def test_amp_target_precision(self):
        boxes = pixel_boxes([[0, 0, 10, 10], [0, 0, 8, 8]])
        gt = torch.tensor([[0., 0., 10., 10.]])
        expected, _ = self.assign(boxes, gt)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual, _ = self.assign(boxes, gt)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_cfg_and_mask(self):
        for topk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                validate_cfg(topk, .5)
        for threshold in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_cfg(5, threshold)
        with self.assertRaises(ValueError):
            self.assign(torch.zeros(2, 4), torch.empty(0, 4),
                        valid_mask=torch.ones(1, 3, dtype=torch.bool))


class CommonQualityHeadTests(unittest.TestCase):
    def test_existing_detection_losses_assignments_and_outputs_exactly_preserved(self):
        for dn in (0, 2):
            for enc, dec in ((0., 0.), (.1, 0.), (0., .1), (.1, .1)):
                with self.subTest(dn=dn, enc=enc, dec=dec):
                    baseline, head = make_head(base=True), make_head(enc, dec)
                    kw = inputs(dn=dn)
                    base_kw = {k:v for k,v in kw.items()
                               if k != 'class_common' and not k.startswith('enc_quality_')}
                    expected = baseline.loss(**base_kw)
                    actual = head.loss(**kw)
                    additions = ({'enc_loss_mu_quality'} if enc else set()) | (
                        {'dec_loss_mu_quality'} if dec else set())
                    self.assertEqual(actual.keys() - expected.keys(), additions)
                    self.assertNotIn('loss_mu_focal', actual)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    self.assertEqual(len(head.assigner.targets), 2 * (3 + 1))
                    for before, after in zip(baseline.assigner.targets, head.assigner.targets):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)
                    args = [kw[k] for k in ('hidden_states', 'references',
                                            'memory_text', 'text_token_mask')]
                    for before, after in zip(baseline(*args), head(*args)):
                        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_quality_formula_ignore_and_independent_normalization(self):
        head = make_head(common_quality_topk=1)
        mu = torch.randn(1, 4, requires_grad=True)
        features = torch.randn(1, 4, 4, requires_grad=True)
        encoder_boxes = pixel_boxes([[0, 0, 10, 10], [1, 1, 9, 9],
                                     [20, 20, 25, 25], [0, 0, 8, 8]]).unsqueeze(0)
        decoder_boxes = pixel_boxes([[0, 0, 10, 10], [20, 20, 25, 25],
                                     [20, 20, 25, 25], [20, 20, 25, 25]]).unsqueeze(0)
        gt = [SimpleNamespace(bboxes=torch.tensor([[0., 0., 10., 10.],
                                                   [20., 20., 25., 25.]]))]
        metas = [dict(img_shape=(10, 10))]
        counts = []
        with patch.dict(Head._common_quality_loss.__globals__, reduce_mean=lambda x: (
                counts.append(x.item()) or x / 2)):
            loss = head._common_quality_loss(features, encoder_boxes, mu, gt, metas,
                                              head.enc_mu_cls)
            head._common_quality_loss(features, decoder_boxes, mu, gt, metas,
                                       head.dec_mu_cls)
        self.assertEqual(counts, [2, 2])
        for call in head.loss_mu_quality.calls:
            self.assertEqual(call[2]['avg_factor'], 1.)
        logits = (features * mu[:, None]).sum(-1) / 2 + head.enc_mu_cls.bias
        targets = torch.tensor([[1., 1.]])
        logits = logits[:, [0, 2]]
        expected = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                    * (targets - logits.sigmoid()).abs().square()).sum()
        expected /= 1 + torch.finfo(torch.float32).eps
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(features.grad[:, [1, 3]].count_nonzero().item(), 0)
        self.assertGreater(features.grad[:, [0, 2]].abs().sum().item(), 0)

    def test_final_detection_scores_labels_and_boxes_preserved(self):
        baseline, head = make_head(base=True), make_head()
        baseline.test_cfg = head.test_cfg = dict(max_per_img=4)
        with torch.no_grad():
            head.enc_mu_cls.bias.fill_(100.)
            head.dec_mu_cls.bias.fill_(-100.)
        kw = inputs(dn=0)
        args = [kw[k] for k in ('hidden_states', 'references', 'memory_text',
                                'text_token_mask')]
        metas = [dict(img_shape=(12, 16), scale_factor=(.5, .5))] * 2
        for mapping in (None, {1:[1], 2:[2, 3], 3:[4, 5]}):
            for rescale in (False, True):
                expected = baseline.predict_by_feat(*baseline(*args),
                    batch_img_metas=metas, batch_token_positive_maps=[mapping]*2,
                    rescale=rescale)
                actual = head.predict_by_feat(*head(*args),
                    batch_img_metas=metas, batch_token_positive_maps=[mapping]*2,
                    rescale=rescale)
                for before, after in zip(expected, actual):
                    for field in ('scores', 'labels', 'bboxes'):
                        torch.testing.assert_close(getattr(before, field),
                                                   getattr(after, field), rtol=0, atol=0)

    def test_encoder_decoder_use_their_own_boxes_and_counts(self):
        head, kw = make_head(common_quality_topk=2), inputs()
        calls = []
        def capture(boxes, gt, metas, valid_mask=None, **cfg):
            calls.append((boxes.detach().clone(), valid_mask))
            targets = boxes.new_zeros(boxes.shape[:2])
            targets[0, :len(calls)] = .5
            return targets, torch.ones_like(targets, dtype=torch.bool)
        with patch.dict(Head._common_quality_loss.__globals__, common_quality_targets=capture):
            losses = head.loss(**kw)
        torch.testing.assert_close(calls[0][0], kw['enc_quality_boxes'])
        self.assertIs(calls[0][1], kw['enc_quality_valid_mask'])
        actual_boxes = head(kw['hidden_states'], kw['references'], kw['memory_text'],
                            kw['text_token_mask'])[1][-1, :, 2:]
        torch.testing.assert_close(calls[1][0], actual_boxes)
        self.assertIsNone(calls[1][1])
        self.assertEqual([c[2]['avg_factor'] for c in head.loss_mu_quality.calls], [1., 2.])
        self.assertTrue(torch.isfinite(losses['enc_loss_mu_quality']))
        self.assertTrue(torch.isfinite(losses['dec_loss_mu_quality']))

    def test_each_loss_gradient_reaches_mu_features_bias_but_not_boxes(self):
        for branch in ('enc', 'dec'):
            head, kw = make_head(), inputs()
            kw['class_common'].retain_grad()
            loss = head.loss(**kw)[f'{branch}_loss_mu_quality']
            loss.backward()
            self.assertGreater(kw['class_common'].grad.abs().sum().item(), 0)
            self.assertGreater(kw['memory_text'].grad[:, 1:6].abs().sum().item(), 0)
            self.assertEqual(kw['memory_text'].grad[:, [0, 6]].count_nonzero().item(), 0)
            torch.testing.assert_close(kw['memory_text'].grad[:, 1],
                                       2 * kw['memory_text'].grad[:, 2])
            self.assertGreater(getattr(head, f'{branch}_mu_cls').bias.grad.abs().sum().item(), 0)
            other = 'dec' if branch == 'enc' else 'enc'
            self.assertIsNone(getattr(head, f'{other}_mu_cls').bias.grad)
            self.assertIsNone(kw['enc_quality_boxes'].grad)
            self.assertTrue(all(p.grad is None for p in head.reg_branches.parameters()))
            if branch == 'enc':
                self.assertIsNone(kw['hidden_states'].grad)
                self.assertGreater(kw['enc_quality_features'].grad[:, :6].abs().sum().item(), 0)
                self.assertEqual(kw['enc_quality_features'].grad[:, 6:].count_nonzero().item(), 0)
            else:
                grad = kw['hidden_states'].grad
                self.assertEqual(grad[:-1].count_nonzero().item(), 0)
                self.assertEqual(grad[-1, :, :2].count_nonzero().item(), 0)
                self.assertGreater(grad[-1, :, 2:].abs().sum().item(), 0)
                self.assertIsNone(kw['enc_quality_features'].grad)

    def test_bias_independence_and_weight_scaling(self):
        head, kw = make_head(), inputs()
        self.assertIsNot(head.enc_mu_cls.bias, head.dec_mu_cls.bias)
        losses = head.loss(**kw)
        doubled = make_head(enc=.2, dec=.3).loss(**inputs())
        torch.testing.assert_close(doubled['enc_loss_mu_quality'], 2*losses['enc_loss_mu_quality'])
        torch.testing.assert_close(doubled['dec_loss_mu_quality'], 3*losses['dec_loss_mu_quality'])
        with torch.no_grad():
            head.enc_mu_cls.bias.add_(1)
        torch.testing.assert_close(head.dec_mu_cls.bias, torch.tensor([-math.log(99)]))

    def test_empty_gt_no_dn_and_amp_are_finite(self):
        for empty in (False, True):
            head, kw = make_head(), inputs(empty=empty, dn=0)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                losses = head.loss(**kw)
            quality = losses['enc_loss_mu_quality'] + losses['dec_loss_mu_quality']
            self.assertTrue(torch.isfinite(quality))
            quality.backward()
            self.assertTrue(torch.isfinite(kw['memory_text'].grad).all())
            if empty:
                for _, target, args in head.loss_mu_quality.calls:
                    self.assertEqual(target.count_nonzero().item(), 0)
                    self.assertEqual(args['avg_factor'], 1.)

    def test_all_invalid_returns_graph_connected_zero(self):
        head, kw = make_head(enc=.1, dec=0.), inputs()
        kw['enc_quality_valid_mask'].fill_(False)
        loss = head.loss(**kw)['enc_loss_mu_quality']
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(kw['enc_quality_features'].grad.count_nonzero().item(), 0)
        self.assertIsNotNone(head.enc_mu_cls.bias.grad)

    def test_invalid_features_do_not_contaminate_mu_gradient(self):
        head = make_head()
        features = torch.tensor([[[1., 2., 3., 4.], [float('nan')] * 4]],
                                 requires_grad=True)
        boxes = torch.ones(1, 2, 4) * .5
        common = torch.randn(1, 4, requires_grad=True)
        loss = head._common_quality_loss(features, boxes, common,
            [SimpleNamespace(bboxes=torch.empty(0, 4))], [dict(img_shape=(10, 10))],
            head.enc_mu_cls, torch.tensor([[True, False]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(common.grad).all())
        self.assertEqual(features.grad[:, 1].count_nonzero().item(), 0)

    def test_constructor_validation_and_missing_inputs(self):
        for name in ('enc_mu_quality_loss_weight', 'dec_mu_quality_loss_weight'):
            for value in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    Head(**{name:value})
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                make_head(common_quality_beta=value)
        head = make_head(enc=0., dec=0.)
        self.assertFalse(hasattr(head, 'enc_mu_cls'))
        self.assertFalse(hasattr(head, 'dec_mu_cls'))
        kw = inputs()
        kw.pop('class_common')
        head.loss(**kw)
        with self.assertRaisesRegex(ValueError, 'class_common'):
            make_head().loss(**kw)
        kw = inputs()
        kw.pop('enc_quality_boxes')
        with self.assertRaisesRegex(ValueError, 'pre-top-k'):
            make_head().loss(**kw)


class CommonQualityDetectorTests(unittest.TestCase):
    def test_all_encoder_candidates_and_validity_preserve_selection_and_dn(self):
        for training in (False, True):
            baseline, model = fixture(), fixture()
            model.bbox_head.enc_mu_quality_loss_weight = .1
            model.train(training)
            baseline.train(training)
            memory = torch.randn(2, 5, 8, requires_grad=True)
            mask = torch.tensor([[False, False, True, False, False]] * 2)
            proposals = torch.zeros(2, 5, 4)
            proposals[:, 3] = float('inf')
            for m in (baseline, model):
                m.gen_encoder_output_proposals = lambda *a: (memory, proposals)
            args = dict(memory=memory, memory_mask=mask, spatial_shapes=torch.tensor([[1, 5]]),
                        memory_text=torch.randn(2, 8, 8),
                        text_token_mask=torch.ones(2, 8, dtype=torch.bool),
                        batch_data_samples=samples())
            torch.manual_seed(37)
            expected_decoder, expected_head = baseline.pre_decoder(**args)
            torch.manual_seed(37)
            decoder, head = model.pre_decoder(**args)
            for key, value in expected_decoder.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, decoder[key], rtol=0, atol=0)
                else:
                    self.assertEqual(value, decoder[key])
            for key, value in expected_head.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, head[key], rtol=0, atol=0)
                else:
                    self.assertEqual(value, head[key])
            if training:
                self.assertIs(head['enc_quality_features'], memory)
                self.assertEqual(head['enc_quality_boxes'].shape, (2, 5, 4))
                self.assertFalse(head['enc_quality_boxes'].requires_grad)
                self.assertEqual(head['enc_quality_valid_mask'].tolist(),
                                  [[True, True, False, False, True]] * 2)
                self.assertEqual(head['enc_outputs_coord'].shape[1], 3)
            else:
                self.assertEqual(head.keys(), expected_head.keys())

    def test_same_final_common_for_both_branches_etf_and_inference_preserved(self):
        for enc, dec in ((.1, 0.), (0., .1), (.1, .1)):
            for etf in (0., 1.):
                model = fixture(weight=etf)
                model.bbox_head.enc_mu_quality_loss_weight = enc
                model.bbox_head.dec_mu_quality_loss_weight = dec
                Detector.loss.__globals__['_second_order_representations'] = (
                    second._second_order_representations)
                losses, _ = run(model, samples())
                seen = model.bbox_head.seen
                tok, _, spans, _ = model.get_tokens_and_prompts(
                    ('crazing', 'inclusion', 'patches', 'pitted_surface',
                     'rolled-in_scale', 'scratches'), True)
                mapping = model._get_second_order_class_token_map(tok, spans)
                prototypes = second._second_order_representations(
                    seen['memory_text'], [mapping] * 2, seen['text_token_mask'])
                torch.testing.assert_close(seen['class_common'], prototypes.mean(1))
                self.assertEqual('enc_quality_features' in seen, bool(enc))
                self.assertEqual('loss_second_order_etf' in losses, bool(etf))
                if etf:
                    torch.testing.assert_close(losses['loss_second_order_etf'],
                        second.second_order_etf_loss(seen['memory_text'], [mapping] * 2,
                                                    seen['text_token_mask']))
                seen['class_common'].square().sum().backward()
                self.assertGreater(model.language_model.embedding.weight.grad.abs().sum().item(), 0)
                model.eval()
                model.predict(torch.zeros(2, 3, 4, 4), samples())
                self.assertNotIn('class_common', model.bbox_head.seen)
                self.assertNotIn('enc_quality_features', model.bbox_head.seen)


if __name__ == '__main__':
    unittest.main()
