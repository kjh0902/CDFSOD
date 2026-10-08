"""Production head/target/loss regressions without MMCV or pretrained weights.

Only the heavy base initialization, Hungarian solver and box-loss modules are
test doubles. The actual head, inherited DINO/DETR loss routing, Hungarian target
construction and Python sigmoid focal implementation execute unchanged.
"""
import ast
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

BASE = '7fbec8da7f6cab0546449021c0437b2284db4257'
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


def multi_apply(fn, *args, **kwargs):
    return tuple(map(list, zip(*(fn(*items, **kwargs) for items in zip(*args)))))


def cxcywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2,
                      box[..., :2] + box[..., 2:] / 2), dim=-1)


def xyxy_to_cxcywh(box):
    return torch.cat(((box[..., :2] + box[..., 2:]) / 2,
                      box[..., 2:] - box[..., :2]), dim=-1)


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
                   MODELS=SimpleNamespace(build=lambda cfg: Focal(**cfg)),
                   QualityFocalLoss=type('QualityFocalLoss', (), {}),
                   InstanceData=SimpleNamespace,
                   bbox_cxcywh_to_xyxy=cxcywh_to_xyxy,
                   bbox_xyxy_to_cxcywh=xyxy_to_cxcywh,
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
    return dict(hidden_states=states, references=[torch.full((2, dn + 4, 4), .5)] * 4,
                memory_text=text, text_token_mask=mask,
                enc_outputs_class=torch.randn(2, 4, 8),
                enc_outputs_coord=torch.rand(2, 4, 4), batch_data_samples=data,
                dn_meta=dict(num_denoising_queries=dn, num_denoising_groups=1),
                class_common=common)


def make_head(base=False, weight=1.):
    torch.manual_seed(17)
    cls = BaseHead if base else Head
    kwargs = {} if base else dict(mu_focal_loss_weight=weight)
    return cls(contrastive_cfg=dict(max_text_len=8, log_scale='auto', bias=True), **kwargs)


class DecoderMuFocalTests(unittest.TestCase):
    def test_base_losses_assignment_count_and_detection_outputs_unchanged(self):
        for dn in (0, 2):
            baseline, head = make_head(base=True), make_head()
            kw = inputs(dn=dn)
            expected = baseline.loss(**{k: v for k, v in kw.items() if k != 'class_common'})
            actual = head.loss(**kw)
            self.assertEqual(len(head.assigner.targets), 2 * (3 + 1))
            self.assertEqual(len(baseline.assigner.targets), len(head.assigner.targets))
            self.assertEqual(actual.keys() - expected.keys(), {'loss_mu_focal'})
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for before, after in zip(baseline.assigner.targets, head.assigner.targets):
                torch.testing.assert_close(before, after)
            args = [kw[k] for k in ('hidden_states', 'references', 'memory_text', 'text_token_mask')]
            for before, after in zip(baseline(*args), head(*args)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_final_hungarian_targets_and_focal_formula(self):
        head, kw = make_head(weight=2.), inputs()
        loss = head.loss(**kw)['loss_mu_focal']
        target = torch.stack(head.assigner.targets[4:6]).gt(0).float()
        self.assertFalse(torch.equal(target, torch.stack(head.assigner.targets[6:8]).gt(0)))
        torch.testing.assert_close(head.loss_mu_focal.last_target, target.reshape(-1, 1))
        logits = (kw['hidden_states'][-1, :, 2:] * kw['class_common'][:, None]).sum(-1)
        logits = logits / math.sqrt(4) + head.mu_cls.bias
        prob = logits.sigmoid()
        error = (1 - prob) * target + prob * (1 - target)
        weight = .25 * target + .75 * (1 - target)
        expected = 2 * (F.binary_cross_entropy_with_logits(
            logits, target, reduction='none') * weight * error.square()).sum()
        expected = expected / (1 + torch.finfo(torch.float32).eps)
        torch.testing.assert_close(loss, expected.reshape_as(loss))

    def test_mu_gradient_excludes_dn_and_earlier_layers_reaches_raw_class_means(self):
        head, kw = make_head(), inputs()
        loss = head.loss(**kw)['loss_mu_focal']
        loss.backward()
        grad = kw['hidden_states'].grad
        self.assertEqual(grad[:-1].count_nonzero().item(), 0)
        self.assertEqual(grad[-1, :, :2].count_nonzero().item(), 0)
        self.assertGreater(grad[-1, :, 2:].abs().sum().item(), 0)
        text_grad = kw['memory_text'].grad
        self.assertEqual(text_grad[:, [0, 6]].count_nonzero().item(), 0)
        self.assertTrue((text_grad[:, 1:6].norm(dim=-1) > 0).all())
        # Equal class weighting, not a mean over all tokens.
        torch.testing.assert_close(text_grad[:, 1], 2 * text_grad[:, 2])
        self.assertGreater(head.mu_cls.bias.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in head.reg_branches.parameters()))

    def test_empty_gt_no_dn_and_autocast_are_finite(self):
        for empty in (False, True):
            head, kw = make_head(), inputs(empty=empty, dn=0)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = head.loss(**kw)['loss_mu_focal']
            self.assertTrue(torch.isfinite(loss))
            if empty:
                self.assertEqual(head.loss_mu_focal.last_target.count_nonzero().item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(kw['memory_text'].grad).all())

    def test_disabled_constructor_and_missing_common(self):
        head = make_head(weight=0.)
        kw = inputs()
        kw.pop('class_common')
        self.assertNotIn('loss_mu_focal', head.loss(**kw))
        self.assertFalse(hasattr(head, 'mu_cls'))
        for weight in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                make_head(weight=weight)
        with self.assertRaisesRegex(ValueError, 'class_common'):
            make_head().loss(**kw)

    def test_detector_uses_final_raw_common_without_changing_etf_or_prediction(self):
        for etf_weight in (0., 1.):
            model = fixture(weight=etf_weight)
            model.bbox_head.mu_focal_loss_weight = 1.
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
            self.assertEqual('loss_second_order_etf' in losses, bool(etf_weight))
            if etf_weight:
                torch.testing.assert_close(losses['loss_second_order_etf'],
                    second.second_order_etf_loss(seen['memory_text'], [mapping] * 2,
                                                seen['text_token_mask']))
            seen['class_common'].square().sum().backward()
            self.assertGreater(model.language_model.embedding.weight.grad.abs().sum().item(), 0)
            model.eval()
            model.predict(torch.zeros(2, 3, 4, 4), samples())
            self.assertNotIn('class_common', model.bbox_head.seen)


if __name__ == '__main__':
    unittest.main()
