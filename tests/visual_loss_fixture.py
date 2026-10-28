"""Real serial detection losses and Hungarian costs on small CPU inputs.

Only module loading, attention and registry construction are replaced. Actual
head loss routing, target generation, focal/L1/GIoU, costs and scipy assignment
are executed from repository source, without importing the MMCV extension.
"""
import ast
from abc import abstractmethod
import functools
from types import SimpleNamespace as NS

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from visual_test_support import ENV, Instances, execute, node, sample, source, visual


env = dict(ENV, F=F, Tensor=torch.Tensor, functools=functools, partial=functools.partial,
           abstractmethod=abstractmethod, InstanceData=Instances,
           QualityFocalLoss=type('QualityFocalLoss', (), {}),
           reduce_mean=lambda x: x, linear_sum_assignment=linear_sum_assignment,
           AssignResult=lambda num_gts, gt_inds, max_overlaps, labels: NS(
               num_gts=num_gts, gt_inds=gt_inds, max_overlaps=max_overlaps, labels=labels))


def functions(rel, names, remove_decorators=False):
    nodes = [node(source(rel), name) for name in names]
    if remove_decorators:
        for n in nodes:
            n.decorator_list = []
    execute(nodes, env)


def klass(rel, name, parent, methods=None):
    cls = node(source(rel), name)
    cls.decorator_list = []
    if parent is not None:
        cls.bases = [ast.parse(parent, mode='eval').body]
    if methods is not None:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    execute([cls], env)


functions('mmdet/structures/bbox/transforms.py', ['bbox_cxcywh_to_xyxy', 'bbox_xyxy_to_cxcywh'])
functions('mmdet/structures/bbox/bbox_overlaps.py', ['fp16_clamp', 'bbox_overlaps'])
functions('mmdet/models/utils/misc.py', ['multi_apply'])
functions('mmdet/models/losses/utils.py', ['reduce_loss', 'weight_reduce_loss', 'weighted_loss'])
functions('mmdet/models/losses/focal_loss.py', ['py_sigmoid_focal_loss'])
functions('mmdet/models/losses/smooth_l1_loss.py', ['l1_loss'])
functions('mmdet/models/losses/iou_loss.py', ['giou_loss'])
for name in ('BaseMatchCost', 'FocalLossCost', 'BinaryFocalLossCost', 'BBoxL1Cost', 'IoUCost'):
    klass('mmdet/models/task_modules/assigners/match_cost.py', name, None)
klass('mmdet/models/task_modules/assigners/hungarian_assigner.py', 'HungarianAssigner',
      'object', ['assign'])
klass('mmdet/models/dense_heads/detr_head.py', 'DETRHead', 'nn.Module', ['loss_by_feat', 'get_targets'])
env['DeformableDETRHead'] = type('DeformableDETRHead', (env['DETRHead'],), {})
klass('mmdet/models/dense_heads/dino_head.py', 'DINOHead', 'DeformableDETRHead',
      ['loss_by_feat', 'loss_dn', 'get_dn_targets', 'split_outputs'])
klass('mmdet/models/dense_heads/grounding_dino_head.py', 'GroundingDINOHead', 'DINOHead')
functions('mmdet/models/dense_heads/grounding_dino_head.py', ['ContrastiveEmbed'])


def head(classes=2):
    cls = env['GroundingDINOHead']
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.num_classes = classes
    model.max_text_len = 16
    model.num_pred_layer = 4
    model.bg_cls_weight = .1
    model.sync_cls_avg_factor = True
    model.cls_branches = nn.ModuleList([env['ContrastiveEmbed'](16) for _ in range(4)])
    model.reg_branches = nn.ModuleList([nn.Linear(8, 4) for _ in range(4)])
    model.loss_cls = env['py_sigmoid_focal_loss']
    model.loss_bbox = lambda pred, target, weight, avg_factor: 5 * env['l1_loss'](
        pred, target, weight, avg_factor=avg_factor)
    model.loss_iou = lambda pred, target, weight, avg_factor: 2 * env['giou_loss'](
        pred, target, weight.mean(-1), avg_factor=avg_factor)
    model.assigner = env['HungarianAssigner']()
    model.assigner.match_costs = [env['BinaryFocalLossCost'](weight=2),
                                  env['BBoxL1Cost'](box_format='xywh', weight=5),
                                  env['IoUCost'](iou_mode='giou', weight=2)]
    return model


def data(classes=2, empty=False):
    samples = [sample([[4, 5, 20, 22]], [classes - 1]), sample([], [])]
    if empty:
        samples = [sample([], []), sample([], [])]
    for s in samples:
        labels = s.gt_instances.labels
        s.gt_instances.positive_maps = F.one_hot(labels, num_classes=16).float()
        s.gt_instances.text_token_mask = torch.ones(len(labels), classes, dtype=torch.bool)
    names = {1: [1, 2]}
    visual_indices = [3]
    if classes == 2:
        names[2] = [5]
        visual_indices.append(6)
    layout = dict(names=[names] * 2, visual=[visual_indices] * 2)
    return samples, layout
