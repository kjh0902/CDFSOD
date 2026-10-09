"""Detached, class-agnostic IoU targets for auxiliary common quality scores."""
import math
from numbers import Integral

import torch
from torch import Tensor

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


def validate_common_quality_cfg(topk, ignore_iou_thr):
    if isinstance(topk, bool) or not isinstance(topk, Integral) or topk < 1:
        raise ValueError('common_quality_topk must be a positive integer.')
    if not math.isfinite(ignore_iou_thr) or not 0 <= ignore_iou_thr <= 1:
        raise ValueError('common_quality_ignore_iou_thr must be in [0, 1].')


@torch.no_grad()
def common_quality_targets(bbox_preds: Tensor, batch_gt_instances,
                           batch_img_metas, valid_mask=None, topk=5,
                           ignore_iou_thr=0.5):
    """Return [B,N] IoU targets and a boolean supervision mask.

    Boxes are normalized cxcywh; GT boxes are absolute xyxy in img_shape.
    Each GT nominates up to topk valid candidates with strictly positive IoU.
    The union is foreground. Its assigned GT is the one with highest IoU
    (ties use the first GT), so a candidate has one class-agnostic quality
    target even when multiple GTs nominate it. Unselected candidates with
    maximum IoU >= ignore_iou_thr are ignored, as are invalid/padded boxes.
    No class logits, Hungarian targets or differentiable boxes are consumed.
    """
    validate_common_quality_cfg(topk, ignore_iou_thr)
    if bbox_preds.ndim != 3 or bbox_preds.shape[-1] != 4:
        raise ValueError('Expected quality boxes with shape [B, N, 4].')
    batch, count = bbox_preds.shape[:2]
    if len(batch_gt_instances) != batch or len(batch_img_metas) != batch:
        raise ValueError('Expected one GT instance and image meta per sample.')
    if valid_mask is None:
        valid_mask = torch.ones((batch, count), device=bbox_preds.device,
                                dtype=torch.bool)
    elif (valid_mask.shape != (batch, count)
          or valid_mask.dtype != torch.bool
          or valid_mask.device != bbox_preds.device):
        raise ValueError('Expected a boolean [B, N] quality validity mask.')

    with torch.autocast(device_type=bbox_preds.device.type, enabled=False):
        boxes = bbox_preds if bbox_preds.dtype == torch.float64 else bbox_preds.float()
        valid = (valid_mask & torch.isfinite(boxes).all(-1)
                 & (boxes[..., 2:] > 0).all(-1))
        targets = boxes.new_zeros((batch, count))
        supervised = valid.clone()
        for b, (gt, meta) in enumerate(zip(batch_gt_instances, batch_img_metas)):
            indices = valid[b].nonzero(as_tuple=True)[0]
            if indices.numel() == 0 or len(gt.bboxes) == 0:
                continue
            img_h, img_w = meta['img_shape'][:2]
            factor = boxes.new_tensor([img_w, img_h, img_w, img_h])
            predicted = bbox_cxcywh_to_xyxy(boxes[b, indices]) * factor
            ious = bbox_overlaps(predicted, gt.bboxes.to(boxes)).clamp(0, 1)
            max_iou = ious.max(dim=1).values
            values, nominees = ious.topk(min(topk, indices.numel()), dim=0)
            selected = torch.zeros(indices.numel(), device=boxes.device,
                                   dtype=torch.bool)
            selected[nominees[values > 0]] = True
            targets[b, indices[selected]] = max_iou[selected]
            supervised[b, indices] = selected | (max_iou < ignore_iou_thr)
    return targets, supervised
