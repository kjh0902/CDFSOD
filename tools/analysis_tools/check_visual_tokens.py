#!/usr/bin/env python3
"""Full-model visual-token smoke test; requires MMCV, BERT and real Support data.

Runs one synthetic query batch through the configured real model and losses,
then inference, without changing a checkpoint or taking an optimizer step.
"""
import argparse
import copy
import os

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules
from mmdet.utils.visual_token_stats import VisualTokenStats


def batch(classes, device):
    sample = DetDataSample(metainfo=dict(
        img_id=-1, ori_shape=(256, 256), img_shape=(256, 256),
        scale_factor=(1., 1.), batch_input_shape=(256, 256)))
    sample.text = tuple(classes)
    sample.custom_entities = True
    sample.gt_instances = InstanceData(
        bboxes=torch.tensor([[40., 40., 100., 120.]], device=device),
        labels=torch.zeros(1, dtype=torch.long, device=device))
    return dict(inputs=[torch.full((3, 256, 256), 114., device=device)],
                data_samples=[sample])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if not cfg.model.get('support_visual_cfg'):
        raise ValueError('Use one of the visual-token finetuning configs.')
    device = torch.device(args.device)
    model = MODELS.build(cfg.model).to(device)
    model.init_weights()
    load_checkpoint(model, args.checkpoint or cfg.load_from, map_location='cpu')
    classes = cfg.model.support_visual_cfg.class_names
    stats = VisualTokenStats(len(classes), device)
    model._visual_stats = model.bbox_head._visual_stats = stats
    model.train()
    train_data = model.data_preprocessor(batch(classes, device), training=True)
    losses = model.loss(**dict(batch_inputs=train_data['inputs'],
                               batch_data_samples=train_data['data_samples']))
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert not any('region_text' in name for name in losses)
    sum(losses.values()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.backbone.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    assert stats.pending is None
    print('Real detection losses/backward OK:', sorted(losses))
    print('Training statistics:', stats.report(classes))

    model.zero_grad(set_to_none=True)
    model.eval()
    stats = VisualTokenStats(len(classes), device)
    model._visual_stats = model.bbox_head._visual_stats = stats
    with torch.no_grad():
        test_data = model.data_preprocessor(batch(classes, device), training=False)
        results = model.predict(test_data['inputs'], test_data['data_samples'])
        assert stats.pending is None
        prototypes = model.support_prototypes()
        assert prototypes.shape == (len(classes), 256) and not prototypes.requires_grad
        assert torch.isfinite(prototypes).all()
        assert torch.isfinite(results[0].pred_instances.scores).all()
        # Loading weights, even identical weights, invalidates the eval cache.
        model.load_state_dict(copy.deepcopy(model.state_dict()))
        assert model._support_prototypes is None
    print('Real inference, prototype dimensions and cache invalidation OK.')
    print('Validation statistics:', stats.report(classes))


if __name__ == '__main__':
    main()
