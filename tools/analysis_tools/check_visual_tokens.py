#!/usr/bin/env python3
"""Full-model visual-token smoke test; requires MMCV, BERT and real Support data.

Runs one synthetic query batch through the configured real model and losses,
then inference, without changing a checkpoint or taking an optimizer step.
"""
import argparse
import os
import json
import time

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


def profile(model, classes, device, steps, amp):
    """Same model/query, support on/off; report full allocated CUDA peak too."""
    if device.type != 'cuda':
        raise ValueError('--profile requires CUDA.')
    cfg, old_stats = model.support_visual_cfg, model._visual_stats
    report = {}
    model._visual_stats = None
    try:
        for enabled in (False, True):
            model.support_visual_cfg = cfg if enabled else None
            model.train()
            timings, peaks, increments = [], [], []
            for i in range(steps + 1):  # first step is warmup
                model.zero_grad(set_to_none=True)
                data = model.data_preprocessor(batch(classes, device), training=True)
                torch.cuda.synchronize(device)
                initial = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    losses = model.loss(data['inputs'], data['data_samples'])
                    loss = sum(losses.values())
                loss.backward()
                torch.cuda.synchronize(device)
                if i:
                    timings.append(1000 * (time.perf_counter() - start))
                    peak = torch.cuda.max_memory_allocated(device)
                    peaks.append(peak / 2**20)
                    increments.append((peak-initial) / 2**20)
                del loss, losses, data
            report['support_on' if enabled else 'support_off'] = dict(
                mean_forward_backward_ms=sum(timings)/steps,
                peak_allocated_mib=max(peaks), peak_increment_mib=max(increments))
    finally:
        model.support_visual_cfg, model._visual_stats = cfg, old_stats
        model.zero_grad(set_to_none=True)
    print('Profile:', json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-steps', type=int, default=3)
    args = parser.parse_args()
    if args.profile_steps < 1:
        parser.error('--profile-steps must be positive')
    register_all_modules()
    cfg = Config.fromfile(args.config)
    if not cfg.model.get('support_visual_cfg'):
        raise ValueError('Use one of the support finetuning configs.')
    device = torch.device(args.device)
    model = MODELS.build(cfg.model).to(device)
    model.init_weights()
    load_checkpoint(model, args.checkpoint or cfg.load_from, map_location='cpu')
    for layer in model.encoder.support_layers:
        assert isinstance(layer.attn, torch.nn.MultiheadAttention)
        assert (layer.attn.embed_dim, layer.attn.num_heads, layer.attn.dropout) == (256, 4, 0.)
        assert not hasattr(layer, 'text_norm')
        if args.checkpoint is None:
            torch.testing.assert_close(layer.gamma, torch.full_like(layer.gamma, 1e-2))
            torch.testing.assert_close(layer.support_norm.weight,
                                       torch.ones_like(layer.support_norm.weight))
            torch.testing.assert_close(layer.support_norm.bias,
                                       torch.zeros_like(layer.support_norm.bias))
    classes = cfg.model.support_visual_cfg.class_names
    stats = VisualTokenStats(len(classes), device, len(model.encoder.support_layers))
    model._visual_stats = stats
    model.train()
    train_data = model.data_preprocessor(batch(classes, device), training=True)
    # --amp=False forces the FP32 check, including backbone feature extraction.
    model.use_autocast = args.amp
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
        losses = model.loss(train_data['inputs'], train_data['data_samples'])
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert not any('region_text' in name for name in losses)
    sum(losses.values()).backward()
    for layer in model.encoder.support_layers:
        for parameter in layer.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.backbone.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    print('Real detection losses/backward OK:', sorted(losses))
    print('Training statistics:', stats.report(classes))
    if device.type == 'cuda':
        print('Training peak allocated MiB:', torch.cuda.max_memory_allocated(device) / 2**20)

    assert model._support_memory is None  # training never retains descriptors
    del losses, train_data
    model.zero_grad(set_to_none=True)
    model.eval()
    evaluation_memory = model.refresh_support_memory()
    stats = VisualTokenStats(len(classes), device, len(model.encoder.support_layers))
    model._visual_stats = stats
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
        test_data = model.data_preprocessor(batch(classes, device), training=False)
        results = model.predict(test_data['inputs'], test_data['data_samples'])
        memory = model.support_memory()
        assert memory is evaluation_memory
        assert memory['tokens'].shape == (int(memory['counts'].sum()), 256)
        assert len(memory['counts']) == len(classes) and (memory['counts'] > 0).all()
        assert (memory['counts'] % 49 == 0).all()
        assert not memory['tokens'].requires_grad
        assert torch.equal(memory['class_ids'].bincount(minlength=len(classes)), memory['counts'])
        assert torch.isfinite(memory['tokens']).all()
        assert torch.isfinite(results[0].pred_instances.scores).all()
        model.load_state_dict(model.state_dict())
        assert model._support_memory is None
        with torch.autocast(device.type, enabled=False):
            reloaded_memory = model.refresh_support_memory()
        assert reloaded_memory is not evaluation_memory
        torch.testing.assert_close(reloaded_memory['tokens'], evaluation_memory['tokens'],
                                   rtol=0, atol=0)
    print('Real inference, evaluation reuse and checkpoint refresh OK.')
    print('Validation statistics:', stats.report(classes))
    del results, test_data, memory, evaluation_memory, reloaded_memory
    if args.profile:
        profile(model, classes, device, args.profile_steps, args.amp)


if __name__ == '__main__':
    main()
