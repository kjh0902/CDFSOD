"""Compare real detection outputs/losses with the pinned serial ACL baseline.

Requires the normal training environment (MMCV ops, BERT and a checkpoint).
Uses synthetic images/GT, the configured full model, and no optimizer steps.
"""
import argparse
import ast
import copy
import os
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = '8926970ebff1a549088b0a4c87c272e1a70fe0dd'
sys.path.insert(0, str(ROOT))


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    return value


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b, (a, b)


def check_zero_weight(model, inputs, samples):
    """Run both implementations on the same weights, buffers, input and RNG."""
    text = subprocess.check_output(
        ['git', 'show', f'{BASE}:mmdet/models/detectors/grounding_dino.py'],
        cwd=ROOT, encoding='utf8')
    baseline_class = next(n for n in ast.parse(text).body
                          if isinstance(n, ast.ClassDef) and n.name == 'GroundingDINO')
    names = ('loss', 'pre_decoder', 'forward_transformer')
    functions = [n for n in baseline_class.body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(model.loss.__func__.__globals__)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
                        names=[ast.alias(name='annotations')], level=0)] + functions,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<serial ACL baseline>', 'exec'),
         namespace)
    originals = {name: getattr(model, name) for name in names}
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    training = model.training
    weight = model.lambda_region_text
    head_forward = model.bbox_head.forward
    captured = []

    def capture(*args, **kwargs):
        output = head_forward(*args, **kwargs)
        captured.append(snapshot(output))
        return output

    def restore_state():
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3]:
            torch.cuda.set_rng_state_all(rng[3])
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])

    try:
        model.train()
        model.lambda_region_text = 0.0
        model.bbox_head.forward = capture
        results = []
        with torch.no_grad():
            for baseline in (True, False):
                for name in names:
                    setattr(model, name, MethodType(namespace[name], model)
                            if baseline else originals[name])
                restore_state()
                captured.clear()
                losses = model.loss(inputs, copy.deepcopy(samples))
                assert 'loss_region_text' not in losses
                assert captured, 'No detection-head output was captured.'
                results.append((snapshot(losses), copy.deepcopy(captured),
                                torch.get_rng_state().clone(),
                                torch.cuda.get_rng_state_all()
                                if torch.cuda.is_available() else []))
        assert_equal(*results)
        return sorted(results[0][0])
    finally:
        for name, method in originals.items():
            setattr(model, name, method)
        model.bbox_head.forward = head_forward
        model.lambda_region_text = weight
        model.train(training)
        restore_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--checkpoint', help='Defaults to config.load_from')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.structures import InstanceData
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device)
    checkpoint = args.checkpoint or cfg.get('load_from')
    if not checkpoint:
        raise ValueError('A checkpoint is required for the full-model regression.')
    load_checkpoint(model, checkpoint, map_location='cpu')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    classes = tuple(cfg.class_names)
    # Enough spatial tokens for the unchanged 900-query config, including DN.
    images = [torch.randint(0, 256, (3, 256, 320), dtype=torch.uint8)
              for _ in range(2)]
    samples = []
    for image_id in range(2):
        sample = DetDataSample(metainfo=dict(
            img_id=image_id, img_shape=(256, 320), ori_shape=(256, 320),
            scale_factor=(1., 1.), text=classes, custom_entities=True))
        sample.gt_instances = InstanceData(
            bboxes=torch.tensor([[24., 32., 100., 144.], [128., 64., 260., 224.]]),
            labels=torch.tensor([0, len(classes) - 1], dtype=torch.long))
        samples.append(sample)
    batch = model.data_preprocessor(dict(inputs=images, data_samples=samples),
                                    training=True)
    keys = check_zero_weight(model, batch['inputs'], batch['data_samples'])
    print('PASS: bitwise identical full detection outputs, losses and torch RNG.')
    print('Compared loss keys:', ', '.join(keys))


if __name__ == '__main__':
    main()
