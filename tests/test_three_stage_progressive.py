"""CPU tests of production scheduler/wrappers/hooks, without MMCV or weights.

Only registry/package loading avoids the heavyweight mmdet __init__. The
implementations under test and MMEngine optimizer/scheduler hooks are real.
"""
import ast
import copy
import importlib.util
import logging
import socket
from pathlib import Path
import subprocess
import sys
import types
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint
from mmengine.config import Config
from mmengine.hooks import ParamSchedulerHook, CheckpointHook
from mmengine.logging import MessageHub
from mmengine.model import MMDistributedDataParallel
from mmengine.optim import OptimWrapper, AmpOptimWrapper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREFIX = '_acl_three_stage_test'
for name in ('', '.engine', '.engine.optimizers', '.engine.schedulers', '.engine.hooks'):
    package = types.ModuleType(PREFIX + name)
    package.__path__ = []
    sys.modules[package.__name__] = package


def load(name, relative):
    spec = importlib.util.spec_from_file_location(PREFIX + '.' + name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


registry = load('registry', 'mmdet/registry.py')
with patch.dict(sys.modules, {'mmdet.registry': registry}):
    wrappers = load('engine.optimizers.active_param_optimizer_wrapper',
                    'mmdet/engine/optimizers/active_param_optimizer_wrapper.py')
    schedulers = load('engine.schedulers.three_stage_plateau_lr',
                      'mmdet/engine/schedulers/three_stage_plateau_lr.py')
    hooks = load('engine.hooks.three_stage_progressive_finetuning_hook',
                 'mmdet/engine/hooks/three_stage_progressive_finetuning_hook.py')
Wrapper = wrappers.ActiveParamOptimWrapper
Scheduler = schedulers.ThreeStagePlateauLR
StageHook = hooks.ThreeStageProgressiveFinetuningHook


def container(**children):
    module = nn.Module()
    for name, child in children.items():
        module.add_module(name, child)
    return module


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        bert = container(embeddings=nn.Linear(2, 2),
                         encoder=container(layer=nn.ModuleList(
                             [nn.Linear(2, 2) for _ in range(12)])))
        self.language_model = container(language_backbone=container(body=container(model=bert)))
        self.backbone = container(
            patch_embed=nn.Linear(2, 2), stages=nn.ModuleList(
                [container(block=nn.Linear(2, 2), downsample=nn.Linear(2, 2))
                 for _ in range(4)]),
            norm1=nn.LayerNorm(2), norm2=nn.LayerNorm(2), norm3=nn.LayerNorm(2))
        for name in ('text_feat_map', 'neck', 'encoder', 'decoder', 'bbox_head'):
            setattr(self, name, nn.Linear(2, 2))
        self.query_embedding = nn.Embedding(2, 2)

    def forward(self, x):
        # Every parameter participates. One unchanged reentrant checkpoint
        # also exercises the original DDP checkpointing combination.
        y = checkpoint(self.encoder, x, use_reentrant=True)
        terms = [p.square().sum() for n, p in self.named_parameters()
                 if not n.startswith('encoder.')]
        return y.square().sum() + x.sum() * torch.stack(terms).sum()


def setup(model=None, **scheduler_options):
    model = model if model is not None else ToyModel()
    bare = model.module if isinstance(model, MMDistributedDataParallel) else model
    groups = [dict(params=[p], lr=2e-5 if name.startswith(
        ('backbone.', 'language_model.')) else 1e-4)
        for name, p in bare.named_parameters()]
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.05)
    wrapper = Wrapper(optimizer, clip_grad=dict(max_norm=0.1, norm_type=2))
    scheduler = Scheduler(wrapper, **scheduler_options)
    stage_hook = StageHook()
    runner = types.SimpleNamespace(
        model=model, optim_wrapper=wrapper, param_schedulers=[scheduler],
        hooks=[ParamSchedulerHook(), stage_hook, CheckpointHook()],
        train_loop=types.SimpleNamespace(stop_training=False), epoch=0,
        logger=logging.getLogger('three_stage_test'),
        message_hub=MessageHub.get_instance('three_stage_test'))
    stage_hook.before_train(runner)
    return runner, stage_hook, scheduler


def validate(runner, stage_hook, value):
    # Match EpochBasedTrainLoop's scheduler count and validation hook order.
    scheduler_hook = runner.hooks[0]
    scheduler_hook.after_train_epoch(runner)
    runner.epoch += 1
    metrics = {'coco/bbox_mAP': value}
    scheduler_hook.after_val_epoch(runner, metrics)
    stage_hook.after_val_epoch(runner, metrics)


class ClippingTests(unittest.TestCase):
    def test_huge_frozen_gradient_does_not_affect_global_clipping(self):
        for frozen_size in (1., 1e12):
            p, q, frozen = [nn.Parameter(torch.ones(1)) for _ in range(3)]
            opt = torch.optim.AdamW([dict(params=[p], lr=1e-4),
                                    dict(params=[q], lr=2e-5),
                                    dict(params=[frozen], lr=0.)])
            wrapper = Wrapper(opt, clip_grad=dict(max_norm=0.1, norm_type=2))
            p.grad, q.grad, frozen.grad = [torch.tensor([v]) for v in (3., 4., frozen_size)]
            wrapper._clip_grad()
            torch.testing.assert_close(p.grad, torch.tensor([0.06]))
            torch.testing.assert_close(q.grad, torch.tensor([0.08]))
            torch.testing.assert_close(frozen.grad, torch.tensor([frozen_size]))
            self.assertEqual(wrapper.message_hub.get_scalar('train/grad_norm').current(), 5.)

    def test_empty_missing_duplicate_and_value_clipping(self):
        p, q = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        opt = torch.optim.SGD([dict(params=[p], lr=0.), dict(params=[q], lr=1.)])
        wrapper = Wrapper(opt, clip_grad=dict(max_norm=1.))
        p.grad = torch.tensor([99.])
        wrapper._clip_grad()  # q has no gradient: empty selection
        torch.testing.assert_close(p.grad, torch.tensor([99.]))
        q.grad = torch.tensor([2.])
        opt.param_groups[1]['params'].append(q)  # deduplicate clipping selection
        wrapper._clip_grad()
        torch.testing.assert_close(q.grad, torch.tensor([1.]))
        wrapper = Wrapper(opt, clip_grad=dict(type='value', clip_value=0.2))
        wrapper._clip_grad()
        torch.testing.assert_close(q.grad, torch.tensor([0.2]))
        torch.testing.assert_close(p.grad, torch.tensor([99.]))

    def test_accumulation_matches_mmengine_and_clipping_can_be_disabled(self):
        for clip in (None, dict(max_norm=0.1, norm_type=2)):
            parameters = [nn.Parameter(torch.tensor([2.])) for _ in range(2)]
            wrappers_to_compare = [
                cls(torch.optim.SGD([p], lr=0.1), accumulative_counts=2,
                    clip_grad=copy.deepcopy(clip))
                for cls, p in zip((Wrapper, OptimWrapper), parameters)]
            for wrapper, p in zip(wrappers_to_compare, parameters):
                wrapper.initialize_count_status(nn.Linear(1, 1), 0, 3)
                for _ in range(3):
                    wrapper.update_params(p.square().sum())
            torch.testing.assert_close(*parameters)

    def test_amp_keeps_mmengine_step_and_cli_selects_matching_wrapper(self):
        self.assertIs(wrappers.ActiveParamAmpOptimWrapper.step, AmpOptimWrapper.step)
        self.assertIs(wrappers.ActiveParamAmpOptimWrapper.backward, AmpOptimWrapper.backward)
        self.assertIs(wrappers.ActiveParamAmpOptimWrapper._clip_grad, Wrapper._clip_grad)
        tree = ast.parse((ROOT / 'tools/train.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        amp = next(n for n in main.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == 'args.amp is True')
        code = compile(ast.Module(body=[amp], type_ignores=[]), '<amp_cli>', 'exec')
        for source, target in [('OptimWrapper', 'AmpOptimWrapper'),
                               ('ActiveParamOptimWrapper', 'ActiveParamAmpOptimWrapper'),
                               ('ActiveParamAmpOptimWrapper', 'ActiveParamAmpOptimWrapper')]:
            cfg = Config(dict(optim_wrapper=dict(type=source)))
            exec(code, dict(args=types.SimpleNamespace(amp=True), cfg=cfg))
            self.assertEqual(cfg.optim_wrapper.type, target)
            self.assertEqual(cfg.optim_wrapper.loss_scale, 'dynamic')


class StageTests(unittest.TestCase):
    def test_patiences_cooldown_lr_masks_and_early_stop(self):
        runner, hook, scheduler = setup()
        self.assertEqual(scheduler.stage_patiences, (3, 5, 8))
        original_refs = scheduler.scheduled_lrs.copy()
        validate(runner, hook, 0.5)
        for _ in range(3):
            validate(runner, hook, 0.5)
            self.assertEqual(scheduler.stage, 1)
        validate(runner, hook, 0.5)
        self.assertEqual(scheduler.stage, 2)
        self.assertEqual(scheduler.patience, 5)
        self.assertEqual(scheduler.scheduled_lrs, [x / 2 for x in original_refs])
        self.assertEqual(scheduler.best, scheduler.rule_worse)
        validate(runner, hook, 0.4)  # new stage baseline, consumes cooldown
        for _ in range(5):
            validate(runner, hook, 0.4)
            self.assertEqual(scheduler.stage, 2)
        validate(runner, hook, 0.4)
        self.assertEqual(scheduler.stage, 3)
        self.assertEqual(scheduler.patience, 8)
        self.assertEqual(scheduler.scheduled_lrs, [x / 4 for x in original_refs])
        validate(runner, hook, 0.3)
        for _ in range(8):
            validate(runner, hook, 0.3)
            self.assertFalse(runner.train_loop.stop_training)
        validate(runner, hook, 0.3)
        self.assertTrue(runner.train_loop.stop_training)
        self.assertEqual(scheduler.scheduled_lrs, [x / 4 for x in original_refs])
        self.assertEqual(runner.optim_wrapper.base_param_settings['lr'], 2.5e-5)
        self.assertEqual(scheduler.get_last_value(),
                         [g['lr'] for g in runner.optim_wrapper.param_groups])

    def test_relative_threshold_improvement_and_invalid_metrics(self):
        runner, hook, scheduler = setup()
        validate(runner, hook, 0.5)
        validate(runner, hook, 0.5 * 1.0001)  # equality is not improvement
        self.assertEqual(scheduler.num_bad_epochs, 1)
        validate(runner, hook, 0.6)
        self.assertEqual(scheduler.num_bad_epochs, 0)
        for metrics in ({}, {'coco/bbox_mAP': float('nan')},
                        {'coco/bbox_mAP': float('inf')}):
            with self.assertRaises(ValueError):
                scheduler.step(metrics)
        for options in (dict(stage_patiences=(3, 5)), dict(stage_patiences=(3, -1, 8)),
                        dict(min_value=1e-3), dict(threshold=float('nan'))):
            with self.assertRaises(ValueError):
                setup(**options)

    def test_updates_modes_moments_and_clipping_membership_across_stages(self):
        runner, hook, scheduler = setup(stage_patiences=(0, 0, 0), cooldown=0)
        model, opt, wrapper = runner.model, runner.optim_wrapper.optimizer, runner.optim_wrapper
        model.backbone.norm1.eval()  # hook must also preserve existing eval modes
        flags = {n: p.requires_grad for n, p in model.named_parameters()}
        modes = {n: m.training for n, m in model.named_modules()}
        identities = (id(model), id(opt), [id(g) for g in opt.param_groups])
        for stage in (1, 2, 3):
            self.assertEqual(scheduler.stage, stage)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            actual_clip = wrapper.clip_func
            with patch.object(wrapper, 'clip_func', wraps=actual_clip) as spy:
                wrapper.step()
                clipped_ids = {id(p) for p in spy.call_args.args[0]}
            for (name, p), group in zip(model.named_parameters(), opt.param_groups):
                if name.startswith('language_model.'):
                    late = any(f'.layer.{i}.' in name for i in range(6, 12))
                    active = stage == 3 or (stage == 2 and late)
                elif name.startswith('backbone.'):
                    late = name.startswith(('backbone.stages.2.', 'backbone.stages.3.',
                                            'backbone.norm2.', 'backbone.norm3.'))
                    active = stage == 3 or (stage == 2 and late)
                else:
                    active = True
                self.assertEqual(group['lr'] > 0, active, name)
                self.assertEqual(id(p) in clipped_ids, active, name)
                self.assertEqual(torch.equal(p, before[name]), not active, name)
                self.assertEqual(int(opt.state[p]['step']), stage)
                self.assertGreater(opt.state[p]['exp_avg'].abs().sum().item(), 0)
            self.assertEqual(flags, {n: p.requires_grad for n, p in model.named_parameters()})
            self.assertEqual(modes, {n: m.training for n, m in model.named_modules()})
            self.assertEqual(identities, (id(model), id(opt), [id(g) for g in opt.param_groups]))
            wrapper.zero_grad()
            validate(runner, hook, 0.5)
            validate(runner, hook, 0.5)

    def test_rejects_mixed_groups_legacy_hooks_and_incompatible_wrapper(self):
        runner, hook, _ = setup()
        opt = runner.optim_wrapper.optimizer
        opt.param_groups[0]['params'].append(runner.model.bbox_head.weight)
        with self.assertRaisesRegex(ValueError, 'mixes'):
            hook._identify_groups(runner.model, opt)
        runner, hook, _ = setup()
        runner.hooks.append(type('AutoLoadBestCheckpointHook', (), {})())
        with self.assertRaisesRegex(ValueError, 'legacy'):
            hook.before_train(runner)
        runner, hook, _ = setup()
        runner.optim_wrapper = OptimWrapper(runner.optim_wrapper.optimizer)
        with self.assertRaises(TypeError):
            hook.before_train(runner)


class ConfigTests(unittest.TestCase):
    def test_all_18_configs_and_preserved_model_sources(self):
        paths = sorted((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            cfg = Config.fromfile(str(path))
            self.assertEqual(cfg.optim_wrapper.type, 'ActiveParamOptimWrapper')
            self.assertEqual(cfg.optim_wrapper.clip_grad, dict(max_norm=0.1, norm_type=2))
            self.assertEqual(cfg.param_scheduler[0].stage_patiences, (3, 5, 8))
            self.assertEqual(cfg.custom_hooks, [dict(type='ThreeStageProgressiveFinetuningHook')])
            self.assertEqual(cfg.model.raw_mean_etf_loss_weight, 0.1)
            self.assertFalse(cfg.resume)
            self.assertEqual(cfg.train_cfg.max_epochs, 100)
            self.assertEqual(cfg.default_hooks.checkpoint.type, 'CheckpointHook')
            self.assertEqual(cfg.default_hooks.checkpoint.save_best, 'auto')
            self.assertFalse(cfg.default_hooks.checkpoint.by_epoch)
            self.assertEqual(cfg.default_hooks.checkpoint.interval, 10000)
            self.assertNotIn('find_unused_parameters', cfg)
            self.assertNotIn('model_wrapper_cfg', cfg)
            self.assertTrue(cfg.model.backbone.with_cp)
            self.assertEqual(cfg.model.encoder.num_cp, 6)
        unchanged = subprocess.check_output(
            ['git', 'diff', '1ae964e', '--', 'mmdet/models'], cwd=ROOT)
        self.assertEqual(unchanged, b'')
        self.assertGreater(StageHook.priority, 70)
        self.assertLess(StageHook.priority, 90)


def ddp_worker(rank, port):
    torch.set_num_threads(1)
    store = dist.TCPStore('127.0.0.1', port, 2, rank == 0,
                          timeout=timedelta(seconds=60), use_libuv=False)
    dist.init_process_group('gloo', store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        model = MMDistributedDataParallel(ToyModel(), broadcast_buffers=False)
        runner, hook, scheduler = setup(model, stage_patiences=(0, 0, 0), cooldown=0)
        original_model_id, optimizer_id = id(model), id(runner.optim_wrapper.optimizer)
        observed = []
        for _ in range(6):
            observed.append(scheduler.stage)
            wrapper = runner.optim_wrapper
            # Different per-rank gradients must be synchronized by unchanged DDP.
            with wrapper.optim_context(model):
                loss = model(torch.full((2, 2), float(rank + 1), requires_grad=True))
            wrapper.update_params(loss)
            for p in model.parameters():
                remote = p.detach().clone()
                dist.broadcast(remote, src=0)
                torch.testing.assert_close(p, remote, rtol=0, atol=0)
            metric = [0.5 if rank == 0 else None]
            dist.broadcast_object_list(metric)
            validate(runner, hook, metric[0])
            assert id(runner.model) == original_model_id
            assert id(wrapper.optimizer) == optimizer_id
        assert observed == [1, 1, 2, 2, 3, 3]
        assert runner.train_loop.stop_training
        states = [None, None]
        dist.all_gather_object(states, (scheduler.stage, scheduler.should_stop,
                                       scheduler.scheduled_lrs))
        assert states[0] == states[1]
    finally:
        dist.destroy_process_group()


class DDPTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_process_original_ddp_and_reentrant_checkpoint(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        mp.spawn(ddp_worker, args=(port,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
