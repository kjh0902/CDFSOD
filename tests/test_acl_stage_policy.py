"""ACL stage LR policy with real MMEngine grouping, scheduler and AdamW steps."""
import copy
import importlib.util
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from visual_test_support import ROOT, execute, node, source
from support_transformer_fixture import SupportAttention


class StageModel(nn.Module):
    def __init__(self, support=True):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.language_model = nn.Linear(8, 8)
        self.bbox_head = nn.Linear(8, 4)
        self.decoder = nn.Linear(8, 8)
        self.neck = nn.Linear(8, 8)
        self.text_feat_map = nn.Linear(8, 8)
        self.dn_query_generator = nn.Embedding(3, 8)
        self.encoder = nn.Module()
        self.encoder.fusion_layers = nn.ModuleList([nn.Linear(8, 8)])
        self.encoder.text_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 8))])
        self.encoder.layers = nn.ModuleList([nn.Linear(8, 8)])
        self.encoder.support_layers = nn.ModuleList([SupportAttention(8, 2)] if support else [])
        self.query_embedding = nn.Embedding(3, 8)
        self.memory_trans_fc = nn.Linear(8, 8)
        self.level_embed = nn.Parameter(torch.ones(2, 8))


@unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine required')
class ACLStagePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmengine.hooks import Hook
        from mmengine.model import is_model_wrapper
        cls.env = dict(nn=nn, Hook=Hook, is_model_wrapper=is_model_wrapper)
        cls.Hook = cls.load_hook(source('mmdet/engine/hooks/stage_lr_hook.py'))

    @classmethod
    def load_hook(cls, text):
        tree = node(text, 'BBoxHeadFirstHook6')
        tree.decorator_list = []
        env = dict(cls.env)
        execute([tree], env)
        return env['BBoxHeadFirstHook6']

    def setup_runner(self, model=None):
        from mmengine.optim import build_optim_wrapper
        from mmengine.optim.scheduler import ReduceOnPlateauParamScheduler
        model = StageModel() if model is None else model
        cfg = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=1e-4,
                                                      weight_decay=.05))
        paramwise = dict(custom_keys={'backbone': dict(lr_mult=.2),
                                     'language_model': dict(lr_mult=.2)})
        with patch('mmengine.optim.optimizer.default_constructor.print_log'):
            wrapper = build_optim_wrapper(model, dict(cfg, paramwise_cfg=paramwise))
        scheduler = ReduceOnPlateauParamScheduler(
            wrapper, 'lr', monitor='coco/bbox_mAP', rule='greater', factor=.5,
            patience=5, cooldown=1, min_value=1e-6)
        runner = NS(model=model, optim_wrapper=wrapper, param_schedulers=[scheduler],
                    epoch=0, logger=Mock())
        hook = self.Hook(adjust_scheduler_patience=True, patience_frozen=3,
                         patience_unfrozen=8)
        return runner, hook, scheduler

    def assert_step(self, runner, frozen):
        before = {name: p.detach().clone() for name, p in runner.model.named_parameters()}
        runner.optim_wrapper.zero_grad(set_to_none=True)
        # Nonzero gradients for every parameter isolate the optimizer LR effect,
        # including AdamW weight decay and accumulated state at LR=0.
        loss = sum((p.square() + p).sum() for p in runner.model.parameters())
        loss.backward()
        optimizer = runner.optim_wrapper.optimizer
        for name, p in runner.model.named_parameters():
            self.assertTrue(p.requires_grad, name)
            self.assertIsNotNone(p.grad, name)
            self.assertGreater(p.grad.abs().sum().item(), 0., name)
        optimizer.step()
        for name, p in runner.model.named_parameters():
            if id(p) in frozen:
                torch.testing.assert_close(p, before[name], rtol=0, atol=0, msg=name)
                self.assertGreater(optimizer.state[p]['step'].item(), 0)
            else:
                self.assertFalse(torch.equal(p, before[name]), name)

    def test_stage1_separate_support_lr_gamma_and_actual_updates(self):
        runner, hook, scheduler = self.setup_runner()
        hook.before_train(runner)
        optimizer = runner.optim_wrapper.optimizer
        self.assertFalse(hook._stage2_started)
        self.assertEqual(scheduler.patience, 3)
        support = {id(p) for p in runner.model.encoder.support_layers.parameters()}
        support_groups = {id(p) for i in hook._support_groups
                          for p in optimizer.param_groups[i]['params']}
        frozen = {id(p) for i in hook._other_groups for p in optimizer.param_groups[i]['params']}
        self.assertEqual(support, support_groups)
        self.assertTrue(support.isdisjoint(frozen))
        self.assertTrue(any(name.endswith('gamma') and id(p) in support
                            for name, p in runner.model.named_parameters()))
        for name, parameter in runner.model.named_parameters():
            if '.support_norm.' in name:
                self.assertIn(id(parameter), support_groups)
                self.assertNotIn(id(parameter), frozen)
        for group in optimizer.param_groups:
            kind = group['acl_group']
            expected = 0. if kind == 'other' else 2e-5 if kind in (
                'backbone', 'language_model') else 1e-4
            self.assertAlmostEqual(group['lr'], expected)
        for _ in range(2):
            self.assert_step(runner, frozen)
        hook.after_train_epoch(runner)
        self.assertIn('Support', runner.logger.info.call_args.args[0])
        self.assertIn('FE/Other', runner.logger.info.call_args.args[0])

    def test_real_plateau_trigger_patience_and_stage2_updates(self):
        runner, hook, scheduler = self.setup_runner()
        hook.before_train(runner)
        optimizer = runner.optim_wrapper.optimizer
        scheduler.step({'coco/bbox_mAP': 1.})
        for _ in range(3):
            scheduler.step({'coco/bbox_mAP': 1.})
            hook.before_train_epoch(runner)
            self.assertFalse(hook._stage2_started)
            self.assertTrue(all(optimizer.param_groups[i]['lr'] == 0 for i in hook._other_groups))
        scheduler.step({'coco/bbox_mAP': 1.})  # first reduction after patience=3
        self.assertAlmostEqual(optimizer.param_groups[hook._lang_model_groups[0]]['lr'], 1e-5)
        hook.before_train_epoch(runner)
        self.assertTrue(hook._stage2_started)
        self.assertEqual(scheduler.patience, 8)
        for group in optimizer.param_groups:
            expected = 1e-5 if group['acl_group'] in ('backbone', 'language_model') else 5e-5
            self.assertAlmostEqual(group['lr'], expected)
        self.assert_step(runner, set())
        # No repeated transition may overwrite later scheduler reductions.
        for group in optimizer.param_groups:
            group['lr'] *= .5
        expected = [g['lr'] for g in optimizer.param_groups]
        hook.before_train_epoch(runner)
        self.assertEqual(expected, [g['lr'] for g in optimizer.param_groups])

    def test_half_initial_lr_boundary_and_no_language_model(self):
        runner, hook, _ = self.setup_runner()
        hook.before_train(runner)
        index = hook._lang_model_groups[0]
        group = runner.optim_wrapper.optimizer.param_groups[index]
        group['lr'] = 2e-5 * .50001
        self.assertFalse(hook._should_start_stage2(runner))
        group['lr'] = 1e-5
        self.assertTrue(hook._should_start_stage2(runner))
        hook._lang_model_groups = []
        self.assertFalse(hook._should_start_stage2(runner))

    def test_resume_keeps_stage_and_scheduler_rates_without_new_groups(self):
        for stage2 in (False, True):
            with self.subTest(stage2=stage2):
                runner, hook, scheduler = self.setup_runner()
                hook.before_train(runner)
                self.assert_step(runner, {id(p) for i in hook._other_groups
                    for p in runner.optim_wrapper.optimizer.param_groups[i]['params']})
                if stage2:
                    for group in runner.optim_wrapper.optimizer.param_groups:
                        group['lr'] *= .5
                    hook.before_train_epoch(runner)
                saved = copy.deepcopy(runner.optim_wrapper.state_dict())
                rates = [g['lr'] for g in runner.optim_wrapper.optimizer.param_groups]
                resumed, new_hook, new_scheduler = self.setup_runner(copy.deepcopy(runner.model))
                resumed.optim_wrapper.load_state_dict(saved)
                new_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
                new_hook.before_train(resumed)
                self.assertEqual(new_hook._stage2_started, stage2)
                self.assertEqual(rates, [g['lr'] for g in resumed.optim_wrapper.optimizer.param_groups])
                self.assertEqual(new_scheduler.patience, 8 if stage2 else 3)
                frozen = set() if stage2 else {id(p) for i in new_hook._other_groups
                    for p in resumed.optim_wrapper.optimizer.param_groups[i]['params']}
                self.assert_step(resumed, frozen)

    def test_no_support_matches_original_acl_lr_policy(self):
        original = subprocess.check_output(
            ['git', 'show', '8926970ebff1a549088b0a4c87c272e1a70fe0dd:mmdet/engine/hooks/stage_lr_hook.py'],
            cwd=ROOT, encoding='utf-8')
        OriginalHook = self.load_hook(original)
        runner, hook, _ = self.setup_runner(StageModel(support=False))
        reference, _, _ = self.setup_runner(copy.deepcopy(runner.model))
        old = OriginalHook(adjust_scheduler_patience=True, patience_frozen=3, patience_unfrozen=8)
        for current, target in ((hook, runner), (old, reference)):
            current.before_train(target)
        self.assertEqual([g['lr'] for g in runner.optim_wrapper.optimizer.param_groups],
                         [g['lr'] for g in reference.optim_wrapper.optimizer.param_groups])
        for current, target in ((hook, runner), (old, reference)):
            for group in target.optim_wrapper.optimizer.param_groups:
                group['lr'] *= .5
            current.before_train_epoch(target)
        self.assertEqual([g['lr'] for g in runner.optim_wrapper.optimizer.param_groups],
                         [g['lr'] for g in reference.optim_wrapper.optimizer.param_groups])

    def test_mixed_support_and_frozen_group_is_rejected(self):
        runner, hook, _ = self.setup_runner()
        params = [next(runner.model.encoder.support_layers.parameters()),
                  next(runner.model.encoder.text_layers.parameters())]
        runner.optim_wrapper = NS(optimizer=torch.optim.AdamW(params, lr=1e-4))
        with self.assertRaisesRegex(ValueError, 'separate optimizer groups'):
            hook.before_train(runner)

    def test_detection_loss_trains_support_through_lr_frozen_fe(self):
        import visual_detector_fixture as fixture
        import visual_loss_fixture as loss_fixture
        from test_visual_integration import enable
        model, _ = fixture.build(real_encoder=True)
        model.bbox_head = loss_fixture.head(classes=3)
        enable(model)
        def dn(samples):
            batch = len(samples)
            return (model.dn_query_generator.label_embedding.weight[None, :1].expand(batch, 4, -1),
                    torch.randn(batch, 4, 4), torch.zeros(7, 7, dtype=torch.bool),
                    dict(num_denoising_queries=4, num_denoising_groups=1))
        model.dn_query_generator.forward = dn
        runner, hook, _ = self.setup_runner(model)
        hook.before_train(runner)
        frozen = {id(p) for i in hook._other_groups
                  for p in runner.optim_wrapper.optimizer.param_groups[i]['params']}
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        losses = model.loss(torch.randn(2, 20, 8), fixture.samples())
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.encoder.support_layers.parameters()))
        self.assertGreater(model.encoder.text_layers[0].self_attn.attn.in_proj_weight.grad.abs().sum(), 0)
        runner.optim_wrapper.optimizer.step()
        for name, parameter in model.named_parameters():
            if id(parameter) in frozen:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0, msg=name)
        for name, parameter in model.encoder.support_layers.named_parameters():
            self.assertFalse(torch.equal(parameter, before['encoder.support_layers.' + name]), name)


if __name__ == '__main__':
    unittest.main()
