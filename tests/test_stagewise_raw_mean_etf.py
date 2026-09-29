"""CPU graph and real-MMEngine scheduler/checkpoint tests for stagewise ETF."""
import copy
import logging
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, ReduceOnPlateauLR
from mmengine.runner import Runner

import test_second_order_etf_integration as integration

ROOT = integration.ROOT
# Import the actual hooks without importing MMDetection/MMCV extensions.
package = types.ModuleType('_stagewise_hook_tests')
package.__path__ = [str(ROOT / 'mmdet/engine/hooks')]
sys.modules[package.__name__] = package
from _stagewise_hook_tests.stage_lr_hook import BBoxHeadFirstHook6
from _stagewise_hook_tests.stagewise_raw_mean_etf_hook import StagewiseRawMeanETFHook


def stage_model(stage=1, weight=1.):
    model = integration.fixture(weight=0.)
    model.stagewise_raw_mean_etf = True
    model.raw_mean_etf_loss_weight = weight
    if stage is not None:
        model.set_raw_mean_etf_stage(stage)
    return model


class StagewiseGraphTests(unittest.TestCase):
    def test_one_solve_at_exact_source_and_no_detection_or_inference_change(self):
        reference = integration.fixture(base=True, weight=0.)
        _, expected = integration.run(reference, integration.samples())
        for stage in (1, 2):
            model = stage_model(stage)
            projected = []
            handle = model.text_feat_map.register_forward_hook(
                lambda module, args, output: projected.append(output))
            with patch.dict(integration.Detector.loss.__globals__,
                            raw_mean_etf_loss=unittest.mock.Mock(
                                wraps=integration.second.raw_mean_etf_loss)):
                solve = integration.Detector.loss.__globals__['raw_mean_etf_loss']
                losses, actual = integration.run(model, integration.samples())
                self.assertEqual(solve.call_count, 1)
                source = projected[0] if stage == 1 else model.bbox_head.seen['memory_text']
                self.assertIs(solve.call_args.args[0], source)
            handle.remove()
            self.assertEqual(reference.state_dict().keys(), model.state_dict().keys())
            self.assertNotIn('loss_second_order_etf', losses)
            actual['losses'] = {k: v for k, v in losses.items()
                                if k not in {'loss_raw_mean_etf', 'etf_raw',
                                             'etf_stage', 'etf_to_detection_ratio'}}
            integration.SecondOrderIntegrationTests().assert_nested_equal(actual, expected)
            total, logs = BaseModel.parse_losses(model, losses)
            expected_total = sum(v for k, v in losses.items() if 'loss' in k)
            torch.testing.assert_close(total, expected_total)
            for key in ('etf_raw', 'etf_stage', 'etf_to_detection_ratio'):
                self.assertFalse(losses[key].requires_grad)
                self.assertIn(key, logs)
            torch.testing.assert_close(losses['etf_to_detection_ratio'],
                losses['loss_raw_mean_etf'].detach() /
                (losses['loss_cls'] + losses['loss_bbox']).detach())
            model.eval()
            with patch.dict(integration.Detector.loss.__globals__,
                            raw_mean_etf_loss=lambda *a: self.fail('inference ETF')):
                prediction = model.predict(torch.zeros(2, 3, 4, 4), integration.samples())
            self.assertEqual(prediction[0].pred_instances.label_names, ['crazing'])

    def test_etf_only_gradients_change_path_on_the_same_model(self):
        model = stage_model(1)
        visual = model.pre_transformer()[0]['feat']
        visual.requires_grad_(True)
        for stage in (1, 2):
            model.zero_grad(set_to_none=True)
            visual.grad = None
            model.set_raw_mean_etf_stage(stage)
            losses, _ = integration.run(model, integration.samples())
            losses['loss_raw_mean_etf'].backward()
            for module in (model.language_model, model.text_feat_map):
                self.assertGreater(sum(p.grad.abs().sum() for p in module.parameters()
                                       if p.grad is not None), 0)
            if stage == 1:
                self.assertIsNone(visual.grad)
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            else:
                self.assertGreater(visual.grad.abs().sum(), 0)
                self.assertGreater(sum(p.grad.abs().sum() for p in model.encoder.parameters()
                                       if p.grad is not None), 0)
            self.assertIsNone(model.query_embedding.weight.grad)
            self.assertTrue(all(p.grad is None for p in model.bbox_head.parameters()))

    def test_zero_weight_missing_hook_and_weight_validation(self):
        with self.assertRaisesRegex(RuntimeError, 'Hook'):
            integration.run(stage_model(None), integration.samples())
        model = stage_model(weight=0.)
        with patch.object(model, '_get_second_order_class_token_map', side_effect=AssertionError), \
                patch.dict(integration.Detector.loss.__globals__,
                           raw_mean_etf_loss=lambda *a: self.fail('disabled ETF')):
            for stage in (1, 2):
                model.set_raw_mean_etf_stage(stage)
                losses, _ = integration.run(model, integration.samples())
                self.assertNotIn('loss_raw_mean_etf', losses)
                self.assertEqual(losses['etf_stage'].item(), stage)
        for value in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                integration.Detector(language_model={}, raw_mean_etf_loss_weight=value)
        with self.assertRaisesRegex(ValueError, 'only one'):
            integration.Detector(language_model={}, raw_mean_etf_loss_weight=0.,
                                 second_order_etf_loss_weight=0.)
        for stage in (0, 3, True):
            with self.assertRaises(ValueError):
                model.set_raw_mean_etf_stage(stage)

    def test_all_configs_resolve_with_unchanged_optimizer_and_pipeline(self):
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(configs), 18)
        for path in configs:
            cfg = Config.fromfile(str(path))
            self.assertEqual(cfg.model.raw_mean_etf_loss_weight, 1.)
            self.assertTrue(cfg.model.stagewise_raw_mean_etf)
            self.assertEqual(cfg.custom_hooks[0].type, 'StagewiseRawMeanETFHook')
            self.assertEqual(cfg.custom_hooks[0].stage2_fe_lr_mult, .5)
            self.assertEqual(cfg.custom_hooks[0].patience_frozen, 3)
            self.assertEqual(cfg.custom_hooks[0].patience_unfrozen, 6)
            self.assertEqual(cfg.optim_wrapper.optimizer.lr, 1e-4)
            self.assertEqual(cfg.optim_wrapper.clip_grad.max_norm, .1)
            self.assertEqual(cfg.param_scheduler[0].factor, .5)
            self.assertEqual(cfg.param_scheduler[0].min_value, 1e-6)


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        for name in ('backbone', 'language_model', 'bbox_head', 'text_feat_map',
                     'neck', 'decoder', 'dn_query_generator', 'encoder', 'memory_trans_fc'):
            setattr(self, name, nn.Linear(2, 2))
        self.level_embed = nn.Parameter(torch.ones(2))
        self.stagewise_raw_mean_etf = True
        self.raw_mean_etf_loss_weight = 1.
        self.raw_mean_etf_stage = None

    set_raw_mean_etf_stage = integration.Detector.set_raw_mean_etf_stage

    def forward(self, inputs=None, data_samples=None, mode='loss'):
        return dict(loss=sum(p.square().sum() for p in self.parameters()))


def toy_runner():
    torch.manual_seed(29)
    model = TinyModel()
    groups = [dict(params=[p], lr=2e-5 if name.startswith(('backbone.', 'language_model.'))
                   else 1e-4) for name, p in model.named_parameters()]
    wrapper = OptimWrapper(torch.optim.AdamW(groups, lr=1e-4),
                           clip_grad=dict(max_norm=.1, norm_type=2))
    scheduler = ReduceOnPlateauLR(wrapper, factor=.5, patience=5, cooldown=1,
                                 min_value=1e-6, monitor='coco/bbox_mAP', rule='greater')
    return SimpleNamespace(model=model, optim_wrapper=wrapper, param_schedulers=[scheduler],
                           epoch=0, iter=0, _resume=False,
                           logger=logging.getLogger('stagewise_test'))


def rates(runner):
    return [g['lr'] for g in runner.optim_wrapper.optimizer.param_groups]


def plateau(runner):
    scheduler = runner.param_schedulers[0]
    # First observation improves; four subsequent observations trigger decay
    # with MMEngine's num_bad_epochs > patience=3 condition.
    for _ in range(5):
        scheduler.step({'coco/bbox_mAP': .2})


class StagewiseHookTests(unittest.TestCase):
    def test_stage1_matches_original_freeze_and_adamw_does_not_move_frozen_params(self):
        original, current = toy_runner(), toy_runner()
        old = BBoxHeadFirstHook6(adjust_scheduler_patience=True, patience_frozen=3,
                                patience_unfrozen=8)
        new = StagewiseRawMeanETFHook()
        old.before_train(original)
        new.before_train(current)
        self.assertEqual(rates(original), rates(current))
        self.assertEqual(old._other_groups, new._other_groups)
        before = [p.detach().clone() for p in current.model.parameters()]
        self.assertTrue(all(p.requires_grad for p in current.model.parameters()))
        current.optim_wrapper.update_params(current.model.forward()['loss'])
        changed = [not torch.equal(p, q) for p, q in zip(before, current.model.parameters())]
        self.assertEqual(changed, [lr > 0 for lr in rates(current)])
        for index in new._other_groups:
            p = current.optim_wrapper.optimizer.param_groups[index]['params'][0]
            self.assertIn('exp_avg', current.optim_wrapper.optimizer.state[p])

    def test_plateau_switch_is_once_and_only_fe_gets_multiplier(self):
        runner = toy_runner()
        hook = StagewiseRawMeanETFHook()
        hook.before_train(runner)
        self.assertEqual(runner.model.raw_mean_etf_stage, 1)
        self.assertEqual(runner.param_schedulers[0].patience, 3)
        plateau(runner)
        self.assertTrue(all(rates(runner)[i] == 0 for i in hook._other_groups))
        self.assertEqual(runner.model.raw_mean_etf_stage, 1)
        runner.epoch, runner.iter = 5, 50
        hook.before_train_epoch(runner)
        self.assertEqual(runner.model.raw_mean_etf_stage, 2)
        self.assertEqual(runner.param_schedulers[0].patience, 6)
        for names, lr in zip(hook._group_names, rates(runner)):
            expected = (1e-5 if names[0].startswith(('backbone.', 'language_model.'))
                        else 2.5e-5 if names[0].startswith('encoder.') else 5e-5)
            self.assertAlmostEqual(lr, expected)
        before = rates(runner)
        hook.before_train_epoch(runner)
        self.assertEqual(rates(runner), before)
        for _ in range(8):
            runner.param_schedulers[0].step({'coco/bbox_mAP': .2})
        self.assertAlmostEqual(rates(runner)[hook._fe_groups[0]], 1.25e-5)

    def test_resume_stage1_boundary_and_stage2_preserves_optimizer_and_scheduler(self):
        for phase in ('stage1', 'boundary', 'stage2'):
            with self.subTest(phase=phase):
                runner = toy_runner()
                hook = StagewiseRawMeanETFHook()
                hook.before_train(runner)
                runner.optim_wrapper.update_params(runner.model.forward()['loss'])
                if phase != 'stage1':
                    plateau(runner)
                if phase == 'stage2':
                    runner.epoch, runner.iter = 5, 50
                    hook.before_train_epoch(runner)
                    for _ in range(8):
                        runner.param_schedulers[0].step({'coco/bbox_mAP': .2})
                checkpoint = dict(optimizer=copy.deepcopy(runner.optim_wrapper.state_dict()),
                                  param_schedulers=[copy.deepcopy(runner.param_schedulers[0].state_dict())])
                hook.before_save_checkpoint(runner, checkpoint)
                restored, new = toy_runner(), StagewiseRawMeanETFHook()
                restored._resume = True
                # Exactly Runner.resume order: load hook first, then optimizer
                # and scheduler restoration, and before_train at loop start.
                new.after_load_checkpoint(restored, checkpoint)
                restored.optim_wrapper.load_state_dict(checkpoint['optimizer'])
                restored.param_schedulers[0].load_state_dict(checkpoint['param_schedulers'][0])
                before = rates(restored)
                scheduler_before = copy.deepcopy(restored.param_schedulers[0].state_dict())
                new.before_train(restored)
                self.assertEqual(rates(restored), before)
                self.assertEqual(restored.param_schedulers[0].state_dict(), scheduler_before)
                self.assertEqual(new._original_lrs, hook._original_lrs)
                hook.before_train_epoch(runner)
                new.before_train_epoch(restored)
                self.assertEqual(rates(restored), rates(runner))
                self.assertEqual(restored.model.raw_mean_etf_stage, runner.model.raw_mean_etf_stage)

    def test_weights_only_missing_metadata_and_legacy_lr_override(self):
        runner, hook = toy_runner(), StagewiseRawMeanETFHook(stage2_fe_lr_mult=1., patience_unfrozen=8)
        hook.after_load_checkpoint(runner, {'meta': {}})
        hook.before_train(runner)
        plateau(runner)
        hook.before_train_epoch(runner)
        self.assertAlmostEqual(rates(runner)[hook._fe_groups[0]], 5e-5)
        self.assertEqual(runner.param_schedulers[0].patience, 8)
        checkpoint = {}
        hook.before_save_checkpoint(runner, checkpoint)
        for saved in (checkpoint, {'meta': {}}):
            fresh, new = toy_runner(), StagewiseRawMeanETFHook()
            new.after_load_checkpoint(fresh, saved)
            new.before_train(fresh)  # load_from ignores even valid Stage 2 metadata
            self.assertEqual(fresh.model.raw_mean_etf_stage, 1)
        fresh, new = toy_runner(), StagewiseRawMeanETFHook()
        fresh._resume = True
        new.after_load_checkpoint(fresh, {'meta': {}})
        with self.assertRaisesRegex(RuntimeError, 'resume=False'):
            new.before_train(fresh)

    def test_actual_runner_checkpoint_roundtrip(self):
        # Exercise Runner.save_checkpoint/resume rather than only a hook stub.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup, \
                patch.dict('os.environ', MPLCONFIGDIR=directory), patch.object(Runner, '_log_env'):
            def close_logger(logger):
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)

            def build(folder):
                toy = toy_runner()
                hook = StagewiseRawMeanETFHook()
                runner = Runner(model=toy.model, work_dir=str(Path(directory) / folder),
                    train_dataloader=torch.utils.data.DataLoader([torch.ones(2)]),
                    train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=10),
                    optim_wrapper=toy.optim_wrapper, param_scheduler=toy.param_schedulers,
                    custom_hooks=[hook], default_hooks=dict(logger=None, timer=None, checkpoint=None),
                    env_cfg=dict(dist_cfg=dict(backend='gloo')), randomness=dict(seed=42),
                    log_level='ERROR')
                cleanup.callback(close_logger, runner.logger)
                # Force the same lazy initialization used by Runner.train.
                runner._train_loop = runner.build_train_loop(runner._train_loop)
                return runner, hook
            first, hook = build('first')
            hook.before_train(first)
            first.optim_wrapper.update_params(first.model.forward()['loss'])
            plateau(first)
            hook.before_train_epoch(first)
            first.save_checkpoint(first.work_dir, 'stage2.pth')
            second, new = build('second')
            second._resume = True
            # PyTorch >=2.6 defaults to weights_only=True, whereas MMEngine
            # 0.10.7 stores trusted MessageHub/HistoryBuffer metadata.
            with patch.dict('os.environ', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'):
                second.resume(str(Path(first.work_dir) / 'stage2.pth'))
            new.before_train(second)
            self.assertEqual(second.model.raw_mean_etf_stage, 2)
            self.assertEqual(rates(first), rates(second))
            self.assertEqual(first.param_schedulers[0].state_dict(), second.param_schedulers[0].state_dict())
            new.before_train_epoch(second)
            self.assertEqual(rates(first), rates(second))


if __name__ == '__main__':
    unittest.main()
