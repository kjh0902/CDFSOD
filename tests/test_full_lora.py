"""CPU numerical, checkpoint, optimizer, ETF and scheduler integration tests."""
import copy
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from mmengine.config import Config
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import ReduceOnPlateauParamScheduler

ROOT = Path(__file__).resolve().parents[1]
BASE = 'd560d78'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lora = load('mmdet.models.layers.lora', 'mmdet/models/layers/lora.py')
# Load production registration modules without importing all MMCV CUDA ops.
Constructor = load('mmdet.engine.optimizers.full_lora_optimizer_constructor',
    'mmdet/engine/optimizers/full_lora_optimizer_constructor.py').FullLoRAOptimWrapperConstructor
Hook = load('mmdet.engine.hooks.full_lora_hook',
            'mmdet/engine/hooks/full_lora_hook.py').FullLoRAHook


def fixture():
    model = nn.Module()
    for component in ('backbone', 'language_model', 'encoder', 'decoder',
                      'bbox_head', 'text_feat_map', 'neck'):
        setattr(model, component, nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8)))
    model.encoder.add_module('attention', nn.MultiheadAttention(8, 2, batch_first=True))
    model.neck.add_module('conv', nn.Conv2d(8, 8, 3, padding=1))
    model.memory_trans_fc = nn.Linear(8, 8)
    model.query_embedding = nn.Embedding(4, 8)
    model.level_embed = nn.Parameter(torch.randn(4, 8))
    # The injection must preserve module aliases and optimizer deduplication.
    model.bbox_head.add_module('shared', model.bbox_head[0])
    return model


def make_optimizer(model):
    return Constructor(dict(type=OptimWrapper, optimizer=dict(
        type=torch.optim.AdamW, lr=2e-4, weight_decay=0.01),
        clip_grad=dict(max_norm=0.1, norm_type=2)))(model)


class FullLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_linear_conv_identity_dropout_and_merged_forward(self):
        for original, data in (
                (nn.Linear(8, 5), torch.randn(2, 4, 8)),
                (nn.Conv2d(3, 5, 1), torch.randn(2, 3, 7, 9)),
                (nn.Conv2d(3, 5, 3, stride=2, padding=1), torch.randn(2, 3, 7, 9)),
                (nn.Conv2d(3, 5, 3, padding=2, dilation=2, padding_mode='reflect'),
                 torch.randn(2, 3, 7, 9))):
            with self.subTest(module=original):
                model = nn.Sequential(copy.deepcopy(original))
                lora.inject_full_lora(model)
                adapted = model[0]
                self.assertEqual(adapted.scaling, 2.)
                self.assertEqual(adapted.lora_A.shape[0], 32)
                self.assertEqual(adapted.lora_dropout.p, .05)
                torch.testing.assert_close(model(data), original(data), rtol=0, atol=0)
                nn.init.normal_(adapted.lora_B, std=.1)
                merged = copy.deepcopy(original)
                with torch.no_grad():
                    merged.weight.add_(2 * (adapted.lora_B @ adapted.lora_A).view_as(merged.weight))
                model.eval()
                torch.testing.assert_close(model(data), merged(data))
                model.train()
                self.assertFalse(torch.equal(model(data), model(data)))

    def test_attention_identity_merged_weights_and_adapter_gradients(self):
        for batch_first in (False, True):
            for need_weights in (False, True):
                for separate in (False, True):
                    for unbatched in (False, True):
                        with self.subTest(batch_first=batch_first, need_weights=need_weights,
                                          separate=separate, unbatched=unbatched):
                            original = nn.MultiheadAttention(8, 2, batch_first=batch_first,
                                kdim=6 if separate else 8, vdim=4 if separate else 8)
                            model = nn.Sequential(copy.deepcopy(original))
                            lora.inject_full_lora(model, rank=3, alpha=6, dropout=0.)
                            adapted = model[0]
                            shape = lambda length, dim: ((length, dim) if unbatched else
                                (2, length, dim) if batch_first else (length, 2, dim))
                            q = torch.randn(shape(3, 8))
                            k = torch.randn(shape(4, original.kdim))
                            v = torch.randn(shape(4, original.vdim))
                            mask = torch.zeros(3, 4, dtype=torch.bool)
                            mask[0, -1] = True
                            padding = torch.zeros(4 if unbatched else (2, 4), dtype=torch.bool)
                            padding[..., -2] = True
                            options = dict(attn_mask=mask, key_padding_mask=padding,
                                           need_weights=need_weights, average_attn_weights=False)
                            before = original(q, k, v, **options)
                            actual = adapted(q, k, v, **options)
                            torch.testing.assert_close(actual, before)
                            for module in adapted.modules():
                                if hasattr(module, 'lora_B'):
                                    nn.init.normal_(module.lora_B, std=.1)
                            merged = copy.deepcopy(original)
                            deltas = [p.scaling * p.lora_B @ p.lora_A for p in adapted.projections]
                            with torch.no_grad():
                                if separate:
                                    for weight, delta in zip((merged.q_proj_weight,
                                            merged.k_proj_weight, merged.v_proj_weight), deltas):
                                        weight.add_(delta)
                                else:
                                    merged.in_proj_weight.add_(torch.cat(deltas))
                                merged.out_proj.weight.add_(
                                    adapted.out_proj.scaling * adapted.out_proj.lora_B @ adapted.out_proj.lora_A)
                            torch.testing.assert_close(adapted(q, k, v, **options),
                                                       merged(q, k, v, **options))
                            adapted(q, k, v, **options)[0].square().mean().backward()
                            for name, parameter in adapted.named_parameters():
                                self.assertEqual(parameter.requires_grad, name.endswith(('lora_A', 'lora_B')))
                                if parameter.requires_grad:
                                    self.assertIsNotNone(parameter.grad, name)
                                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_attention_additive_masks_bias_tokens_zero_attention_and_dropout(self):
        original = nn.MultiheadAttention(8, 2, batch_first=True,
                                         add_bias_kv=True, add_zero_attn=True)
        model = nn.Sequential(copy.deepcopy(original))
        lora.inject_full_lora(model)
        data = torch.randn(2, 3, 8)
        mask = torch.zeros(4, 3, 3)
        mask[:, 0, 1] = float('-inf')
        padding = torch.zeros(2, 3)
        padding[0, -1] = float('-inf')
        for need_weights in (False, True):
            for average in (False, True):
                options = dict(attn_mask=mask, key_padding_mask=padding,
                               need_weights=need_weights, average_attn_weights=average)
                torch.testing.assert_close(model[0](data, data, data, **options),
                                           original(data, data, data, **options))
        for module in model.modules():
            if hasattr(module, 'lora_B'):
                nn.init.normal_(module.lora_B, std=.1)
        self.assertFalse(torch.equal(model[0](data, data, data)[0],
                                     model[0](data, data, data)[0]))

    def test_checkpoint_keys_freezing_optimizer_step_and_strict_round_trip(self):
        model = fixture()
        pretrained = copy.deepcopy(model.state_dict())
        lora.inject_full_lora(model)
        self.assertIs(model.bbox_head[0], model.bbox_head.shared)
        missing, unexpected = model.load_state_dict(pretrained, strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(key.endswith(('lora_A', 'lora_B')) for key in missing))
        wrapper = make_optimizer(model)
        groups = wrapper.optimizer.param_groups
        self.assertEqual([g['lr'] for g in groups], [1e-4, 1e-4] + [2e-4] * 5)
        params = [p for g in groups for p in g['params']]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters() if p.requires_grad})
        for _ in range(2):
            # Exercise all adapter types, including attention output projection.
            loss = sum(module(torch.randn(2, 8)).square().mean()
                       for module in model.modules() if isinstance(module, lora.LoRALinear))
            x = torch.randn(2, 3, 8)
            loss = loss + model.encoder.attention(x, x, x)[0].square().mean()
            loss = loss + model.neck.conv(torch.randn(2, 8, 4, 4)).square().mean()
            wrapper.update_params(loss)
        for name, before in pretrained.items():
            torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)
        self.assertTrue(any(p.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('lora_B')))
        restored = fixture()
        lora.inject_full_lora(restored)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored_wrapper = make_optimizer(restored)
        restored_wrapper.load_state_dict(copy.deepcopy(wrapper.state_dict()))
        model.eval()
        restored.eval()
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(model.encoder.attention(x, x, x),
                                   restored.encoder.attention(x, x, x))
        model.level_embed.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'freeze state'):
            lora.lora_parameter_groups(model)

    def test_production_plateau_scheduler_ratios_floor_and_resume(self):
        model = fixture()
        lora.inject_full_lora(model)
        wrapper = make_optimizer(model)
        scheduler = ReduceOnPlateauParamScheduler(wrapper, param_name='lr',
            monitor='coco/bbox_mAP', rule='greater', factor=.5, patience=8,
            threshold=1e-4, threshold_rule='rel', cooldown=1, min_value=1e-6)
        runner = SimpleNamespace(model=model, optim_wrapper=wrapper,
            param_schedulers=[scheduler], logger=SimpleNamespace(info=lambda _: None))
        hook = Hook()
        hook.before_train(runner)
        initial = [g['lr'] for g in wrapper.optimizer.param_groups]
        history = []
        for epoch in range(100):
            hook.before_train_epoch(runner)
            scheduler.step()
            scheduler.step(dict(**{'coco/bbox_mAP': .2}))
            current = [g['lr'] for g in wrapper.optimizer.param_groups]
            history.append(current)
            self.assertTrue(all(abs(lr / current[0] - start / initial[0]) < 1e-10
                                for lr, start in zip(current, initial)))
            if epoch == 8:
                self.assertEqual(current, initial)
            if epoch == 9:
                self.assertEqual(current, [lr * .5 for lr in initial])
            if epoch == 45:
                restored = make_optimizer(model)
                restored.load_state_dict(copy.deepcopy(wrapper.state_dict()))
                resumed_scheduler = ReduceOnPlateauParamScheduler(restored, param_name='lr')
                resumed_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
                resumed_runner = SimpleNamespace(model=model, optim_wrapper=restored,
                    param_schedulers=[resumed_scheduler], logger=runner.logger)
                hook.before_train(resumed_runner)
                resumed_scheduler.step()
                resumed_scheduler.step(dict(**{'coco/bbox_mAP': .2}))
                resumed_values = [g['lr'] for g in restored.optimizer.param_groups]
        self.assertEqual(history[46], resumed_values)
        self.assertEqual(history[-1], [1e-6, 1e-6] + [2e-6] * 5)

    def test_all_18_configs_preserve_dataset_model_and_fixed_experiment(self):
        paths = sorted((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            with self.subTest(config=path.name):
                current = Config.fromfile(str(path))
                before = subprocess.check_output(['git', 'show',
                    f'{BASE}:{path.relative_to(ROOT).as_posix()}'], cwd=ROOT).decode('utf-8')
                before = before.replace("'../grounding_dino_swin-b_pretrain_all.py'",
                    repr((ROOT / 'configs_cdfsod/grounding_dino_swin-b_pretrain_all.py').as_posix()))
                old = Config.fromstring(before, '.py')
                for key in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                            'train_pipeline', 'test_pipeline', 'val_evaluator',
                            'test_evaluator', 'load_from', 'randomness', 'env_cfg'):
                    self.assertEqual(current[key], old[key], key)
                previous_model, new_model = copy.deepcopy(old.model), copy.deepcopy(current.model)
                for key in ('type', 'lora_cfg', 'bert_etf_loss_weight', 'bert_orth_loss_weight',
                            'fe_etf_loss_weight', 'fe_orth_loss_weight'):
                    previous_model.pop(key, None)
                    new_model.pop(key, None)
                self.assertEqual(new_model, previous_model)
                self.assertEqual(current.model.type, 'GroundingDINOFullLoRA')
                self.assertEqual(current.model.lora_cfg, dict(rank=32, alpha=64, dropout=.05))
                self.assertEqual([current.model[key] for key in ('bert_etf_loss_weight',
                    'bert_orth_loss_weight', 'fe_etf_loss_weight', 'fe_orth_loss_weight')], [0., 0., 5., 0.])
                self.assertEqual(current.train_cfg.max_epochs, 100)
                self.assertEqual(current.randomness.seed, 42)
                self.assertEqual(current.train_dataloader.batch_size, 4)
                self.assertEqual(current.custom_hooks, [dict(type='FullLoRAHook')])
                self.assertEqual(current.optim_wrapper.optimizer,
                                 dict(type='AdamW', lr=2e-4, weight_decay=.01))
                self.assertEqual(current.optim_wrapper.clip_grad.max_norm, .1)
                self.assertFalse(current.auto_scale_lr.enable)
                self.assertEqual(current.param_scheduler, [dict(
                    type='ReduceOnPlateauParamScheduler', param_name='lr', monitor='coco/bbox_mAP',
                    rule='greater', factor=.5, patience=8, threshold=1e-4, threshold_rule='rel',
                    cooldown=1, min_value=1e-6, verbose=True)])
                self.assertEqual(current.default_hooks.checkpoint.save_best, 'coco/bbox_mAP')

    def test_fe_etf_reuses_production_loss_with_gradients_only_in_adapters(self):
        from test_raw_mean_etf_integration import fixture as detector_fixture, samples
        model = detector_fixture(weight=0.)
        model.fe_etf_loss_weight = 5.
        # Give the lightweight language/neck stand-ins the projections that
        # the real BERT and ChannelMapper contain, retaining production loops.
        model.language_model.embedding = nn.Sequential(model.language_model.embedding, nn.Linear(8, 8))
        model.neck = nn.Linear(8, 8)
        extract = model.extract_feat
        model.extract_feat = lambda images: (model.neck(extract(images)[0]),)
        lora.inject_full_lora(model)
        wrapper = make_optimizer(model)
        losses = model.loss(torch.zeros(2, 3, 4, 4), samples())
        self.assertEqual(set(losses), {'loss_cls', 'loss_bbox', 'loss_fe_etf'})
        losses['loss_fe_etf'].backward(retain_graph=True)
        for component in ('backbone', 'language_model', 'encoder', 'text_feat_map', 'neck'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in getattr(model, component).parameters() if p.requires_grad), component)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        wrapper.zero_grad()
        wrapper.update_params(sum(losses.values()))
        self.assertTrue(any(p.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith('lora_B')))


if __name__ == '__main__':
    unittest.main()
