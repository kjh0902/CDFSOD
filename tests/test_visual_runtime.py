import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from visual_test_support import (ROOT, ENV, compat, execute, load, node, source,
                                 stats_module, text_fixture)


class RandomnessTests(unittest.TestCase):
    def test_stock_patched_and_kwargs_runner_configs(self):
        class Stock:
            def set_randomness(self, seed, deterministic=False):
                pass
        class Patched:
            def set_randomness(self, seed, deterministic=False, warn_only=True):
                pass
        class Flexible:
            def set_randomness(self, **kwargs):
                pass
        for cls in (Stock, Patched, Flexible):
            cfg = dict(randomness=dict(seed=42, deterministic=True, warn_only=True))
            out, setting = compat.runner_randomness_config(cfg, cls)
            self.assertIn('warn_only', cfg['randomness'])
            self.assertEqual('warn_only' in out['randomness'], cls is not Stock)
            self.assertEqual(setting, True if cls is Stock else None)
        with patch.object(torch, 'use_deterministic_algorithms') as call:
            compat.apply_warn_only(False)
            call.assert_called_once_with(True, warn_only=False)
            compat.apply_warn_only(None)
            call.assert_called_once()
        _, setting = compat.runner_randomness_config(
            dict(randomness=dict(seed=42, deterministic=False, warn_only=True)), Stock)
        self.assertIsNone(setting)

    @unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine not installed')
    def test_actual_stock_mmengine_signature(self):
        from mmengine.runner import Runner
        from mmengine.config import Config
        cfg, setting = compat.runner_randomness_config(Config(dict(
            randomness=dict(seed=42, deterministic=True, warn_only=True))), Runner)
        self.assertNotIn('warn_only', cfg.randomness)
        self.assertTrue(setting)


class StatisticsTests(unittest.TestCase):
    def test_memory_counts_norms_and_ddp_padding_samples(self):
        from test_support_visual import memory_fixture
        memory = memory_fixture(channels=256)
        memory['tokens'].fill_(1.)
        stats = stats_module.VisualTokenStats(2, 'cpu')
        stats.valid = [True, False]
        text, maps = text_fixture(2)
        stats.support_features(memory, 2)
        stats.text_features(text['embedded'], maps, 'text_norm')
        stats.text_features(text['embedded'] * 2, maps, 'enhanced_text_norm')
        result = stats.report(('a', 'b'))
        self.assertEqual(result['overall']['support_token_count'], 5)
        self.assertEqual(result['classes']['a']['support_token_count'], 3)
        self.assertEqual(result['overall']['support_token_norm'], dict(value=16., count=5))
        self.assertEqual(result['overall']['text_norm']['count'], 3)
        self.assertAlmostEqual(result['overall']['enhanced_text_norm']['value'],
                               2 * result['overall']['text_norm']['value'])
        # Simulate DDP reduction of scalar accumulators. Memory counts stay local.
        stats.sums.mul_(2)
        stats.counts.mul_(2)
        self.assertEqual(stats.report(('a', 'b'))['overall']['support_token_count'], 5)
        stats.finish_batch()
        self.assertIsNone(stats.valid)
        self.assertTrue(all(value.grad_fn is None for value in
                            (stats.sums, stats.counts, stats.support_counts)))
        json.dumps(stats.report(('a', 'b')), allow_nan=False)

    def test_epoch_tail_only_and_validation_streaming_without_tensors(self):
        env = dict(torch=torch, dist=torch.distributed, Path=Path, json=json,
                   Hook=object, is_model_wrapper=lambda model: False,
                   VisualTokenStats=stats_module.VisualTokenStats)
        cls = node(source('mmdet/engine/hooks/visual_token_hook.py'), 'VisualTokenHook')
        cls.decorator_list = []
        execute([cls], env)
        model = torch.nn.Linear(1, 1)
        model.visual_enabled = True
        model.clear_support_cache = Mock()
        model.bbox_head = NS(num_classes=2)
        model.support_visual_cfg = dict(class_names=('a', 'b'))
        with tempfile.TemporaryDirectory() as tmp:
            runner = NS(model=model, train_dataloader=[0, 1, 2],
                        val_dataloader=NS(dataset=[0, 1, 2]), epoch=2, iter=9,
                        work_dir=tmp, logger=Mock())
            hook = env['VisualTokenHook']('toy', 5)
            for i in range(2):
                hook.before_train_iter(runner, i)
                self.assertIsNone(hook.stats)
            hook.before_train_iter(runner, 2)
            self.assertIsNotNone(hook.stats)
            hook.after_train_iter(runner, 2)
            self.assertIsNone(model._visual_stats)
            hook._prepare_cache = Mock()
            hook.before_val_epoch(runner)
            model.clear_support_cache.assert_not_called()
            hook.before_val_iter(runner, 0, dict(data_samples=[0, 1]))
            self.assertEqual(hook.stats.valid, [True, True])
            self.assertTrue(hook.stats.capture_support)
            hook.after_val_iter(runner, 0)
            hook.before_val_iter(runner, 1, dict(data_samples=[2, 3]))
            self.assertEqual(hook.stats.valid, [True, False])
            self.assertFalse(hook.stats.capture_support)
            hook.after_val_iter(runner, 1)
            hook.before_val_iter(runner, 50, dict(data_samples=[0]))
            self.assertTrue(hook.stats.capture_support)
            hook.after_val_iter(runner, 50)
            hook.after_val_epoch(runner)
            rows = [json.loads(line) for line in (Path(tmp) / 'visual_token_stats.jsonl').read_text().splitlines()]
            self.assertEqual([row['phase'] for row in rows], ['train', 'val'])
            self.assertTrue(all(row['dataset'] == 'toy' and row['shot'] == 5 for row in rows))
            self.assertTrue(all(row['schema_version'] == 3 and row['architecture'] == 'support_local_cross_attention' for row in rows))
            self.assertLess((Path(tmp) / 'visual_token_stats.jsonl').stat().st_size, 6000)


class ConfigTests(unittest.TestCase):
    def test_serial_layers_and_progressive_hooks_unchanged_from_start(self):
        for rel in ('mmdet/engine/hooks/stage_lr_hook.py',
                    'mmdet/models/layers/transformer/dino_layers.py'):
            original = subprocess.check_output(['git', 'show', '449e69a:' + rel], cwd=ROOT, encoding='utf8')
            self.assertEqual(source(rel), original)

    def test_decoder_layers_and_all_18_configs_unchanged_from_base(self):
        rel = 'mmdet/models/layers/transformer/grounding_dino_layers.py'
        original = subprocess.check_output(['git', 'show', '72bced2:' + rel], cwd=ROOT, encoding='utf8')
        for name in ('GroundingDinoTransformerDecoderLayer', 'GroundingDinoTransformerDecoder'):
            self.assertEqual(ast.dump(node(source(rel), name)), ast.dump(node(original, name)))
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '72bced2',
                                         'configs_cdfsod'], cwd=ROOT, encoding='utf8').splitlines()
        for path in paths:
            if path.endswith('.py'):
                content = subprocess.check_output(['git', 'show', '72bced2:' + path], cwd=ROOT, encoding='utf8')
                self.assertEqual(source(path), content)

    @unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine not installed')
    def test_all_18_configs_parse_and_preserve_query_training_settings(self):
        from mmengine.config import Config
        configs = sorted((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(configs), 18)
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            # Mirror original relative base imports in a private temporary tree.
            for rel in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
                                                '449e69a', 'configs_cdfsod'], cwd=ROOT,
                                               encoding='utf8').splitlines():
                if rel.endswith('.py'):
                    dest = temp_root / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(subprocess.check_output(['git', 'show', '449e69a:' + rel], cwd=ROOT))
            for path in configs:
                with self.subTest(config=path.name):
                    cfg = Config.fromfile(str(path))
                    old = Config.fromfile(str(temp_root / path.relative_to(ROOT)))
                    support = cfg.model.support_visual_cfg
                    self.assertEqual(support.dataset.ann_file, cfg.train_dataloader.dataset.ann_file)
                    self.assertEqual(support.dataset.data_prefix, cfg.train_dataloader.dataset.data_prefix)
                    self.assertEqual(tuple(support.class_names), cfg.class_names)
                    self.assertTrue(support.dataset.test_mode)
                    self.assertIsNone(support.dataset.filter_cfg)
                    self.assertEqual([t.type for t in support.dataset.pipeline],
                                     ['LoadImageFromFile', 'LoadAnnotations', 'FixScaleResize', 'PackDetInputs'])
                    self.assertEqual(cfg.custom_hooks[-1].type, 'VisualTokenHook')
                    for key in ('train_pipeline', 'train_dataloader', 'val_dataloader',
                                'test_dataloader', 'optim_wrapper', 'param_scheduler',
                                'train_cfg', 'default_hooks', 'randomness'):
                        self.assertEqual(cfg[key], old[key], key)
                    self.assertEqual(cfg.custom_hooks[:-1], old.custom_hooks)
                    model, old_model = copy.deepcopy(cfg.model), copy.deepcopy(old.model)
                    model.pop('support_visual_cfg')
                    for key in ('lambda_region_text', 'region_text_roi_size', 'region_text_featmap_strides'):
                        old_model.pop(key)
                    self.assertEqual(model, old_model)
                    pipeline = cfg.val_dataloader.dataset.pipeline
                    self.assertGreater(next(i for i,t in enumerate(pipeline) if t.type == 'LoadAnnotations'),
                                       next(i for i,t in enumerate(pipeline) if 'Resize' in t.type))


if __name__ == '__main__':
    unittest.main()
