"""Behavioral coverage for the compact pre-enhancer token/ETF fork."""
import ast
import inspect
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import test_qwen_offline_sanity as sanity

BASE = '2c0892a2f35da0813182720049b0866f553f394e'


class PreEnhancerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        fixture = sanity.QwenSanity()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.model = fixture.model(
            entries={'pitted_surface': 'small uneven holes',
                     'other': 'dark round shape', 'beetles': 'shiny wings'},
            names=['pitted_surface', 'other', 'beetles'])

    def test_projection_receives_only_ordered_class_subwords(self):
        model = self.model
        hidden = model._encode_support_prompt_features(
            model._prepare_cached_tokenized('cpu')).detach().requires_grad_()
        model._encode_support_prompt_features = lambda _: hidden
        seen = []
        hook = model.text_feat_map.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        self.addCleanup(hook.remove)
        text = model.build_prototype_text_dict(3, 'cpu')
        expected = torch.cat([hidden[0, [1, 2]], hidden[1, [1]], hidden[2, [1, 2]]])
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], expected)
        torch.testing.assert_close(text['embedded'][0], model.text_feat_map(expected))
        self.assertEqual(text['embedded'].shape, (3, 5, 3))
        self.assertEqual(text['position_ids'][0].tolist(), [1, 2, 1, 1, 2])
        groups = torch.tensor([0, 0, 1, 2, 2])
        torch.testing.assert_close(text['masks'][0], groups[:, None] == groups[None, :])
        text['embedded'].sum().backward()
        selected = torch.zeros(hidden.shape[:2], dtype=torch.bool)
        selected[0, [1, 2]] = True
        selected[1, 1] = True
        selected[2, [1, 2]] = True
        self.assertTrue((hidden.grad[~selected] == 0).all())
        self.assertTrue((hidden.grad[selected].abs().sum(-1) > 0).all())

    def test_etf_runs_before_enhancer_on_all_classes_with_empty_gt(self):
        model = self.model
        model.nearest_etf_loss_weight = 0.25
        model.extract_feat = lambda images: (images,)
        samples = [SimpleNamespace(text=tuple(model.support_class_names),
                   gt_instances=SimpleNamespace(labels=labels))
                   for labels in [torch.tensor([1]), torch.tensor([], dtype=torch.long)]]
        seen = {}

        def auxiliary(features, maps, mask):
            self.assertNotIn('enhancer', seen)
            self.assertEqual(maps, [{1: [0, 1], 2: [2], 3: [3, 4]}] * 2)
            seen['raw'] = features
            seen['snapshot'] = features.detach().clone()
            prototypes = torch.stack([features[:, :2].mean(1), features[:, 2],
                                      features[:, 3:].mean(1)], dim=1)
            seen['expected'] = sanity.etf.nearest_etf_loss(prototypes)
            return sanity.raw_etf.raw_mean_etf_loss(features, maps, mask)

        def forward(features, text, data):
            self.assertIn('raw', seen)
            self.assertIs(text['embedded'], seen['raw'])
            seen['enhancer'] = True
            return dict(memory_text=text['embedded'] * 2,
                        text_token_mask=text['text_token_mask'])

        model.forward_transformer = forward
        model.bbox_head.loss = lambda **kw: dict(loss_detection=kw['memory_text'].square().mean())
        with patch.dict(sanity.Detector.loss.__globals__, raw_mean_etf_loss=auxiliary):
            losses = model.loss(torch.zeros(2, 3, 4, 4), samples)
        torch.testing.assert_close(losses['loss_nearest_etf'], 0.25 * seen['expected'])
        torch.testing.assert_close(seen['raw'], seen['snapshot'])
        self.assertEqual(samples[1].gt_instances.text_token_mask.shape, (0, 5))

    def test_default_weight_and_invalid_weights(self):
        signature = inspect.signature(sanity.Detector.__init__)
        self.assertEqual(signature.parameters['nearest_etf_loss_weight'].default, 1.0)
        for weight in [-1, float('nan'), float('inf')]:
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                sanity.Detector(language_model={}, nearest_etf_loss_weight=weight)

    def test_nonprototype_mode_skips_etf_even_with_positive_weight(self):
        model = self.model
        model.use_class_name_token_prototypes = False
        model.nearest_etf_loss_weight = 1.0

        class Sample(dict):
            __getattr__ = dict.__getitem__

        samples = [Sample(text='other', gt_instances=SimpleNamespace(labels=torch.tensor([0])))]
        model.get_tokens_and_prompts = lambda *a: (None, 'other.', [[[0, 5]]], ['other'])
        model.get_positive_map = lambda *a: ({1: [0]}, torch.ones(1, 8))
        model.language_model.forward = lambda *a: dict(
            embedded=torch.randn(1, 2, 4), text_token_mask=torch.ones(1, 2, dtype=torch.bool))
        model.extract_feat = lambda x: (x,)
        model.forward_transformer = lambda visual, text, data: dict(memory_text=text['embedded'])
        model.bbox_head.loss = lambda **kw: dict(loss_detection=kw['memory_text'].sum())
        with patch.dict(sanity.Detector.loss.__globals__,
                        raw_mean_etf_loss=lambda *a: self.fail('ETF in nonprototype mode')):
            losses = model.loss(torch.zeros(1, 3, 4, 4), samples)
        self.assertEqual(set(losses), {'loss_detection'})

    def test_configs_differ_from_exact_base_only_in_etf_weight(self):
        for path in (sanity.ROOT / 'configs_cdfsod/final_configs_bs4').glob('*shot.py'):
            relative = path.relative_to(sanity.ROOT).as_posix()
            original = subprocess.check_output(['git', 'show', f'{BASE}:{relative}'],
                                               cwd=sanity.ROOT).decode()
            self.assertEqual(path.read_text(encoding='utf-8'),
                             original.replace('nearest_etf_loss_weight=0.1',
                                              'nearest_etf_loss_weight=1.0'))
        for relative in ['mmdet/engine/hooks/stage_lr_hook.py',
                         'mmdet/models/losses/nearest_etf_loss.py',
                         'mmdet/models/layers/transformer/grounding_dino_layers.py',
                         'configs_cdfsod/grounding_dino_swin-b_pretrain_all.py']:
            original = subprocess.check_output(['git', 'show', f'{BASE}:{relative}'],
                                               cwd=sanity.ROOT).decode()
            self.assertEqual((sanity.ROOT / relative).read_text(encoding='utf-8'), original)

    def test_existing_pft_lr_freeze_and_unfreeze_with_auxiliary_loss(self):
        model = self.model
        model.encoder = sanity.recording_encoder()
        path = sanity.ROOT / 'mmdet/engine/hooks/stage_lr_hook.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'BBoxHeadFirstHook6')
        cls.bases = [ast.Name(id='object', ctx=ast.Load())]
        cls.decorator_list = []
        env = dict(nn=nn, is_model_wrapper=lambda _: False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     str(path), 'exec'), env)
        optimizer = torch.optim.AdamW([dict(params=[p], lr=1e-3)
                                       for p in model.parameters()])
        runner = SimpleNamespace(model=model, epoch=0,
                                 optim_wrapper=SimpleNamespace(optimizer=optimizer),
                                 logger=SimpleNamespace(info=lambda *a: None))
        hook = env['BBoxHeadFirstHook6']()
        hook.before_train(runner)
        for stage in [1, 2]:
            if stage == 2:
                for i in hook._lang_model_groups:
                    optimizer.param_groups[i]['lr'] *= 0.5
                hook.before_train_epoch(runner)
                self.assertTrue(hook._stage2_started)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            text = model.build_prototype_text_dict(2, 'cpu')
            auxiliary = sanity.raw_etf.raw_mean_etf_loss(
                text['embedded'], [model.build_prototype_token_positive_map()] * 2,
                text['text_token_mask'])
            features = torch.randn(2, 4, 3)
            output = model.forward_encoder(
                feat=features, feat_mask=torch.zeros(2, 4, dtype=torch.bool),
                feat_pos=torch.zeros_like(features), spatial_shapes=torch.tensor([[2, 2]]),
                level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2),
                text_dict=text)
            loss = auxiliary + output['memory_text'].square().mean() + output['memory'].square().mean()
            loss.backward()
            self.assertTrue(all(p.requires_grad and p.grad is not None for p in model.parameters()))
            optimizer.step()
            self.assertFalse(torch.equal(before['text_feat_map.weight'], model.text_feat_map.weight))
            for name, p in model.encoder.named_parameters():
                unchanged = torch.equal(before['encoder.' + name], p)
                if stage == 1:
                    self.assertTrue(unchanged)
            if stage == 2:
                self.assertFalse(torch.equal(before['encoder.text_layers.0.attn.in_proj_weight'],
                                             model.encoder.text_layers[0].attn.in_proj_weight))


if __name__ == '__main__':
    unittest.main()
