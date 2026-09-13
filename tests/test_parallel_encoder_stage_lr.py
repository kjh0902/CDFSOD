"""Exercise the production stage hook with CPU AdamW, without MMEngine.

Run: python -m unittest discover -s tests -p 'test_parallel_encoder*.py' -v
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


def load_hook():
    path = (Path(__file__).resolve().parents[1] / 'mmdet' / 'engine' /
            'hooks' / 'stage_lr_hook.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'BBoxHeadFirstHook6')
    node.decorator_list = []
    namespace = dict(nn=nn, Hook=object,
                     is_model_wrapper=lambda model: hasattr(model, 'module'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
         namespace)
    return namespace['BBoxHeadFirstHook6']


BBoxHeadFirstHook6 = load_hook()


def make_runner(num_layers=6, wrapped=False):
    model = nn.Module()
    for name in ('backbone', 'language_model', 'bbox_head', 'decoder', 'neck',
                 'text_feat_map', 'dn_query_generator', 'query_embedding'):
        model.add_module(name, nn.Linear(2, 2))
    model.encoder = nn.Module()
    for name in ('layers', 'text_layers', 'fusion_layers'):
        model.encoder.add_module(name, nn.ModuleList(
            [nn.Linear(2, 2) for _ in range(num_layers)]))
    model.encoder.extra = nn.Linear(2, 2)
    groups = []
    for name, parameter in model.named_parameters():
        lr = 2e-5 if name.startswith(('backbone.', 'language_model.')) else 1e-4
        groups.append(dict(params=[parameter], lr=lr, name=name))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    runner = SimpleNamespace(
        model=SimpleNamespace(module=model) if wrapped else model,
        optim_wrapper=SimpleNamespace(optimizer=optimizer), epoch=0,
        logger=logging.getLogger(__name__),
        param_schedulers=[SimpleNamespace(patience=5)])
    return runner, model, optimizer


def optimizer_step(model, optimizer):
    # Nonzero gradients and weight decay distinguish LR=0 from true updates.
    optimizer.zero_grad()
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()


class TestParallelEncoderStageLR(unittest.TestCase):
    def test_stage1_lrs_and_actual_updates(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                runner, model, optimizer = make_runner(wrapped=wrapped)
                original_lrs = [group['lr'] for group in optimizer.param_groups]
                original = {name: p.detach().clone()
                            for name, p in model.named_parameters()}
                hook = BBoxHeadFirstHook6(adjust_scheduler_patience=True)
                hook.before_train(runner)
                self.assertEqual(runner.param_schedulers[0].patience, 3)
                optimizer_step(model, optimizer)
                for group, original_lr in zip(optimizer.param_groups, original_lrs):
                    name = group['name']
                    frozen = (name.startswith('query_embedding.') or
                              name.startswith('encoder.extra.') or
                              any(name.startswith(f'encoder.{kind}.0.') for kind
                                  in ('layers', 'text_layers', 'fusion_layers')))
                    self.assertEqual(group['lr'], 0 if frozen else original_lr)
                    parameter = group['params'][0]
                    self.assertTrue(parameter.requires_grad)
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(torch.equal(parameter, original[name]), frozen)
                self.assertEqual(len(hook._parallel_encoder_groups), 5 * 3 * 2)

    def test_stage2_threshold_and_unfreeze(self):
        runner, model, optimizer = make_runner()
        hook = BBoxHeadFirstHook6(adjust_scheduler_patience=True,
                                 patience_unfrozen=8)
        hook.before_train(runner)
        hook.before_train_epoch(runner)
        self.assertFalse(hook._stage2_started)
        # Simulate the existing plateau scheduler's factor=0.5 reduction.
        for group in optimizer.param_groups:
            group['lr'] *= 0.5
        hook.before_train_epoch(runner)
        self.assertTrue(hook._stage2_started)
        self.assertEqual(runner.param_schedulers[0].patience, 8)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer_step(model, optimizer)
        for group in optimizer.param_groups:
            name = group['name']
            expected = 1e-5 if name.startswith(('backbone.', 'language_model.')) else 5e-5
            self.assertEqual(group['lr'], expected)
            self.assertFalse(torch.equal(group['params'][0], before[name]))
        # Stage 2 is entered once; later scheduler rates must not be reset.
        for group in optimizer.param_groups:
            group['lr'] *= 0.5
        reduced = [group['lr'] for group in optimizer.param_groups]
        hook.before_train_epoch(runner)
        self.assertEqual([group['lr'] for group in optimizer.param_groups], reduced)

    def test_layer_indices_are_not_hardcoded(self):
        for count in (1, 12):
            with self.subTest(num_layers=count):
                runner, _, optimizer = make_runner(num_layers=count)
                hook = BBoxHeadFirstHook6()
                hook.before_train(runner)
                self.assertEqual(len(hook._parallel_encoder_groups), (count - 1) * 6)
                for index in hook._parallel_encoder_groups:
                    self.assertEqual(optimizer.param_groups[index]['lr'], 1e-4)


if __name__ == '__main__':
    unittest.main()
