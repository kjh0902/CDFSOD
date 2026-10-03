"""Joint ETF regression checks using the production detector and ACL LR hook."""
import ast
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import test_raw_mean_etf_integration as integration


BASE = '30090c3d48ba0fe7aabb04546a6cf72dbbca1118'
BERT_LOSS = 'loss_bert_raw_mean_etf'
FE_LOSS = 'loss_fe_raw_mean_etf'


def load_hook():
    node = next(n for n in ast.parse(integration.source(
        'mmdet/engine/hooks/stage_lr_hook.py')).body
        if isinstance(n, ast.ClassDef) and n.name == 'BBoxHeadFirstHook6')
    node.decorator_list = []
    return integration.execute(
        [node], Hook=object, is_model_wrapper=lambda model: False)[node.name]()


class BertFEETFTests(unittest.TestCase):
    assert_nested_equal = integration.RawMeanIntegrationTests.assert_nested_equal
    def assert_gradient(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_both_sources_share_all_class_name_tokens(self):
        for mode in ['shared', 'different', 'explicit_list', 'explicit_dict',
                     'description']:
            with self.subTest(mode=mode):
                model = integration.fixture(weight=0.3, fe_weight=0.7)
                data = integration.samples()  # Includes an empty-GT sample.
                if mode == 'different':
                    data[1].text = integration.NAMES[::-1]
                elif mode.startswith('explicit'):
                    _, caption, spans, _ = model.get_tokens_and_prompts(
                        integration.NAMES, True)
                    for sample in data:
                        sample.text = caption
                        sample.tokens_positive = (
                            dict(enumerate(spans)) if mode.endswith('dict') else spans)
                elif mode == 'description':
                    caption, spans = '', []
                    for name in integration.NAMES:
                        spans.append([[len(caption), len(caption) + len(name)]])
                        caption += name + ' with visible texture. '
                    for sample in data:
                        sample.text, sample.tokens_positive = caption, spans
                with patch.dict(integration.Detector.loss.__globals__,
                                raw_mean_etf_loss=Mock(
                                    wraps=integration.raw_mean.raw_mean_etf_loss)):
                    solver = integration.Detector.loss.__globals__['raw_mean_etf_loss']
                    with patch.object(integration.raw_mean, 'nearest_etf_loss',
                                      wraps=integration.raw_mean.nearest_etf_loss) as nearest:
                        losses, _ = integration.run(model, data)
                self.assertEqual(solver.call_count, 2)
                self.assertEqual(nearest.call_count, 2)
                bert_call, fe_call = solver.call_args_list
                self.assertIs(bert_call.args[0], model.text_feat_map.output)
                self.assertIs(fe_call.args[0], model.bbox_head.seen['memory_text'])
                self.assertIs(bert_call.args[1], fe_call.args[1])
                self.assertIs(fe_call.args[2], model.bbox_head.seen['text_token_mask'])
                for call, prototype_call, key, weight in zip(
                        [bert_call, fe_call], nearest.call_args_list,
                        [BERT_LOSS, FE_LOSS], [0.3, 0.7]):
                    features, maps, mask = call.args
                    self.assertTrue(all(set(m) == set(range(1, 7)) for m in maps))
                    means = torch.stack([
                        torch.stack([row[m[c]].mean(0) for c in range(1, 7)])
                        for row, m in zip(features, maps)])
                    self.assertEqual(means.shape, (2, 6, integration.FEATURE_DIM))
                    torch.testing.assert_close(prototype_call.args[0], means)
                    torch.testing.assert_close(losses[key], weight *
                        integration.raw_mean.raw_mean_etf_loss(features, maps, mask))
                    if mode == 'description':
                        for m in maps:
                            for indices in m.values():
                                self.assertNotEqual(indices, list(range(features.size(1))))
                        tokenized = model.language_model.tokenizer([caption])
                        description_token = tokenized.char_to_token(caption.index('visible'))
                        self.assertNotIn(description_token,
                                         {i for m in maps for ids in m.values() for i in ids})

    def test_weights_independently_enable_and_scale_losses(self):
        for bert_weight, fe_weight in [(0., 0.), (0.3, 0.), (0., 0.7), (0.3, 0.7)]:
            with self.subTest(bert=bert_weight, fe=fe_weight):
                model = integration.fixture(bert_weight, fe_weight)
                with patch.dict(integration.Detector.loss.__globals__,
                                raw_mean_etf_loss=Mock(
                                    wraps=integration.raw_mean.raw_mean_etf_loss)):
                    solver = integration.Detector.loss.__globals__['raw_mean_etf_loss']
                    with patch.object(model, '_get_raw_mean_class_token_map',
                                      wraps=model._get_raw_mean_class_token_map) as mapping:
                        losses, _ = integration.run(model, integration.samples())
                self.assertEqual(solver.call_count, int(bert_weight > 0) + int(fe_weight > 0))
                self.assertEqual(mapping.call_count, int(bool(bert_weight or fe_weight)))
                self.assertEqual(BERT_LOSS in losses, bert_weight > 0)
                self.assertEqual(FE_LOSS in losses, fe_weight > 0)
                for call, weight, key in zip(solver.call_args_list,
                        [w for w in [bert_weight, fe_weight] if w > 0],
                        [k for k, w in [(BERT_LOSS, bert_weight), (FE_LOSS, fe_weight)] if w > 0]):
                    torch.testing.assert_close(losses[key], weight *
                        integration.raw_mean.raw_mean_etf_loss(*call.args))

    def test_fe_gradient_and_optimizer_behavior_in_both_acl_stages(self):
        model = integration.fixture(weight=1., fe_weight=1.)
        optimizer = torch.optim.AdamW([
            dict(params=[p], lr=0.001) for p in model.parameters()], weight_decay=0.)
        runner = SimpleNamespace(model=model, optim_wrapper=SimpleNamespace(optimizer=optimizer),
                                 logger=Mock(), epoch=0)
        hook = load_hook()
        hook.before_train(runner)
        hook.before_train_epoch(runner)
        self.assertFalse(hook._stage2_started)
        for stage in [1, 2]:
            with self.subTest(stage=stage):
                if stage == 2:
                    for index in hook._lang_model_groups:
                        optimizer.param_groups[index]['lr'] *= 0.5
                    runner.epoch = 1
                    hook.before_train_epoch(runner)
                    self.assertTrue(hook._stage2_started)
                self.assertTrue(all(optimizer.param_groups[i]['lr'] ==
                    (0. if stage == 1 else 0.001) for i in hook._other_groups))
                self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
                before = copy.deepcopy(model.state_dict())
                optimizer.zero_grad(set_to_none=True)
                losses, _ = integration.run(model, integration.samples())
                self.assertIn(BERT_LOSS, losses)
                self.assertIn(FE_LOSS, losses)
                final = model.bbox_head.seen['memory_text']
                final.retain_grad()
                model.text_feat_map.output.retain_grad()
                # Backpropagate FE alone to prove BERT/backbone gradients come from FE.
                losses[FE_LOSS].backward()
                for parameter in [final, model.text_feat_map.output,
                                  model.text_feat_map.weight,
                                  model.language_model.embedding.weight,
                                  model.backbone.weight]:
                    self.assert_gradient(parameter)
                for layer in [*model.encoder.fusion_layers, *model.encoder.text_layers]:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in layer.parameters()))
                for module in [model.decoder, model.bbox_head, model.query_embedding]:
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                optimizer.step()
                encoder_changed = any(not torch.equal(before[name], tensor)
                    for name, tensor in model.state_dict().items() if name.startswith('encoder.'))
                self.assertEqual(encoder_changed, stage == 2)
                for name in ['backbone.weight', 'text_feat_map.weight',
                             'language_model.embedding.weight']:
                    self.assertFalse(torch.equal(before[name], model.state_dict()[name]))

    def test_joint_total_gradient_is_sum_of_detection_bert_and_fe(self):
        model = integration.fixture(weight=0.3, fe_weight=0.7)
        losses, _ = integration.run(model, integration.samples())
        parameters = tuple(model.parameters())
        terms = [losses['loss_cls'] + losses['loss_bbox'], losses[BERT_LOSS], losses[FE_LOSS]]
        gradients = [torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                     for term in terms]
        sum(terms).backward()
        for index, parameter in enumerate(parameters):
            expected = sum((g[index] for g in gradients if g[index] is not None),
                           torch.zeros_like(parameter))
            if parameter.grad is None:
                self.assertEqual(expected.abs().sum().item(), 0.)
            else:
                torch.testing.assert_close(parameter.grad, expected)

    def test_joint_auxiliary_keeps_detection_values_and_gradients(self):
        reference = integration.fixture(0., 0.)
        _, expected = integration.run(reference, integration.samples())
        model = integration.fixture(0.3, 0.7)
        losses, actual = integration.run(model, integration.samples())
        actual['losses'] = {k: v for k, v in losses.items() if k not in {BERT_LOSS, FE_LOSS}}
        self.assert_nested_equal(actual, expected)
        self.assertEqual(reference.state_dict().keys(), model.state_dict().keys())
        for detector, snapshot in [(reference, expected), (model, actual)]:
            (snapshot['losses']['loss_cls'] + snapshot['losses']['loss_bbox']).backward()
        self.assert_nested_equal({k: p.grad for k, p in reference.named_parameters()},
                                 {k: p.grad for k, p in model.named_parameters()})

    def test_joint_inference_never_builds_mapping_or_solves(self):
        model = integration.fixture(1., 1.)
        model.eval()
        with patch.object(model, '_get_raw_mean_class_token_map', side_effect=AssertionError), \
                patch.dict(integration.Detector.loss.__globals__, raw_mean_etf_loss=Mock(
                    side_effect=AssertionError)):
            result = model.predict(torch.zeros(2, 3, 4, 4), integration.samples())
        self.assertEqual(result[0].pred_instances.label_names, ['crazing'])

    def test_both_weights_reject_nonfinite_or_negative_values(self):
        model = integration.Detector(language_model={})
        self.assertEqual(model.bert_raw_mean_etf_loss_weight, 0.)
        self.assertEqual(model.fe_raw_mean_etf_loss_weight, 0.)
        for name in ['bert_raw_mean_etf_loss_weight', 'fe_raw_mean_etf_loss_weight']:
            for value in [-1., float('nan'), float('inf'), -float('inf')]:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    integration.Detector(language_model={}, **{name: value})

    def test_latest_base_cleanup_and_etf_math_are_unchanged(self):
        with patch.object(integration, 'BASE', BASE):
            for path in ['mmdet/utils/distributed_runtime.py', 'tools/train.py', 'tools/test.py',
                         'UODD_SHUTDOWN.md', 'mmdet/engine/hooks/stage_lr_hook.py',
                         'mmdet/engine/hooks/__init__.py',
                         'mmdet/models/losses/nearest_etf_loss.py',
                         'configs_cdfsod/grounding_dino_swin-b_pretrain_all.py']:
                self.assertEqual(integration.source(path), integration.source(path, True), path)
            raw_path = 'mmdet/models/losses/raw_mean_etf_loss.py'
            # Only the module description changes to mention both feature sources.
            self.assertEqual(integration.source(raw_path).splitlines()[1:],
                             integration.source(raw_path, True).splitlines()[1:])


if __name__ == '__main__':
    unittest.main()
