"""CPU regressions using production detector/head/encoder methods and IO.

No CUDA, pretrained weights, transformers or MMCV extensions are required.
"""
import ast
from collections import OrderedDict
import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from acl_diagnostic_test_support import ROOT, events, diagnostic_io as dio, capture_module as cap
import test_raw_mean_etf_integration as reference


def load_hook(hook_base=object, checkpoint_type=None):
    path = ROOT / 'mmdet/engine/hooks/acl_diagnostic_hook.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef))
    node.decorator_list = []
    env = dict(__file__=str(path), Hook=hook_base,
               CheckpointHook=checkpoint_type or type('CheckpointHook', (), {}),
               Path=Path, math=math, np=np, subprocess=subprocess,
               DiagnosticWriter=dio.DiagnosticWriter, ACLBatchCapture=cap.ACLBatchCapture,
               activate_diagnostics=events.activate_diagnostics,
               deactivate_diagnostics=events.deactivate_diagnostics,
               diagnostic_event=events.diagnostic_event,
               is_model_wrapper=lambda _: False, get_dist_info=lambda: (0, 1),
               load=lambda path, **kw: json.loads(Path(path).read_text(encoding='utf-8')))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env['ACLDiagnosticHook']


DiagnosticHook = load_hook()


class Instances(SimpleNamespace):
    def __len__(self):
        return len(self.labels)

    @staticmethod
    def cat(items):
        return Instances(**{key: torch.cat([getattr(item, key) for item in items])
                            for key in ('bboxes', 'labels', 'scores')})


def xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), -1)


PREDICT_SINGLE = reference.method(reference.HEAD, 'GroundingDINOHead_ParallelDecoder_DN',
                                  '_predict_by_feat_single', bbox_cxcywh_to_xyxy=xyxy,
                                  InstanceData=Instances)
GENERATE_PROPOSALS = reference.method('mmdet/models/detectors/deformable_detr.py',
                                      'DeformableDETR', 'gen_encoder_output_proposals')


def full_fixture():
    model = reference.fixture(weight=0.)
    original_extract = model.extract_feat
    model.extract_feat = lambda inputs: tuple(value[:len(inputs)] for value in original_extract(inputs))
    def pre_transformer(features, samples):
        feat = features[0]
        batch, points, _ = feat.shape
        padding = torch.zeros(batch, points, dtype=torch.bool)
        shapes = torch.tensor([[1, points]])
        ratios = torch.ones(batch, 1, 2)
        return (dict(feat=feat, feat_mask=padding, feat_pos=torch.zeros_like(feat),
                     spatial_shapes=shapes, level_start_index=torch.tensor([0]), valid_ratios=ratios),
                dict(memory_mask=padding, spatial_shapes=shapes,
                     level_start_index=torch.tensor([0]), valid_ratios=ratios))
    model.pre_transformer = pre_transformer
    model.memory_trans_fc = nn.Identity()
    model.memory_trans_norm = nn.Identity()
    model.gen_encoder_output_proposals = MethodType(GENERATE_PROPOSALS, model)
    language = model.language_model
    class Backbone(nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
        def forward(self, inputs):
            return dict(embedded=self.embedding(inputs['input_ids']),
                        masks=inputs['attention_mask'])
    language.language_backbone = Backbone(language.embedding)
    def language_forward(self, prompts):
        tokens = self.tokenizer(prompts, truncation=True, max_length=self.max_tokens)
        batch, length = tokens['input_ids'].shape
        positions = torch.arange(length).expand(batch, -1)
        output = self.language_backbone(dict(
            input_ids=tokens['input_ids'], position_ids=positions,
            attention_mask=torch.ones(batch, length, length, dtype=torch.bool)))
        output.update(text_token_mask=tokens['attention_mask'].bool(), position_ids=positions)
        return output
    language.forward = MethodType(language_forward, language)
    language.tokenizer.convert_ids_to_tokens = lambda ids: [str(i) for i in ids]
    head = model.bbox_head
    head.test_cfg = dict(max_per_img=5)
    head._predict_by_feat_single = MethodType(PREDICT_SINGLE, head)
    # Use the actual token-to-class conversion.
    def convert(logits, positive_maps):
        output = torch.zeros(*logits.shape[:2], len(positive_maps[0]))
        for i, mapping in enumerate(positive_maps):
            for label, tokens in mapping.items():
                output[i, :, label - 1] = logits[i, :, tokens].mean(-1)
        return output
    head.predict_by_feat = MethodType(reference.method(
        reference.HEAD, 'GroundingDINOHead_ParallelDecoder_DN', 'predict_by_feat',
        convert_grounding_to_cls_scores=convert), head)
    head.predict = MethodType(reference.method(
        reference.HEAD, 'GroundingDINOHead_ParallelDecoder_DN', 'predict'), head)
    return model


def data_samples():
    data = reference.samples()
    for i, sample in enumerate(data):
        sample.custom_entities = True
        sample.metainfo = dict(img_id=101 + i, img_path=f'{i}.jpg',
                               ori_shape=(40, 60), img_shape=(20, 30),
                               scale_factor=(.5, .5), batch_input_shape=(20, 30))
    return data


class Config(dict):
    def __getattr__(self, name):
        return self[name]
    pretty_text = 'fixture baseline config'


def runner_fixture(directory):
    model = full_fixture()
    categories = [dict(id=i + 10, name=name) for i, name in enumerate(reference.NAMES)]
    annotation = dict(images=[dict(id=i, width=60, height=40) for i in (101, 102)],
                      categories=categories,
                      annotations=[dict(id=1, image_id=101, category_id=10,
                                        bbox=[20, 10, 10, 10], area=100, iscrowd=0)])
    annotation_path = Path(directory) / 'annotations.json'
    annotation_path.write_text(json.dumps(annotation), encoding='utf-8')
    info = [dict(img_id=i, text=reference.NAMES) for i in (101, 102)]
    dataset = SimpleNamespace(full_init=lambda: None, get_data_info=info.__getitem__,
                              ann_file=annotation_path, backend_args=None,
                              cat_ids=list(range(10, 16)))
    class Dataset:
        def __len__(self):
            return 2
        def __getattr__(self, name):
            return getattr(dataset, name)
    loader = SimpleNamespace(dataset=Dataset(), sampler=[0, 1])
    logger_hook = type('LoggerHook', (), dict(interval=50, ignore_last=True))()
    stage_hook = type('BBoxHeadFirstHook6', (), dict(_stage2_started=False))()
    cfg = Config(resume=False, val_dataloader=Config(dataset=Config(pipeline=[
        dict(type='FixScaleResize', scale=(800, 1333))])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    runner = SimpleNamespace(
        model=model, val_dataloader=loader, hooks=[logger_hook, stage_hook],
        cfg=cfg, max_epochs=2, timestamp='cpu_fixture', work_dir=directory,
        seed=42, epoch=0, iter=0,
        optim_wrapper=SimpleNamespace(optimizer=optimizer),
        logger=SimpleNamespace(info=lambda _: None),
        train_dataloader=[0, 1, 2],
        log_processor=SimpleNamespace(window_size=50),
        message_hub=SimpleNamespace(log_scalars={}))
    return runner


class DiagnosticTests(unittest.TestCase):
    def test_all_18_configs_have_only_allowed_changes(self):
        paths = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'))
        self.assertEqual(len(paths), 18)
        for path in paths:
            old = subprocess.check_output(['git', 'show', 'd560d78:' +
                                          path.relative_to(ROOT).as_posix()], cwd=ROOT).decode('utf-8')
            expected = old.replace('bert_etf_loss_weight=1.0', 'bert_etf_loss_weight=0.0')
            expected = expected.replace(
                "default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))",
                'default_hooks = dict(checkpoint=None)')
            expected = expected.replace('custom_hooks = [',
                                        "custom_hooks = [\n    dict(type='ACLDiagnosticHook'),")
            expected = expected.replace('val_evaluator = dict(\n',
                                        'val_evaluator = dict(\n    classwise=True,\n')
            expected = expected.replace("default_hooks = dict(checkpoint=None) \n", "default_hooks = dict(checkpoint=None)\n")
            self.assertEqual(path.read_text(encoding='utf-8'), expected, path.name)

    def test_classwise_ap_uses_actual_thresholds_and_invalid_gt(self):
        precision = np.full((3, 5, 3, 1, 1), -1.)
        precision[:, :, 0, 0, 0] = np.array([.75, .5, .9])[:, None]
        precision[:, :, 1, 0, 0] = 0.
        rows = dio.classwise_ap(precision, [.75, .50, .90], [8, 3, 11])
        self.assertAlmostEqual(rows[0]['AP'], (.75 + .5 + .9) / 3)
        self.assertEqual(rows[0]['AP50'], .5)
        self.assertEqual(rows[0]['AP75'], .75)
        self.assertEqual(rows[1]['AP50'], 0.)
        self.assertIsNone(rows[2]['AP'])
        missing = dio.classwise_ap(precision[:1], [.9], [8, 3, 11])[0]
        self.assertEqual(missing['reasons']['AP50'], 'iou_threshold_unavailable')

    def test_coco_metric_returns_class_ap50_ap75_without_changing_overall_metrics(self):
        precision = np.full((2, 5, 2, 4, 1), .3)
        precision[0, :, 0] = .75
        precision[1, :, 0] = .50
        precision[:, :, 1] = -1
        api = SimpleNamespace(loadRes=lambda _: object(),
                              loadCats=lambda cat: [dict(name=f'class{cat}')])
        coco_eval = SimpleNamespace(params=SimpleNamespace(), eval={'precision': precision},
                                    stats=np.arange(12) / 100, evaluate=lambda: None,
                                    accumulate=lambda: None, summarize=lambda: None)
        metric = SimpleNamespace(
            outfile_prefix='unused', _coco_api=api, cat_ids=[8, 3], img_ids=[101],
            metrics=['bbox'], format_only=False, use_mp_eval=False,
            proposal_nums=[100], iou_thrs=np.array([.75, .5]), metric_items=None,
            results2json=lambda *args: dict(bbox='unused'), classwise=True)
        method = reference.method('mmdet/evaluation/metrics/coco_metric.py', 'CocoMetric',
                                  'compute_metrics', OrderedDict=OrderedDict, np=np,
                                  MMLogger=SimpleNamespace(get_current_instance=lambda: Mock()),
                                  load=lambda _: [], COCOeval=lambda *a: coco_eval,
                                  AsciiTable=lambda _: SimpleNamespace(table='table'),
                                  itertools=itertools, classwise_ap=dio.classwise_ap)
        result = method(metric, [({}, {})])
        self.assertEqual(result['bbox_class_8_AP50'], .5)
        self.assertEqual(result['bbox_class_8_AP75'], .75)
        self.assertTrue(math.isnan(result['bbox_class_3_AP75']))
        metric.classwise = False
        baseline = method(metric, [({}, {})])
        for key, value in baseline.items():
            self.assertEqual(result[key], value)

    def test_detection_topk_provenance_with_ties_multiple_classes_and_no_mapping(self):
        model = SimpleNamespace(test_cfg=dict(max_per_img=5))
        scores, boxes = torch.full((3, 2), .5), torch.rand(3, 4)
        metadata = dict(img_shape=(20, 30), scale_factor=(.5, .5))
        for mapping in ({1: [0], 2: [1]}, None):
            model.test_cfg['max_per_img'] = 5 if mapping else 3
            baseline = PREDICT_SINGLE(model, scores, boxes, mapping, metadata, True)
            records = []
            token = events.activate_diagnostics(lambda event, **data: records.append(data))
            try:
                prediction = PREDICT_SINGLE(model, scores, boxes, mapping, metadata, True)
            finally:
                events.deactivate_diagnostics(token)
            torch.testing.assert_close(prediction.bboxes, baseline.bboxes, rtol=0, atol=0)
            torch.testing.assert_close(prediction.scores, baseline.scores, rtol=0, atol=0)
            payload = records[0]
            expected = (scores.flatten().topk(5).indices if mapping else
                        scores.max(-1).values.topk(3).indices)
            torch.testing.assert_close(payload['flat_index'], expected)
            if mapping:
                torch.testing.assert_close(payload['query_index'], expected // 2)
                torch.testing.assert_close(payload['class_index'], expected % 2)
                self.assertLess(len(torch.unique(payload['query_index'])), 5)
            self.assertEqual(set(vars(prediction)), {'bboxes', 'labels', 'scores'})

    def test_proposal_recipe_matches_actual_model_with_padding(self):
        shapes = torch.tensor([[2, 3], [1, 2]])
        memory = torch.rand(2, 8, 4)
        for padding in (None, torch.tensor([[0, 0, 1, 0, 0, 1, 0, 1],
                                           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)):
            model = SimpleNamespace(memory_trans_fc=nn.Identity(), memory_trans_norm=nn.Identity())
            _, raw = GENERATE_PROPOSALS(model, memory, padding, shapes)
            for i in range(2):
                mask = np.zeros(8, bool) if padding is None else padding[i].numpy()
                offset, hw = 0, []
                for h, w in shapes.tolist():
                    level = mask[offset:offset + h * w].reshape(h, w)
                    hw.append([(~level[:, 0]).sum(), (~level[0]).sum()])
                    offset += h * w
                anchors, valid = dio.reconstruct_proposals(shapes.numpy(), hw, mask)
                np.testing.assert_allclose(anchors, raw[i].sigmoid().numpy(), atol=2e-7)
                np.testing.assert_array_equal(valid, torch.isfinite(raw[i]).all(-1).numpy())
                np.testing.assert_array_equal(mask, dio.unpack_mask(dio.pack_mask(mask), mask.shape))

    def test_full_hook_two_epochs_reader_and_offline_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture(directory)
            hook = DiagnosticHook(reserve_gib=0)
            hook.before_train(runner)
            for epoch in (1, 2):
                runner.epoch = epoch - 1
                hook.before_train_epoch(runner)
                runner.epoch = epoch
                hook.before_val_epoch(runner)
                hook.before_val_iter(runner, 0)
                with torch.no_grad():
                    runner.model.eval()
                    outputs = runner.model.predict(torch.zeros(2, 3, 4, 4), data_samples())
                hook.after_val_iter(runner, 0, outputs=outputs)
                metrics = {'coco/bbox_mAP': .2, 'coco/bbox_class_10_AP': .3,
                           'coco/bbox_class_10_AP50': .5, 'coco/bbox_class_10_AP75': .1}
                hook.after_val_epoch(runner, metrics)
            hook.after_run(runner)
            reader = dio.DiagnosticReader(Path(directory) / 'diagnostics')
            self.assertEqual(reader.verify(), dict(observations=4, epochs=2, status='complete'))
            records = list(reader.observations())
            self.assertEqual(records[0]['text_ref'], records[1]['text_ref'])
            self.assertNotEqual(records[0]['text_ref'], records[2]['text_ref'])
            arrays = reader.load(records[0])
            self.assertEqual(arrays['fe_text'].shape[0], 6)
            self.assertEqual(arrays['decoder_boxes'].shape[0], 6)
            self.assertEqual(arrays['encoder_class_scores'].shape, (5, 6))
            self.assertNotIn('encoder_token_logits', arrays)
            self.assertNotIn('base_proposals', arrays)
            self.assertEqual(arrays['det_boxes'].dtype, np.float32)
            self.assertEqual(arrays['decoder_token_logits'].dtype, np.float16)
            self.assertEqual(arrays['selection_max_token_logit'].dtype, np.float32)
            saved = list(reader.records('metrics.jsonl'))
            self.assertEqual(saved[0]['per_class'][0]['AP50'], .5)
            self.assertIsNone(saved[0]['per_class'][1]['AP50'])
            spec = importlib.util.spec_from_file_location('inspect_acl', ROOT /
                'tools/analysis_tools/inspect_acl_diagnostics.py')
            inspector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inspector)
            result = inspector.analyze(reader)
            self.assertEqual(result[1]['observations'], 2)
            self.assertIn('candidate_gt_coverage_at_0_5', result[1])
            self.assertIn('query_class_confusion', result[1])
            self.assertIn('mean_objectness_proxy', result[1])
            self.assertIn('mean_bert_to_fe_token_cosine', result[1])

    def test_logging_preserves_loss_gradients_updates_predictions_and_rng(self):
        snapshots = []
        for enabled in (False, True):
            model = full_fixture()
            hook = DiagnosticHook()
            hook.model = model
            if enabled:
                hook._install_hooks()
            random.seed(73)
            np.random.seed(73)
            torch.manual_seed(73)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
            model.train()
            losses = model.loss(torch.zeros(2, 3, 4, 4), data_samples())
            sum(losses.values()).backward()
            gradients = [p.grad.detach().clone() if p.grad is not None else None
                         for p in model.parameters()]
            norm = nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
            capture = cap.ACLBatchCapture()
            token = events.activate_diagnostics(capture) if enabled else None
            try:
                model.eval()
                with torch.no_grad():
                    outputs = model.predict(torch.zeros(2, 3, 4, 4), data_samples())
            finally:
                if token is not None:
                    events.deactivate_diagnostics(token)
            if enabled:
                records = capture.finish()
                self.assertEqual(len(records), 2)
                self.assertFalse(any(isinstance(value, torch.Tensor) for row in records
                                     for value in row['arrays'].values()))
            snapshots.append(dict(
                loss={k: v.detach().clone() for k, v in losses.items()}, gradients=gradients,
                parameters=[p.detach().clone() for p in model.parameters()], norm=norm,
                boxes=[o.pred_instances.bboxes for o in outputs],
                scores=[o.pred_instances.scores for o in outputs],
                rng=torch.get_rng_state(), random=random.getstate(),
                numpy=np.random.get_state()[1].tolist()))
            for handle in hook.handles:
                handle.remove()
        reference.RawMeanIntegrationTests().assert_nested_equal(snapshots[0], snapshots[1])

    def test_train_cadence_stage_lrs_loss_window_and_norm(self):
        history = lambda value: SimpleNamespace(current=lambda: value, mean=lambda _: value / 2)
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture(directory)
            hook = DiagnosticHook(reserve_gib=0)
            hook.before_train(runner)
            runner.message_hub.log_scalars = {'train/loss': history(8.), 'train/grad_norm': history(3.)}
            for index in range(3):
                runner.iter = index
                if index == 2:
                    hook.stage_hook._stage2_started = True
                hook.before_train_iter(runner, index)
                hook.after_train_iter(runner, index, outputs={'loss': 8., 'd0.loss_cls': 2.})
            rows = list(dio.DiagnosticReader(Path(directory) / 'diagnostics').records('train.jsonl'))
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0]['epoch'], rows[0]['step'], rows[0]['stage']), (1, 3, 2))
            self.assertEqual(rows[0]['grad_norm'], 3.)
            self.assertEqual(rows[0]['loss_window_mean']['loss'], 4.)
            self.assertEqual(rows[0]['lr'], [.0001])
            for handle in hook.handles:
                handle.remove()

    def test_disk_failures_incomplete_atomic_write_and_no_epoch_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'diagnostics'
            writer = dio.DiagnosticWriter(root, {}, {}, reserve_gib=0)
            with patch.object(dio.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(OSError, 'insufficient disk space'):
                    writer.save_arrays('epoch_001/batch.npz', {'x': np.zeros(2)})
            with patch.object(dio.np, 'savez_compressed', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    writer.save_arrays('epoch_001/batch.npz', {'x': np.zeros(2)})
            self.assertFalse((root / 'epoch_001/batch.npz').exists())
            self.assertFalse((root / 'epoch_001/complete.json').exists())
            self.assertEqual(json.loads((root / 'manifest.json').read_text())['status'], 'running')
            with self.assertRaisesRegex(ValueError, 'partial'):
                dio.DiagnosticReader(root).verify()
            with self.assertRaises(FileExistsError):
                dio.DiagnosticWriter(root, {}, {}, reserve_gib=0)

    def test_capture_incomplete_events_and_nonfinite_json(self):
        capture = cap.ACLBatchCapture()
        with self.assertRaisesRegex(RuntimeError, 'before prompt'):
            capture('head', outputs=())
        capture('prompt', samples=data_samples(), captions=['a', 'b'],
                mappings=[{1: [1]}, {1: [1]}], chunk=0, class_offset=0)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            capture.finish()
        self.assertEqual(json.loads(dio.json_bytes({'x': float('nan'), 'y': float('inf')})),
                         dict(x=None, y=None))

    def test_chunked_prompt_preserves_local_and_global_class_indices(self):
        model = full_fixture().eval()
        model.test_cfg['chunked_size'] = 2
        model.bbox_head.test_cfg['max_per_img'] = 2
        hook = DiagnosticHook()
        hook.model = model
        hook._install_hooks()
        capture = cap.ACLBatchCapture()
        token = events.activate_diagnostics(capture)
        try:
            with torch.no_grad():
                outputs = model.predict(torch.zeros(1, 3, 4, 4), data_samples()[:1])
        finally:
            events.deactivate_diagnostics(token)
            for handle in hook.handles:
                handle.remove()
        rows = capture.finish()
        self.assertEqual([row['class_offset'] for row in rows], [0, 2, 4])
        self.assertEqual([row['chunk'] for row in rows], [0, 1, 2])
        labels = np.concatenate([row['arrays']['det_class_index'] for row in rows])
        np.testing.assert_array_equal(labels, outputs[0].pred_instances.labels.numpy())
        for row in rows:
            arrays = row['arrays']
            np.testing.assert_array_equal(arrays['det_flat_topk_index'],
                                          arrays['det_query_index'] * 2 + arrays['det_class_index_local'])

    def test_variable_prompt_lengths_and_single_class_scores(self):
        model = full_fixture().eval()
        samples = data_samples()
        samples[1].text = tuple(name + ' extra words' for name in reference.NAMES)
        hook = DiagnosticHook()
        hook.model = model
        hook._install_hooks()
        capture = cap.ACLBatchCapture()
        token = events.activate_diagnostics(capture)
        try:
            with torch.no_grad():
                model.predict(torch.zeros(2, 3, 4, 4), samples)
        finally:
            events.deactivate_diagnostics(token)
            for handle in hook.handles:
                handle.remove()
        rows = capture.finish()
        self.assertLess(rows[0]['token_length'], rows[1]['token_length'])
        for row in rows:
            self.assertEqual(row['arrays']['decoder_token_logits'].shape[-1], row['token_length'])
            self.assertEqual(len(row['text']['input_ids']), row['token_length'])
        logits = torch.tensor([[1., -2., 3.]])
        expected = logits[:, [0, 2]].sigmoid().mean(-1)
        actual = cap.class_scores(logits, {1: [0, 2]})
        torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
        np.testing.assert_allclose(cap.numpy_copy(actual, np.float16), actual.numpy(), rtol=1e-3)

    def test_short_final_batch_and_duplicate_ids_keep_distinct_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture(directory)
            hook = DiagnosticHook(reserve_gib=0)
            hook.before_train(runner)
            runner.epoch = 1
            hook.before_val_epoch(runner)
            for batch, sample in enumerate(data_samples()):
                hook.before_val_iter(runner, batch)
                with torch.no_grad():
                    runner.model.eval()
                    outputs = runner.model.predict(torch.zeros(1, 3, 4, 4), [sample])
                hook.after_val_iter(runner, batch, outputs=outputs)
            hook.after_val_epoch(runner, {})
            self.assertEqual(hook.observed_samples, 2)
            self.assertEqual(hook.observed_ids, {101, 102})
            # Repeat the same ID in a batch: compare by original batch position.
            hook.before_val_iter(runner, 2)
            samples = data_samples()
            samples[1].metainfo['img_id'] = 101
            with torch.no_grad():
                outputs = runner.model.predict(torch.zeros(2, 3, 4, 4), samples)
            hook.after_val_iter(runner, 2, outputs=outputs)
            for handle in hook.handles:
                handle.remove()

    def test_real_mmengine_config_inheritance_and_grad_norm(self):
        try:
            from mmengine.config import Config as EngineConfig
            from mmengine.optim import OptimWrapper
        except ImportError:
            self.skipTest('Optional integration requires MMEngine 0.10.7')
        for path in (ROOT / 'configs_cdfsod/final_configs_bs4').glob('*.py'):
            cfg = EngineConfig.fromfile(str(path))
            self.assertIsNone(cfg.default_hooks.checkpoint)
            self.assertTrue(cfg.val_evaluator.classwise)
            self.assertEqual([cfg.model[key] for key in ('bert_etf_loss_weight',
                'bert_orth_loss_weight', 'fe_etf_loss_weight', 'fe_orth_loss_weight')], [0.] * 4)
            self.assertEqual(cfg.train_cfg.val_interval, 1)
            self.assertEqual(cfg.optim_wrapper.clip_grad, dict(max_norm=.1, norm_type=2))
        parameter = nn.Parameter(torch.tensor(2.))
        wrapper = OptimWrapper(torch.optim.SGD([parameter], lr=.1),
                               clip_grad=dict(max_norm=.1, norm_type=2))
        wrapper.update_params(parameter.square())
        self.assertEqual(wrapper.message_hub.get_scalar('train/grad_norm').current(), 4.)
        self.assertAlmostEqual(parameter.item(), 1.99, places=5)

    def test_real_mmengine_hook_lifecycle_and_message_hub(self):
        try:
            from mmengine.config import Config as EngineConfig
            from mmengine.hooks import Hook, CheckpointHook, LoggerHook, RuntimeInfoHook
            from mmengine.logging import MessageHub
            from mmengine.optim import OptimWrapper
            from mmengine.runner.log_processor import LogProcessor
        except ImportError:
            self.skipTest('Optional integration requires MMEngine 0.10.7')
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture(directory)
            runner.cfg = EngineConfig(dict(runner.cfg))
            runner.max_epochs = 1
            runner.message_hub = MessageHub.get_instance('acl_diag_' + Path(directory).name)
            runner.log_processor = LogProcessor(window_size=50)
            runner.optim_wrapper = OptimWrapper(runner.optim_wrapper.optimizer,
                clip_grad=dict(max_norm=.1, norm_type=2))
            hook = load_hook(Hook, CheckpointHook)(reserve_gib=0)
            runner.hooks = [LoggerHook(interval=50), runner.hooks[1], hook]
            hook.before_train(runner)
            for index in range(3):
                runner.iter = index
                hook.before_train_iter(runner, index)
                loss = sum(p.square().mean() for p in runner.model.parameters())
                runner.optim_wrapper.update_params(loss)
                outputs = {'loss': loss.item()}
                RuntimeInfoHook().after_train_iter(runner, index, outputs=outputs)
                hook.after_train_iter(runner, index, outputs=outputs)
            runner.epoch = 1
            hook.before_val_epoch(runner)
            hook.before_val_iter(runner, 0)
            with torch.no_grad():
                runner.model.eval()
                outputs = runner.model.predict(torch.zeros(2, 3, 4, 4), data_samples())
            hook.after_val_iter(runner, 0, outputs=outputs)
            hook.after_val_epoch(runner, {})
            hook.after_run(runner)
            reader = dio.DiagnosticReader(Path(directory) / 'diagnostics')
            self.assertEqual(reader.verify()['status'], 'complete')
            training = list(reader.records('train.jsonl'))
            self.assertGreater(training[0]['grad_norm'], 0.)
            self.assertEqual(training[0]['grad_norm'],
                             runner.message_hub.get_scalar('train/grad_norm').current())

    def test_zero_weights_checkpoint_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture(directory)
            runner.model.fe_etf_loss_weight = 1.
            with self.assertRaisesRegex(ValueError, 'fe_etf_loss_weight'):
                DiagnosticHook().before_train(runner)
            runner.model.fe_etf_loss_weight = 0.
            runner.cfg['resume'] = True
            with self.assertRaisesRegex(ValueError, 'resume'):
                DiagnosticHook().before_run(runner)
            runner.cfg['resume'] = False
            runner.hooks.append(type('AutoLoadBestCheckpointHook', (), {})())
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                DiagnosticHook().before_train(runner)


if __name__ == '__main__':
    unittest.main()
