"""Exercise production detector/BERT/FE methods without MMCV or downloaded weights."""
import ast
import copy
import re
import subprocess
import unittest
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from test_background_anchored_etf_loss import geometry


ROOT = Path(__file__).resolve().parents[1]
BASE = '4118d18c3eb59ad5eed51bc228aff1492cd624d5'
DETECTOR = 'mmdet/models/detectors/grounding_dino_HED.py'
KEY = 'loss_bg_anchored_nearest_deformed_etf'
NAMES = ('seaurchin', 'cracked surface', 'scallop')
DIM = 8


def execute(nodes, **env):
    env.update(torch=torch, nn=nn, copy=copy, re=re, Integral=Integral)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<production>', 'exec'), env)
    return env


def production_method(path, cls, name, **env):
    node = next(n for n in ast.parse((ROOT / path).read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return execute([method], **env)[name]


glip = ast.parse((ROOT / 'mmdet/models/detectors/glip.py').read_text(encoding='utf-8'))
map_env = execute([n for n in glip.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('create_positive_map', 'create_positive_map_label_to_token')])
tree = ast.parse((ROOT / DETECTOR).read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
cls = next(n for n in nodes if isinstance(n, ast.ClassDef))
cls.bases = [ast.Name(id='Module', ctx=ast.Load())]
cls.decorator_list = []
Detector = execute(nodes, Module=nn.Module,
    background_anchored_etf_loss=geometry.background_anchored_etf_loss,
    validate_nonnegative_scalar=geometry.validate_nonnegative_scalar,
    create_positive_map=map_env['create_positive_map'],
    create_positive_map_label_to_token=map_env['create_positive_map_label_to_token'])[cls.name]


class Batch(dict):
    __getattr__ = dict.__getitem__

    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})

    def char_to_token(self, character):
        for index, (start, end) in enumerate(self['offset_mapping'][0].tolist()):
            if start <= character < end:
                return index
        return None


class Tokenizer:
    """Fast-tokenizer API double, including multi-subword names and offsets."""
    def __init__(self):
        self.vocab = {'[CLS]': 1, '[SEP]': 2, '.': 3, '?': 4, 'background': 5}

    def __call__(self, captions, padding='longest', max_length=256,
                 truncation=False, **kwargs):
        rows, offsets = [], []
        for caption in captions:
            words, spans = [], []
            for match in re.finditer(r'\w+|[^\w\s]', caption):
                word = match.group().lower()
                a, b = match.span()
                if word == 'seaurchin':
                    words.extend(['sea', '##urchin'])
                    spans.extend([(a, a + 3), (a + 3, b)])
                else:
                    words.append(word)
                    spans.append((a, b))
            for word in words:
                self.vocab.setdefault(word, len(self.vocab) + 1)
            if truncation:
                words, spans = words[:max_length - 2], spans[:max_length - 2]
            rows.append([1] + [self.vocab[w] for w in words] + [2])
            offsets.append([(0, 0)] + spans + [(0, 0)])
        length = max(map(len, rows))
        if padding == 'max_length':
            length = max(length, max_length)
        return Batch(
            input_ids=torch.tensor([r + [0] * (length - len(r)) for r in rows]),
            attention_mask=torch.tensor([[1] * len(r) + [0] * (length - len(r)) for r in rows]),
            token_type_ids=torch.zeros(len(rows), length, dtype=torch.long),
            offset_mapping=torch.tensor([r + [(0, 0)] * (length - len(r)) for r in offsets]))

    batch_encode_plus = __call__


bert_path = 'mmdet/models/language_models/bert.py'
bert_tree = ast.parse((ROOT / bert_path).read_text(encoding='utf-8'))
generate_masks = execute([next(n for n in bert_tree.body if isinstance(n, ast.FunctionDef)
    and n.name == 'generate_masks_with_special_tokens_and_transfer_map')])[
        'generate_masks_with_special_tokens_and_transfer_map']


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1024, DIM)

    def forward(self, inputs):
        x = self.embedding(inputs['input_ids'])
        self.input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        x = x + (mask.to(x.dtype) @ x) / mask.sum(-1, keepdim=True).clamp_min(1)
        return dict(embedded=x, masks=mask)


class TinyLanguage(nn.Module):
    forward = production_method(bert_path, 'BertModel', 'forward',
        generate_masks_with_special_tokens_and_transfer_map=generate_masks)

    def __init__(self, pad_to_max=False):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.language_backbone = TinyBackbone()
        self.max_tokens = 256
        self.pad_to_max = pad_to_max
        self.use_sub_sentence_represent = True
        self.special_tokens = [1, 2, 3, 4]


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_text = nn.Linear(DIM, DIM)
        self.to_visual = nn.Linear(DIM, DIM)

    def forward(self, visual_feature, lang_feature, attention_mask_l, **kwargs):
        self.text_length = lang_feature.size(1)
        self.mask = attention_mask_l
        valid = (~attention_mask_l).unsqueeze(-1)
        text_mean = (lang_feature * valid).sum(1) / valid.sum(1)
        return (visual_feature + self.to_visual(text_mean).unsqueeze(1),
                lang_feature + self.to_text(visual_feature.mean(1)).unsqueeze(1))


class TextLayer(nn.Linear):
    def __init__(self):
        super().__init__(DIM, DIM)
        self.self_attn_cfg = SimpleNamespace(num_heads=1)

    def forward(self, query, **kwargs):
        return torch.tanh(super().forward(query))


class VisualLayer(nn.Linear):
    def __init__(self):
        super().__init__(DIM, DIM)

    def forward(self, query, **kwargs):
        return super().forward(query)


fe_forward = production_method(
    'mmdet/models/layers/transformer/grounding_dino_layers_HED.py',
    'GroundingDinoTransformerEncoder', 'forward',
    get_text_sine_pos_embed=lambda x, **kwargs: x.expand(-1, -1, DIM))


class TinyFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([VisualLayer() for _ in range(6)])
        self.text_layers = nn.ModuleList([TextLayer() for _ in range(6)])
        self.fusion_layers = nn.ModuleList([Fusion() for _ in range(6)])

    def get_encoder_reference_points(self, shapes, ratios, device):
        return torch.zeros(1, 5, 1, 2, device=device)

    def forward(self, **kwargs):
        visual, text = fe_forward(self, **kwargs)
        self.final_text = text
        text.retain_grad() if text.requires_grad else None
        return visual, text


class Contrastive(nn.Module):
    max_text_len = 256

    def forward(self, visual, text, mask):
        self.seen = (text, mask)
        logits = visual @ text.transpose(-2, -1)
        logits = logits.masked_fill(~mask[:, None], float('-inf'))
        return nn.functional.pad(logits, (0, 256 - logits.size(-1)), value=float('-inf'))


class Instances(SimpleNamespace):
    def __len__(self):
        return self.labels.numel()

    def cat(self, instances):
        return Instances(labels=torch.cat([item.labels for item in instances]))


class Head(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.num_classes = classes
        self.cls_branches = nn.ModuleList([Contrastive() for _ in range(7)])
        self.reg_branches = nn.ModuleList([nn.Linear(DIM, 4) for _ in range(7)])

    def loss(self, **kwargs):
        self.seen = kwargs
        assert 'bg_anchored_etf_loss' not in kwargs
        return dict(loss_detection=kwargs['memory_text'].sum() * 0)

    def predict(self, batch_data_samples, **kwargs):
        self.seen = kwargs
        assert 'bg_anchored_etf_loss' not in kwargs
        return [Instances(labels=torch.arange(len(sample.token_positive_map)))
                for sample in batch_data_samples]


class Sample(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def samples(names=NAMES):
    return [Sample(text=names, custom_entities=True,
                   gt_instances=SimpleNamespace(labels=torch.tensor([0, 0, len(names) - 1]))),
            Sample(text=names, custom_entities=True,
                   gt_instances=SimpleNamespace(labels=torch.empty(0, dtype=torch.long)))]


def fixture(enabled=True, weight=0.1, classes=3, pad_to_max=False):
    torch.manual_seed(42)
    model = Detector(language_model={}, use_background_anchor=enabled,
                     bg_anchored_etf_loss_weight=weight)
    model.language_model = TinyLanguage(pad_to_max)
    model.text_feat_map = nn.Linear(DIM, DIM)
    model.encoder = TinyFE()
    model.backbone = nn.Linear(DIM, DIM)
    model.decoder = nn.Module()
    model.decoder.num_layers = 6
    model.bbox_head = Head(classes)
    model.query_embedding = nn.Embedding(2, DIM)
    model.num_queries = 2
    model.test_cfg = {}
    model.extract_feat = lambda x: (model.backbone(x),)
    model.pre_transformer = lambda feats, data: (dict(
        feat=feats[0], feat_mask=None, feat_pos=torch.zeros_like(feats[0]),
        spatial_shapes=torch.tensor([[1, 5]]), level_start_index=torch.tensor([0]),
        valid_ratios=torch.ones(2, 1, 2)), dict(memory_mask=None,
            spatial_shapes=torch.tensor([[1, 5]]), level_start_index=torch.tensor([0]),
            valid_ratios=torch.ones(2, 1, 2)))
    model.gen_encoder_output_proposals = lambda memory, *args: (
        memory, torch.zeros(*memory.shape[:2], 4))
    model.dn_query_generator = Mock(return_value=(
        torch.zeros(2, 0, DIM), torch.zeros(2, 0, 4), None,
        dict(num_denoising_queries=0, num_denoising_groups=0)))
    def decode(**kwargs):
        model.decoder_seen = kwargs
        return dict(hidden_states=kwargs['query'].unsqueeze(0),
                    references=kwargs['reference_points'].unsqueeze(0))
    model.forward_decoder = decode
    return model


def run(model, data):
    return model.loss(torch.randn(len(data), 5, DIM), data)


class BackgroundETFIntegrationTests(unittest.TestCase):
    def assert_gradient(self, parameter):
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_full_fe_but_foreground_only_query_selection_decoder_head(self):
        for mode in ['shared', 'different', 'explicit_list', 'explicit_dict', 'padding']:
            with self.subTest(mode=mode):
                model = fixture(pad_to_max=mode == 'padding')
                data = samples()
                if mode == 'different':
                    data[1].text = ('large seaurchin', 'cracked surface', 'scallop')
                elif mode.startswith('explicit'):
                    _, caption, spans, _ = model.get_tokens_and_prompts(NAMES, True)
                    for sample in data:
                        sample.text = caption
                        sample.tokens_positive = dict(enumerate(spans)) if mode.endswith('dict') else spans
                before = copy.deepcopy(data)
                with patch.dict(Detector.forward_transformer.__globals__,
                    background_anchored_etf_loss=Mock(wraps=geometry.background_anchored_etf_loss)):
                    solve = Detector.forward_transformer.__globals__['background_anchored_etf_loss']
                    losses = run(model, data)
                self.assertEqual(solve.call_count, 1)
                full, maps, bg_indices, mask, labels = solve.call_args.args
                self.assertIs(full, model.encoder.final_text)
                self.assertIsNot(full, model.bbox_head.seen['memory_text'])
                for fusion in model.encoder.fusion_layers:
                    self.assertEqual(fusion.text_length, full.size(1))
                    self.assertFalse(fusion.mask[0, bg_indices[0]].any())
                originals = [model.get_tokens_and_prompts(sample.text, True)[1]
                             if not isinstance(sample.text, str) else sample.text for sample in before]
                _, metadata = model._prepare_background_text(originals)
                detection = model.bbox_head.seen['memory_text']
                torch.testing.assert_close(model.bbox_head.seen['text_token_mask'],
                    metadata['detection_text_token_mask'])
                for b, indices in enumerate(metadata['detection_token_indices']):
                    torch.testing.assert_close(detection[b, :len(indices)], full[b, indices])
                    self.assertTrue(set(indices).isdisjoint(bg_indices[b]))
                self.assertIs(model.decoder_seen['memory_text'], detection)
                self.assertIs(model.bbox_head.cls_branches[-1].seen[0], detection)
                self.assertTrue(all(set(mapping) == {1, 2, 3} for mapping in maps))
                torch.testing.assert_close(losses[KEY], 0.1 * geometry.background_anchored_etf_loss(
                    full, maps, bg_indices, mask, labels))
                # The original foreground positive map and GT mask match the no-anchor path.
                baseline_data = copy.deepcopy(before)
                run(fixture(enabled=False, pad_to_max=mode == 'padding'), baseline_data)
                for current, baseline in zip(data, baseline_data):
                    torch.testing.assert_close(current.gt_instances.positive_maps,
                                               baseline.gt_instances.positive_maps)
                    torch.testing.assert_close(current.gt_instances.text_token_mask,
                                               baseline.gt_instances.text_token_mask)
                losses[KEY].backward()
                self.assertGreater(full.grad[:, bg_indices[0]].abs().sum().item(), 0)
                self.assertEqual(model.dn_query_generator.call_count, 1)

    def test_weight_zero_and_disabled_flag_skip_svd_with_distinct_text_paths(self):
        for enabled, weight in [(True, 0), (False, 0.1)]:
            model = fixture(enabled, weight)
            with patch.object(torch.linalg, 'svd', side_effect=AssertionError('SVD')):
                losses = run(model, samples())
            self.assertNotIn(KEY, losses)
            ids = model.language_model.language_backbone.input_ids
            self.assertEqual(bool((ids == 5).any()), enabled)

    def test_auxiliary_gradient_and_acl_optimizer_stages(self):
        model = fixture()
        node = next(n for n in ast.parse((ROOT / 'mmdet/engine/hooks/stage_lr_hook.py')
                    .read_text(encoding='utf-8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'BBoxHeadFirstHook6')
        node.decorator_list = []
        Hook = execute([node], Hook=object, is_model_wrapper=lambda x: False)['BBoxHeadFirstHook6']
        hook = Hook(adjust_scheduler_patience=True, patience_frozen=3, patience_unfrozen=8)
        optimizer = torch.optim.AdamW([dict(params=[p], lr=0.001)
                                      for p in model.parameters()], weight_decay=0)
        runner = SimpleNamespace(model=model, optim_wrapper=SimpleNamespace(optimizer=optimizer),
            logger=Mock(), epoch=0, param_schedulers=[SimpleNamespace(patience=5)])
        hook.before_train(runner)
        self.assertEqual(runner.param_schedulers[0].patience, 3)
        for stage in [1, 2]:
            optimizer.zero_grad()
            if stage == 2:
                for index in hook._lang_model_groups:
                    optimizer.param_groups[index]['lr'] *= 0.5
                hook.before_train_epoch(runner)
                self.assertTrue(hook._stage2_started)
                self.assertEqual(runner.param_schedulers[0].patience, 8)
            before = model.encoder.fusion_layers[0].to_text.weight.detach().clone()
            losses = run(model, samples())
            losses[KEY].backward()
            self.assert_gradient(model.language_model.language_backbone.embedding.weight)
            self.assert_gradient(model.text_feat_map.weight)
            self.assert_gradient(model.backbone.weight)
            self.assert_gradient(model.encoder.fusion_layers[0].to_text.weight)
            optimizer.step()
            same = torch.equal(before, model.encoder.fusion_layers[0].to_text.weight)
            self.assertEqual(same, stage == 1)
        disabled = fixture(enabled=False)
        self.assertEqual(model.state_dict().keys(), disabled.state_dict().keys())

    def test_inference_labels_and_maps_stay_foreground_without_gt_or_svd(self):
        for chunked in (False, True):
            model = fixture().eval()
            if chunked:
                model.test_cfg['chunked_size'] = 2
            data = [Sample(text=NAMES, custom_entities=True)]
            # The fixture's spatial tensors work with either batch size.
            with torch.no_grad(), patch.object(torch.linalg, 'svd', side_effect=AssertionError('SVD')):
                result = model.predict(torch.randn(1, 5, DIM), data)
            self.assertEqual(result[0].pred_instances.labels.tolist(), [0, 1, 2])
            self.assertEqual(result[0].pred_instances.label_names, list(NAMES))
            self.assertFalse(model.dn_query_generator.called)
            ids = model.language_model.language_backbone.input_ids
            self.assertTrue((ids == 5).any())
            self.assertLess(model.decoder_seen['memory_text'].size(1), ids.size(1))

    def test_single_class_fish_keeps_background_and_zero_auxiliary(self):
        model = fixture(classes=1)
        with patch.object(torch.linalg, 'svd', side_effect=AssertionError('SVD')):
            losses = run(model, samples(('fish',)))
        self.assertEqual(losses[KEY].item(), 0)
        self.assertTrue(losses[KEY].requires_grad)
        losses[KEY].backward()
        self.assertIsNotNone(model.text_feat_map.weight.grad)
        self.assertTrue((model.language_model.language_backbone.input_ids == 5).any())

    def test_invalid_options_gt_and_truncated_or_missing_spans(self):
        for option in ['bg_anchored_etf_alpha', 'bg_anchored_etf_loss_weight']:
            for value in [-1, float('nan'), float('inf'), True, [0.5]]:
                with self.assertRaises(ValueError):
                    Detector(language_model={}, **{option: value})
        model = fixture()
        model.language_model.max_tokens = 8
        with self.assertRaisesRegex(ValueError, 'untruncated'):
            run(model, samples())
        model = fixture()
        with self.assertRaises(ValueError):
            model._encode_text(['seaurchin. cracked surface. scallop. '],
                               [[[[0, 9]], [], [[28, 35]]]])
        invalid = samples()
        invalid[0].gt_instances.labels = torch.tensor([3])
        with self.assertRaisesRegex(ValueError, 'GT label'):
            run(model, invalid)
        for spans in [[[[0, 9]]], {1: [[0, 9]], 2: [[11, 26]]}]:
            invalid = samples()
            for sample in invalid:
                sample.text = 'seaurchin. cracked surface. scallop. '
                sample.tokens_positive = spans
            with self.assertRaisesRegex(ValueError, 'all foreground'):
                run(model, invalid)

    def test_configs_and_training_infrastructure_preserved(self):
        def baseline(path):
            return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode('utf-8')
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*shot.py'))
        self.assertEqual(len(configs), 18)
        old_options = {'use_class_name_token_prototypes', 'support_class_names', 'support_caption_file'}
        new_options = {'use_background_anchor', 'bg_anchored_etf_alpha', 'bg_anchored_etf_loss_weight'}
        for path in configs:
            after = ast.parse(path.read_text(encoding='utf-8'))
            before = ast.parse(baseline(path.relative_to(ROOT).as_posix()))
            def model_node(tree):
                return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
            after_model = model_node(after)
            options = {k.arg: ast.literal_eval(k.value) for k in after_model.keywords
                       if k.arg in new_options}
            self.assertEqual(options, dict(use_background_anchor=True,
                bg_anchored_etf_alpha=0.5, bg_anchored_etf_loss_weight=0.1))
            after_model.keywords = [k for k in after_model.keywords if k.arg not in new_options]
            before_model = model_node(before)
            before_model.keywords = [k for k in before_model.keywords if k.arg not in old_options]
            self.assertEqual(ast.dump(after), ast.dump(before), path.name)
        for path in ['mmdet/engine/hooks/stage_lr_hook.py', 'tools/train.py', 'tools/test.py',
                     'configs_cdfsod/grounding_dino_swin-b_pretrain_all.py',
                     'configs_cdfsod/grounding_dino_swin-t_pretrain_obj365.py']:
            self.assertEqual((ROOT / path).read_text(encoding='utf-8'), baseline(path))
        self.assertFalse((ROOT / 'tools/generate_instance_captions.py').exists())
        self.assertNotIn('support_caption_file', (ROOT / DETECTOR).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
