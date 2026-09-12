"""CPU sanity checks without MMDetection extensions or downloaded weights.

Run: python -m unittest discover -s tests -p test_qwen_offline_sanity.py -v
The actual detector methods are loaded via AST; only its heavy base class and
registry decorator are removed. Detection integration uses recording doubles.
"""
import ast
import copy
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = 'mmdet/models/detectors/grounding_dino_HED.py'
BASE = '8926970ebff1a549088b0a4c87c272e1a70fe0dd'


def load_detector():
    tree = ast.parse((ROOT / DETECTOR).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef))
    cls.bases = [ast.Name(id='Module', ctx=ast.Load())]
    cls.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    env = dict(torch=torch, nn=nn, Module=nn.Module, re=re, json=json,
               defaultdict=defaultdict, copy=copy)
    exec(compile(ast.fix_missing_locations(module), DETECTOR, 'exec'), env)
    return env[cls.name]


Detector = load_detector()
spec = importlib.util.spec_from_file_location(
    'caption_generator', ROOT / 'tools/generate_instance_captions.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class Tokenizer:
    def batch_encode_plus(self, prompts, max_length, **kwargs):
        self.prompts = prompts
        rows = []
        for prompt in prompts:
            spans = [(m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', prompt)]
            # Simulate WordPiece offsets for a single-word class name.
            if prompt.startswith('beetles:'):
                spans[:1] = [(0, 3), (3, 7)]
            rows.append([(0, 0)] + spans[:max_length - 2] + [(0, 0)])
        size = max(map(len, rows))
        offsets, ids, masks = [], [], []
        for row in rows:
            offsets.append(row + [(0, 0)] * (size - len(row)))
            ids.append(list(range(1, len(row) + 1)) + [0] * (size - len(row)))
            masks.append([1] * len(row) + [0] * (size - len(row)))
        return dict(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks),
                    token_type_ids=torch.zeros(len(rows), size, dtype=torch.long),
                    offset_mapping=torch.tensor(offsets))


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 4)

    def forward(self, input_ids, attention_mask, **kwargs):
        assert attention_mask.ndim == 2  # Whole prompt, no subsentence splitting.
        x = self.embedding(input_ids)
        context = (x * attention_mask[..., None]).sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=x + context)


def recording_encoder():
    """Run the production six-layer loop with lightweight attention doubles."""
    path = ROOT / 'mmdet/models/layers/transformer/grounding_dino_layers_HED.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'GroundingDinoTransformerEncoder')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'forward')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), forward], type_ignores=[])
    env = dict(torch=torch, get_text_sine_pos_embed=lambda x, **kw: x.expand(-1, -1, 3))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)

    class Fusion(nn.Module):
        def forward(self, visual_feature, lang_feature, attention_mask_l, **kw):
            self.token_mask = attention_mask_l.clone()
            self.length = lang_feature.size(1)
            return (visual_feature + lang_feature.mean(1, keepdim=True) * 0.01,
                    lang_feature + visual_feature.mean(1, keepdim=True) * 0.01)

    class TextLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn_cfg = SimpleNamespace(num_heads=1)
            self.attn = nn.MultiheadAttention(3, 1, batch_first=True)

        def forward(self, query, query_pos, attn_mask, **kw):
            self.mask = attn_mask.clone()
            self.output = query + self.attn(query + query_pos, query + query_pos,
                                          query, attn_mask=attn_mask)[0] * 0.1
            return self.output

    class VisualLayer(nn.Module):
        def forward(self, query, **kw):
            return query

    class Encoder(nn.Module):
        forward = env['forward']
        get_encoder_reference_points = staticmethod(lambda *a, **kw: None)

    encoder = Encoder()
    encoder.layers = nn.ModuleList([VisualLayer() for _ in range(6)])
    encoder.text_layers = nn.ModuleList([TextLayer() for _ in range(6)])
    encoder.fusion_layers = nn.ModuleList([Fusion() for _ in range(6)])
    return encoder


class QwenSanity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'captions.json'
        self.entries = {'other': 'dark round shape',
                        'pitted_surface': 'small uneven holes'}

    def model(self, entries=None, names=None, limit=8):
        self.path.write_text(json.dumps(self.entries if entries is None else entries))
        model = Detector.__new__(Detector)
        nn.Module.__init__(model)
        model.use_class_name_token_prototypes = True
        model.use_autocast = False
        model.support_caption_file = str(self.path)
        model.support_class_names = names if names is not None else ['pitted_surface', 'other']
        model.support_prompt_bank = None
        model.decoder = SimpleNamespace(num_layers=1)
        model.bbox_head = SimpleNamespace(cls_branches=[None, SimpleNamespace(max_text_len=limit)])
        model.language_model = nn.Module()
        model.language_model.tokenizer = Tokenizer()
        model.language_model.max_tokens = 32
        model.language_model.pad_to_max = False
        model.language_model.language_backbone = nn.Module()
        model.language_model.language_backbone.body = nn.Module()
        model.language_model.language_backbone.body.model = TinyBert()
        model.text_feat_map = nn.Linear(4, 3)
        model.build_support_prompt_bank()
        return model

    def test_full_prompt_context_and_gradient(self):
        model = self.model()
        self.assertEqual(model.support_prompt_texts[0], 'pitted surface: small uneven holes.')
        self.assertEqual(model.support_prompt_class_token_positions, [[1, 2], [1]])
        encoded = model._encode_support_prompt_features(model._prepare_cached_tokenized('cpu'))
        result = model.build_prototype_text_dict(2, 'cpu')
        valid = model.support_tokenized['attention_mask'].bool()
        torch.testing.assert_close(result['embedded'][0], model.text_feat_map(encoded[valid]))
        self.assertEqual(result['class_token_indices'].tolist(), [1, 2, 10])
        result['embedded'].sum().backward()
        self.assertGreater(model.text_feat_map.weight.grad.abs().sum().item(), 0)
        grad = model.language_model.language_backbone.body.model.embedding.weight.grad
        self.assertGreater(grad[5].abs().sum().item(), 0)  # Description influences class features.
        self.path.unlink()  # JSON is read only once.
        model.build_support_prompt_bank()

    def test_batch_masks_and_no_stale_cache(self):
        model = self.model().eval()
        first = model.build_prototype_text_dict(4, 'cpu')
        second = model.build_prototype_text_dict(1, 'cpu')
        self.assertEqual(first['embedded'].shape, (4, 17, 3))
        self.assertEqual(second['embedded'].shape, (1, 17, 3))
        self.assertTrue(second['text_token_mask'].all())
        expected = torch.block_diag(torch.ones(9, 9, dtype=torch.bool),
                                    torch.ones(8, 8, dtype=torch.bool))
        torch.testing.assert_close(second['masks'][0], expected)
        self.assertEqual(second['position_ids'].tolist(), [list(range(9)) + list(range(8))])
        with torch.no_grad():
            model.text_feat_map.bias.add_(1)
        third = model.build_prototype_text_dict(1, 'cpu')
        torch.testing.assert_close(third['embedded'], second['embedded'] + 1)

    def test_validation(self):
        for entries in [{'other': 'shape'}, [], {'other': ' '},
                        {'unknown': 'shape'}]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.model(entries)
        for names in [[], ['other', 'other']]:
            with self.assertRaises(ValueError):
                self.model(names=names)
        with self.assertRaises(ValueError):
            self.model(limit=1)
        with self.assertRaisesRegex(ValueError, 'subword count'):
            self.model(limit=2)  # Two classes, but three class-name tokens.
        self.assertEqual(self.model(limit=3).build_prototype_text_dict(
            1, 'cpu')['embedded'].shape[1], 17)
        model = self.model()
        with self.assertRaises(RuntimeError):
            model._find_class_name_token_positions(torch.zeros(1, 2, 2), [(0, 3)])
        for label in [-1, 2]:
            with self.assertRaises(ValueError):
                model.build_prototype_positive_maps([torch.tensor([label])], 'cpu')

    def test_six_layers_then_query_selection_and_decoder(self):
        torch.manual_seed(4)
        model = self.model().eval()
        model.encoder = recording_encoder()
        model.num_queries = 2
        model.query_embedding = nn.Embedding(2, 3)
        text = model.build_prototype_text_dict(2, 'cpu')
        text['embedded'].retain_grad()
        inputs = dict(feat=torch.randn(2, 4, 3), feat_mask=torch.zeros(2, 4, dtype=torch.bool),
                      feat_pos=torch.zeros(2, 4, 3), spatial_shapes=torch.tensor([[2, 2]]),
                      level_start_index=torch.tensor([0]), valid_ratios=torch.ones(2, 1, 2))
        model.pre_transformer = lambda *a: (inputs, {})
        model.gen_encoder_output_proposals = lambda memory, *a: (
            memory, memory.new_zeros(2, 4, 4))
        seen = {}

        class Classifier:
            max_text_len = 8
            def __call__(self, memory, memory_text, mask):
                seen['selection'] = memory_text
                seen['selection_mask'] = mask
                scores = memory @ memory_text.transpose(1, 2)
                return torch.cat([scores, scores.new_full((2, 4, 5), -float('inf'))], -1)

        model.bbox_head.cls_branches[1] = Classifier()
        model.bbox_head.reg_branches = [None, lambda x: x.new_zeros(2, 4, 4)]
        def decoder(**kw):
            seen['decoder'] = kw['memory_text']
            self.assertFalse(kw['text_attention_mask'].any())
            return {}
        model.forward_decoder = decoder
        result = model.forward_transformer((inputs['feat'],), text)
        for fusion, layer in zip(model.encoder.fusion_layers, model.encoder.text_layers):
            self.assertEqual(fusion.length, 17)
            self.assertFalse(fusion.token_mask.any())
            torch.testing.assert_close(layer.mask, ~text['masks'])
        expected = model.encoder.text_layers[-1].output[:, [1, 2, 10]]
        for actual in [seen['selection'], seen['decoder'], result['memory_text']]:
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.shape, (2, 3, 3))
        self.assertFalse(torch.allclose(expected[:, 0], expected[:, 1]))
        self.assertEqual(seen['selection_mask'].shape, (2, 3))
        result['memory_text'].square().sum().backward()
        self.assertGreater(text['embedded'].grad[:, 4:7].abs().sum().item(), 0)
        self.assertGreater(model.encoder.text_layers[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.text_feat_map.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.language_model.language_backbone.body.model.embedding.weight.grad.abs().sum().item(), 0)

        # Ordinary text dictionaries keep all tokens at the same boundary.
        ordinary = {k: v for k, v in text.items() if k != 'class_token_indices'}
        self.assertEqual(model.forward_encoder(**inputs, text_dict=ordinary)['memory_text'].shape[1], 17)

    def test_independent_bert_rows_and_truncation(self):
        model = self.model()
        tokenized = model._prepare_cached_tokenized('cpu')
        together = model._encode_support_prompt_features(tokenized)
        for i in range(2):
            alone = model._encode_support_prompt_features({k: v[i:i + 1] for k, v in tokenized.items()})
            torch.testing.assert_close(alone[0], together[i])
        model.support_prompt_bank = None
        model.language_model.max_tokens = 5
        model.build_support_prompt_bank()
        self.assertEqual(model.build_prototype_text_dict(1, 'cpu')['embedded'].shape[1], 10)
        model.support_prompt_bank = None
        model.language_model.max_tokens = 3
        with self.assertRaisesRegex(RuntimeError, 'truncated'):
            model.build_support_prompt_bank()

    def test_single_word_subwords_are_not_pooled(self):
        model = self.model(entries={'beetles': 'shiny wings'}, names=['beetles'])
        text = model.build_prototype_text_dict(1, 'cpu')
        self.assertEqual(text['class_token_indices'].tolist(), [1, 2])
        self.assertEqual(model.build_prototype_token_positive_map(), {1: [0, 1]})
        self.assertFalse(torch.allclose(text['embedded'][:, 1], text['embedded'][:, 2]))
        targets = model.build_prototype_positive_maps([torch.tensor([0])], 'cpu')
        self.assertEqual(targets[0][0].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])

    def test_loss_predict_contract(self):
        model = self.model()
        samples = [SimpleNamespace(text=tuple(model.support_class_names),
                   gt_instances=SimpleNamespace(labels=labels))
                   for labels in [torch.tensor([1, 0]), torch.tensor([], dtype=torch.long)]]
        model.extract_feat = lambda images: (images,)
        calls = []
        def forward(features, text, data):
            calls.append(text)
            return dict(memory_text=text['embedded'].index_select(1, text['class_token_indices']),
                        text_token_mask=text['text_token_mask'].index_select(1, text['class_token_indices']))
        model.forward_transformer = forward
        model.bbox_head.loss = lambda **kw: {'loss_mock': kw['memory_text'].sum()}
        losses = model.loss(torch.zeros(2, 3, 4, 4), samples)
        self.assertIn('loss_mock', losses)
        self.assertEqual(samples[0].gt_instances.positive_maps[:, :3].tolist(), [[0, 0, 1], [1, 1, 0]])
        self.assertEqual(samples[1].gt_instances.positive_maps.shape, (0, 8))
        self.assertEqual(samples[1].gt_instances.text_token_mask.shape, (0, 3))
        class Prediction:
            labels = torch.tensor([1, 0])
            def __len__(self):
                return 2
        model.bbox_head.predict = lambda **kw: [Prediction() for _ in kw['batch_data_samples']]
        output = model.predict(torch.zeros(2, 3, 4, 4), samples)
        self.assertEqual(output[0].token_positive_map, {1: [0, 1], 2: [2]})
        self.assertEqual(output[0].pred_instances.label_names, ['other', 'pitted_surface'])
        self.assertEqual(len(calls), 2)

    def test_crop_and_mock_generation(self):
        root = Path(self.tmp.name)
        Image.new('RGB', (10, 10), 'red').save(root / 'image.png')
        coco = dict(annotations=[dict(id=i, image_id=1, category_id=c, bbox=b)
                    for i, c, b in [(1, 7, [-2, -2, 5, 5]), (2, 7, [2, 2, 4, 4]),
                                    (3, 9, [8, 8, 5, 5])]])
        groups = generator.build_class_groups(coco, {1: {'file_name': 'image.png'}},
                                               {7: 'one', 9: 'two'}, root)
        self.assertEqual([len(g['instances']) for g in groups], [2, 1])
        self.assertEqual(groups[0]['instances'][0]['image'].size, (3, 3))
        self.assertEqual(groups[1]['instances'][0]['image'].size, (2, 2))
        class Inputs(dict):
            @property
            def input_ids(self):
                return self['input_ids']
            def to(self, device):
                return self
        class Processor:
            def apply_chat_template(self, conversations, **kwargs):
                self.conversations = conversations
                return Inputs(input_ids=torch.ones(len(conversations), 2, dtype=torch.long))
            def batch_decode(self, ids, **kwargs):
                return ['rounded red surface'] * len(ids)
        processor = Processor()
        model = SimpleNamespace(generate=lambda **kw: torch.ones(2, 3, dtype=torch.long))
        captions = []
        generator.flush_batch(groups, processor, model, 'cpu', 10, captions)
        self.assertEqual(groups, [])
        self.assertEqual(captions[0]['ann_ids'], [1, 2])
        self.assertEqual(len(processor.conversations[0][0]['content']), 3)
        self.assertEqual(len(json.loads(json.dumps({'captions': captions}))['captions']), 2)
        # Exercise the CLI entrypoint and real JSON writer with model loading mocked.
        coco['images'] = [dict(id=1, file_name='image.png')]
        coco['categories'] = [dict(id=7, name='one'), dict(id=9, name='two')]
        (root / 'support.json').write_text(json.dumps(coco))
        args = SimpleNamespace(dataset_root=str(root), ann_file='support.json',
                               img_prefix='.', output='generated.json', device='cpu',
                               batch_size=2, model_name='mock-qwen', max_new_tokens=10)
        model.to = lambda device: model
        model.eval = lambda: model
        fake_transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda name: processor),
            Qwen3VLForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda *a, **kw: model))
        with patch.dict(sys.modules, {'transformers': fake_transformers}), \
                patch.object(generator, 'parse_args', return_value=args):
            generator.main()
        output = json.loads((root / 'generated.json').read_text())
        self.assertEqual(output['captions'], captions)
        self.assertEqual(output['ann_file'], 'support.json')
        self.assertEqual(output['model_name'], 'mock-qwen')
        with patch.object(processor, 'batch_decode', return_value=[' ']):
            with self.assertRaises(ValueError):
                generator.flush_batch([dict(category_name='one', instances=[])],
                                      processor, model, 'cpu', 10, [])
        for bbox in [[0, 0, -1, 3], [20, 20, 2, 2]]:
            self.assertIsNone(generator.crop_bbox(Image.new('RGB', (10, 10)), bbox))
        with self.assertRaises(ValueError):
            generator.crop_bbox(Image.new('RGB', (10, 10)), [0, 0, float('nan'), 1])

    def test_acl_preservation_and_configs(self):
        def git_file(path):
            return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], cwd=ROOT).decode()
        before = ast.parse(git_file(DETECTOR))
        after = ast.parse((ROOT / DETECTOR).read_text(encoding='utf-8'))
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        original, updated = methods(before), methods(after)
        for name in original:
            if name not in ['__init__', 'loss', 'predict', 'forward_encoder']:
                self.assertEqual(ast.dump(original[name]), ast.dump(updated[name]), name)
        encoder = copy.deepcopy(updated['forward_encoder'])
        encoder.body = [n for n in encoder.body if not (
            isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Constant)
            and n.test.left.value == 'class_token_indices')]
        self.assertEqual(ast.dump(encoder), ast.dump(original['forward_encoder']))
        for name in ['loss', 'predict']:
            node = copy.deepcopy(updated[name])
            node.body = [n for n in node.body if not (
                isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
                and n.test.attr == 'use_class_name_token_prototypes')]
            self.assertEqual(ast.dump(node), ast.dump(original[name]), name)
        for path in ['mmdet/engine/hooks/stage_lr_hook.py',
                     'mmdet/models/layers/transformer/grounding_dino_layers_HED.py',
                     'mmdet/models/dense_heads/grounding_dino_head_HED.py']:
            self.assertEqual((ROOT / path).read_text(encoding='utf-8'), git_file(path))
        configs = list((ROOT / 'configs_cdfsod/final_configs_bs4').glob('*shot.py'))
        self.assertEqual(len(configs), 18)
        for path in configs:
            text = path.read_text(encoding='utf-8')
            compile(text, str(path), 'exec')
            shot = re.search(r'_(1|5|10)shot', path.name)[1]
            self.assertIn(f'/annotations/{shot}_shot_captions.json', text)
            tree = ast.parse(text)
            model = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
            model.keywords = [k for k in model.keywords if k.arg not in {
                'support_caption_file', 'support_class_names', 'use_class_name_token_prototypes'}]
            self.assertEqual(ast.dump(tree), ast.dump(ast.parse(git_file(path.relative_to(ROOT).as_posix()))))


if __name__ == '__main__':
    unittest.main()
