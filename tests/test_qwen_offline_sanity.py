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


class QwenSanity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'captions.json'
        self.entries = [dict(category_name='other', caption='dark round shape'),
                        dict(category_name='pitted_surface', caption='small uneven holes')]

    def model(self, entries=None, names=None, limit=8):
        self.path.write_text(json.dumps({'captions': self.entries if entries is None else entries}))
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

    def test_pooling_context_and_gradient(self):
        model = self.model()
        self.assertEqual(model.support_prompt_texts[0], 'pitted surface: small uneven holes.')
        self.assertEqual(model.support_prompt_class_token_positions, [[1, 2], [1]])
        encoded = model._encode_support_prompt_features(model._prepare_cached_tokenized('cpu'))
        prototypes = model.compute_class_text_prototypes()
        torch.testing.assert_close(prototypes[0], encoded[0, [1, 2]].mean(0))
        self.assertFalse(torch.allclose(prototypes[0], encoded[0].mean(0)))
        result = model.build_prototype_text_dict(2, 'cpu')
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
        self.assertEqual(first['embedded'].shape, (4, 2, 3))
        self.assertEqual(second['embedded'].shape, (1, 2, 3))
        self.assertTrue(second['text_token_mask'].all())
        torch.testing.assert_close(second['masks'][0], torch.eye(2, dtype=torch.bool))
        self.assertEqual(second['position_ids'].tolist(), [[0, 1]])
        with torch.no_grad():
            model.text_feat_map.bias.add_(1)
        third = model.build_prototype_text_dict(1, 'cpu')
        torch.testing.assert_close(third['embedded'], second['embedded'] + 1)

    def test_validation(self):
        for entries in [self.entries[:1], self.entries + self.entries[:1],
                        [dict(category_name='other', caption=' ')],
                        [dict(category_name='unknown', caption='shape')]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.model(entries)
        for names in [[], ['other', 'other']]:
            with self.assertRaises(ValueError):
                self.model(names=names)
        with self.assertRaises(ValueError):
            self.model(limit=1)
        model = self.model()
        with self.assertRaises(RuntimeError):
            model._find_class_name_token_positions(torch.zeros(1, 2, 2), [(0, 3)])
        for label in [-1, 2]:
            with self.assertRaises(ValueError):
                model.build_prototype_positive_maps([torch.tensor([label])], 'cpu')

    def test_loss_predict_contract(self):
        model = self.model()
        samples = [SimpleNamespace(text=tuple(model.support_class_names),
                   gt_instances=SimpleNamespace(labels=labels))
                   for labels in [torch.tensor([1, 0]), torch.tensor([], dtype=torch.long)]]
        model.extract_feat = lambda images: (images,)
        calls = []
        def forward(features, text, data):
            calls.append(text)
            return dict(memory_text=text['embedded'], text_token_mask=text['text_token_mask'])
        model.forward_transformer = forward
        model.bbox_head.loss = lambda **kw: {'loss_mock': kw['memory_text'].sum()}
        losses = model.loss(torch.zeros(2, 3, 4, 4), samples)
        self.assertIn('loss_mock', losses)
        self.assertEqual(samples[0].gt_instances.positive_maps[:, :2].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(samples[1].gt_instances.positive_maps.shape, (0, 8))
        self.assertEqual(samples[1].gt_instances.text_token_mask.shape, (0, 2))
        class Prediction:
            labels = torch.tensor([1, 0])
            def __len__(self):
                return 2
        model.bbox_head.predict = lambda **kw: [Prediction() for _ in kw['batch_data_samples']]
        output = model.predict(torch.zeros(2, 3, 4, 4), samples)
        self.assertEqual(output[0].token_positive_map, {1: [0], 2: [1]})
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
            if name not in ['__init__', 'loss', 'predict']:
                self.assertEqual(ast.dump(original[name]), ast.dump(updated[name]), name)
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
