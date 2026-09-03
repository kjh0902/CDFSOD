import json
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from mmdet.models.detectors.grounding_dino_HED import (
    GroundingDINO_ParallelDecoder_15_DNQuery_rand)


Detector = GroundingDINO_ParallelDecoder_15_DNQuery_rand


def _make_detector_stub(num_classes=2, hidden_size=8, max_text_len=256):
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    detector._cached_eval_support_prototypes = None
    detector.support_class_names = [
        f'class_{idx}' for idx in range(num_classes)
    ]
    detector.decoder = SimpleNamespace(num_layers=0)
    detector.bbox_head = SimpleNamespace(
        cls_branches=[SimpleNamespace(max_text_len=max_text_len)])
    detector.embed_dims = hidden_size
    detector.text_feat_map = nn.Linear(768, hidden_size)
    return detector


def test_eval_blip_prototype_is_detached_and_cached():
    detector = _make_detector_stub().eval()
    calls = []

    def compute(self, device):
        calls.append(device)
        return torch.arange(
            16, dtype=torch.float32, device=device,
            requires_grad=True).reshape(2, 8)

    detector.compute_caption_support_prototypes = MethodType(compute, detector)
    first = detector.build_prototype_text_dict(1, torch.device('cpu'))
    second = detector.build_prototype_text_dict(4, torch.device('cpu'))

    assert len(calls) == 1
    assert detector._cached_eval_support_prototypes.shape == (2, 8)
    assert not detector._cached_eval_support_prototypes.requires_grad
    assert not first['embedded'].requires_grad
    assert first['embedded'].shape == (1, 2, 8)
    assert second['embedded'].shape == (4, 2, 8)
    assert first['text_token_mask'].all()
    torch.testing.assert_close(first['embedded'][0], second['embedded'][0])


def test_training_recomputes_blip_prototypes_and_clears_eval_cache():
    detector = _make_detector_stub().train()
    calls = []

    def compute(self, device):
        calls.append(device)
        return torch.ones(2, 8, requires_grad=True)

    detector.compute_caption_support_prototypes = MethodType(compute, detector)
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    assert len(calls) == 2

    detector._cached_eval_support_prototypes = torch.ones(2, 8)
    detector.train(True)
    assert detector._cached_eval_support_prototypes is None


def test_caption_features_are_averaged_by_class():
    features = torch.tensor([[1., 2.], [3., 4.], [10., 20.]])
    labels = torch.tensor([0, 0, 1])

    prototypes = Detector.aggregate_support_caption_features(
        features, labels, num_classes=2)

    torch.testing.assert_close(
        prototypes, torch.tensor([[2., 3.], [10., 20.]]))


def test_support_blip_and_bert_run_inside_dedicated_autocast():
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    active = {'value': False}
    calls = []

    class _AutocastContext:

        def __enter__(self):
            active['value'] = True

        def __exit__(self, exc_type, exc_value, traceback):
            active['value'] = False

    class _Captioner(nn.Module):

        def forward(self, pixel_values):
            calls.append(('blip', active['value']))
            return {'caption': pixel_values.mean()}

    language_model = nn.Module()
    language_model.support_blip_captioner = _Captioner()
    detector.language_model = language_model

    def encode(self, caption_outputs, labels):
        calls.append(('bert', active['value']))
        return caption_outputs['caption'] + labels.float().mean()

    detector._encode_caption_enriched_class_features = MethodType(
        encode, detector)
    with patch(
            'mmdet.models.detectors.grounding_dino_HED.autocast',
            return_value=_AutocastContext()) as mocked_autocast:
        detector._compute_support_batch_caption_features(
            torch.ones(2, 3, 4, 4), torch.tensor([0, 1]))

    mocked_autocast.assert_called_once_with(enabled=True)
    assert calls == [('blip', True), ('bert', True)]
    assert not active['value']


def test_each_class_maps_to_its_single_caption_prototype():
    detector = _make_detector_stub()
    labels = [torch.tensor([0, 1])]

    positive_map = detector.build_prototype_positive_maps(
        labels, torch.device('cpu'))[0]
    token_map = detector.build_prototype_token_positive_map()

    assert token_map == {1: [0], 2: [1]}
    for row_idx, token_idx in enumerate((0, 1)):
        expected = torch.zeros(256)
        expected[token_idx] = 1
        torch.testing.assert_close(positive_map[row_idx], expected)


class _FakeGroundingTokenizer:

    cls_token_id = 5
    sep_token_id = 4

    def encode(self, text, add_special_tokens=False):
        return {
            'pitted surface': [2, 3],
            'scratch': [8],
            ':': [7],
        }[text]


class _FakeGroundingBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(10, 4)
        self.scale = nn.Parameter(torch.tensor(0.5))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds, attention_mask, token_type_ids,
                output_hidden_states, return_dict):
        masked = inputs_embeds * attention_mask.unsqueeze(-1)
        context = masked.sum(dim=1, keepdim=True)
        return SimpleNamespace(
            last_hidden_state=inputs_embeds * self.scale + context)


def test_caption_embeddings_flow_through_bert_to_class_tokens():
    detector = Detector.__new__(Detector)
    nn.Module.__init__(detector)
    bert = _FakeGroundingBert()
    language_model = nn.Module()
    language_model.tokenizer = _FakeGroundingTokenizer()
    language_model.max_tokens = 20
    language_model.language_backbone = SimpleNamespace(
        body=SimpleNamespace(model=bert))
    detector.language_model = language_model
    detector.support_class_names = ['pitted_surface', 'scratch']
    detector._support_class_token_ids = None
    detector._support_colon_token_ids = None
    detector.blip_shared_vocab_size = 10
    detector.text_feat_map = nn.Linear(4, 2)
    logits = torch.randn(3, 2, 12, requires_grad=True)
    distributions = torch.softmax(logits, dim=-1)
    caption_outputs = {
        'token_distributions': distributions,
        'caption_mask': torch.tensor([
            [True, False],
            [True, True],
            [True, False],
        ]),
    }
    labels = torch.tensor([0, 0, 1])

    object_features = detector._encode_caption_enriched_class_features(
        caption_outputs, labels)
    prototypes = detector.aggregate_support_caption_features(
        object_features, labels, num_classes=2)
    projected = detector.text_feat_map(prototypes)
    projected.square().mean().backward()

    assert object_features.shape == (3, 4)
    assert prototypes.shape == (2, 4)
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0
    assert bert.scale.grad is not None
    assert bert.embeddings.weight.grad is not None
    assert detector.text_feat_map.weight.grad is not None


def test_transformer_receives_one_prototype_per_class():
    detector = _make_detector_stub(num_classes=2)
    prototypes = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    text_dict = detector._prototypes_to_text_dict(prototypes, batch_size=3)

    assert text_dict['embedded'].shape == (3, 2, 8)
    assert text_dict['text_token_mask'].all()
    torch.testing.assert_close(text_dict['embedded'][0], prototypes)


def test_total_prototype_length_cannot_exceed_max_text_len():
    detector = _make_detector_stub(num_classes=4, max_text_len=3)

    with pytest.raises(RuntimeError, match='exceed max_text_len'):
        detector._prototypes_to_text_dict(
            torch.ones(4, 8), batch_size=1)


def _write_support_json(path, bbox=(1, 2, 3, 4)):
    path.write_text(
        json.dumps({
            'images': [{'id': 17, 'file_name': 'a.png'}],
            'categories': [{'id': 5, 'name': 'class_0'}],
            'annotations': [{
                'id': 99,
                'image_id': 17,
                'category_id': 5,
                'bbox': bbox,
            }],
        }), encoding='utf-8')


def test_support_annotation_resolves_coco_ids_and_class_names(tmp_path):
    ann_file = tmp_path / '1_shot.json'
    _write_support_json(ann_file)
    detector = _make_detector_stub(num_classes=1)
    detector.support_entries = None
    detector.support_ann_file = str(ann_file)

    detector.build_support_object_bank()

    assert detector.support_entries == [{
        'file_name': 'a.png',
        'bbox': (1.0, 2.0, 3.0, 4.0),
        'class_idx': 0,
    }]


class _CaptureImageProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.crop_sizes = None

    def preprocess_images(self, images):
        self.crop_sizes = [image.size for image in images]
        return torch.ones(len(images), 3, 4, 4)


def test_support_objects_are_cropped_from_ground_truth_boxes(tmp_path):
    image_root = tmp_path / 'train'
    image_root.mkdir()
    Image.new('RGB', (10, 10)).save(image_root / 'a.png')
    captioner = _CaptureImageProcessor()
    detector = _make_detector_stub(num_classes=1)
    detector.support_entries = [{
        'file_name': 'a.png',
        'bbox': (1.0, 2.0, 3.0, 4.0),
        'class_idx': 0,
    }]
    detector.support_image_root = str(image_root)
    detector._support_pixel_values = None
    detector._support_pixel_labels = None
    detector.language_model = nn.Module()
    detector.language_model.support_blip_captioner = captioner

    detector._prepare_support_image_inputs()

    assert captioner.crop_sizes == [(3, 4)]
    assert detector._support_pixel_values.shape == (1, 3, 4, 4)
    assert detector._support_pixel_labels.tolist() == [0]


@pytest.mark.parametrize('bbox, expected_message', [
    ([0, 0, 2], 'invalid xywh bbox'),
    ([0, 0, 0, 2], 'non-positive bbox size'),
])
def test_support_annotation_validation(tmp_path, bbox, expected_message):
    ann_file = tmp_path / '1_shot.json'
    _write_support_json(ann_file, bbox=bbox)
    detector = _make_detector_stub(num_classes=1)
    detector.support_entries = None
    detector.support_ann_file = str(ann_file)

    with pytest.raises(ValueError, match=expected_message):
        detector.build_support_object_bank()
