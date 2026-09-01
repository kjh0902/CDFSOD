import json
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mmdet.models.detectors.grounding_dino import GroundingDINO


def _make_detector_stub(num_classes=2, hidden_size=8, max_text_len=256):
    detector = GroundingDINO.__new__(GroundingDINO)
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


def _support_feature_inputs():
    features = torch.tensor([
        [1., 2.],
        [3., 4.],
        [10., 20.],
    ])
    labels = torch.tensor([0, 0, 1])
    return features, labels


def _make_support_pipeline_detector_stub():
    detector = GroundingDINO.__new__(GroundingDINO)
    nn.Module.__init__(detector)
    detector.support_image_batch_size = 2
    detector.support_class_names = ['class_0', 'class_1']
    detector._support_pixel_values = torch.stack([
        torch.ones(3, 2, 2),
        torch.full((3, 2, 2), 2.0),
    ])
    detector._support_pixel_labels = torch.tensor([0, 1])
    detector.support_scale = nn.Parameter(torch.tensor(1.0))

    def prepare(self):
        return None

    def compute_batch(self, pixel_values, labels):
        values = pixel_values.mean(dim=(1, 2, 3)) * self.support_scale
        return values[:, None].expand(-1, 2)

    detector._prepare_support_image_inputs = MethodType(prepare, detector)
    detector._compute_support_batch_caption_features = MethodType(
        compute_batch, detector)
    return detector


def test_direct_training_support_pipeline_preserves_parameter_gradients():
    detector = _make_support_pipeline_detector_stub()
    detector.training = True

    features = detector.compute_support_caption_features(torch.device('cpu'))
    features.square().mean().backward()

    assert detector.support_scale.grad is not None
    assert detector.support_scale.grad.abs().item() > 0


def test_caption_features_are_averaged_by_class():
    inputs = _support_feature_inputs()

    prototypes = GroundingDINO.aggregate_support_caption_features(
        *inputs, num_classes=2)

    assert prototypes.shape == (2, 2)
    torch.testing.assert_close(
        prototypes, torch.tensor([[2., 3.], [10., 20.]]))


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
        tokens = {
            'pitted surface': [2, 3],
            'scratch': [8],
            ':': [7],
        }
        return tokens[text]


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
    detector = GroundingDINO.__new__(GroundingDINO)
    nn.Module.__init__(detector)
    bert = _FakeGroundingBert()
    detector.language_model = SimpleNamespace(
        tokenizer=_FakeGroundingTokenizer(),
        max_tokens=20,
        language_backbone=SimpleNamespace(
            body=SimpleNamespace(model=bert)))
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
    embedding_weight = bert.embeddings.weight
    first_caption = distributions[0, 0, :10] @ embedding_weight
    first_context = embedding_weight[
        torch.tensor([5, 2, 3, 7, 4])].sum(dim=0) + first_caption
    expected_first = embedding_weight[
        torch.tensor([2, 3])].mean(dim=0) * bert.scale + first_context
    torch.testing.assert_close(object_features[0], expected_first)
    prototypes = detector.aggregate_support_caption_features(
        object_features, labels, num_classes=2)
    projected = detector.text_feat_map(prototypes)
    projected.square().mean().backward()

    assert object_features.shape == (3, 4)
    assert prototypes.shape == (2, 4)
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0
    assert bert.scale.grad is not None
    assert bert.embeddings.weight.grad is not None
    assert detector.text_feat_map.weight.grad is not None


def test_transformer_receives_one_prototype_per_class():
    detector = _make_detector_stub(num_classes=2)
    prototypes = torch.arange(16, dtype=torch.float32).reshape(2, 8)

    text_dict = detector._prototypes_to_text_dict(
        prototypes, batch_size=3)

    assert text_dict['embedded'].shape == (3, 2, 8)
    assert text_dict['text_token_mask'].all()
    torch.testing.assert_close(text_dict['embedded'][0], prototypes)


def test_total_prototype_length_cannot_exceed_max_text_len():
    detector = _make_detector_stub(num_classes=4, max_text_len=3)

    with pytest.raises(RuntimeError, match='exceed max_text_len'):
        detector._prototypes_to_text_dict(
            torch.ones(4, 8), batch_size=1)


@pytest.mark.parametrize('bbox, expected_message', [
    ([0, 0, 2], 'invalid xywh bbox'),
    ([0, 0, 0, 2], 'non-positive bbox size'),
])
def test_support_annotation_validation(tmp_path, bbox, expected_message):
    ann_file = tmp_path / '1_shot.json'
    ann_file.write_text(
        json.dumps({
            'images': [{
                'id': 17,
                'file_name': 'a.jpg'
            }],
            'categories': [{
                'id': 5,
                'name': 'class_0'
            }],
            'annotations': [{
                'id': 99,
                'image_id': 17,
                'category_id': 5,
                'bbox': bbox,
            }],
        }), encoding='utf-8')
    detector = _make_detector_stub(num_classes=1)
    detector.support_entries = None
    detector.support_ann_file = str(ann_file)
    detector.support_image_root = str(tmp_path / 'train')

    with pytest.raises(ValueError, match=expected_message):
        detector.build_support_object_bank()


def test_support_annotation_resolves_coco_ids_and_class_names(tmp_path):
    ann_file = tmp_path / '1_shot.json'
    ann_file.write_text(
        json.dumps({
            'images': [{
                'id': 17,
                'file_name': 'nested/a.jpg'
            }],
            'categories': [{
                'id': 5,
                'name': 'class_0'
            }],
            'annotations': [{
                'id': 99,
                'image_id': 17,
                'category_id': 5,
                'bbox': [1, 2, 3, 4],
            }],
        }), encoding='utf-8')
    detector = _make_detector_stub(num_classes=1)
    detector.support_entries = None
    detector.support_ann_file = str(ann_file)
    detector.support_image_root = str(tmp_path / 'train')

    detector.build_support_object_bank()

    assert detector.support_entries == [{
        'file_name': 'nested/a.jpg',
        'bbox': (1.0, 2.0, 3.0, 4.0),
        'class_idx': 0,
    }]


def test_support_annotation_requires_every_class(tmp_path):
    ann_file = tmp_path / '1_shot.json'
    ann_file.write_text(
        json.dumps({
            'images': [{
                'id': 17,
                'file_name': 'a.jpg'
            }],
            'categories': [{
                'id': 5,
                'name': 'class_0'
            }, {
                'id': 9,
                'name': 'class_1'
            }],
            'annotations': [{
                'id': 99,
                'image_id': 17,
                'category_id': 5,
                'bbox': [0, 0, 2, 2],
            }],
        }), encoding='utf-8')
    detector = _make_detector_stub(num_classes=2)
    detector.support_entries = None
    detector.support_ann_file = str(ann_file)
    detector.support_image_root = str(tmp_path / 'train')

    with pytest.raises(ValueError, match='class_1'):
        detector.build_support_object_bank()
