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
    detector._prototype_token_counts = None
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
        return [
            torch.arange(
                8, dtype=torch.float32, device=device,
                requires_grad=True).reshape(1, 8),
            torch.arange(
                16, dtype=torch.float32, device=device,
                requires_grad=True).reshape(2, 8),
        ]

    detector.compute_visual_support_prototypes = MethodType(compute, detector)
    first = detector.build_prototype_text_dict(1, torch.device('cpu'))
    second = detector.build_prototype_text_dict(4, torch.device('cpu'))

    assert len(calls) == 1
    assert [item.shape for item in detector._cached_eval_support_prototypes] \
        == [(1, 8), (2, 8)]
    assert all(not item.requires_grad
               for item in detector._cached_eval_support_prototypes)
    assert not first['embedded'].requires_grad
    assert first['embedded'].shape == (1, 3, 8)
    assert second['embedded'].shape == (4, 3, 8)
    torch.testing.assert_close(first['embedded'][0], second['embedded'][0])
    assert detector._prototype_token_counts == [1, 2]


def test_training_recomputes_blip_prototypes_and_clears_eval_cache():
    detector = _make_detector_stub().train()
    calls = []

    def compute(self, device):
        calls.append(device)
        return [
            torch.ones(1, 8, requires_grad=True),
            torch.ones(2, 8, requires_grad=True),
        ]

    detector.compute_visual_support_prototypes = MethodType(compute, detector)
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    assert len(calls) == 2

    detector._cached_eval_support_prototypes = (torch.ones(1, 8), )
    detector.train(True)
    assert detector._cached_eval_support_prototypes is None


def _support_token_inputs():
    tokens = torch.tensor([
        [[0., 0.], [1., 2.], [3., 4.], [5., 6.], [0., 0.]],
        [[2., 2.], [3., 4.], [5., 6.], [7., 8.], [0., 0.]],
        [[10., 10.], [20., 20.], [30., 30.], [0., 0.], [0., 0.]],
    ])
    labels = torch.tensor([0, 0, 1])
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
    ])
    special_tokens_mask = torch.tensor([
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 1, 1],
    ])
    return tokens, labels, attention_mask, special_tokens_mask


def test_class_avg_averages_shots_and_class_tokens():
    inputs = _support_token_inputs()

    prototypes = GroundingDINO.aggregate_support_multimodal_tokens(
        *inputs, num_classes=2, mode='class_avg')

    assert [prototype.shape for prototype in prototypes] == [(1, 2), (1, 2)]
    torch.testing.assert_close(prototypes[0], torch.tensor([[3., 4.]]))
    torch.testing.assert_close(prototypes[1], torch.tensor([[20., 20.]]))


def test_class_tokens_preserves_original_token_count():
    inputs = _support_token_inputs()

    prototypes = GroundingDINO.aggregate_support_multimodal_tokens(
        *inputs, num_classes=2, mode='class_tokens')

    assert [prototype.shape for prototype in prototypes] == [(2, 2), (1, 2)]
    torch.testing.assert_close(
        prototypes[0], torch.tensor([[2., 3.], [4., 5.]]))
    torch.testing.assert_close(prototypes[1], torch.tensor([[20., 20.]]))


def test_all_tokens_includes_special_tokens_and_excludes_padding():
    inputs = _support_token_inputs()

    prototypes = GroundingDINO.aggregate_support_multimodal_tokens(
        *inputs, num_classes=2, mode='all_tokens')

    assert [prototype.shape for prototype in prototypes] == [(4, 2), (3, 2)]
    torch.testing.assert_close(
        prototypes[0],
        torch.tensor([[1., 1.], [2., 3.], [4., 5.], [6., 7.]]))
    torch.testing.assert_close(
        prototypes[1], _support_token_inputs()[0][2, :3])


def test_invalid_prototype_mode_is_rejected():
    inputs = _support_token_inputs()

    with pytest.raises(ValueError, match='Prototype mode'):
        GroundingDINO.aggregate_support_multimodal_tokens(
            *inputs, num_classes=2, mode='invalid')


def test_variable_token_ranges_map_to_their_class():
    detector = _make_detector_stub()
    detector._prototype_token_counts = [1, 3]
    labels = [torch.tensor([0, 1])]

    positive_map = detector.build_prototype_positive_maps(
        labels, torch.device('cpu'))[0]
    token_map = detector.build_prototype_token_positive_map()

    assert positive_map[0, 0] == 1 and positive_map[0, 1:].sum() == 0
    assert positive_map[1, 1:4].all() and positive_map[1, :1].sum() == 0
    assert token_map == {1: [0], 2: [1, 2, 3]}


def test_total_prototype_length_cannot_exceed_max_text_len():
    detector = _make_detector_stub(max_text_len=3)

    with pytest.raises(RuntimeError, match='exceed max_text_len'):
        detector._prototypes_to_text_dict(
            [torch.ones(2, 8), torch.ones(2, 8)], batch_size=1)


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
