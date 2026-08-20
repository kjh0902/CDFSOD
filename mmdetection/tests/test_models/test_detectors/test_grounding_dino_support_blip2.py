import json
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mmdet.models.detectors.grounding_dino import GroundingDINO


def _make_detector_stub(num_classes=2, hidden_size=8):
    detector = GroundingDINO.__new__(GroundingDINO)
    nn.Module.__init__(detector)
    detector._cached_eval_support_prototypes = None
    detector.prototype_tokens_per_class = 32
    detector.support_class_names = [
        f'class_{idx}' for idx in range(num_classes)
    ]
    detector.decoder = SimpleNamespace(num_layers=0)
    detector.bbox_head = SimpleNamespace(
        cls_branches=[SimpleNamespace(max_text_len=max(
            256, num_classes * 32))])
    detector.text_feat_map = nn.Linear(768, hidden_size)
    return detector


def test_eval_visual_prototype_is_detached_and_cached():
    detector = _make_detector_stub().eval()
    calls = []

    def compute(self, device):
        calls.append(device)
        return torch.arange(
            2 * 32 * 8,
            dtype=torch.float32,
            device=device,
            requires_grad=True).reshape(2, 32, 8)

    detector.compute_visual_support_prototypes = MethodType(compute, detector)
    first = detector.build_prototype_text_dict(1, torch.device('cpu'))
    second = detector.build_prototype_text_dict(4, torch.device('cpu'))

    assert len(calls) == 1
    assert detector._cached_eval_support_prototypes.shape == (2, 32, 8)
    assert not detector._cached_eval_support_prototypes.requires_grad
    assert not first['embedded'].requires_grad
    assert first['embedded'].shape == (1, 64, 8)
    assert second['embedded'].shape == (4, 64, 8)
    torch.testing.assert_close(first['embedded'][0], second['embedded'][0])


def test_training_recomputes_visual_prototypes_and_clears_eval_cache():
    detector = _make_detector_stub().train()
    calls = []

    def compute(self, device):
        calls.append(device)
        return torch.ones(2, 32, 8, requires_grad=True)

    detector.compute_visual_support_prototypes = MethodType(compute, detector)
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    detector.build_prototype_text_dict(1, torch.device('cpu'))
    assert len(calls) == 2

    detector._cached_eval_support_prototypes = torch.ones(2, 32, 8)
    detector.train(True)
    assert detector._cached_eval_support_prototypes is None


def test_k_shot_aggregation_preserves_query_indices():
    queries = torch.stack([
        torch.arange(32 * 3).reshape(32, 3),
        torch.arange(32 * 3).reshape(32, 3) + 2,
        torch.arange(32 * 3).reshape(32, 3) + 100,
    ]).float()
    labels = torch.tensor([0, 0, 1])

    aggregated = GroundingDINO.aggregate_support_query_tokens(
        queries, labels, num_classes=2)

    assert aggregated.shape == (2, 32, 3)
    torch.testing.assert_close(aggregated[0], (queries[0] + queries[1]) / 2)
    torch.testing.assert_close(aggregated[1], queries[2])


def test_all_32_tokens_map_to_their_class():
    detector = _make_detector_stub()
    labels = [torch.tensor([0, 1])]

    positive_map = detector.build_prototype_positive_maps(
        labels, torch.device('cpu'))[0]
    token_map = detector.build_prototype_token_positive_map()

    assert positive_map[0, :32].all() and not positive_map[0, 32:].any()
    assert positive_map[1, 32:64].all()
    assert positive_map[1, :32].sum() == 0
    assert token_map == {1: list(range(32)), 2: list(range(32, 64))}


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
