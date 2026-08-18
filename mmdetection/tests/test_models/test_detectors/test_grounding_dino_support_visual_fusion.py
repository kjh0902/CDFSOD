import json
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mmdet.models.detectors.grounding_dino import GroundingDINO


def _make_detector_stub():
    detector = GroundingDINO.__new__(GroundingDINO)
    nn.Module.__init__(detector)
    detector._cached_eval_support_prototypes = None
    detector.decoder = SimpleNamespace(num_layers=0)
    detector.bbox_head = SimpleNamespace(
        cls_branches=[SimpleNamespace(max_text_len=16)])
    return detector


def test_eval_fused_prototype_is_detached_and_cached():
    detector = _make_detector_stub().eval()
    calls = []

    def compute(self, device):
        calls.append(device)
        return torch.arange(
            24, dtype=torch.float32, device=device,
            requires_grad=True).reshape(3, 8)

    detector.compute_fused_support_prototypes = MethodType(compute, detector)
    first = detector.build_prototype_text_dict(1, torch.device('cpu'))
    second = detector.build_prototype_text_dict(4, torch.device('cpu'))

    assert len(calls) == 1
    assert detector._cached_eval_support_prototypes.shape == (3, 8)
    assert not detector._cached_eval_support_prototypes.requires_grad
    assert not first['embedded'].requires_grad
    assert first['embedded'].shape == (1, 3, 8)
    assert second['embedded'].shape == (4, 3, 8)
    torch.testing.assert_close(first['embedded'][0], second['embedded'][0])


def test_train_mode_clears_eval_support_cache():
    detector = _make_detector_stub().eval()
    detector._cached_eval_support_prototypes = torch.ones(2, 8)
    detector.train(True)
    assert detector._cached_eval_support_prototypes is None


def test_zero_visual_gate_exactly_preserves_text_prototype():
    detector = _make_detector_stub()
    detector.visual_gate = nn.Parameter(torch.tensor(0.0))
    text = torch.randn(4, 8)
    visual = torch.randn(4, 8)

    fused = detector.fuse_text_visual_prototypes(text, visual)

    assert torch.equal(fused, text)


@pytest.mark.parametrize(
    'caption_entry, expected_message', [
        ({
            'category_name': 'class_a',
            'caption': 'description',
            'file_names': ['a.jpg'],
            'bboxes': [],
        }, 'file_names/bboxes lengths differ'),
        ({
            'category_name': 'class_a',
            'caption': 'description',
            'file_names': ['a.jpg'],
            'bboxes': [[0, 0, 0, 2]],
        }, 'non-positive bbox size'),
    ])
def test_support_caption_visual_metadata_validation(
        tmp_path, caption_entry, expected_message):
    caption_file = tmp_path / 'captions.json'
    caption_file.write_text(
        json.dumps({
            'img_prefix': 'train',
            'captions': [caption_entry],
        }),
        encoding='utf-8')
    detector = _make_detector_stub()
    detector.support_prompt_bank = None
    detector.support_caption_file = str(caption_file)
    detector.support_class_names = ['class_a']
    detector.support_image_root = str(tmp_path / 'train')

    with pytest.raises(ValueError, match=expected_message):
        detector.build_support_prompt_bank()


def test_support_caption_requires_visual_objects_for_every_class(tmp_path):
    caption_file = tmp_path / 'captions.json'
    caption_file.write_text(
        json.dumps({
            'img_prefix': 'train',
            'captions': [{
                'category_name': 'class_a',
                'caption': 'description',
                'file_names': ['a.jpg'],
                'bboxes': [[0, 0, 2, 2]],
            }],
        }),
        encoding='utf-8')
    detector = _make_detector_stub()
    detector.support_prompt_bank = None
    detector.support_caption_file = str(caption_file)
    detector.support_class_names = ['class_a', 'class_b']
    detector.support_image_root = str(tmp_path / 'train')

    with pytest.raises(ValueError, match='class_b'):
        detector.build_support_prompt_bank()
