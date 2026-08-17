import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mmdet.models.detectors.grounding_dino import GroundingDINO


TOOLS_DIR = Path(__file__).resolve().parents[2] / 'tools'


def load_tool_module(name):
    spec = importlib.util.spec_from_file_location(
        name, TOOLS_DIR / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_tool_module('generate_instance_captions')
selector = load_tool_module('select_attribute_descriptions')


def test_attribute_order_and_prompt_contract():
    assert [item[0] for item in generator.ATTRIBUTE_SPECS] == [
        'shape', 'texture', 'boundary', 'internal_structure',
        'color_intensity_material'
    ]
    prompt = generator.build_attribute_description_prompt(
        'rolled-in_scale', 'shape')
    assert 'Class name: rolled-in_scale' in prompt
    assert 'overall form, length, orientation' in prompt
    assert 'exactly one concise sentence' in prompt
    assert 'Do not mention the images, crops, regions, or class name' in prompt


def test_description_validation():
    assert generator.normalize_and_validate_description(
        'Long, thin, gently curved forms', 'scratches') == (
            'Long, thin, gently curved forms.')
    with pytest.warns(UserWarning, match='mentions class name'):
        description = generator.normalize_and_validate_description(
            'The rolled-in scale has a long boundary.', 'rolled-in_scale')
    assert description == 'The rolled-in scale has a long boundary.'
    with pytest.raises(ValueError, match='exactly one sentence'):
        generator.normalize_and_validate_description(
            'Dark and smooth. It is also thin.', 'inclusion')


def test_multi_token_class_features_are_averaged_before_selection():
    fake_model = SimpleNamespace(
        support_class_names=['sea cucumber', 'other'],
        text_feat_map=nn.Identity(),
        decoder=SimpleNamespace(num_layers=0),
        bbox_head=SimpleNamespace(
            cls_branches=[SimpleNamespace(max_text_len=256)]))
    fake_model._format_support_prompt = lambda name, description: \
        GroundingDINO._format_support_prompt(fake_model, name, description)
    tokenized = {
        'input_ids': torch.zeros((2, 4), dtype=torch.long),
        'attention_mask': torch.ones((2, 4), dtype=torch.long),
        'token_type_ids': torch.zeros((2, 4), dtype=torch.long),
        'special_tokens_mask': torch.zeros((2, 4), dtype=torch.long),
        'offset_mapping': torch.tensor([
            [[0, 0], [0, 3], [4, 12], [13, 14]],
            [[0, 0], [0, 5], [5, 6], [0, 0]],
        ]),
    }
    hidden = torch.tensor([
        [[0., 0.], [2., 4.], [6., 8.], [0., 0.]],
        [[0., 0.], [10., 12.], [0., 0.], [0., 0.]],
    ])
    fake_model._tokenize_support_prompts = lambda prompts: tokenized
    def find_token_positions(offsets, spans, prompts):
        return GroundingDINO._find_class_name_token_positions(
            fake_model, offsets, spans, prompts)

    fake_model._find_class_name_token_positions = find_token_positions
    fake_model._encode_support_prompt_features = lambda inputs: hidden

    text_dict = GroundingDINO.build_description_selection_text_dict(
        fake_model, 0, 'Long and slightly curved.', torch.device('cpu'))
    assert torch.equal(
        text_dict['embedded'][0], torch.tensor([[4., 6.], [10., 12.]]))


def test_query_count_and_first_max_tie_break():
    centers = torch.tensor([[[0.1, 0.1], [0.2, 0.2], [0.8, 0.8],
                             [0.15, 0.15]]])
    argmax_tokens = torch.tensor([[0, 1, 0, 0]])
    boxes = torch.tensor([[0.0, 0.0, 0.25, 0.25],
                          [0.7, 0.7, 0.9, 0.9]])
    assert selector.count_queries_in_bboxes(
        centers, argmax_tokens, 0, boxes) == [2, 1]

    scores = {
        key: 3.0 for key, _, _ in generator.ATTRIBUTE_SPECS
    }
    assert selector.choose_top_attribute(scores) == 'shape'


def test_selected_caption_remains_compatible_with_prompt_bank(tmp_path):
    caption_path = tmp_path / 'captions.json'
    caption_path.write_text(json.dumps({
        'captions': [{
            'category_name': 'sea cucumber',
            'selected_attribute': 'shape',
            'selected_description': 'Long and gently curved.',
            'caption': 'Long and gently curved.',
        }]
    }), encoding='utf-8')
    fake_model = SimpleNamespace(
        support_prompt_bank=None,
        support_caption_file=str(caption_path),
        support_class_names=['sea cucumber'])
    fake_model._format_support_prompt = lambda name, description: \
        GroundingDINO._format_support_prompt(fake_model, name, description)
    fake_model._tokenize_support_prompts = lambda prompts: {
        'input_ids': torch.zeros((1, 4), dtype=torch.long),
        'attention_mask': torch.ones((1, 4), dtype=torch.long),
        'offset_mapping': torch.tensor(
            [[[0, 0], [0, 3], [4, 12], [13, 14]]]),
    }
    fake_model._find_class_name_token_positions = lambda offsets, spans: \
        GroundingDINO._find_class_name_token_positions(
            fake_model, offsets, spans, fake_model.support_prompt_texts)

    GroundingDINO.build_support_prompt_bank(fake_model)
    assert fake_model.support_prompt_texts == [
        'sea cucumber: Long and gently curved.'
    ]
