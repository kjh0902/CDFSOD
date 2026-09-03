from types import SimpleNamespace

import torch
import torch.nn as nn

from mmdet.engine.hooks.stage_lr_hook import BBoxHeadFirstHook6


class _Logger:

    def info(self, message):
        pass


class _Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.language_model = nn.Module()
        self.language_model.bert = nn.Linear(1, 1)
        self.language_model.support_blip_captioner = nn.Module()
        self.language_model.support_blip_captioner.vision_model = nn.Linear(
            1, 1)
        self.language_model.support_blip_captioner.text_decoder = nn.Linear(
            1, 1)
        self.bbox_head = nn.Linear(1, 1)
        self.other = nn.Linear(1, 1)


def test_blip_and_bert_are_identified_as_language_model_groups():
    model = _Model()
    named_parameters = dict(model.named_parameters())
    optimizer = torch.optim.SGD([
        {'params': [parameter]}
        for parameter in named_parameters.values()
    ], lr=0.1)
    runner = SimpleNamespace(
        optim_wrapper=SimpleNamespace(optimizer=optimizer), logger=_Logger())
    hook = BBoxHeadFirstHook6()

    hook.before_train(runner)

    param_names = list(named_parameters)
    language_group_names = {
        param_names[group_idx] for group_idx in hook._lang_model_groups
    }
    assert 'language_model.bert.weight' in language_group_names
    assert ('language_model.support_blip_captioner.vision_model.weight'
            in language_group_names)
    assert ('language_model.support_blip_captioner.text_decoder.weight'
            in language_group_names)
    assert all(
        optimizer.param_groups[group_idx]['lr'] > 0
        for group_idx in hook._lang_model_groups)

    hook._set_stage2_lr(runner)

    assert all(
        optimizer.param_groups[group_idx]['lr'] > 0
        for group_idx in hook._lang_model_groups)
