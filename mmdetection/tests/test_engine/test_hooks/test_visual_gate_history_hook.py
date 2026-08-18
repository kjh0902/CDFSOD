import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from mmdet.engine.hooks import VisualGateHistoryHook


def _make_runner(tmp_path, gate_value=0.0, epoch=0):
    model = nn.Module()
    model.visual_gate = nn.Parameter(torch.tensor(gate_value))
    return SimpleNamespace(
        model=model,
        epoch=epoch,
        work_dir=str(tmp_path),
        logger=Mock())


def test_visual_gate_history_records_and_replaces_epochs(tmp_path):
    hook = VisualGateHistoryHook()
    runner = _make_runner(tmp_path, gate_value=0.1, epoch=0)
    hook.after_train_epoch(runner)

    runner.epoch = 1
    runner.model.visual_gate.data.fill_(0.2)
    hook.after_train_epoch(runner)

    runner.epoch = 0
    runner.model.visual_gate.data.fill_(0.3)
    hook.after_train_epoch(runner)

    history_path = tmp_path / 'visual_gate_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    assert [entry['epoch'] for entry in history] == [1, 2]
    assert history[0]['visual_gate'] == pytest.approx(0.3)
    assert history[1]['visual_gate'] == pytest.approx(0.2)
    assert runner.logger.info.call_count == 3
    assert not (tmp_path / 'visual_gate_history.json.tmp').exists()


def test_visual_gate_history_skips_models_without_gate(tmp_path):
    hook = VisualGateHistoryHook()
    runner = SimpleNamespace(
        model=nn.Module(),
        epoch=0,
        work_dir=str(tmp_path),
        logger=Mock())

    hook.after_train_epoch(runner)

    assert not (tmp_path / 'visual_gate_history.json').exists()
    runner.logger.info.assert_not_called()
