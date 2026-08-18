# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp

from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class VisualGateHistoryHook(Hook):
    """Persist the learned visual gate once after every training epoch."""

    def __init__(self,
                 file_name: str = 'visual_gate_history.json') -> None:
        self.file_name = file_name

    @master_only
    def after_train_epoch(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        visual_gate = getattr(model, 'visual_gate', None)
        if visual_gate is None:
            return

        epoch = int(runner.epoch) + 1
        gate_value = float(visual_gate.detach().cpu().item())
        history_path = osp.join(runner.work_dir, self.file_name)
        history = []
        if osp.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as history_file:
                history = json.load(history_file)
            if (not isinstance(history, list) or
                    any(not isinstance(entry, dict) for entry in history)):
                raise ValueError(
                    'Visual gate history must be a list of dictionaries: '
                    f'{history_path}')

        history = [
            entry for entry in history
            if entry.get('epoch') != epoch
        ]
        history.append(dict(epoch=epoch, visual_gate=gate_value))
        history.sort(key=lambda entry: entry.get('epoch', -1))

        os.makedirs(runner.work_dir, exist_ok=True)
        temporary_path = history_path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as history_file:
            json.dump(history, history_file, ensure_ascii=False, indent=2)
            history_file.write('\n')
        os.replace(temporary_path, history_path)
        runner.logger.info(
            f'visual_gate after epoch {epoch}: {gate_value:.10g}')
