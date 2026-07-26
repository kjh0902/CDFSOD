# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os
from typing import Any, Optional, TextIO

import torch
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class LossGradientFileLoggerHook(Hook):
    """Write per-iteration losses and gradient norm to a JSONL file.

    The gradient norm is read from the value recorded by ``OptimWrapper``
    while applying gradient clipping, so this hook does not add gradient
    computation or change the optimization procedure.

    Args:
        filename (str): File name under ``runner.work_dir``.
            Defaults to ``loss_gradient_per_iter.jsonl``.
    """

    priority = 'VERY_LOW'

    def __init__(
            self,
            filename: str = 'loss_gradient_per_iter.jsonl') -> None:
        self.filename = filename
        self.file: Optional[TextIO] = None

    def before_train(self, runner: Runner) -> None:
        """Open the log file on the main process before training."""
        if not is_main_process():
            return

        log_path = os.path.join(runner.work_dir, self.filename)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        mode = 'a' if runner.iter > 0 else 'w'
        self.file = open(log_path, mode, encoding='utf-8')
        runner.logger.info(
            f'Per-iteration loss/gradient log: {log_path}')

    @staticmethod
    def _to_scalar(value: Any) -> Optional[float]:
        """Convert a scalar tensor or number to a Python float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _json_safe(value: float) -> Any:
        """Keep the JSONL valid while preserving non-finite diagnostics."""
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return 'NaN'
        return 'Infinity' if value > 0 else '-Infinity'

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Write one loss/gradient record after every train iteration."""
        if self.file is None:
            return

        losses = {}
        for name, value in (outputs or {}).items():
            if 'loss' not in name:
                continue
            scalar = self._to_scalar(value)
            if scalar is not None:
                losses[name] = self._json_safe(scalar)

        grad_norm = None
        message_hub = runner.message_hub
        if 'train/grad_norm' in message_hub.log_scalars:
            current_norm = message_hub.get_scalar('train/grad_norm').current()
            grad_norm = self._json_safe(float(current_norm))

        record = dict(
            epoch=runner.epoch + 1,
            iteration=runner.iter + 1,
            iteration_in_epoch=batch_idx + 1,
            losses=losses,
            grad_norm=grad_norm)
        self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.file.flush()

    def after_train(self, runner: Runner) -> None:
        """Close the loss/gradient log file after training."""
        if self.file is not None:
            self.file.close()
            self.file = None
