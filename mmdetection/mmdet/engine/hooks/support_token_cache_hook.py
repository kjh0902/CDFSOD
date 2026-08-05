# Copyright (c) OpenMMLab. All rights reserved.
import re
from copy import deepcopy
from typing import Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SupportTokenCacheHook(Hook):
    """Build textualized visual tokens once before the test loop."""

    def __init__(self,
                 support_dataloader: dict,
                 support_shots: Optional[int] = None) -> None:
        self.support_dataloader = deepcopy(support_dataloader)
        self.support_shots = support_shots

    def _resolve_support_shots(self, runner: Runner) -> int:
        if self.support_shots is not None:
            if self.support_shots <= 0:
                raise ValueError('support_shots must be a positive integer.')
            return self.support_shots

        candidates = [runner.work_dir, getattr(runner, '_load_from', None)]
        for candidate in candidates:
            if candidate is None:
                continue
            match = re.search(r'(?<!\d)(\d+)[_-]?shot', str(candidate), re.I)
            if match is not None:
                return int(match.group(1))
        raise ValueError(
            'The support shot count is unavailable. Set CDFSOD_SHOT or use '
            'a support annotation/work directory containing "{N}_shot" or '
            '"{N}shot".')

    def before_test(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if model.has_support_token_cache:
            return

        support_dataloader = Runner.build_dataloader(
            self.support_dataloader, seed=runner.seed)
        class_names = support_dataloader.dataset.metainfo['classes']
        model.build_support_token_cache(
            support_dataloader,
            support_shots=self._resolve_support_shots(runner),
            class_names=class_names)
