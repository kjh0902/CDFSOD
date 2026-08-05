# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SupportTokenCacheHook(Hook):
    """Build textualized visual tokens once before the test loop."""

    def __init__(self, support_dataloader: dict) -> None:
        self.support_dataloader = deepcopy(support_dataloader)

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
            class_names=class_names)
