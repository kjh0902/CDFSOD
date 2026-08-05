# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SupportTokenCacheHook(Hook):
    """Provide support data for training and cache tokens before testing."""

    def __init__(self, support_dataloader: dict) -> None:
        self.support_dataloader = deepcopy(support_dataloader)
        self._support_dataloader_instance = None

    def _get_model(self, runner: Runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def _get_support_dataloader(self, runner: Runner):
        if self._support_dataloader_instance is None:
            self._support_dataloader_instance = Runner.build_dataloader(
                self.support_dataloader, seed=runner.seed)
        return self._support_dataloader_instance

    def before_train(self, runner: Runner) -> None:
        model = self._get_model(runner)
        support_dataloader = self._get_support_dataloader(runner)
        class_names = support_dataloader.dataset.metainfo['classes']
        model.set_support_dataloader(
            support_dataloader, class_names=class_names)

    def before_test(self, runner: Runner) -> None:
        model = self._get_model(runner)
        if model.has_support_token_cache:
            return

        support_dataloader = self._get_support_dataloader(runner)
        class_names = support_dataloader.dataset.metainfo['classes']
        model.build_support_token_cache(
            support_dataloader,
            class_names=class_names)
