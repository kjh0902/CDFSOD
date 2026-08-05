# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TextualizerOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Build an optimizer containing only textualizer parameters."""

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        textualizer = getattr(module,
                              'textualized_visual_token_generator', None)
        if textualizer is None:
            raise RuntimeError(
                'TextualizerOptimizerConstructor requires an enabled '
                'textualized visual token generator.')

        trainable_parameters = [
            parameter for parameter in textualizer.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError('The textualizer has no trainable parameters.')
        params.append(dict(params=trainable_parameters))
