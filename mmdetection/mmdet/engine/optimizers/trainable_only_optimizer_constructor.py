# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class TrainableOnlyOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Build an optimizer containing only trainable parameters."""

    def __call__(self, model):
        optim_wrapper = super().__call__(model)
        optimizer = optim_wrapper.optimizer

        trainable_param_groups = []
        for param_group in optimizer.param_groups:
            param_group['params'] = [
                param for param in param_group['params']
                if param.requires_grad
            ]
            if param_group['params']:
                trainable_param_groups.append(param_group)
        optimizer.param_groups[:] = trainable_param_groups

        return optim_wrapper
