from mmengine.model import is_model_wrapper
from mmengine.optim import DefaultOptimWrapperConstructor

from mmdet.models.layers.lora import lora_parameter_groups
from mmdet.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class FullLoRAOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """AdamW sees only adapters, in seven explicit module LR groups."""
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        # A nonempty paramwise_cfg makes the parent use add_params().
        super().__init__(optim_wrapper_cfg, paramwise_cfg=dict(full_lora=True))

    def add_params(self, params, module, prefix='', **kwargs):
        if is_model_wrapper(module):
            module = module.module
        params.extend(lora_parameter_groups(module))
