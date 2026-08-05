import torch.nn as nn

from mmdet.engine.optimizers import TextualizerOptimizerConstructor
from mmdet.models.utils import TextualizedVisualTokenGenerator


def test_textualizer_optimizer_contains_only_textualizer_parameters():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.detector = nn.Linear(4, 4)
            self.detector.requires_grad_(False)
            self.textualized_visual_token_generator = \
                TextualizedVisualTokenGenerator()

    model = Model()
    constructor = TextualizerOptimizerConstructor(
        optim_wrapper_cfg=dict(
            type='OptimWrapper',
            optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4)),
        paramwise_cfg=dict(custom_keys={
            'textualized_visual_token_generator': dict(lr_mult=1.0)
        }))
    optim_wrapper = constructor(model)

    optimized_parameters = {
        id(parameter)
        for group in optim_wrapper.optimizer.param_groups
        for parameter in group['params']
    }
    textualizer_parameters = {
        id(parameter)
        for parameter in model.textualized_visual_token_generator.parameters()
        if parameter.requires_grad
    }
    assert optimized_parameters == textualizer_parameters
