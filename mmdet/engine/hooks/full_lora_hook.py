import math

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.optim.scheduler import ReduceOnPlateauParamScheduler

from mmdet.models.layers.lora import lora_parameter_groups
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FullLoRAHook(Hook):
    """Audit freezing and retain module LR ratios down to the LR floor."""
    def before_train(self, runner):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        expected = lora_parameter_groups(model)
        actual = runner.optim_wrapper.optimizer.param_groups
        if len(expected) != len(actual):
            raise RuntimeError('Full LoRA optimizer must contain seven adapter groups.')
        for group, target in zip(actual, expected):
            if ({id(p) for p in group['params']} != {id(p) for p in target['params']}
                    or group.get('name') != target['name']):
                raise RuntimeError('Full LoRA optimizer contains unexpected parameters.')
            runner.logger.info(f"Full LoRA {group['name']}: "
                               f"{sum(p.numel() for p in group['params']):,} parameters, "
                               f"lr={group['lr']:.8g}")
        schedulers = runner.param_schedulers
        if not isinstance(schedulers, list) or len(schedulers) != 1:
            raise RuntimeError('Full LoRA requires one ReduceOnPlateau scheduler.')
        scheduler = schedulers[0]
        if not isinstance(scheduler, ReduceOnPlateauParamScheduler):
            raise RuntimeError('Full LoRA requires ReduceOnPlateauParamScheduler.')
        # A shared absolute floor would collapse the 1:2 ratio late in training.
        # Keep the requested 1e-6 minimum for the slow groups, and use 2e-6
        # for the fast groups. All groups still stay >= configured min_value.
        initial = [group['initial_lr'] for group in scheduler.optimizer.param_groups]
        reference = min(initial)
        floor = min(scheduler.min_values)
        scheduler.min_values = [floor * lr / reference for lr in initial]
        self._check_ratios(actual)
        runner.logger.info(f'Full LoRA proportional LR floors: {scheduler.min_values}')

    @staticmethod
    def _check_ratios(groups):
        factors = [g['lr'] / g['initial_lr'] for g in groups]
        if not all(math.isclose(factor, factors[0], rel_tol=1e-7) for factor in factors):
            raise RuntimeError('Full LoRA module learning-rate ratios changed.')

    def before_train_epoch(self, runner):
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        lora_parameter_groups(model)
        self._check_ratios(runner.optim_wrapper.optimizer.param_groups)
