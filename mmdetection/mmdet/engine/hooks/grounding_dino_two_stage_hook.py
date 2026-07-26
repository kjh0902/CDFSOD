# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class GroundingDinoTwoStageHook(Hook):
    """Switch Grounding DINO from MLP-only to full fine-tuning.

    The optimizer is built once with every model parameter. This hook only
    changes ``requires_grad`` and module modes at the epoch boundary.
    """

    priority = 'VERY_HIGH'

    def __init__(self, mlp_only_epochs: int = 5) -> None:
        if mlp_only_epochs <= 0:
            raise ValueError('mlp_only_epochs must be positive.')
        self.mlp_only_epochs = mlp_only_epochs
        self._last_logged_stage = None

    @staticmethod
    def _unwrap_model(runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def before_train(self, runner) -> None:
        """Verify that the single optimizer owns every model parameter."""
        model = self._unwrap_model(runner)
        optimizer_param_ids = {
            id(param)
            for group in runner.optim_wrapper.optimizer.param_groups
            for param in group['params']
        }
        model_params = list(model.parameters())
        missing_names = [
            name for name, param in model.named_parameters()
            if id(param) not in optimizer_param_ids
        ]
        if missing_names:
            raise RuntimeError(
                'The two-stage optimizer must contain every model parameter, '
                f'but it is missing: {missing_names}')
        runner.logger.info(
            'Two-stage optimizer contains all model parameters: '
            f'{len(optimizer_param_ids)} tensors, '
            f'{sum(param.numel() for param in model_params):,} elements.')

    def before_train_epoch(self, runner) -> None:
        """Apply the stage required by the upcoming one-based epoch."""
        model = self._unwrap_model(runner)
        epoch = runner.epoch + 1
        target_stage = ('mlp_only' if epoch <= self.mlp_only_epochs else
                        'full_finetune')

        if model.training_stage != target_stage:
            if target_stage == 'full_finetune':
                model.clear_support_caches()
                runner.logger.info(
                    'Cleared Stage 1 support prototype/feature caches.')
            model.set_training_stage(target_stage)

        # EpochBasedTrainLoop calls train() immediately after this hook, but
        # apply it here as well so the logged state already reflects reality.
        model.train(True)

        if self._last_logged_stage != target_stage:
            self._log_trainable_parameters(runner, model, epoch)
            self._last_logged_stage = target_stage

    @staticmethod
    def _log_trainable_parameters(runner, model, epoch: int) -> None:
        trainable = [(name, param) for name, param in model.named_parameters()
                     if param.requires_grad]
        names = [name for name, _ in trainable]
        num_elements = sum(param.numel() for _, param in trainable)
        runner.logger.info(
            f'Epoch {epoch} training stage: {model.training_stage}. '
            f'Trainable parameters: {len(trainable)} tensors, '
            f'{num_elements:,} elements.\n' + '\n'.join(names))
