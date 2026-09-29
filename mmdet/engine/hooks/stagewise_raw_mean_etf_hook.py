"""Synchronize raw mean ETF placement with ACL progressive fine-tuning."""
import copy
import math

from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

from .stage_lr_hook import BBoxHeadFirstHook6


@HOOKS.register_module()
class StagewiseRawMeanETFHook(BBoxHeadFirstHook6):
    """Keep ACL's Stage 1 freeze groups and switch BERT ETF to FE ETF once.

    Only encoder.* groups receive the extra Stage 2 LR multiplier. Checkpoint
    metadata owns hook state, leaving all model state_dict keys unchanged.
    """

    checkpoint_key = 'stagewise_raw_mean_etf'

    def __init__(self, adjust_scheduler_patience=True, patience_frozen=3,
                 patience_unfrozen=6, stage2_fe_lr_mult=0.5):
        super().__init__(adjust_scheduler_patience, patience_frozen,
                         patience_unfrozen)
        if not math.isfinite(stage2_fe_lr_mult) or stage2_fe_lr_mult <= 0:
            raise ValueError('stage2_fe_lr_mult must be finite and positive.')
        self.stage2_fe_lr_mult = float(stage2_fe_lr_mult)
        self._fe_groups = []
        self._group_names = []
        self._loaded_state = None
        self._checkpoint_seen = False
        self._transition_epoch = None
        self._transition_iter = None

    @staticmethod
    def _model(runner):
        return runner.model.module if is_model_wrapper(runner.model) else runner.model

    def _identify_param_groups(self, runner, model):
        # Reuse the exact original classification, including backbone and neck
        # being trainable in Stage 1. Do not modify requires_grad or group order.
        self._head_groups, self._lang_model_groups = [], []
        self._backbone_groups, self._other_groups = [], []
        super()._identify_param_groups(runner, model)
        names = {id(p): name for name, p in model.named_parameters()}
        self._group_names = [
            [names[id(p)] for p in group['params']]
            for group in runner.optim_wrapper.optimizer.param_groups]
        self._fe_groups = []
        for index, group in enumerate(self._group_names):
            if any(name.startswith('encoder.') for name in group):
                if not all(name.startswith('encoder.') for name in group):
                    raise ValueError('Stagewise ETF needs separate encoder parameter groups.')
                self._fe_groups.append(index)
        if not self._lang_model_groups or not self._fe_groups:
            raise ValueError('Stagewise ETF requires language_model and encoder groups.')

    def before_train(self, runner):
        model = self._model(runner)
        if not getattr(model, 'stagewise_raw_mean_etf', False):
            raise ValueError('StagewiseRawMeanETFHook requires stagewise_raw_mean_etf=True.')
        if getattr(runner, '_resume', False) and self._checkpoint_seen:
            state = self._loaded_state
            if state is None:
                raise RuntimeError('Cannot resume stagewise ETF without stage metadata. '
                                   'Use load_from with resume=False for a new Stage 1 run.')
            self._identify_param_groups(runner, model)
            if (state.get('version') != 1 or state.get('stage') not in (1, 2)
                    or state.get('group_names') != self._group_names
                    or len(state.get('original_lrs', [])) != len(self._group_names)):
                raise RuntimeError('Incompatible stagewise ETF checkpoint metadata.')
            self._original_lrs = dict(enumerate(state['original_lrs']))
            self._stage2_started = state['stage'] == 2
            self._transition_epoch = state['transition_epoch']
            self._transition_iter = state['transition_iter']
            model.set_raw_mean_etf_stage(state['stage'])
            # Optimizer and scheduler were restored by Runner.resume *after*
            # after_load_checkpoint. Never overwrite their restored LR/state.
            self._log_status(runner, 'Resumed')
        else:
            self._stage2_started = False
            self._transition_epoch = self._transition_iter = None
            super().before_train(runner)
            self._log_status(runner, 'Initialized')

    def _set_stage1_lr(self, runner):
        super()._set_stage1_lr(runner)
        self._model(runner).set_raw_mean_etf_stage(1)

    def _set_stage2_lr(self, runner):
        super()._set_stage2_lr(runner)
        optimizer = runner.optim_wrapper.optimizer
        for index in self._fe_groups:
            optimizer.param_groups[index]['lr'] *= self.stage2_fe_lr_mult
        self._model(runner).set_raw_mean_etf_stage(2)
        self._transition_epoch, self._transition_iter = runner.epoch, runner.iter
        self._log_status(runner, 'Switched')

    def after_load_checkpoint(self, runner, checkpoint):
        # This hook also runs for ordinary load_from and inference. Defer the
        # resume decision until before_train and never change model weights.
        self._checkpoint_seen = True
        self._loaded_state = copy.deepcopy(
            checkpoint.get('meta', {}).get(self.checkpoint_key))

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})[self.checkpoint_key] = dict(
            version=1, stage=self._model(runner).raw_mean_etf_stage,
            original_lrs=[self._original_lrs[i] for i in range(len(self._group_names))],
            group_names=copy.deepcopy(self._group_names),
            transition_epoch=self._transition_epoch,
            transition_iter=self._transition_iter)

    def _log_status(self, runner, action):
        model = self._model(runner)
        position = 'bert_projected' if model.raw_mean_etf_stage == 1 else 'fe_final'
        rates = {}
        for names, group in zip(self._group_names, runner.optim_wrapper.optimizer.param_groups):
            for name in names:
                rates.setdefault(name.split('.')[0], set()).add(group['lr'])
        rates = ', '.join(f'{name}={sorted(values)}' for name, values in sorted(rates.items()))
        runner.logger.info(
            f'{action} ETF stage={model.raw_mean_etf_stage} position={position} '
            f'lambda={model.raw_mean_etf_loss_weight:g} epoch={runner.epoch} '
            f'iter={runner.iter}; module LRs: {rates}')

    def after_train_epoch(self, runner):
        self._log_status(runner, 'Current')
