"""Three-stage ACL fine-tuning without changing autograd or model wrappers."""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS
from ..optimizers.active_param_optimizer_wrapper import (
    ActiveParamAmpOptimWrapper, ActiveParamOptimWrapper)
from ..schedulers.three_stage_plateau_lr import ThreeStagePlateauLR


@HOOKS.register_module()
class ThreeStageProgressiveFinetuningHook(Hook):
    # ParamSchedulerHook=70; CheckpointHook=90.
    priority = 75

    def before_train(self, runner):
        if getattr(runner, '_resume', False):
            raise ValueError('Three-stage resume is not supported; use '
                             'load_from with resume=False for a new Stage 1 run.')
        if not isinstance(runner.optim_wrapper,
                          (ActiveParamOptimWrapper, ActiveParamAmpOptimWrapper)):
            raise TypeError('Three-stage training requires active-only clipping wrapper.')
        schedulers = runner.param_schedulers
        if (not isinstance(schedulers, list) or len(schedulers) != 1
                or not isinstance(schedulers[0], ThreeStagePlateauLR)):
            raise ValueError('Use exactly one ThreeStagePlateauLR scheduler.')
        forbidden = {'BBoxHeadFirstHook6', 'StageWiseFreezeHook',
                     'StagewiseRawMeanETFHook', 'AutoLoadBestCheckpointHook'}
        if any(type(h).__name__ in forbidden for h in runner.hooks):
            raise ValueError('Remove legacy progressive/rollback hooks.')
        if sum(isinstance(h, type(self)) for h in runner.hooks) != 1:
            raise ValueError('Use exactly one three-stage hook.')
        if not hasattr(runner.train_loop, 'stop_training'):
            raise TypeError('Three-stage training requires EpochBasedTrainLoop.')
        self.scheduler = schedulers[0]
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        self._identify_groups(model, runner.optim_wrapper.optimizer)
        self._apply_stage(runner)
        self._log_status(runner, 'Start')

    def _identify_groups(self, model, optimizer):
        bert = model.language_model.language_backbone.body.model
        backbone = model.backbone
        if len(bert.encoder.layer) != 12 or len(backbone.stages) != 4:
            raise ValueError('Expected 12 BERT layers and four Swin stages.')
        activation = {id(p): 1 for p in model.parameters()}
        labels = {id(p): 'Other' for p in model.parameters()}
        for module, label in ((model.language_model, 'BERT'), (backbone, 'Swin')):
            for p in module.parameters():
                activation[id(p)] = 3
                labels[id(p)] = label
        partial = list(bert.encoder.layer[6:12]) + list(backbone.stages[2:4])
        partial += [backbone.norm2, backbone.norm3]
        for module in partial:
            for p in module.parameters():
                activation[id(p)] = 2
        for name in ('text_feat_map', 'neck', 'encoder', 'decoder', 'bbox_head'):
            for p in getattr(model, name).parameters():
                labels[id(p)] = name

        self.group_stages, self.group_labels = [], []
        seen = set()
        for group in optimizer.param_groups:
            ids = [id(p) for p in group['params']]
            if not ids or any(i not in activation or i in seen for i in ids):
                raise ValueError('Optimizer must contain unique, nonempty model groups.')
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate parameter in optimizer group.')
            stages = {activation[i] for i in ids}
            if len(stages) != 1:
                raise ValueError('Optimizer group mixes different stage activation targets.')
            self.group_stages.append(stages.pop())
            self.group_labels.append('+'.join(sorted({labels[i] for i in ids})))
            seen.update(ids)
        required = {id(p) for p in model.parameters() if p.requires_grad}
        if not required.issubset(seen):
            raise ValueError('Optimizer must include all trainable model parameters.')

    def _apply_stage(self, runner):
        groups = runner.optim_wrapper.optimizer.param_groups
        scheduled = self.scheduler.scheduled_lrs
        wrapper_groups = runner.optim_wrapper.param_groups
        if len(scheduled) != len(wrapper_groups):
            raise ValueError('Optimizer groups changed after scheduler construction.')
        for group, first_stage, lr in zip(groups, self.group_stages, scheduled):
            group['lr'] = lr if self.scheduler.stage >= first_stage else 0.
        if runner.optim_wrapper.base_param_settings is not None:
            runner.optim_wrapper.base_param_settings['lr'] = scheduled[-1]
        self.scheduler._last_value = [g['lr'] for g in wrapper_groups]
        runner.message_hub.update_scalar('train/finetuning_stage', self.scheduler.stage)

    def before_train_epoch(self, runner):
        self._log_status(runner, 'Epoch start')

    def after_val_epoch(self, runner, metrics=None):
        self._apply_stage(runner)
        event = self.scheduler.last_event
        if event:
            runner.logger.info(
                f"Three-stage plateau: Stage {event['from_stage']} -> "
                f"Stage {event['to_stage']}; {self.scheduler.monitor}="
                f"{event['metric']:.6g}, best={event['best']:.6g}, "
                f"bad_count={event['bad_count']}, patience={event['patience']}; "
                f"reference LR ranges {min(event['old_lrs']):.6g}.."
                f"{max(event['old_lrs']):.6g} -> "
                f"{min(event['new_lrs']):.6g}..{max(event['new_lrs']):.6g}")
        if self.scheduler.should_stop:
            runner.train_loop.stop_training = True
            runner.logger.info('Stage 3 plateau: early stopping at current weights; '
                               'no further LR reduction or checkpoint rollback.')
        self._log_status(runner, 'Validation complete')

    def _log_status(self, runner, context):
        counts = [0, 0]
        rates = {}
        for label, group in zip(self.group_labels,
                                runner.optim_wrapper.optimizer.param_groups):
            active = group['lr'] > 0
            counts[int(active)] += sum(p.numel() for p in group['params'])
            rates.setdefault(label, set()).add(group['lr'])
        lr_text = '; '.join(f'{name}=' + ','.join(f'{lr:.6g}' for lr in sorted(lrs))
                            for name, lrs in sorted(rates.items()))
        runner.logger.info(
            f'{context}: epoch={runner.epoch}, Stage {self.scheduler.stage}, '
            f'patience={self.scheduler.patience}, update-active={counts[1]:,}, '
            f'LR-zero={counts[0]:,}; actual LRs: {lr_text}')
