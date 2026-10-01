"""Plateau scheduling with reference LRs independent of LR-zero freezing."""
import math

from mmengine.optim.scheduler import ReduceOnPlateauParamScheduler

from mmdet.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class ThreeStagePlateauLR(ReduceOnPlateauParamScheduler):
    """Halve reference LRs twice, then request early stopping.

    The companion hook applies the stage mask to actual optimizer LRs.
    Plateau counting, relative improvement and cooldown use MMEngine's
    implementation. All state is plain data compatible with its state_dict.
    """

    def __init__(self, optimizer, stage_patiences=(3, 5, 8),
                 monitor='coco/bbox_mAP', threshold=1e-4, cooldown=1,
                 min_value=1e-6, **kwargs):
        if (len(stage_patiences) != 3 or any(
                isinstance(p, bool) or not isinstance(p, int) or p < 0
                for p in stage_patiences)):
            raise ValueError('stage_patiences must contain three nonnegative integers.')
        if not math.isfinite(min_value) or min_value < 0:
            raise ValueError('min_value must be finite and nonnegative.')
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('threshold must be finite and nonnegative.')
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
            raise ValueError('cooldown must be a nonnegative integer.')
        self.stage_patiences = tuple(stage_patiences)
        self.stage = 1
        self.should_stop = False
        self.last_event = None
        super().__init__(
            optimizer, param_name='lr', monitor=monitor, rule='greater',
            factor=0.5, patience=stage_patiences[0], threshold=threshold,
            threshold_rule='rel', cooldown=cooldown, min_value=min_value,
            eps=0., **kwargs)
        self.scheduled_lrs = [float(g['lr']) for g in optimizer.param_groups]
        if any(not math.isfinite(lr) or lr <= 0 or lr * 0.25 < floor
               for lr, floor in zip(self.scheduled_lrs, self.min_values)):
            raise ValueError(
                'All initial reference LRs must be positive and allow two '
                'exact halvings without crossing min_value; check auto-scale LR.')

    def step(self, metrics=None):
        if metrics is None:
            return super().step()
        if not isinstance(metrics, dict) or self.monitor not in metrics:
            raise ValueError(f'Validation must provide metric {self.monitor!r}.')
        metric = float(metrics[self.monitor])
        if not math.isfinite(metric):
            raise ValueError(f'{self.monitor} must be finite, got {metric}.')
        self.last_event = None
        if self.should_stop:
            return
        self._current_metric = metric
        super().step({**metrics, self.monitor: metric})
        if self.last_event is not None and not self.should_stop:
            # Keep the cooldown set by the parent, but compare the next stage
            # against its own first validation, not the preceding stage's best.
            self.best = self.rule_worse
            self.patience = self.stage_patiences[self.stage - 1]

    def _get_value(self):
        old_stage = self.stage
        old_lrs = self.scheduled_lrs.copy()
        if self.stage < 3:
            self.scheduled_lrs = [lr * self.factor for lr in old_lrs]
            self.stage += 1
        else:
            self.should_stop = True
        self.last_event = dict(
            from_stage=old_stage, to_stage=self.stage,
            metric=self._current_metric, best=self.best,
            bad_count=self.num_bad_epochs, patience=self.patience,
            old_lrs=old_lrs, new_lrs=self.scheduled_lrs.copy(),
            stop=self.should_stop)
        # Keep frozen groups zero until the hook applies the new stage mask.
        # The final entry may be OptimWrapper's synthetic base LR settings.
        return [lr if group['lr'] > 0 else 0.
                for group, lr in zip(self.optimizer.param_groups,
                                     self.scheduled_lrs)]
