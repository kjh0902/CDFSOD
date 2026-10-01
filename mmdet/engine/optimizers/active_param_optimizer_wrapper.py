"""Clip only parameters whose optimizer groups currently permit updates."""
from mmengine.optim import AmpOptimWrapper, OptimWrapper

from mmdet.registry import OPTIM_WRAPPERS


class _ActiveParamClipping:
    def _clip_grad(self):
        params, seen = [], set()
        for group in self.optimizer.param_groups:
            if group['lr'] <= 0:
                continue
            for param in group['params']:
                if (param.requires_grad and param.grad is not None
                        and id(param) not in seen):
                    params.append(param)
                    seen.add(id(param))
        if params:
            norm = self.clip_func(params, **self.clip_grad_kwargs)
            if norm is not None:
                self.message_hub.update_scalar(
                    f'train/{self.grad_name}', float(norm))


@OPTIM_WRAPPERS.register_module()
class ActiveParamOptimWrapper(_ActiveParamClipping, OptimWrapper):
    """OptimWrapper with LR-positive-only global gradient clipping.

    Frozen gradients remain intact for backward, DDP and optimizer moments.
    Accumulation, stepping and zeroing follow MMEngine without modification.
    """


@OPTIM_WRAPPERS.register_module()
class ActiveParamAmpOptimWrapper(_ActiveParamClipping, AmpOptimWrapper):
    """AMP variant; MMEngine unscales gradients before active-only clipping."""
