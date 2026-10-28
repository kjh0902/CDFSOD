"""Support randomness.warn_only with stock and patched MMEngine runners."""
import copy
import inspect

import torch


def runner_randomness_config(cfg, runner_cls):
    """Return a private config and an optional post-construction setting."""
    cfg = copy.deepcopy(cfg)
    randomness = cfg.get('randomness', {})
    setting = None
    if 'warn_only' in randomness:
        parameters = inspect.signature(runner_cls.set_randomness).parameters
        supports = 'warn_only' in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if not supports:
            warn_only = randomness.pop('warn_only')
            if randomness.get('deterministic', False):
                setting = bool(warn_only)
    return cfg, setting


def apply_warn_only(setting):
    if setting is not None:
        torch.use_deterministic_algorithms(True, warn_only=setting)
