"""Epoch-tail training and streaming whole-validation visual-token logging."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS
from mmdet.utils.visual_token_stats import VisualTokenStats


@HOOKS.register_module()
class VisualTokenHook(Hook):
    priority = 'LOW'

    def __init__(self, dataset, shot, enabled=True):
        self.dataset = dataset
        self.shot = shot
        self.enabled = enabled
        self.stats = None
        self.loaded_metadata = None
        self.cache_prepared = False

    def _model(self, runner):
        return runner.model.module if is_model_wrapper(runner.model) else runner.model

    def _attach(self, runner, stats):
        model = self._model(runner)
        model._visual_stats = stats

    def _new_stats(self, runner):
        model = self._model(runner)
        return VisualTokenStats(model.bbox_head.num_classes,
                                next(model.parameters()).device,
                                len(model.encoder.support_layers) if hasattr(model, 'encoder') else 0)

    def after_load_checkpoint(self, runner, checkpoint):
        self.loaded_metadata = checkpoint.get('meta', {}).get('support_local_memory')
        self.cache_prepared = False

    def before_save_checkpoint(self, runner, checkpoint):
        model = self._model(runner)
        if model.visual_enabled:
            checkpoint.setdefault('meta', {})['support_local_memory'] = model._support_cache_metadata

    def _prepare_cache(self, runner, require_existing=False):
        model = self._model(runner)
        if not model.visual_enabled or self.cache_prepared:
            return
        path = model.support_visual_cfg.get(
            'memory_path', str(Path(runner.work_dir) / 'support_local_memory.pt'))
        metadata = model.support_cache_metadata(runner.cfg.get('load_from'), self.loaded_metadata)
        if self.loaded_metadata is not None:
            identity = lambda m: {k: v for k, v in m.items() if k != 'source_checkpoint'}
            if identity(metadata) != identity(self.loaded_metadata):
                raise ValueError('Checkpoint Support metadata does not match this experiment.')
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        # Only rank zero extracts/writes; all ranks reuse exactly the same artifact.
        error = [None]
        if rank == 0:
            try:
                model.prepare_support_cache(path, metadata, require_existing)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        if distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])
        if rank != 0:
            model.prepare_support_cache(path, metadata, require_existing=True)
        self.cache_prepared = True

    def before_train(self, runner):
        self._prepare_cache(runner, require_existing=bool(
            self.loaded_metadata or getattr(runner, '_resume', False)))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if (self.enabled and self._model(runner).visual_enabled and
                batch_idx + 1 == len(runner.train_dataloader)):
            self.stats = self._new_stats(runner)
            self._attach(runner, self.stats)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.stats is not None:
            self.stats.finish_batch()
            self._attach(runner, None)
            self._write(runner, 'train')

    def _before_eval(self, runner, phase):
        self._prepare_cache(runner, require_existing=True)
        self.cursor = 0
        if self.enabled and self._model(runner).visual_enabled:
            self.stats = self._new_stats(runner)
            self._attach(runner, self.stats)
            loader = getattr(runner, f'{phase}_dataloader')
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                from mmengine.dataset import DefaultSampler
                if not isinstance(loader.sampler, DefaultSampler) or loader.sampler.shuffle:
                    raise ValueError('DDP visual statistics require an unshuffled DefaultSampler.')

    def _before_eval_iter(self, runner, phase, batch_idx, data_batch):
        if self.stats is None:
            return
        rank, world = (dist.get_rank(), dist.get_world_size()) if (
            dist.is_available() and dist.is_initialized()) else (0, 1)
        size = len(getattr(runner, f'{phase}_dataloader').dataset)
        batch = len(data_batch['data_samples'])
        # DefaultSampler strides global indices by rank and pads only at the end.
        self.stats.capture_support = batch_idx % 50 == 0
        self.stats.valid = [rank + (self.cursor + i) * world < size for i in range(batch)]
        self.cursor += batch

    def before_val_epoch(self, runner):
        self._before_eval(runner, 'val')

    def before_test_epoch(self, runner):
        self._before_eval(runner, 'test')

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        self._before_eval_iter(runner, 'val', batch_idx, data_batch)

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        self._before_eval_iter(runner, 'test', batch_idx, data_batch)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.stats is not None:
            self.stats.finish_batch()

    after_test_iter = after_val_iter

    def after_val_epoch(self, runner, metrics=None):
        self._attach(runner, None)
        if self.stats is not None:
            self._write(runner, 'val')

    def after_test_epoch(self, runner, metrics=None):
        self._attach(runner, None)
        if self.stats is not None:
            self._write(runner, 'test')

    def _write(self, runner, phase):
        stats, self.stats = self.stats, None
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats.sums)
            dist.all_reduce(stats.counts)
            dist.all_reduce(stats.layer_sums)
            dist.all_reduce(stats.layer_counts)
            if dist.get_rank() != 0:
                return
        names = self._model(runner).support_visual_cfg['class_names']
        row = dict(schema_version=3, architecture='support_local_cross_attention',
                   dataset=self.dataset, shot=self.shot, phase=phase,
                   epoch=runner.epoch + 1, iteration=runner.iter,
                   **stats.report(names))
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        path = Path(runner.work_dir) / 'visual_token_stats.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        runner.logger.info('[visual-token] ' + line)
