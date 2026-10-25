"""ACL diagnostics without changing training, predictions or optimizer state."""
import math
from pathlib import Path
import subprocess

import numpy as np
from mmengine.dist import get_dist_info
from mmengine.fileio import load
from mmengine.hooks import Hook, CheckpointHook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS
from mmdet.utils.acl_diagnostic_capture import ACLBatchCapture
from mmdet.utils.acl_diagnostic_events import (
    activate_diagnostics, deactivate_diagnostics, diagnostic_event)
from mmdet.utils.acl_diagnostic_io import (
    DiagnosticWriter, select_token_probes, snapshot_reasons)


@HOOKS.register_module()
class ACLDiagnosticHook(Hook):
    """Fail-fast diagnostics for the 18 single-GPU ACL baseline configs.

    ``out_dir`` defaults to work_dir/diagnostics. Existing output is rejected.
    ``reserve_gib`` and ``capacity_margin`` control disk capacity checks.
    Per-rank subdirectories avoid collisions if the runner is distributed.
    """
    priority = 'LOW'

    def __init__(self, out_dir=None, reserve_gib=10., capacity_margin=1.2,
                 internal_interval=5, token_probe_images=32):
        if not isinstance(internal_interval, int) or internal_interval < 1:
            raise ValueError('internal_interval must be a positive integer')
        if not isinstance(token_probe_images, int) or token_probe_images < 0:
            raise ValueError('token_probe_images must be a nonnegative integer')
        self.internal_interval = internal_interval
        self.token_probe_images = token_probe_images
        self.out_dir = out_dir
        self.reserve_gib = reserve_gib
        self.capacity_margin = capacity_margin
        self.handles = []
        self.token = None
        self.capture = None
        self.writer = None
        self.completed_epochs = 0
        self.measured_rates = {}

    def before_run(self, runner):
        # Runner calls before_run before load_or_resume. Reject resume before
        # it can restore model/optimizer state, even with the native train CLI.
        self._check_baseline(runner)

    def _check_baseline(self, runner):
        self.model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        for name in ('bert_etf_loss_weight', 'bert_orth_loss_weight',
                     'fe_etf_loss_weight', 'fe_orth_loss_weight'):
            if getattr(self.model, name) != 0.:
                raise ValueError(f'ACL diagnostic baseline requires {name}=0.0')
        if runner.cfg.get('resume', False):
            raise ValueError('ACL diagnostic baseline does not support resume')
        if any(isinstance(hook, CheckpointHook) or
               type(hook).__name__ == 'AutoLoadBestCheckpointHook' for hook in runner.hooks):
            raise ValueError('ACL diagnostics require all checkpoint save/reload hooks disabled')

    def before_train(self, runner):
        self._check_baseline(runner)
        self.logger = next(hook for hook in runner.hooks
                           if type(hook).__name__ == 'LoggerHook')
        self.stage_hook = next(hook for hook in runner.hooks
                               if type(hook).__name__ == 'BBoxHeadFirstHook6')
        dataset = runner.val_dataloader.dataset
        dataset.full_init()
        info = [dataset.get_data_info(i) for i in range(len(dataset))]
        ids = {row['img_id'] for row in info}
        annotation = load(dataset.ann_file, backend_args=dataset.backend_args)
        categories = {row['id']: row for row in annotation['categories']}
        self.classes = [dict(class_index=i, category_id=int(category_id),
                             class_name=categories[category_id]['name'])
                        for i, category_id in enumerate(dataset.cat_ids)]
        evaluator_cfg = runner.cfg.get('val_evaluator', {})
        thresholds = evaluator_cfg.get('iou_thrs')
        self.iou_thrs = np.asarray(np.linspace(.5, .95, 10) if thresholds is None else thresholds)
        self.expected_ids = ids
        self.rank, self.world_size = get_dist_info()
        self.epoch_samples = len(runner.val_dataloader.sampler)
        snapshot = dict(
            categories=self.classes,
            images=[row for row in annotation['images'] if row['id'] in ids],
            annotations=[{key: value for key, value in row.items() if key in
                          ('id', 'image_id', 'category_id', 'bbox', 'area', 'iscrowd', 'ignore')}
                         for row in annotation['annotations'] if row['image_id'] in ids],
            gt_box_format='xywh_original_pixels',
            annotation_file=str(dataset.ann_file))
        self.ground_truth = {}
        for entry in snapshot['annotations']:
            self.ground_truth.setdefault(entry['image_id'], []).append(entry)
        self.category_map = {entry['category_id']: entry['class_index'] for entry in self.classes}
        self.probe_ids = select_token_probes(snapshot, self.token_probe_images)
        self.current_stage = 2 if self.stage_hook._stage2_started else 1
        self.last_validation_stage = self.current_stage
        param_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        groups = [dict(index=i, names=[param_names.get(id(p), '<unnamed>')
                                      for p in group['params']],
                       weight_decay=group.get('weight_decay'))
                  for i, group in enumerate(runner.optim_wrapper.optimizer.param_groups)]
        try:
            repo = Path(__file__).resolve().parents[3]
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                             stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        estimated = self._estimate_capacity(runner, info)
        manifest = dict(
            git_commit=commit, run_id=runner.timestamp, rank=self.rank,
            world_size=self.world_size, max_epochs=runner.max_epochs,
            config=runner.cfg.pretty_text, seed=runner.seed,
            parameter_groups=groups, classes=self.classes,
            gradient_norm='pre_clipping_L2_from_optim_wrapper_message_hub',
            stage='BBoxHeadFirstHook6._stage2_started; LR=0 is the original freeze mechanism',
            raw_logits='final decoder token logits only on fixed token probe images at internal snapshots',
            class_scores='mean(sigmoid(token_logits)) over mapped class tokens',
            objectness='no native objectness head; use max class score as a proxy',
            proposal_recipe_version=1, mask_bitorder='little',
            decoder_box_format='normalized_cxcywh_resized_image',
            detection_box_format='xyxy_original_pixels',
            decoder_final_layer='decoder_class_scores[-1], decoder_boxes[-1]; not duplicated',
            fe_features='fe_class_prototypes[layer,class,channel]; token probes retain fe_final_text',
            storage_policy=dict(internal_interval=self.internal_interval,
                extra_epochs=['initial', 'stage_transition', 'final'], token_probe_ids=self.probe_ids,
                encoder_boundary_candidates=128, encoder_oracle_per_gt=5,
                encoder_scope='selected + boundary + GT-IoU oracle; omitted candidates summarized',
                detections='all images every epoch; original float32 boxes/scores',
                compression='NPZ DEFLATE; float16 internals; smallest lossless unsigned indices'),
            large_float_dtype='float16', detection_and_gt_dtype='float32',
            indices='zero_based; class_mapping keys retain model one_based labels')
        root = Path(self.out_dir or Path(runner.work_dir) / 'diagnostics')
        if self.world_size > 1:
            root /= f'rank_{self.rank:04d}'
        self.writer = DiagnosticWriter(root, manifest, snapshot, self.reserve_gib,
                                       self.capacity_margin, estimated)
        self.estimated_remaining = estimated
        runner.logger.info(f'ACL diagnostics: {root}; conservative uncompressed '
                           f'estimate={estimated / 1024**3:.2f} GiB, '
                           f'reserve={self.reserve_gib:.2f} GiB. Storage failures stop training.')
        self._install_hooks()

    def _estimate_capacity(self, runner, info):
        # Supported configs all use FixScaleResize. Account for a mixed-aspect
        # padded batch, ceil rounding through Swin, all candidates and layers.
        pipeline = runner.cfg.val_dataloader.dataset.pipeline
        resize = next((item for item in pipeline if item['type'] == 'FixScaleResize'), None)
        if resize is None:
            raise ValueError('Capacity estimate requires the baseline FixScaleResize pipeline')
        longest = max(resize['scale']) + 64  # conservative padding/rounding bound
        self.max_candidates = sum(math.ceil(longest / stride) ** 2 for stride in (8, 16, 32, 64))
        lengths, chunks = [], 1
        for text in {tuple(row['text']) if not isinstance(row['text'], str)
                     else row['text'] for row in info}:
            _, captions, _, _ = self.model.get_tokens_positive_and_prompts(
                text, custom_entities=True)
            captions = captions if isinstance(captions, list) else [captions]
            chunks = max(chunks, len(captions))
            for caption in captions:
                tokens = self.model.language_model.tokenizer([caption], return_tensors='pt')
                lengths.append(min(tokens['input_ids'].shape[-1],
                                   self.model.language_model.max_tokens))
        tokens = max(lengths)
        dim = self.model.text_feat_map.out_features
        bert_dim = self.model.text_feat_map.in_features
        q, c = self.model.num_queries, len(self.classes)
        layers = self.model.decoder.num_layers
        enhancer_layers = len(self.model.encoder.text_layers)
        gt_count = max((len(rows) for rows in self.ground_truth.values()), default=0)
        retained = min(self.max_candidates, q + 128 + 5 * gt_count)
        detections = self.model.bbox_head.test_cfg.get('max_per_img', q)
        detection_bytes = detections * 26 + 2048
        internal_bytes = (detection_bytes + retained * (2 * c + 16) + q * 4 +
                          layers * q * (2 * c + 8) + enhancer_layers * c * dim * 2 +
                          tokens * (bert_dim + dim) * 2 + tokens ** 2 +
                          gt_count * 64 + 8192)
        # Conservatively budget token probes as all images. Real storage has
        # at most token_probe_images probes, shared BERT features and compression.
        if self.token_probe_images:
            internal_bytes += 2 * (q * tokens + tokens * dim)
        self.initial_rates = dict(detections=detection_bytes * chunks,
                                  internal=internal_bytes * chunks)
        return self._remaining_capacity(runner, 0, 0)

    def _remaining_capacity(self, runner, epoch, remaining_images):
        future = runner.max_epochs - epoch
        snapshots = sum(bool(snapshot_reasons(e, runner.max_epochs, self.internal_interval))
                        for e in range(epoch + 1, runner.max_epochs + 1))
        if self.current_stage == 1:
            snapshots = min(future, snapshots + 1)  # one possible progressive transition
        rates = {key: self.measured_rates.get(key, value) for key, value in self.initial_rates.items()}
        current = getattr(self, 'capture_level', 'detections')
        return int(remaining_images * rates[current] + self.epoch_samples *
                   (snapshots * rates['internal'] + (future - snapshots) * rates['detections']))

    def _install_hooks(self):
        def tokenizer(module, args):
            diagnostic_event('tokenizer', inputs=args[0])
        def language(module, args, output):
            diagnostic_event('language', outputs=output)
        def feature_map(module, args, output):
            diagnostic_event('feat_map', before=args[0], after=output)
        def head(module, args, output):
            diagnostic_event('head', outputs=output)
        self.handles.append(self.model.language_model.language_backbone.register_forward_pre_hook(tokenizer))
        self.handles.append(self.model.language_model.register_forward_hook(language))
        self.handles.append(self.model.text_feat_map.register_forward_hook(feature_map))
        for i, layer in enumerate(self.model.encoder.text_layers):
            def text_layer(module, args, output, final=i == len(self.model.encoder.text_layers) - 1):
                diagnostic_event('fe_layer', outputs=output, final_layer=final)
            self.handles.append(layer.register_forward_hook(text_layer))
        self.handles.append(self.model.bbox_head.register_forward_hook(head))

    def before_train_epoch(self, runner):
        stage = 2 if self.stage_hook._stage2_started else 1
        if stage != self.current_stage:
            self.writer.append('stage_events.jsonl', dict(
                epoch=runner.epoch + 1, step=runner.iter, previous_stage=self.current_stage,
                stage=stage, timing='before_train_epoch'))
        self.current_stage = stage
        self.epoch_loss_sums = {}
        self.epoch_train_steps = 0
        self.epoch_grad_sum = 0.
        self.epoch_grad_count = 0
        self.writer.check_space(self.estimated_remaining)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.step_lrs = [float(group['lr']) for group in runner.optim_wrapper.optimizer.param_groups]
        self.step_stage = 2 if self.stage_hook._stage2_started else 1

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # Aggregate every iteration, even when LoggerHook ignores the final
        # partial logging window. No optimizer/gradient operation is added.
        if not hasattr(self, 'epoch_loss_sums'):
            self.epoch_loss_sums, self.epoch_train_steps = {}, 0
            self.epoch_grad_sum, self.epoch_grad_count = 0., 0
        self.epoch_train_steps += 1
        for name, value in (outputs or {}).items():
            self.epoch_loss_sums[name] = self.epoch_loss_sums.get(name, 0.) + float(value)
        norm = runner.message_hub.log_scalars.get('train/grad_norm')
        if norm is not None:
            self.epoch_grad_sum += float(norm.current())
            self.epoch_grad_count += 1
        interval = self.logger.interval
        log_now = ((batch_idx + 1) % interval == 0 or
                   (batch_idx + 1 == len(runner.train_dataloader) and
                    (not self.logger.ignore_last or len(runner.train_dataloader) <= interval)))
        if not log_now:
            return
        scalars = runner.message_hub.log_scalars
        means = {key.removeprefix('train/'): float(value.mean(
            runner.log_processor.window_size)) for key, value in scalars.items()
                 if key.startswith('train/') and 'loss' in key}
        norm = scalars.get('train/grad_norm')
        self.writer.append('train.jsonl', dict(
            epoch=runner.epoch + 1, step=runner.iter + 1, epoch_step=batch_idx + 1,
            stage=self.step_stage, losses={k: float(v) for k, v in (outputs or {}).items()},
            loss_window_mean=means, lr=self.step_lrs,
            grad_norm=None if norm is None else float(norm.current()),
            grad_norm_reason='not_available' if norm is None else None))

    def after_train_epoch(self, runner):
        self.writer.append('train_epochs.jsonl', dict(
            epoch=runner.epoch + 1, step=runner.iter, stage=self.step_stage,
            steps=self.epoch_train_steps,
            losses={key: value / self.epoch_train_steps for key, value in self.epoch_loss_sums.items()},
            lr=[float(group['lr']) for group in runner.optim_wrapper.optimizer.param_groups],
            grad_norm_mean=(self.epoch_grad_sum / self.epoch_grad_count
                            if self.epoch_grad_count else None)))

    def before_val_epoch(self, runner):
        # EpochBasedTrainLoop increments epoch before invoking validation.
        self.epoch = runner.epoch
        self.current_stage = 2 if self.stage_hook._stage2_started else 1
        self.snapshot_reasons = snapshot_reasons(
            self.epoch, runner.max_epochs, self.internal_interval,
            self.current_stage != self.last_validation_stage)
        self.last_validation_stage = self.current_stage
        self.capture_level = 'internal' if self.snapshot_reasons else 'detections'
        self.observed_ids = set()
        self.observed_samples = 0
        self.epoch_observations = 0
        self.epoch_start_bytes = self.writer.written_bytes
        self.writer.check_space(self.estimated_remaining)

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        if self.token is not None:
            raise RuntimeError('ACL diagnostic capture was not closed')
        self.capture = ACLBatchCapture(
            level=self.capture_level, num_queries=self.model.num_queries,
            logit_width=self.model.bbox_head.max_text_len,
            ground_truth=self.ground_truth, category_map=self.category_map, probe_ids=self.probe_ids)
        self.token = activate_diagnostics(self.capture)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        deactivate_diagnostics(self.token)
        self.token = None
        records = self.capture.finish()
        self.capture = None
        # Confirm chunk concatenation and global class offsets against the
        # runner's actual final outputs, without attaching fields to them.
        for image_position, output in enumerate(outputs):
            matching = [row for row in records
                        if row['image_batch_position'] == image_position]
            prediction = output.pred_instances
            for stored, actual in [('det_boxes', prediction.bboxes),
                                   ('det_scores', prediction.scores),
                                   ('det_class_index', prediction.labels)]:
                np.testing.assert_array_equal(
                    np.concatenate([row['arrays'][stored] for row in matching]),
                    actual.detach().cpu().numpy())
        for row in records:
            if row['text']:
                row['tokens'] = self.model.language_model.tokenizer.convert_ids_to_tokens(
                    row['text']['input_ids'].tolist())
            self.observed_ids.add(row['metadata']['img_id'])
        self.writer.write_batch(self.epoch, batch_idx, records,
                                runner.timestamp, runner.iter, self.rank)
        self.observed_samples += len(outputs)
        self.epoch_observations += len(records)
        byte_count = self.writer.written_bytes - self.epoch_start_bytes
        measured = byte_count / self.observed_samples
        raw = sum(value.nbytes for row in records for value in
                  list(row['arrays'].values()) + list(row['text'].values())) + 2048 * len(outputs)
        batch_bytes = self.writer.last_batch_bytes
        ratio = min(1.1, batch_bytes / max(1, raw))
        projected = max(measured, self.initial_rates[self.capture_level] * ratio)
        self.measured_rates[self.capture_level] = max(
            self.measured_rates.get(self.capture_level, 0.), projected)
        self.estimated_remaining = self._remaining_capacity(
            runner, self.epoch, self.epoch_samples - self.observed_samples)
        self.writer.check_space(self.estimated_remaining)

    def after_val_epoch(self, runner, metrics=None):
        if self.observed_samples != self.epoch_samples:
            raise RuntimeError('Validation diagnostic sample count mismatch')
        if self.world_size == 1 and self.observed_ids != self.expected_ids:
            raise RuntimeError('Validation diagnostics did not cover every dataset image')
        per_class = []
        metrics = metrics or {}
        for entry in self.classes:
            row = dict(entry, reasons={})
            for name in ('AP', 'AP50', 'AP75'):
                suffix = f'/bbox_class_{entry["category_id"]}_{name}'
                key = next((key for key in metrics if key.endswith(suffix)), None)
                value = None if key is None else float(metrics[key])
                row[name] = value
                if value is None or not math.isfinite(value):
                    threshold = {'AP50': .5, 'AP75': .75}.get(name)
                    row['reasons'][name] = (
                        'metric_unavailable' if key is None else
                        'iou_threshold_unavailable' if threshold is not None and not np.any(
                            np.isclose(self.iou_thrs, threshold)) else 'no_evaluable_ground_truth')
            per_class.append(row)
        self.writer.append('metrics.jsonl', dict(
            epoch=self.epoch, step=runner.iter, metrics=metrics, per_class=per_class,
            observations=self.epoch_observations, unique_images=len(self.observed_ids),
            tensor_bytes=self.writer.written_bytes - self.epoch_start_bytes,
            stage=self.current_stage, capture_level=self.capture_level,
            snapshot_reasons=self.snapshot_reasons,
            estimated_remaining_bytes=self.estimated_remaining))
        self.writer.atomic_json(f'epoch_{self.epoch:03d}/complete.json', dict(
            epoch=self.epoch, samples=self.observed_samples,
            observations=self.epoch_observations, unique_images=len(self.observed_ids),
            capture_level=self.capture_level, snapshot_reasons=self.snapshot_reasons))
        self.completed_epochs += 1
        self.writer.check_space(self.estimated_remaining)

    def after_run(self, runner):
        if self.token is not None:
            deactivate_diagnostics(self.token)
            self.token = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.writer is not None:
            if self.completed_epochs != runner.max_epochs:
                raise RuntimeError('ACL diagnostic run ended before all validation epochs completed')
            self.writer.finish()
