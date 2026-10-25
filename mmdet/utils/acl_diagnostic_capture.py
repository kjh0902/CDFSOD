"""Validation-only tensor capture. Never retain autograd graphs/GPU states."""
import numpy as np
import torch

from .acl_diagnostic_io import compact_uint, pack_mask, rectangular_padding


def numpy_copy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device='cpu').numpy()
    return np.array(value, dtype=dtype, copy=True)


def class_scores(logits, mapping):
    """Same mean(sigmoid(token logits)) as Grounding DINO, no NxT copy.

    Reduction is before storage quantization, and includes every class.
    """
    if mapping is None:
        return logits.sigmoid().float()
    result = torch.zeros((*logits.shape[:-1], len(mapping)),
                         device=logits.device, dtype=torch.float32)
    for label, tokens in mapping.items():
        result[..., int(label) - 1] = logits[..., tokens].sigmoid().mean(-1)
    return result


class ACLBatchCapture:
    def __init__(self, level='full', num_queries=None, logit_width=256,
                 ground_truth=None, category_map=None, probe_ids=(),
                 boundary_candidates=128, oracle_candidates=5):
        if level not in ('full', 'internal', 'detections'):
            raise ValueError('Unknown ACL capture level')
        self.level = level
        self.num_queries = num_queries
        self.logit_width = logit_width
        self.ground_truth = ground_truth or {}
        self.category_map = category_map or {}
        self.probe_ids = set(probe_ids)
        self.boundary_candidates = boundary_candidates
        self.oracle_candidates = oracle_candidates
        self.records = []
        self.current = None

    def _finish_forward(self):
        if self.current is not None:
            for row in self.current:
                required = {'det_query_index', 'det_boxes', 'det_scores', 'det_class_index'}
                if self.level != 'detections':
                    required |= {'encoder_boxes', 'encoder_class_scores', 'selected_indices',
                                 'decoder_class_scores', 'decoder_boxes'}
                if not required <= row['arrays'].keys() or (self.level != 'detections' and not row['fe']):
                    raise RuntimeError('Incomplete ACL diagnostic forward capture')
                if self.level != 'detections':
                    key = 'fe_text' if self.level == 'full' else 'fe_class_prototypes'
                    row['arrays'][key] = np.stack(row['fe'])
                row.pop('fe')
                if self.level != 'full':
                    arrays = row['arrays']
                    # Exact integer identities, checked against native top-k
                    # before discarding duplicate local/flat index arrays.
                    local = arrays['det_class_index_local'].astype(np.int64)
                    query = arrays['det_query_index'].astype(np.int64)
                    classes = len(row['class_mapping']) if row['class_mapping'] is not None else None
                    expected = query * classes + local if classes is not None else query
                    np.testing.assert_array_equal(expected, arrays['det_flat_topk_index'])
                    del arrays['det_flat_topk_index'], arrays['det_class_index_local']
                    for key in ('det_query_index', 'det_class_index', 'selected_indices',
                                'encoder_candidate_indices', 'selection_argmax_token'):
                        if key in arrays:
                            arrays[key] = compact_uint(arrays[key])
                self.records.append(row)
        self.current = None

    @torch.no_grad()
    def __call__(self, event, **data):
        if event == 'prompt':
            self._finish_forward()
            self.current = []
            for index, sample in enumerate(data['samples']):
                mapping = data['mappings'][index]
                self.current.append(dict(
                    metadata=dict(sample.metainfo), caption=data['captions'][index],
                    image_batch_position=index,
                    class_mapping=mapping, class_offset=int(data['class_offset']),
                    chunk=int(data['chunk']), arrays={}, text={}, fe=[],
                    capture_level=self.level, num_queries=self.num_queries,
                    original_logit_width=self.logit_width,
                    token_probe=self.level == 'internal' and sample.metainfo['img_id'] in self.probe_ids))
        elif self.current is None:
            raise RuntimeError(f'ACL diagnostic event {event} arrived before prompt')
        elif self.level == 'detections' and event != 'detections':
            return  # No reductions, feature copies or internal GPU->CPU transfer.
        elif event == 'tokenizer':
            value = data['inputs']
            for index, row in enumerate(self.current):
                for name in ('input_ids', 'position_ids', 'token_type_ids'):
                    if value.get(name) is not None:
                        row['text'][name] = numpy_copy(value[name][index], np.int32)
                row['text']['text_attention_mask'] = numpy_copy(
                    value['attention_mask'][index], bool)
        elif event == 'language':
            for index, row in enumerate(self.current):
                mask = data['outputs']['text_token_mask'][index]
                length = int(mask.sum().item())
                row['token_length'] = length
                row['text']['text_token_mask'] = numpy_copy(mask[:length], bool)
                for name in ('input_ids', 'position_ids', 'token_type_ids'):
                    if name in row['text']:
                        row['text'][name] = row['text'][name][:length]
                mask = row['text']['text_attention_mask']
                row['text']['text_attention_mask'] = (
                    mask[:length, :length] if mask.ndim == 2 else mask[:length])
        elif event == 'feat_map':
            for index, row in enumerate(self.current):
                length = row['token_length']
                row['text']['bert_before_map'] = numpy_copy(
                    data['before'][index, :length], np.float16)
                row['text']['bert_after_map'] = numpy_copy(
                    data['after'][index, :length], np.float16)
        elif event == 'fe_layer':
            for index, row in enumerate(self.current):
                features = data['outputs'][index, :row['token_length']].detach()
                if self.level == 'internal':
                    features = torch.stack([features[tokens].mean(0)
                                            for _, tokens in sorted(row['class_mapping'].items())])
                    if row['token_probe'] and data.get('final_layer', False):
                        row['arrays']['fe_final_text'] = numpy_copy(
                            data['outputs'][index, :row['token_length']], np.float16)
                row['fe'].append(numpy_copy(features, np.float16))
        elif event == 'candidates':
            shapes = numpy_copy(data['spatial_shapes'], np.int32)
            n = int(np.prod(shapes, axis=-1).sum())
            for index, row in enumerate(self.current):
                arrays = row['arrays']
                logits = data['logits'][index].detach()
                maximum, token_index = logits.max(-1)
                if self.level == 'internal':
                    self._compact_candidates(row, data, index, shapes, maximum, token_index)
                    continue
                arrays['encoder_class_scores'] = numpy_copy(
                    class_scores(logits, row['class_mapping']), np.float16)
                arrays['encoder_boxes'] = numpy_copy(
                    data['boxes_unact'][index].detach().sigmoid(), np.float16)
                arrays['selection_max_token_logit'] = numpy_copy(maximum, np.float32)
                arrays['selection_argmax_token'] = numpy_copy(token_index, np.uint16)
                arrays['selected_indices'] = numpy_copy(data['selected'][index], np.int32)
                arrays['spatial_shapes'] = shapes.copy()
                padding = (np.zeros(n, dtype=bool) if data['memory_mask'] is None
                           else numpy_copy(data['memory_mask'][index], bool))
                arrays['padding_mask'] = pack_mask(padding)
                arrays['proposal_valid'] = pack_mask(numpy_copy(
                    torch.isfinite(data['base_proposals'][index]).all(-1), bool))
                valid_hw, offset = [], 0
                for height, width in shapes:
                    mask = padding[offset:offset + height * width].reshape(height, width)
                    valid_hw.append([(~mask[:, 0]).sum(), (~mask[0]).sum()])
                    offset += height * width
                arrays['valid_hw'] = np.asarray(valid_hw, dtype=np.int32)
        elif event == 'head':
            logits, boxes = data['outputs']
            for index, row in enumerate(self.current):
                arrays = row['arrays']
                if self.level == 'full':
                    arrays['decoder_token_logits'] = numpy_copy(
                        logits[:, index, :, :row['token_length']], np.float16)
                elif row['token_probe']:
                    arrays['decoder_final_token_logits'] = numpy_copy(
                        logits[-1, index, :, :row['token_length']], np.float16)
                arrays['decoder_class_scores'] = numpy_copy(
                    class_scores(logits[:, index].detach(), row['class_mapping']), np.float16)
                arrays['decoder_boxes'] = numpy_copy(boxes[:, index], np.float16)
                row['original_logit_width'] = logits.shape[-1]
                row['num_queries'] = boxes.shape[2]
        elif event == 'detections':
            # The head processes images in order; never match scores/boxes to
            # infer origins (ties and multiple classes per query are legal).
            row = next(row for row in self.current
                       if 'det_query_index' not in row['arrays'])
            arrays = row['arrays']
            arrays['det_query_index'] = numpy_copy(data['query_index'], np.int32)
            arrays['det_class_index_local'] = numpy_copy(data['class_index'], np.int32)
            arrays['det_class_index'] = arrays['det_class_index_local'] + row['class_offset']
            arrays['det_flat_topk_index'] = numpy_copy(data['flat_index'], np.int32)
            arrays['det_boxes'] = numpy_copy(data['results'].bboxes, np.float32)
            arrays['det_scores'] = numpy_copy(data['results'].scores, np.float32)
        else:
            raise ValueError(f'Unknown ACL diagnostic event {event}')

    def finish(self):
        self._finish_forward()
        records, self.records = self.records, []
        return records

    def _compact_candidates(self, row, data, index, shapes, maximum, token_index):
        arrays = row['arrays']
        logits = data['logits'][index].detach()
        boxes = data['boxes_unact'][index].detach().sigmoid()
        scores = class_scores(logits, row['class_mapping'])
        selected = data['selected'][index]
        n = len(boxes)
        valid = torch.isfinite(data['base_proposals'][index]).all(-1)
        valid_ids = torch.nonzero(valid).flatten()
        unselected = torch.ones(n, dtype=torch.bool, device=boxes.device)
        unselected[selected] = False
        boundary_count = min(self.boundary_candidates, n - len(selected))
        boundary = maximum.masked_fill(~unselected, -torch.inf).topk(boundary_count).indices
        offset, classes = row['class_offset'], scores.shape[-1]
        gt_rows = [gt for gt in self.ground_truth.get(row['metadata']['img_id'], [])
                   if not gt.get('ignore') and not gt.get('iscrowd') and
                   gt['category_id'] in self.category_map and
                   offset <= self.category_map[gt['category_id']] < offset + classes]
        gt = boxes.new_tensor([entry['bbox'] for entry in gt_rows]).reshape(-1, 4)
        gt[:, 2:] += gt[:, :2]
        gt_classes = torch.tensor([self.category_map[entry['category_id']] - offset
                                   for entry in gt_rows], device=boxes.device, dtype=torch.long)
        h, w = row['metadata']['img_shape'][:2]
        scale = boxes.new_tensor([w, h, w, h]) / boxes.new_tensor(row['metadata']['scale_factor']).repeat(2)
        xyxy = torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2], -1) * scale
        g = len(gt)
        best = boxes.new_zeros(g)
        best_selected = boxes.new_zeros(g)
        best_class = boxes.new_zeros((2, g))
        counts = torch.zeros(2, g, device=boxes.device, dtype=torch.int64)
        top_values = boxes.new_full((0, g), -1.)
        top_indices = torch.empty(0, g, device=boxes.device, dtype=torch.long)
        k = min(self.oracle_candidates, len(valid_ids))
        gt_area = (gt[:, 2:] - gt[:, :2]).clamp(min=0).prod(-1)
        for start in range(0, len(valid_ids), 2048):
            ids = valid_ids[start:start + 2048]
            if not g:
                break
            chunk = xyxy[ids]
            intersection = (torch.minimum(chunk[:, None, 2:], gt[None, :, 2:]) -
                            torch.maximum(chunk[:, None, :2], gt[None, :, :2])).clamp(min=0).prod(-1)
            area = (chunk[:, 2:] - chunk[:, :2]).clamp(min=0).prod(-1)
            iou = intersection / (area[:, None] + gt_area[None] - intersection).clamp(min=1e-12)
            best = torch.maximum(best, iou.max(0).values)
            selected_iou = iou.masked_fill(unselected[ids, None], 0.)
            best_selected = torch.maximum(best_selected, selected_iou.max(0).values)
            for t, threshold in enumerate((.5, .75)):
                eligible = iou >= threshold
                counts[t] += eligible.sum(0)
                class_score = scores[ids][:, gt_classes].masked_fill(~eligible, 0.)
                best_class[t] = torch.maximum(best_class[t], class_score.max(0).values)
            merged_values = torch.cat([top_values, iou])
            merged_indices = torch.cat([top_indices, ids[:, None].expand(-1, g)])
            top_values, locations = merged_values.topk(min(k, len(merged_values)), dim=0)
            top_indices = merged_indices.gather(0, locations)
        retained = torch.unique(torch.cat([selected, boundary, top_indices.flatten()]), sorted=True)
        arrays['encoder_candidate_indices'] = numpy_copy(retained)
        arrays['encoder_boxes'] = numpy_copy(boxes[retained], np.float16)
        arrays['encoder_class_scores'] = numpy_copy(scores[retained], np.float16)
        arrays['selection_max_token_logit'] = numpy_copy(maximum[retained], np.float32)
        arrays['selection_argmax_token'] = numpy_copy(token_index[retained])
        arrays['selected_indices'] = numpy_copy(selected)
        finite = maximum[torch.isfinite(maximum)]
        arrays['candidate_max_logit_quantiles'] = numpy_copy(
            torch.quantile(finite.float(), finite.new_tensor([0., .25, .5, .75, .9, .99, 1.]).float())
            if len(finite) else maximum.new_full((7,), -torch.inf), np.float32)
        arrays['gt_annotation_ids'] = np.asarray([entry['id'] for entry in gt_rows], dtype=np.int64)
        arrays['gt_candidate_iou_max'] = numpy_copy(best, np.float32)
        arrays['gt_selected_iou_max'] = numpy_copy(best_selected, np.float32)
        arrays['gt_candidate_class_score_iou50_75'] = numpy_copy(best_class, np.float16)
        arrays['gt_candidate_count_iou50_75'] = numpy_copy(counts, np.uint32)
        arrays['spatial_shapes'] = shapes.copy()
        padding = (np.zeros(n, dtype=bool) if data['memory_mask'] is None
                   else numpy_copy(data['memory_mask'][index], bool))
        valid_hw, cursor = [], 0
        for height, width in shapes:
            mask = padding[cursor:cursor + height * width].reshape(height, width)
            valid_hw.append([(~mask[:, 0]).sum(), (~mask[0]).sum()])
            cursor += height * width
        arrays['valid_hw'] = np.asarray(valid_hw, dtype=np.int32)
        if not np.array_equal(padding, rectangular_padding(shapes, arrays['valid_hw'])):
            arrays['padding_mask'] = pack_mask(padding)
