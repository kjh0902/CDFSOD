"""Validation-only tensor capture. Never retain autograd graphs/GPU states."""
import numpy as np
import torch

from .acl_diagnostic_io import pack_mask


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
    def __init__(self):
        self.records = []
        self.current = None

    def _finish_forward(self):
        if self.current is not None:
            for row in self.current:
                required = {'encoder_boxes', 'encoder_class_scores', 'selected_indices',
                            'decoder_token_logits', 'decoder_class_scores',
                            'decoder_boxes', 'det_query_index', 'det_boxes'}
                if not required <= row['arrays'].keys() or not row['fe']:
                    raise RuntimeError('Incomplete ACL diagnostic forward capture')
                row['arrays']['fe_text'] = np.stack(row.pop('fe'))
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
                    chunk=int(data['chunk']), arrays={}, text={}, fe=[]))
        elif self.current is None:
            raise RuntimeError(f'ACL diagnostic event {event} arrived before prompt')
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
                row['fe'].append(numpy_copy(data['outputs'][index, :row['token_length']],
                                            np.float16))
        elif event == 'candidates':
            shapes = numpy_copy(data['spatial_shapes'], np.int32)
            n = int(np.prod(shapes, axis=-1).sum())
            for index, row in enumerate(self.current):
                arrays = row['arrays']
                logits = data['logits'][index].detach()
                maximum, token_index = logits.max(-1)
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
                arrays['decoder_token_logits'] = numpy_copy(
                    logits[:, index, :, :row['token_length']], np.float16)
                arrays['decoder_class_scores'] = numpy_copy(
                    class_scores(logits[:, index].detach(), row['class_mapping']), np.float16)
                arrays['decoder_boxes'] = numpy_copy(boxes[:, index], np.float16)
                row['original_logit_width'] = logits.shape[-1]
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
