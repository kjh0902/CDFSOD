# Copyright (c) OpenMMLab. All rights reserved.
"""Object-wise, all-class raw-dot-product alignment after query projection."""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.ops import roi_align


def class_name_prototypes(memory_text, class_token_maps, text_token_mask):
    """Raw mean of each class's name tokens in the final FE output.

    Maps use ACL's one-based class keys; dataset GT labels are zero-based.
    Neither special tokens nor padding is pooled. Missing/truncated classes
    are errors, since silently dropping one would change the negative set.
    """
    if len(class_token_maps) != memory_text.shape[0]:
        raise ValueError('Expected one all-class token map per image.')
    prototypes = []
    for image, mapping in enumerate(class_token_maps):
        if not mapping or set(mapping) != set(range(1, len(mapping) + 1)):
            raise ValueError('Expected every class key in 1..C.')
        per_image = []
        for class_id in range(1, len(mapping) + 1):
            indices = sorted(set(mapping[class_id]))
            if (not indices or min(indices) < 0
                    or max(indices) >= memory_text.shape[1]
                    or not text_token_mask[image, indices].all()):
                raise ValueError('Every class needs valid, untruncated name tokens.')
            per_image.append(memory_text[image, indices].float().mean(dim=0))
        prototypes.append(torch.stack(per_image))
    return prototypes


def pool_gt_regions(output_memory, spatial_shapes, level_start_index,
                    batch_data_samples, featmap_strides, roi_size=7):
    """RoIAlign every object at every level, then mean over space and levels.

    GT xyxy boxes are already in augmented/resized input pixel coordinates.
    Use the actual feature strides (Swin-B/ChannelMapper: 8,16,32,64), not
    img_shape ratios: the latter stretch boxes in padded or rounded-up maps.
    Averaging levels is parameter-free and preserves one anchor per object.
    """
    shapes = spatial_shapes.tolist()
    starts = level_start_index.tolist()
    if len(shapes) != len(starts) or len(shapes) != len(featmap_strides):
        raise ValueError('Shapes, level starts and feature strides must agree.')
    expected_start = 0
    for (height, width), start in zip(shapes, starts):
        if start != expected_start or height <= 0 or width <= 0:
            raise ValueError('Invalid flattened feature layout.')
        expected_start += height * width
    if expected_start != output_memory.shape[1]:
        raise ValueError('Spatial shapes do not cover output_memory.')
    if len(batch_data_samples) != output_memory.shape[0]:
        raise ValueError('Expected one data sample per image.')

    rois, labels, image_ids = [], [], []
    for image, sample in enumerate(batch_data_samples):
        boxes = sample.gt_instances.bboxes
        # Accept both Tensor boxes and MMDetection HorizontalBoxes.
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        boxes = boxes.to(device=output_memory.device, dtype=torch.float32)
        targets = sample.gt_instances.labels.to(output_memory.device)
        if (boxes.shape != (targets.numel(), 4) or not torch.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError('GT boxes must be finite, nondegenerate xyxy boxes.')
        rois.append(torch.cat([boxes.new_full((len(boxes), 1), image), boxes], 1))
        labels.append(targets)
        image_ids.append(targets.new_full((len(boxes),), image))
    rois = torch.cat(rois)
    labels = torch.cat(labels)
    image_ids = torch.cat(image_ids)
    if len(rois) == 0:
        return output_memory.new_empty((0, output_memory.shape[-1])), labels, image_ids

    pooled_levels = []
    for (height, width), start, stride in zip(shapes, starts, featmap_strides):
        feature = output_memory[:, start:start + height * width]
        feature = feature.transpose(1, 2).reshape(
            output_memory.shape[0], output_memory.shape[2], height, width)
        # FP32 avoids half precision overflow without changing raw-dot geometry.
        regions = roi_align(feature.float(), rois, output_size=roi_size,
                            spatial_scale=1.0 / stride, sampling_ratio=2,
                            aligned=True)
        pooled_levels.append(regions.mean(dim=(-2, -1)))
    return torch.stack(pooled_levels).mean(dim=0), labels, image_ids


def region_text_loss(output_memory, memory_text, spatial_shapes,
                     level_start_index, text_token_mask, class_token_maps,
                     batch_data_samples, featmap_strides=(8, 16, 32, 64),
                     roi_size=7):
    """Mean CE over GT objects, using all dataset classes for each image.

    No normalization, temperature, projection, class balancing or stop-gradient.
    For DDP, normalize by the global GT count (with world-size compensation for
    DDP's gradient averaging). Empty ranks still participate in the reduction.
    """
    with torch.autocast(device_type=output_memory.device.type, enabled=False):
        prototypes = class_name_prototypes(
            memory_text, class_token_maps, text_token_mask)
        objects, labels, image_ids = pool_gt_regions(
            output_memory, spatial_shapes, level_start_index, batch_data_samples,
            featmap_strides, roi_size)
        # Keep both FE branches in the graph even for a completely empty batch.
        total = output_memory.float().sum() * 0 + memory_text.float().sum() * 0
        for image, text in enumerate(prototypes):
            selected = image_ids == image
            if selected.any():
                targets = labels[selected]
                if (targets.dtype != torch.long
                        or ((targets < 0) | (targets >= len(text))).any()):
                    raise ValueError('GT class IDs must index all-class prototypes.')
                logits = objects[selected] @ text.transpose(0, 1)
                total = total + F.cross_entropy(logits, targets, reduction='sum')
        count = total.new_tensor(labels.numel())
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(count)
            count = count / world_size
        # For an empty local/global batch, total remains a differentiable zero.
        return total / count.clamp(min=1.0 / world_size)
