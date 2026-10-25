#!/usr/bin/env python3
"""Compare real compressed v1/v2 files from identical synthetic tensors.

This is a CPU storage benchmark, not a trained-model or GPU accuracy test.
The historical capture/writer are loaded verbatim from commit 09efa0c.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = '09efa0ca01319a09d269727682836b476c191f7d'


def load_modules(legacy=False):
    name = '_acl_benchmark_v1' if legacy else '_acl_benchmark_v2'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / 'mmdet/utils')]
    sys.modules[name] = package
    modules = []
    for filename in ('acl_diagnostic_io', 'acl_diagnostic_capture'):
        qualified = name + '.' + filename
        if legacy:
            source = subprocess.check_output(['git', 'show', f'{BASE}:mmdet/utils/{filename}.py'],
                                              cwd=ROOT).decode('utf-8')
            module = types.ModuleType(qualified)
            module.__file__ = str(ROOT / f'mmdet/utils/{filename}.py')
            module.__package__ = name
            sys.modules[qualified] = module
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
        else:
            module = importlib.import_module(qualified)
        modules.append(module)
    return modules


def synthetic_payload(epoch, image, candidates=20000, queries=900, tokens=26,
                      classes=6, dimension=256, detections=300):
    io, _ = load_modules_cached()
    height = math.isqrt(candidates)
    while candidates % height:
        height -= 1
    shapes = np.array([[height, candidates // height]], dtype=np.int32)
    anchors, valid = io.reconstruct_proposals(shapes, shapes)
    rng = torch.Generator().manual_seed(100000 * epoch + image)
    logits = torch.randn(1, candidates, tokens, generator=rng) * 2 - 5
    logits[:, ~torch.from_numpy(valid)] = -12
    base = torch.logit(torch.from_numpy(anchors)).unsqueeze(0)
    boxes_unact = base + torch.randn(1, candidates, 4, generator=rng) * .5
    selected = logits.max(-1).values.topk(queries, dim=1).indices
    query_boxes = boxes_unact[0, selected[0]].sigmoid()
    decoder_logits = torch.randn(6, 1, queries, tokens, generator=rng) * 2 - 5
    decoder_boxes = query_boxes[None, None].repeat(6, 1, 1, 1)
    mapping = {i + 1: [2 * i + 1, 2 * i + 2] for i in range(classes)}
    scores = torch.stack([decoder_logits[-1, 0, :, value].sigmoid().mean(-1)
                          for value in mapping.values()], -1)
    det_scores, flat = scores.flatten().topk(min(detections, queries * classes))
    query = flat // classes
    chosen = query_boxes[query]
    det_boxes = torch.cat([chosen[:, :2] - chosen[:, 2:] / 2,
                           chosen[:, :2] + chosen[:, 2:] / 2], -1)
    det_boxes *= torch.tensor([1333, 800, 1333, 800])
    det_boxes[:, 0::2].clamp_(0, 1333)
    det_boxes[:, 1::2].clamp_(0, 800)
    prediction = types.SimpleNamespace(bboxes=det_boxes, scores=det_scores, labels=flat % classes)
    metadata = dict(img_id=image, img_path=f'{image}.jpg', ori_shape=(800, 1333),
                    img_shape=(800, 1333), scale_factor=(1., 1.), batch_input_shape=(800, 1333))
    text_rng = torch.Generator().manual_seed(epoch)
    before = torch.randn(1, tokens, 768, generator=text_rng)
    after = torch.randn(1, tokens, dimension, generator=text_rng)
    ids = torch.arange(tokens).unsqueeze(0)
    events = [
        ('prompt', dict(samples=[types.SimpleNamespace(metainfo=metadata)],
                        captions=['class prompt'], mappings=[mapping], chunk=0, class_offset=0)),
        ('tokenizer', dict(inputs=dict(input_ids=ids, position_ids=ids,
                                      token_type_ids=torch.zeros_like(ids),
                                      attention_mask=torch.ones(1, tokens, tokens, dtype=torch.bool)))),
        ('language', dict(outputs=dict(text_token_mask=torch.ones(1, tokens, dtype=torch.bool)))),
        ('feat_map', dict(before=before, after=after))]
    for layer in range(6):
        events.append(('fe_layer', dict(outputs=after + torch.randn(
            1, tokens, dimension, generator=rng) * (.1 + layer * .1), final_layer=layer == 5)))
    events += [
        ('candidates', dict(logits=logits, boxes_unact=boxes_unact, selected=selected,
                            base_proposals=base, memory_mask=None, spatial_shapes=torch.from_numpy(shapes))),
        ('head', dict(outputs=(decoder_logits, decoder_boxes))),
        ('detections', dict(query_index=query, class_index=flat % classes, flat_index=flat,
                            results=prediction))]
    return events


_cached = None


def load_modules_cached():
    global _cached
    if _cached is None:
        _cached = load_modules()
    return _cached


def emit(capture, payload):
    for event, data in payload:
        capture(event, **data)
    rows = capture.finish()
    for row in rows:
        if row['text']:
            row['tokens'] = [f'token_{i}' for i in row['text']['input_ids'].tolist()]
    return rows


def write_legacy_batch(writer, epoch, batch, records):
    arrays, index = {}, []
    shard = f'epoch_{epoch:03d}/batch_{batch:06d}.npz'
    for position, row in enumerate(records):
        text_ref = writer.shared_text(epoch, row['text'], dict(
            caption=row['caption'], class_mapping=row['class_mapping'], tokens=row['tokens']))
        prefix = f'obs_{position:04d}__'
        arrays.update({prefix + key: value for key, value in row['arrays'].items()})
        index.append(dict(run_id='storage_benchmark', epoch=epoch, step=epoch,
            rank=0, batch=batch, batch_position=position,
            image_batch_position=row['image_batch_position'], img_id=row['metadata']['img_id'],
            chunk=row['chunk'], class_offset=row['class_offset'],
            image_meta=row['metadata'], shard=shard, prefix=prefix, text_ref=text_ref,
            original_logit_width=row['original_logit_width'],
            final_decoder_layer=row['arrays']['decoder_boxes'].shape[0] - 1))
    writer.save_arrays(shard, arrays)
    for record in index:
        writer.append(f'epoch_{epoch:03d}/index.jsonl', record)


def size_by_kind(root):
    result = {}
    for path in Path(root).rglob('*'):
        if path.is_file():
            kind = 'tensors' if path.suffix == '.npz' else 'metadata_and_scalars'
            result[kind] = result.get(kind, 0) + path.stat().st_size
    result['total'] = sum(result.values())
    return result


def benchmark(root, epochs=9, images=8, max_epochs=100, stage_epoch=7,
              candidates=20000, queries=900, tokens=26, classes=6, dimension=256):
    root = Path(root)
    legacy_io, legacy_cap = load_modules(legacy=True)
    io, capture = load_modules_cached()
    dataset = dict(categories=[dict(category_id=10 + c, class_index=c, class_name=f'class{c}')
                               for c in range(classes)],
                   images=[dict(id=i, width=1333, height=800) for i in range(images)],
                   annotations=[dict(id=i * 10 + g, image_id=i, category_id=10 + g % classes,
                                     bbox=[300 + g * 50, 200, 100, 100], area=10000, iscrowd=0)
                                for i in range(images) for g in range(3)])
    gt = {i: [row for row in dataset['annotations'] if row['image_id'] == i] for i in range(images)}
    categories = {10 + c: c for c in range(classes)}
    manifest = dict(run_id='synthetic_cpu_storage_benchmark', max_epochs=max_epochs)
    old = legacy_io.DiagnosticWriter(root / 'v1', manifest, dataset, reserve_gib=0)
    new = io.DiagnosticWriter(root / 'v2', dict(manifest, storage_policy=dict(
        internal_interval=5, token_probe_ids=io.select_token_probes(dataset))), dataset, reserve_gib=0)
    snapshots = []
    for epoch in range(1, epochs + 1):
        reasons = io.snapshot_reasons(epoch, max_epochs, stage_changed=epoch == stage_epoch)
        if reasons:
            snapshots.append(epoch)
        level = 'internal' if reasons else 'detections'
        for image in range(images):
            payload = synthetic_payload(epoch, image, candidates, queries, tokens, classes, dimension)
            rows = emit(legacy_cap.ACLBatchCapture(), payload)
            write_legacy_batch(old, epoch, image, rows)
            compact = capture.ACLBatchCapture(level=level, num_queries=queries, logit_width=tokens,
                ground_truth=gt, category_map=categories, probe_ids=io.select_token_probes(dataset))
            rows2 = emit(compact, payload)
            np.testing.assert_array_equal(rows[0]['arrays']['det_boxes'], rows2[0]['arrays']['det_boxes'])
            np.testing.assert_array_equal(rows[0]['arrays']['det_scores'], rows2[0]['arrays']['det_scores'])
            np.testing.assert_array_equal(rows[0]['arrays']['det_query_index'], rows2[0]['arrays']['det_query_index'])
            new.write_batch(epoch, image, rows2, 'storage_benchmark', epoch)
        for writer in (old, new):
            writer.append('metrics.jsonl', dict(epoch=epoch, metrics={'coco/bbox_mAP': .1}))
            writer.append('train_epochs.jsonl', dict(epoch=epoch, loss=1., lr=.0001))
            writer.atomic_json(f'epoch_{epoch:03d}/complete.json', dict(
                epoch=epoch, samples=images, observations=images, unique_images=images))
    if epochs == max_epochs:
        old.finish()
        new.finish()
        io.DiagnosticReader(root / 'v1').verify()
        io.DiagnosticReader(root / 'v2').verify()
    v1, v2 = size_by_kind(root / 'v1'), size_by_kind(root / 'v2')
    epoch_bytes = {version: {epoch: sum(path.stat().st_size for path in
        (root / version / f'epoch_{epoch:03d}').rglob('*') if path.is_file())
        for epoch in range(1, epochs + 1)} for version in ('v1', 'v2')}
    light = [size for epoch, size in epoch_bytes['v2'].items() if epoch not in snapshots]
    internal = [size for epoch, size in epoch_bytes['v2'].items() if epoch in snapshots]
    full_snapshots = sum(bool(io.snapshot_reasons(epoch, max_epochs,
        stage_changed=epoch == stage_epoch)) for epoch in range(1, max_epochs + 1))
    projected = None
    if light and internal:
        scalar_bytes = {version: sum((root / version / filename).stat().st_size
            for filename in ('metrics.jsonl', 'train_epochs.jsonl')) for version in ('v1', 'v2')}
        fixed = {version: totals['total'] - sum(epoch_bytes[version].values()) - scalar_bytes[version]
                 for version, totals in [('v1', v1), ('v2', v2)]}
        original = ((sum(epoch_bytes['v1'].values()) + scalar_bytes['v1']) *
                    max_epochs / epochs + fixed['v1'])
        compact = (np.mean(internal) * full_snapshots + np.mean(light) *
                   (max_epochs - full_snapshots) + scalar_bytes['v2'] * max_epochs / epochs + fixed['v2'])
        projected = dict(epochs=max_epochs, internal_epochs=full_snapshots,
            v1_bytes=int(original), v2_bytes=int(compact),
            reduction_percent=100 * (1 - compact / original),
            note='Extrapolated from measured synthetic snapshot/light epochs, not a full training run')
    result = dict(kind='synthetic_cpu_compressed_file_measurement', baseline_commit=BASE,
        parameters=dict(epochs=epochs, images=images, max_epochs=max_epochs, stage_epoch=stage_epoch,
                        candidates=candidates, queries=queries, tokens=tokens, classes=classes,
                        dimension=dimension), internal_snapshot_epochs=snapshots,
        v1_bytes=v1, v2_bytes=v2, reduction_percent=100 * (1 - v2['total'] / v1['total']),
        per_epoch_bytes=epoch_bytes, projected_full_run=projected,
        native_detection_arrays_equal=True)
    (root / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new output directory')
    for name, default in [('epochs', 9), ('images', 8), ('max-epochs', 100), ('stage-epoch', 7),
                          ('candidates', 20000), ('queries', 900), ('tokens', 26),
                          ('classes', 6), ('dimension', 256)]:
        parser.add_argument('--' + name, type=int, default=default)
    args = vars(parser.parse_args())
    output = args.pop('output')
    if not (1 <= args['epochs'] <= args['max_epochs'] and args['images'] > 0 and
            1 <= args['queries'] <= args['candidates'] and args['classes'] > 0 and
            args['dimension'] > 0 and args['tokens'] >= 2 * args['classes'] + 2):
        parser.error('Require positive sizes, epochs <= max-epochs, queries <= candidates and tokens >= 2*classes+2')
    torch.set_num_threads(1)
    print(json.dumps(benchmark(output, **args), indent=2))


if __name__ == '__main__':
    main()
