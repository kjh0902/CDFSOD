# Auto-generated config for UODD 1-shot

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/UODD/'

metainfo = dict(
    classes = ('seacucumber', 'seaurchin', 'scallop')
)

train_ann_file = 'annotations/1_shot.json'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0, prob=0.5),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.2),
        pad_val=(114, 114, 114),
        prob=0.2),
    dict(
        type='RandomChoiceResize',
        scales=[(512, 1333), (640, 1333), (768, 1333)],
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.5, 0.5),
        allow_negative_crop=False,
        prob=0.2),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        return_classes=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)
