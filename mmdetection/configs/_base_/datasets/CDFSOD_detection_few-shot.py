# Dataset settings for the NEU-DET GroundingDINO baseline.
#
# Expected layout from the repository root:
# datasets/
#   NEU-DET/
#     annotations/
#       train.json
#       test.json
#     train/
#     test/

dataset_type = 'CocoDataset'
data_root = 'datasets/NEU-DET/'
backend_args = None

metainfo = dict(
    classes=(
        'crazing',
        'inclusion',
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches',
    ))

train_ann_file = 'annotations/train.json'
val_ann_file = 'annotations/test.json'
test_ann_file = 'annotations/test.json'

neu_det_domain_attribute = (
    'gray-scale hot-rolled steel surface with metallic texture, low color '
    'variation, and subtle industrial defect patterns')

enriched_text_cfg = dict(
    enabled=True,
    domain_attribute=neu_det_domain_attribute,
    model_id='Salesforce/blip-image-captioning-base',
    support_ann_file='auto',
    support_img_prefix='train/',
    device='auto',
    log_progress=True)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
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
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False),
        enriched_text_cfg=enriched_text_cfg,
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
        ann_file=val_ann_file,
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        enriched_text_cfg=enriched_text_cfg,
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
        ann_file=test_ann_file,
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        enriched_text_cfg=enriched_text_cfg,
        return_classes=True))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + test_ann_file,
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args)
