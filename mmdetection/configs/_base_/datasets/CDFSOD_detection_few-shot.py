import os

# Dataset settings for the compact CDFSOD GroundingDINO baseline.
# Override with:
#   CDFSOD_DATA_ROOT=/path/to/datasets
#   CDFSOD_DATASET=NEU-DET
#   CDFSOD_TRAIN_ANN=annotations/1_shot.json
#   CDFSOD_CAPTION_FILE=annotations/1_shot_captions.json
#   CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES=1

dataset_type = 'CocoDataset'
datasets_root = os.getenv(
    'CDFSOD_DATA_ROOT',
    '/home/aislab5090/CDFSOD/junhyung/datasets')
dataset_name = os.getenv('CDFSOD_DATASET', 'NEU-DET')
data_root = os.path.join(datasets_root, dataset_name) + '/'
backend_args = None

classes_by_dataset = {
    'NEU-DET': (
        'crazing',
        'inclusion',
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches',
    ),
    'clipart1k': (
        'sheep',
        'chair',
        'boat',
        'bottle',
        'diningtable',
        'sofa',
        'cow',
        'motorbike',
        'car',
        'aeroplane',
        'cat',
        'train',
        'person',
        'bicycle',
        'pottedplant',
        'bird',
        'dog',
        'bus',
        'tvmonitor',
        'horse',
    ),
    'UODD': (
        'seacucumber',
        'seaurchin',
        'scallop',
    ),
}

metainfo = dict(classes=classes_by_dataset[dataset_name])

domain_attributes_by_dataset = {
    'NEU-DET': 'industrial steel surface defect image',
    'clipart1k': 'clipart style object image',
    'UODD': 'underwater object detection image',
}

train_ann_file = os.getenv('CDFSOD_TRAIN_ANN', 'annotations/train.json')
val_ann_file = 'annotations/test.json'
test_ann_file = 'annotations/test.json'
default_caption_file = train_ann_file.rsplit('.', 1)[0] + '_captions.json'
instance_caption_file = os.getenv('CDFSOD_CAPTION_FILE',
                                  default_caption_file)
use_class_name_token_prototypes = os.getenv(
    'CDFSOD_USE_CLASS_NAME_TOKEN_PROTOTYPES',
    os.getenv('CDFSOD_USE_ENRICHED_CLASS_TOKENS',
              os.getenv('CDFSOD_USE_CLASS_PROTOTYPES', '1'))) != '0'
use_enriched_class_tokens = use_class_name_token_prototypes
debug_text_tokens = os.getenv(
    'CDFSOD_DEBUG_TEXT_TOKENS',
    os.getenv('CDFSOD_DEBUG_TEXT_PROTOTYPE', '0')) == '1'
domain_attribute = os.getenv(
    'CDFSOD_DOMAIN_ATTRIBUTE',
    domain_attributes_by_dataset[dataset_name])

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
