_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth' 
data_root = '/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/'
class_name = ('sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')
num_classes = len(class_name)
# metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

metainfo = dict(
    classes=('sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')
)
# ‰ø?îπÊ®°Âûã?çÁΩÆÔºåÁ°Æ‰øùÊ?Á°?§Ñ?ÜÁ±ª?´Êï∞?èÂíålabel_embedding
model = dict(
    type='GroundingDINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=0,
        init_cfg=None),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        num_cp=0,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0)),
        fusion_layer_cfg=dict(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=20,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    language_model=dict(
        type='BertModel',
        name='bert-base-uncased',
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False),  # Á°?øù‰∏éclipart1k_1shot_train.py‰∏Ä??    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),
    test_cfg=dict(max_per_img=300),
    use_autocast=False,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        datasets=[
            dict(
                ann_file='annotations/1_shot.json',
                data_prefix=dict(img='train/'),
                data_root='/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/',
                filter_cfg=dict(filter_empty_gt=False),
                metainfo=dict(
                    classes=(
                        'sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(
                        transforms=[
                            [
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            480,
                                            1333,
                                        ),
                                        (
                                            512,
                                            1333,
                                        ),
                                        (
                                            544,
                                            1333,
                                        ),
                                        (
                                            576,
                                            1333,
                                        ),
                                        (
                                            608,
                                            1333,
                                        ),
                                        (
                                            640,
                                            1333,
                                        ),
                                        (
                                            672,
                                            1333,
                                        ),
                                        (
                                            704,
                                            1333,
                                        ),
                                        (
                                            736,
                                            1333,
                                        ),
                                        (
                                            768,
                                            1333,
                                        ),
                                        (
                                            800,
                                            1333,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                            [
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            400,
                                            4200,
                                        ),
                                        (
                                            500,
                                            4200,
                                        ),
                                        (
                                            600,
                                            4200,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                                dict(
                                    allow_negative_crop=True,
                                    crop_size=(
                                        384,
                                        600,
                                    ),
                                    crop_type='absolute_range',
                                    type='RandomCrop'),
                                dict(
                                    keep_ratio=True,
                                    scales=[
                                        (
                                            480,
                                            1333,
                                        ),
                                        (
                                            512,
                                            1333,
                                        ),
                                        (
                                            544,
                                            1333,
                                        ),
                                        (
                                            576,
                                            1333,
                                        ),
                                        (
                                            608,
                                            1333,
                                        ),
                                        (
                                            640,
                                            1333,
                                        ),
                                        (
                                            672,
                                            1333,
                                        ),
                                        (
                                            704,
                                            1333,
                                        ),
                                        (
                                            736,
                                            1333,
                                        ),
                                        (
                                            768,
                                            1333,
                                        ),
                                        (
                                            800,
                                            1333,
                                        ),
                                    ],
                                    type='RandomChoiceResize'),
                            ],
                        ],
                        type='RandomChoice'),
                    dict(
                        meta_keys=(
                            'img_id',
                            'img_path',
                            'ori_shape',
                            'img_shape',
                            'scale_factor',
                            'flip',
                            'flip_direction',
                            'text',
                            'custom_entities',
                        ),
                        type='PackDetInputs'),
                ],
                return_classes=True,
                type='CocoDataset'),
        ],
        type='ConcatDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'text',
            'custom_entities',
        ),
        type='PackDetInputs'),
]
train_real_dataset = dict(
    ann_file='annotations/1_shot.json',
    data_prefix=dict(img='train/'),
    data_root='/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/',
    filter_cfg=dict(filter_empty_gt=False),
    metainfo=dict(
        classes=(
                        'sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(prob=0.5, type='RandomFlip'),
        dict(
            transforms=[
                [
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                480,
                                1333,
                            ),
                            (
                                512,
                                1333,
                            ),
                            (
                                544,
                                1333,
                            ),
                            (
                                576,
                                1333,
                            ),
                            (
                                608,
                                1333,
                            ),
                            (
                                640,
                                1333,
                            ),
                            (
                                672,
                                1333,
                            ),
                            (
                                704,
                                1333,
                            ),
                            (
                                736,
                                1333,
                            ),
                            (
                                768,
                                1333,
                            ),
                            (
                                800,
                                1333,
                            ),
                        ],
                        type='RandomChoiceResize'),
                ],
                [
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                400,
                                4200,
                            ),
                            (
                                500,
                                4200,
                            ),
                            (
                                600,
                                4200,
                            ),
                        ],
                        type='RandomChoiceResize'),
                    dict(
                        allow_negative_crop=True,
                        crop_size=(
                            384,
                            600,
                        ),
                        crop_type='absolute_range',
                        type='RandomCrop'),
                    dict(
                        keep_ratio=True,
                        scales=[
                            (
                                480,
                                1333,
                            ),
                            (
                                512,
                                1333,
                            ),
                            (
                                544,
                                1333,
                            ),
                            (
                                576,
                                1333,
                            ),
                            (
                                608,
                                1333,
                            ),
                            (
                                640,
                                1333,
                            ),
                            (
                                672,
                                1333,
                            ),
                            (
                                704,
                                1333,
                            ),
                            (
                                736,
                                1333,
                            ),
                            (
                                768,
                                1333,
                            ),
                            (
                                800,
                                1333,
                            ),
                        ],
                        type='RandomChoiceResize'),
                ],
            ],
            type='RandomChoice'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'flip',
                'flip_direction',
                'text',
                'custom_entities',
            ),
            type='PackDetInputs'),
    ],
    return_classes=True,
    type='CocoDataset')

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        data_root='/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/',
        metainfo=dict(
            classes=(
                        'sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                1333,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/annotations/test.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)