# configs/mm_grounding_dino/grounding_dino_swin-b_finetune.py

_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

# ä½¿ç”¨é¢„è?ç»ƒçš„Grounding DINOæ¨¡å‹ä½œä¸ºèµ·ç‚¹
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth'  # noqa

data_root = '/home/aislab5090/CDFSOD/junhyung/datasets/clipart1k/'
class_name = ('sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# æ·»åŠ  launcher ?Œåˆ†å¸ƒå¼è®?»ƒ?ç½®
launcher = 'pytorch'

# ?ç½® find_unused_parameters=True è§£å†³?ªä½¿?¨å‚?°çš„??¢˜
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# æ·»åŠ ?™æ€å›¾è®¾ç½®
_use_static_graph = True
_set_static_graph = True

# ä¿?”¹æ¨¡å‹?ç½®ï¼Œç¡®ä¿æ?ç¡?¤„?†ç±»?«æ•°?å’Œlabel_embedding
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
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    language_model=dict(
        type='BertModel',
        name='bert-base-uncased',
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False),  # ç¡?¿ä¸clipart1k_1shot_train.pyä¸€??    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),
    test_cfg=dict(max_per_img=300),
    use_autocast=False,
)

# ?ªå®šä¹‰é’ˆå¯¹clipart1kæ£€æµ‹çš„?°æ®å¢å¼ºç®¡é“
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

# æµ‹è¯•?ŒéªŒè¯çš„ç®¡é“
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

# å®šä¹‰Clipart1k?°æ®??train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='CocoDataset',
                data_root=data_root,
                ann_file='annotations/1_shot.json',
                data_prefix=dict(img='train/'),
                filter_cfg=dict(filter_empty_gt=False),
                metainfo=dict(classes=class_name),
                pipeline=train_pipeline,
                return_classes=True,
            )
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=class_name),
        pipeline=test_pipeline,
        return_classes=True,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

# è°ƒæ•´ä¼˜åŒ–?¨å’Œå­¦ä¹ ??optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1),
            language_model=dict(lr_mult=0.0))),  # ?»ç»“BERT
    type='OptimWrapper')

# å­¦ä¹ ?‡è°ƒåº?param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[11],
        type='MultiStepLR'),
]

# è¯„ä¼°è®¾ç½®
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox')

test_evaluator = val_evaluator

# è®?»ƒ?ç½®
max_epoch = 20
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=1)
val_cfg = dict(type='ValLoop')

# é»˜è??©å­
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))

# ?¥å¿—è®¾ç½®
log_processor = dict(by_epoch=True)
log_level = 'INFO'
resume = False

# ??§†?–è?ç½?vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# ä¿?”¹?†å¸ƒå¼è?ç»ƒè?ç½?find_unused_parameters = True