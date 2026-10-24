"""Fixed Full LoRA + FE ETF experiment shared by all 18 dataset/shot configs."""
_base_ = 'grounding_dino_swin-b_pretrain_all.py'

model = dict(
    type='GroundingDINOFullLoRA',
    lora_cfg=dict(rank=32, alpha=64, dropout=0.05),
    bert_etf_loss_weight=0.0,
    bert_orth_loss_weight=0.0,
    fe_etf_loss_weight=5.0,
    fe_orth_loss_weight=0.0)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    constructor='FullLoRAOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2))

max_epochs = 100
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

# Fixed absolute module LRs; batch size is four in every dataset config.
auto_scale_lr = dict(enable=False, base_batch_size=4)

param_scheduler = [
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',
        monitor='coco/bbox_mAP',
        rule='greater',
        factor=0.5,
        patience=8,
        threshold=1e-4,
        threshold_rule='rel',
        cooldown=1,
        min_value=1e-6,
        verbose=True)
]

default_hooks = dict(checkpoint=dict(
    max_keep_ckpts=1, save_best='coco/bbox_mAP', rule='greater'))
custom_hooks = [dict(type='FullLoRAHook')]
