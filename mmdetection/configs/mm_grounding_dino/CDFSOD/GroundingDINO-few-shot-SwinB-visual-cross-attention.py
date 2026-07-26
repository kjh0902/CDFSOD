_base_ = [
    './GroundingDINO-few-shot-SwinB.py',
]

# Train BERT, the detector, and the visual cross-attention jointly from the
# first epoch with a single FP32 OptimWrapper.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))

max_epochs = 30
param_scheduler = []
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=max_epochs + 1,
    val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=max_epochs))
custom_hooks = [
    dict(
        type='LossGradientFileLoggerHook',
        filename='loss_gradient_per_iter.jsonl')
]
