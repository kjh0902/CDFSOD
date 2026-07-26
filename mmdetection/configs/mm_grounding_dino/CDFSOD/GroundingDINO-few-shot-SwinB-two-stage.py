_base_ = [
    './GroundingDINO-few-shot-SwinB.py',
]

# Build the model in Stage 1 mode. With no paramwise_cfg, MMEngine registers
# model.parameters() in full even while most parameters require no gradient.
model = dict(training_stage='mlp_only')

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
        type='GroundingDinoTwoStageHook',
        mlp_only_epochs=5,
        priority='VERY_HIGH')
]
