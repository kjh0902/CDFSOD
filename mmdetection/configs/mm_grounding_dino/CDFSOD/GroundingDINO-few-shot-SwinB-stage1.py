_base_ = [
    './GroundingDINO-few-shot-SwinB.py',
]

# Stage 1: train only the newly added 49 -> 128 -> 1 spatial projection MLP.
model = dict(training_stage='mlp_only')

stage1_epochs = 5
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = []
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=stage1_epochs,
    val_begin=stage1_epochs + 1,
    val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=stage1_epochs))
