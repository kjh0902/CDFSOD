_base_ = [
    './GroundingDINO-few-shot-SwinB.py',
]

# Stage 2: fine-tune every module except the frozen BERT text encoder.
model = dict(training_stage='full_finetune')

stage2_epochs = 25
optim_wrapper = dict(optimizer=dict(lr=1e-5))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=stage2_epochs,
        by_epoch=True,
        # Stage 2 epoch 15 is global epoch 20 after the five-epoch Stage 1.
        milestones=[15],
        gamma=0.1)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=stage2_epochs,
    val_begin=stage2_epochs + 1,
    val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=stage2_epochs))
