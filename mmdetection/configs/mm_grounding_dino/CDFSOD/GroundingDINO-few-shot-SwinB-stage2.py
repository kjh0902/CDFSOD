_base_ = [
    './GroundingDINO-few-shot-SwinB.py',
]

# Stage 2: fine-tune every module except the frozen BERT text encoder.
model = dict(training_stage='full_finetune')

stage2_epochs = 25
# NEU-DET 1-shot has only three iterations per epoch. Keep a stable learning
# rate for all 75 Stage 2 updates instead of starving the fine-tuning phase
# with the previous 1e-5 -> 1e-6 schedule.
optim_wrapper = dict(optimizer=dict(lr=5e-5))
param_scheduler = []
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=stage2_epochs,
    val_begin=stage2_epochs + 1,
    val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=stage2_epochs,
        # Stage 2 uses local epochs 1--25, but this is global epoch 30.
        filename_tmpl='epoch_30.pth'))
