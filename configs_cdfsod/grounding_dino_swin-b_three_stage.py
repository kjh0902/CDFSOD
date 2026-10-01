"""Shared ACL fine-tuning policy; detection and checkpoint settings are inherited."""
_base_ = 'grounding_dino_swin-b_pretrain_all.py'

param_scheduler = [
    dict(type='ThreeStagePlateauLR', monitor='coco/bbox_mAP',
         stage_patiences=(3, 5, 8), threshold=1e-4,
         cooldown=1, min_value=1e-6)
]

custom_hooks = [dict(type='ThreeStageProgressiveFinetuningHook')]

# Every dataset config replaces optim_wrapper, including its concrete type.
resume = False
