"""Single-stage Full LoRA over the existing ACL detector and ETF loss."""
from mmdet.registry import MODELS
from ..layers.lora import (LoRAConv2d, LoRALinear, LoRAProjection,
                           inject_full_lora)
from .grounding_dino_HED import GroundingDINO_ParallelDecoder_15_DNQuery_rand


@MODELS.register_module()
class GroundingDINOFullLoRA(GroundingDINO_ParallelDecoder_15_DNQuery_rand):
    def __init__(self, *args, lora_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        inject_full_lora(self, **(lora_cfg or {}))

    def init_weights(self):
        super().init_weights()
        # DINO initializes all encoder/decoder matrices with Xavier, including
        # newly inserted adapters. Restore A=Kaiming, B=0 before load_from.
        for module in self.modules():
            if isinstance(module, (LoRALinear, LoRAConv2d, LoRAProjection)):
                module.reset_lora_parameters()
