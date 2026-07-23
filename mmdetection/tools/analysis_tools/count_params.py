from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS

# 替换为你的 config 路径
config_path = '/home/add_disk2/qiuxingyu/mmdetection/configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py'  # ← 修改此处

# 加载并初始化
cfg = Config.fromfile(config_path)
init_default_scope(cfg.get('default_scope', 'mmdet'))

# 构建模型，不需要 forward，不需要数据
model = MODELS.build(cfg.model)
model.eval()

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f'\nTotal parameters: {total_params / 1e6:.2f} M\n')

# 可选：打印前几个子模块参数占比（定位大头模块）
print('Top-level modules and their param count:')
for name, module in model.named_children():
    module_params = sum(p.numel() for p in module.parameters())
    print(f'{name}: {module_params / 1e6:.2f} M')
