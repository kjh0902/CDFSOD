# MMDetection 训练流程指南

本指南介绍如何使用 `copy_from_json.py` 和 `auto_modify_swin_t_config.py` 脚本进行模型训练。

## 1. 准备数据

首先，使用 `copy_from_json.py` 从 JSON 文件中提取生成的图片并复制到训练文件夹：

```bash
# 使用方法
python copy_from_json.py --json <outpaint结果JSON文件> --target <目标目录> --dataset <数据集名称>

# 示例
# 将生成的图片复制到 FISH 数据集的训练文件夹
python copy_from_json.py --json ./outpaint/fish_results.json --target ./data/FISH/train --dataset FISH
```

### 参数说明
- `--json`: outpaint 结果 JSON 文件路径
- `--target`: 目标目录路径，通常是数据集的训练文件夹
- `--dataset`: 数据集名称，用于重命名图片文件

脚本会将生成的图片以 `{dataset_name}_{sample_id}_outpaint_{rank_number}.png` 的格式复制到目标目录。

## 2. 执行训练

准备好数据后，使用 `auto_modify_swin_t_config.py` 修改配置并执行训练：

```bash
# 直接运行脚本
python auto_modify_swin_t_config.py
```

### 配置说明

在运行前，你可能需要修改 `auto_modify_swin_t_config.py` 中的以下参数：

```python
# 修改数据集和 shot 数量
DATASETS = [ "FISH" ]  # 要训练的数据集列表
SHOTS = ["1"]          # few-shot 设置（1-shot, 5-shot, 10-shot 等）
MAX_EPOCHS = 50        # 最大训练 epoch 数
```

### 工作流程

脚本会自动完成以下操作：

1. 备份原始 Swin-T 配置文件
2. 为每个数据集和 shot 设置修改配置文件
3. 创建特定的工作目录 `cat_work_dir/{数据集}_{shot}shot`
4. 设置训练的最大 epoch 数
5. 使用 4 个 GPU 进行分布式训练
6. 训练完成后恢复原始配置

## 注意事项

1. 确保 `configs/grounding_dino/CDFSOD_detection_few-shot_{dataset}_{shot}shot.py` 配置文件已存在
2. 训练结果将保存在 `cat_work_dir/{数据集}_{shot}shot` 目录中
3. 脚本会使用 `tools/dist_train.sh` 进行分布式训练，确保环境已正确配置

## 完整流程示例

```bash
# 1. 复制 FISH 数据集的生成图片到训练文件夹
python copy_from_json.py --json ./outpaint/fish_results.json --target ./data/FISH/train --dataset FISH

# 2. 执行训练
python auto_modify_swin_t_config.py
```

训练完成后，模型权重和日志将保存在 `cat_work_dir/{数据集}_{shot}shot` 目录中。 