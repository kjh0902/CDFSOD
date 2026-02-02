#!/usr/bin/env python
import os
import shutil
import re
import time
import tempfile

# 配置参数
SWIN_T_CONFIG = "configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py"
SWIN_B_CONFIG = "configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py"
DATASETS = [ "ArTaxOr"]
# DATASETS = [ "UODD" ,"FISH", "NEU-DET", "ArTaxOr", "clipart1k", "DIOR"]
SHOTS = ["1"]
# DATASETS = [ "NWPU_VHR-10"]
# SHOTS = ["3", "5", "10", "20"]
# DATASETS = [ "Camouflage"]
# SHOTS = [ "2", "3", "5", "1"]
FEWSHOT_CONFIGS_DIR = "configs/grounding_dino"  # 放置few-shot configs的目录
MAX_EPOCHS = 30  # 设置为30

# 备份原始swin-t配置
backup_file = f"{SWIN_T_CONFIG}.original.bak"
if not os.path.exists(backup_file):
    print(f"备份原始配置文件: {SWIN_T_CONFIG} -> {backup_file}")
    shutil.copy2(SWIN_T_CONFIG, backup_file)

# 检查fewshot配置文件是否存在
all_configs_exist = True
for dataset in DATASETS:
    for shot in SHOTS:
        config_path = f"{FEWSHOT_CONFIGS_DIR}/CDFSOD_detection_few-shot_{dataset}_{shot}shot.py"
        if not os.path.exists(config_path):
            print(f"错误: 找不到few-shot配置文件: {config_path}")
            all_configs_exist = False

if not all_configs_exist:
    print("请先生成所有few-shot配置文件。程序终止。")
    exit(1)

# 读取原始swin-t配置内容
with open(backup_file, 'r') as f:
    original_content = f.read()

# 执行实验
for dataset in DATASETS:
    for shot in SHOTS:
        # 构建新的_base_路径
        fewshot_config = f"CDFSOD_detection_few-shot_{dataset}_{shot}shot.py"
        fewshot_path = f"{FEWSHOT_CONFIGS_DIR}/{fewshot_config}"
        
        # 相对路径（从swin-t配置文件的角度）
        rel_path = os.path.relpath(fewshot_path, os.path.dirname(SWIN_T_CONFIG))
        
        # 修改swin-t配置中的_base_[0]
        new_content = re.sub(
            r"(_base_ = \[\s*)'[^']*'", 
            r"\1'" + rel_path.replace('\\', '/') + "'", 
            original_content, 
            count=1
        )
        
        # 写入修改后的配置
        with open(SWIN_T_CONFIG, 'w') as f:
            f.write(new_content)
        
        print(f"\n===== 运行实验: {dataset} {shot}-shot =====")
        print(f"已修改 {SWIN_T_CONFIG} 数据集为 {dataset}_{shot}shot")
        
        # 创建特定的工作目录，包含数据集和shot信息
        work_dir = f"cat_work_dir/{dataset}_{shot}shot"
        os.makedirs(work_dir, exist_ok=True)
        
        # 创建临时配置文件来设置max_epochs
        temp_config = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
        temp_config_path = temp_config.name
        
        with open(temp_config_path, 'w') as f:
            f.write(f'''
# 临时配置文件，设置max_epochs为{MAX_EPOCHS}
_base_ = ['{os.path.abspath(SWIN_B_CONFIG)}']
train_cfg = dict(max_epochs={MAX_EPOCHS})
''')
        
        print(f"创建临时配置文件，设置max_epochs={MAX_EPOCHS}")
        
        # 运行训练命令
        cmd = f"./tools/dist_train.sh {temp_config_path} 4 --work-dir {work_dir}"
        print(f"执行命令: {cmd}")
        
        # 实际运行命令
        os.system(cmd)
        
        # 删除临时配置文件
        os.unlink(temp_config_path)
        
        # 等待一段时间确保文件被正确读取（可选）
        time.sleep(1)

# 实验结束后恢复原始配置
print("\n===== 所有实验完成 =====")
print(f"恢复原始配置: {backup_file} -> {SWIN_T_CONFIG}")
shutil.copy2(backup_file, SWIN_T_CONFIG) 