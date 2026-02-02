#!/bin/bash

# 设置目标服务器信息（替换为实际信息）
TARGET_SERVER="liyu@10.176.42.44"
TARGET_DIR="/mnt/data/liyu"
SSH_PORT="22"

# 创建排除文件列表
cat > exclude_list.txt << EOL
.git/
weights/
work_dirs/
*.pth
*.pt
*.bin
*.model
*.onnx
*.npy
data/*/images
**/__pycache__/
*.pyc
outputs/
cat_dataset.zip
EOL

# 创建包含列表，明确指定要传输的数据集文件夹
cat > include_list.txt << EOL
+ data/*/annotations/
+ data/*/train/
+ data/*/val/
+ data/*/test/
+ data/*/labels/
+ data/*/meta/
+ data/*/*.json
EOL

# 执行rsync命令，使用-e选项指定SSH端口
rsync -avz -e "ssh -p ${SSH_PORT}" --exclude-from=exclude_list.txt --include-from=include_list.txt --progress ./ ${TARGET_SERVER}:${TARGET_DIR}/

echo "传输完成！" 