#!/bin/bash

# 设置目标服务器信息
TARGET_SERVER="liyu@10.176.42.44"
TARGET_DIR="/mnt/data/liyu/mmdetection_data"
SSH_PORT="22"

# 创建目标目录
ssh -p ${SSH_PORT} ${TARGET_SERVER} "mkdir -p ${TARGET_DIR}"

echo "开始传输data目录中的train、test和annotations文件夹..."

# 为每个数据集创建相应的目录并传输特定文件夹
for dataset in clipart1k NEU-DET FISH DIOR UODD ArTaxOr coco cat; do
  # 创建目标数据集目录
  ssh -p ${SSH_PORT} ${TARGET_SERVER} "mkdir -p ${TARGET_DIR}/${dataset}"
  
  # 传输annotations文件夹
  if [ -d "./data/${dataset}/annotations" ]; then
    echo "传输 ${dataset}/annotations..."
    rsync -avz -e "ssh -p ${SSH_PORT}" "./data/${dataset}/annotations/" "${TARGET_SERVER}:${TARGET_DIR}/${dataset}/annotations/"
  fi
  
  # 传输train文件夹
  if [ -d "./data/${dataset}/train" ]; then
    echo "传输 ${dataset}/train..."
    rsync -avz -e "ssh -p ${SSH_PORT}" "./data/${dataset}/train/" "${TARGET_SERVER}:${TARGET_DIR}/${dataset}/train/"
  fi
  
  # 传输test文件夹
  if [ -d "./data/${dataset}/test" ]; then
    echo "传输 ${dataset}/test..."
    rsync -avz -e "ssh -p ${SSH_PORT}" "./data/${dataset}/test/" "${TARGET_SERVER}:${TARGET_DIR}/${dataset}/test/"
  fi
done

echo "data目录传输完成！" 