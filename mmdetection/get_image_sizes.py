#!/usr/bin/env python
import os
from PIL import Image
import sys

def get_image_sizes(folder_path):
    """获取指定文件夹中所有图片的尺寸"""
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return
    
    # 获取所有图片文件
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(folder_path, filename))
    
    if not image_files:
        print(f"文件夹 '{folder_path}' 中没有找到图片文件")
        return
    
    print(f"{'文件名':<40} {'宽度':<10} {'高度':<10}")
    print("-" * 60)
    
    # 获取并打印每个图片的尺寸
    for img_path in sorted(image_files):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                filename = os.path.basename(img_path)
                print(f"{filename:<40} {width:<10} {height:<10}")
        except Exception as e:
            print(f"无法打开 {img_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "/home/add_disk2/qiuxingyu/mmdetection/process_2/UODD/results_coco_0.8_target_1_cocotext_1_targettext_1.2_20250502_170225/003200"
    
    get_image_sizes(folder_path) 