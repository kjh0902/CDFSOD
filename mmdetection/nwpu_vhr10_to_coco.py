#!/usr/bin/env python
import os
import re
import json
import shutil
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def parse_annotation(anno_file):
    """解析NWPU_VHR-10标注文件，将(x1,y1),(x2,y2),class_id格式转换为COCO格式"""
    annotations = []
    if not os.path.exists(anno_file):
        print(f"Warning: Annotation file not found: {anno_file}")
        return annotations
    
    with open(anno_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析(x1,y1),(x2,y2),class_id格式
            match = re.match(r'\((\d+),(\d+)\),\((\d+),(\d+)\),(\d+)', line)
            if match:
                x1, y1, x2, y2, class_id = map(int, match.groups())
                # 转换为COCO格式 [x, y, width, height]
                width = x2 - x1
                height = y2 - y1
                
                # 确保宽度和高度为正
                if width <= 0 or height <= 0:
                    print(f"Warning: Invalid bbox in {anno_file}: ({x1},{y1}),({x2},{y2})")
                    continue
                
                annotations.append({
                    "bbox": [x1, y1, width, height],
                    "category_id": int(class_id)
                })
            else:
                print(f"Warning: Could not parse line in {anno_file}: {line}")
    
    return annotations

def get_image_info(image_path):
    """获取图像信息，包括尺寸"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 1024, 1024  # 默认值

def get_few_shot_samples(shot_num, few_shot_dir, category_list):
    """获取few-shot训练样本ID列表"""
    shot_dir = os.path.join(few_shot_dir, f"benchmark_{shot_num}shot")
    train_ids = set()
    
    # 读取每个类别的训练样本ID
    for category in category_list:
        category_name = category["name"]
        filename = f"box_{shot_num}shot_{category_name}_train.txt"
        file_path = os.path.join(shot_dir, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]
                train_ids.update(ids)
        else:
            print(f"Warning: File not found: {file_path}")
    
    return list(train_ids)

def get_all_image_ids(image_dir):
    """获取所有图像ID（不包含扩展名）"""
    all_ids = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_id = os.path.splitext(filename)[0]
            all_ids.append(image_id)
    return all_ids

def create_coco_json(sample_ids, anno_dir, image_dir, output_json, output_image_dir, categories):
    """创建COCO格式的JSON文件并复制图像"""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 初始化COCO JSON结构
    coco_json = {
        "info": {
            "description": "Converted NWPU_VHR-10 dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # 添加图片和标注
    annotation_id = 1
    
    for image_id, sample_id in enumerate(tqdm(sample_ids, desc=f"Processing {os.path.basename(output_json)}")):
        # 确保图像文件存在
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 获取图像尺寸
        width, height = get_image_info(image_path)
        
        # 添加图像信息
        coco_json["images"].append({
            "id": image_id + 1,  # COCO格式中的ID从1开始
            "file_name": f"{sample_id}.jpg",
            "width": width,
            "height": height,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
        # 解析标注
        anno_file = os.path.join(anno_dir, f"{sample_id}.txt")
        annotations = parse_annotation(anno_file)
        
        if not annotations:
            print(f"Warning: No valid annotations found for {sample_id}")
        
        # 添加标注信息
        for anno in annotations:
            coco_json["annotations"].append({
                "id": annotation_id,
                "image_id": image_id + 1,
                "category_id": anno["category_id"],
                "bbox": anno["bbox"],
                "area": anno["bbox"][2] * anno["bbox"][3],
                "iscrowd": 0,
                "segmentation": []  # NWPU_VHR-10没有分割标注
            })
            annotation_id += 1
        
        # 复制图像文件
        try:
            dest_path = os.path.join(output_image_dir, f"{sample_id}.jpg")
            shutil.copy(image_path, dest_path)
        except Exception as e:
            print(f"Error copying image {image_path}: {e}")
    
    # 保存JSON文件
    with open(output_json, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Created COCO JSON: {output_json}")
    print(f"Copied {len(sample_ids)} images to {output_image_dir}")
    print(f"Added {annotation_id-1} annotations")

def generate_statistics(json_file):
    """生成数据集统计信息"""
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 按类别统计实例数量
    category_stats = {}
    for category in data["categories"]:
        category_stats[category["id"]] = {
            "name": category["name"],
            "count": 0
        }
    
    for anno in data["annotations"]:
        if anno["category_id"] in category_stats:
            category_stats[anno["category_id"]]["count"] += 1
    
    print(f"\nStatistics for {os.path.basename(json_file)}:")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print("Category distribution:")
    
    for cat_id, stats in sorted(category_stats.items()):
        print(f"  {stats['name']} (id={cat_id}): {stats['count']} instances")

def convert_nwpu_vhr10_to_coco(data_dir, output_dir, few_shot_dir=None):
    """将NWPU_VHR-10数据集转换为COCO格式"""
    # 设置路径
    image_dir = os.path.join(data_dir, "positive image set")
    anno_dir = os.path.join(data_dir, "ground truth")
    
    if not few_shot_dir:
        few_shot_dir = os.path.join(data_dir, "few_shot_ann/vhr10")
    
    # 定义NWPU_VHR-10类别
    categories = [
        {"id": 1, "name": "airplane", "supercategory": "vehicle"},
        {"id": 2, "name": "ship", "supercategory": "vehicle"},
        {"id": 3, "name": "storage-tank", "supercategory": "infrastructure"},
        {"id": 4, "name": "baseball-diamond", "supercategory": "sports"},
        {"id": 5, "name": "tennis-court", "supercategory": "sports"},
        {"id": 6, "name": "basketball-court", "supercategory": "sports"},
        {"id": 7, "name": "ground-track-field", "supercategory": "sports"},
        {"id": 8, "name": "harbor", "supercategory": "infrastructure"},
        {"id": 9, "name": "bridge", "supercategory": "infrastructure"},
        {"id": 10, "name": "vehicle", "supercategory": "vehicle"}
    ]
    
    # 获取所有图像ID
    all_image_ids = get_all_image_ids(image_dir)
    print(f"Found {len(all_image_ids)} images in total")
    
    # 处理不同shot设置
    shot_nums = [3, 5, 10, 20]
    
    for shot_num in shot_nums:
        print(f"\nProcessing {shot_num}-shot setting...")
        
        # 获取训练样本ID
        train_ids = get_few_shot_samples(shot_num, few_shot_dir, categories)
        print(f"Found {len(train_ids)} training samples for {shot_num}-shot")
        
        # 测试样本 = 所有样本 - 训练样本
        test_ids = [img_id for img_id in all_image_ids if img_id not in train_ids]
        print(f"Using {len(test_ids)} test samples for {shot_num}-shot")
        
        # 创建输出目录
        train_output_dir = os.path.join(output_dir, f"train_{shot_num}shot")
        test_output_dir = os.path.join(output_dir, f"test_{shot_num}shot")
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # 创建训练集COCO JSON
        train_json_path = os.path.join(annotations_dir, f"instances_train_{shot_num}shot.json")
        create_coco_json(train_ids, anno_dir, image_dir, train_json_path, train_output_dir, categories)
        
        # 创建测试集COCO JSON
        test_json_path = os.path.join(annotations_dir, f"instances_test_{shot_num}shot.json")
        create_coco_json(test_ids, anno_dir, image_dir, test_json_path, test_output_dir, categories)
        
        # 生成统计信息
        generate_statistics(train_json_path)
        generate_statistics(test_json_path)

def main():
    parser = argparse.ArgumentParser(description="Convert NWPU_VHR-10 dataset to COCO format with different few-shot settings")
    parser.add_argument("--data_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NWPU_VHR-10",
                        help="Path to NWPU_VHR-10 dataset directory")
    parser.add_argument("--output_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NWPU_VHR-10_coco",
                        help="Output directory for COCO format dataset")
    parser.add_argument("--few_shot_dir", type=str, default=None,
                        help="Path to few-shot annotation directory (default: data_dir/few_shot_ann/vhr10)")
    
    args = parser.parse_args()
    
    convert_nwpu_vhr10_to_coco(args.data_dir, args.output_dir, args.few_shot_dir)
    
    print("\nConversion completed!")
    print(f"COCO format dataset saved to: {args.output_dir}")
    print("Directory structure:")
    print(f"{args.output_dir}/")
    print("├── annotations/")
    print("│   ├── instances_train_3shot.json")
    print("│   ├── instances_test_3shot.json")
    print("│   ├── ...")
    print("├── train_3shot/")
    print("├── test_3shot/")
    print("├── ...")

if __name__ == "__main__":
    main() 