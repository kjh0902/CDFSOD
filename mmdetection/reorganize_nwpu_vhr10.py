#!/usr/bin/env python
import os
import json
import shutil
import argparse
from tqdm import tqdm

def reorganize_dataset(input_dir, output_dir, shot_num=5):
    """
    重新组织NWPU_VHR-10_coco数据集，使其结构与NEU-DET类似
    将shot_num-shot的数据结构重组为train/, test/, annotations/目录结构
    """
    print(f"Reorganizing NWPU_VHR-10_coco {shot_num}-shot dataset...")
    
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    annotations_dir = os.path.join(output_dir, "annotations")
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 源目录
    src_train_dir = os.path.join(input_dir, f"train_{shot_num}shot")
    src_test_dir = os.path.join(input_dir, f"test_{shot_num}shot")
    src_annotations_dir = os.path.join(input_dir, "annotations")
    
    # 检查源目录是否存在
    if not os.path.exists(src_train_dir):
        print(f"Error: Source train directory not found: {src_train_dir}")
        return
    
    if not os.path.exists(src_test_dir):
        print(f"Error: Source test directory not found: {src_test_dir}")
        return
    
    if not os.path.exists(src_annotations_dir):
        print(f"Error: Source annotations directory not found: {src_annotations_dir}")
        return
    
    # 复制训练集图像
    print(f"Copying training images from {src_train_dir} to {train_dir}...")
    for filename in tqdm(os.listdir(src_train_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            src_path = os.path.join(src_train_dir, filename)
            dst_path = os.path.join(train_dir, filename)
            shutil.copy(src_path, dst_path)
    
    # 复制测试集图像
    print(f"Copying test images from {src_test_dir} to {test_dir}...")
    for filename in tqdm(os.listdir(src_test_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            src_path = os.path.join(src_test_dir, filename)
            dst_path = os.path.join(test_dir, filename)
            shutil.copy(src_path, dst_path)
    
    # 重组标注文件
    train_json_path = os.path.join(src_annotations_dir, f"instances_train_{shot_num}shot.json")
    test_json_path = os.path.join(src_annotations_dir, f"instances_test_{shot_num}shot.json")
    
    # 创建NEU-DET风格的标注文件
    if os.path.exists(train_json_path) and os.path.exists(test_json_path):
        # 读取原始标注文件
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
        
        # 创建训练集标注文件（类似NEU-DET的5_shot.json）
        train_output_path = os.path.join(annotations_dir, f"{shot_num}_shot.json")
        with open(train_output_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # 创建测试集标注文件（类似NEU-DET的test.json）
        test_output_path = os.path.join(annotations_dir, f"test_{shot_num}shot.json")
        with open(test_output_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # 创建类别映射文件（类似NEU-DET的label_map.json）
        categories = train_data.get("categories", [])
        label_map = {}
        for cat in categories:
            label_map[str(cat["id"])] = cat["name"]
        
        label_map_path = os.path.join(annotations_dir, "label_map.json")
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"Created annotation files:")
        print(f"  - {train_output_path}")
        print(f"  - {test_output_path}")
        print(f"  - {label_map_path}")
    
    print("\nReorganization completed!")
    print(f"New dataset structure saved to: {output_dir}")
    print("Directory structure:")
    print(f"{output_dir}/")
    print("├── annotations/")
    print(f"│   ├── {shot_num}_shot.json")
    print("│   ├── test.json")
    print("│   └── label_map.json")
    print("├── train/")
    print("└── test/")

def main():
    parser = argparse.ArgumentParser(description="Reorganize NWPU_VHR-10_coco dataset to match NEU-DET structure")
    parser.add_argument("--input_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NWPU_VHR-10_coco",
                        help="Path to NWPU_VHR-10_coco dataset directory")
    parser.add_argument("--output_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NWPU_VHR-10",
                        help="Output directory for reorganized dataset")
    parser.add_argument("--shot_num", type=int, default=5,
                        help="Shot number to use (default: 5)")
    
    args = parser.parse_args()
    
    reorganize_dataset(args.input_dir, args.output_dir, args.shot_num)

if __name__ == "__main__":
    main() 