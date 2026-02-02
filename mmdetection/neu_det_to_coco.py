#!/usr/bin/env python
import os
import json
import shutil
import argparse
from PIL import Image
from tqdm import tqdm

def get_image_info(image_path):
    """获取图像信息，包括尺寸"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 200, 200  # 默认值，NEU-DET的图像通常为200x200

def create_directory_structure(output_dir, shot_nums):
    """创建输出目录结构"""
    # 创建annotations目录
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # 创建各个shot设置的训练和测试目录
    for shot_num in shot_nums:
        train_dir = os.path.join(output_dir, f"train_{shot_num}shot")
        test_dir = os.path.join(output_dir, f"test_{shot_num}shot")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    return annotations_dir

def process_shot_setting(shot_num, data_dir, output_dir, annotations_dir):
    """处理特定shot设置的数据"""
    # 加载shot设置的JSON文件
    shot_json_path = os.path.join(data_dir, "annotations", f"{shot_num}_shot.json")
    test_json_path = os.path.join(data_dir, "annotations", "test.json")
    
    if not os.path.exists(shot_json_path):
        print(f"Error: Shot JSON file not found: {shot_json_path}")
        return
    
    if not os.path.exists(test_json_path):
        print(f"Error: Test JSON file not found: {test_json_path}")
        return
    
    # 加载shot和test的JSON数据
    with open(shot_json_path, 'r') as f:
        shot_data = json.load(f)
    
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # 获取类别信息
    category_map_path = os.path.join(data_dir, "annotations", "label_map.json")
    if os.path.exists(category_map_path):
        with open(category_map_path, 'r') as f:
            category_map = json.load(f)
    else:
        # 如果没有找到类别映射文件，使用默认的NEU-DET类别
        category_map = {
            "1": "crazing", 
            "2": "inclusion", 
            "3": "patches", 
            "4": "pitted_surface", 
            "5": "rolled-in_scale", 
            "6": "scratches"
        }
    
    # 创建COCO格式的类别列表
    categories = []
    for cat_id, cat_name in category_map.items():
        categories.append({
            "id": int(cat_id),
            "name": cat_name,
            "supercategory": "defect"
        })
    
    # 创建训练集COCO JSON
    train_images = shot_data.get("images", [])
    train_annotations = shot_data.get("annotations", [])
    
    # 获取训练图像的ID集合
    train_image_ids = {img["id"] for img in train_images}
    
    # 筛选出训练集中的标注
    train_annotations = [anno for anno in train_annotations if anno.get("image_id") in train_image_ids]
    
    # 创建训练集COCO JSON
    train_coco = {
        "info": {
            "description": f"NEU-DET {shot_num}-shot train dataset",
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
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    
    # 创建测试集COCO JSON
    test_images = test_data.get("images", [])
    test_annotations = test_data.get("annotations", [])
    
    # 获取测试图像的ID集合
    test_image_ids = {img["id"] for img in test_images}
    
    # 筛选出测试集中的标注
    test_annotations = [anno for anno in test_annotations if anno.get("image_id") in test_image_ids]
    
    # 创建测试集COCO JSON
    test_coco = {
        "info": {
            "description": f"NEU-DET {shot_num}-shot test dataset",
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
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories
    }
    
    # 保存训练集和测试集COCO JSON
    train_output_path = os.path.join(annotations_dir, f"instances_train_{shot_num}shot.json")
    test_output_path = os.path.join(annotations_dir, f"instances_test_{shot_num}shot.json")
    
    with open(train_output_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(test_output_path, 'w') as f:
        json.dump(test_coco, f, indent=2)
    
    print(f"\nCreated COCO JSON files for {shot_num}-shot setting:")
    print(f"  - Training: {train_output_path}")
    print(f"  - Testing: {test_output_path}")
    
    # 复制图像文件
    copy_images(data_dir, output_dir, train_images, test_images, shot_num)
    
    # 生成统计信息
    generate_statistics(train_output_path, test_output_path)

def copy_images(data_dir, output_dir, train_images, test_images, shot_num):
    """复制图像文件到相应的目录"""
    # 设置源目录和目标目录
    image_dirs = [
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "IMAGES")
    ]
    
    # 确定可用的图像目录
    valid_image_dirs = [d for d in image_dirs if os.path.exists(d)]
    if not valid_image_dirs:
        print("Warning: No valid image directories found. Cannot copy images.")
        return
    
    # 设置目标目录
    train_output_dir = os.path.join(output_dir, f"train_{shot_num}shot")
    test_output_dir = os.path.join(output_dir, f"test_{shot_num}shot")
    
    # 复制训练图像
    print(f"\nCopying {len(train_images)} training images for {shot_num}-shot setting...")
    for img in tqdm(train_images, desc=f"Copying train images ({shot_num}-shot)"):
        file_name = img["file_name"]
        found = False
        
        for src_dir in valid_image_dirs:
            src_path = os.path.join(src_dir, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(train_output_dir, file_name)
                try:
                    shutil.copy(src_path, dst_path)
                    found = True
                    break
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
        
        if not found:
            print(f"Warning: Image not found: {file_name}")
    
    # 复制测试图像
    print(f"\nCopying {len(test_images)} test images for {shot_num}-shot setting...")
    for img in tqdm(test_images, desc=f"Copying test images ({shot_num}-shot)"):
        file_name = img["file_name"]
        found = False
        
        for src_dir in valid_image_dirs:
            src_path = os.path.join(src_dir, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(test_output_dir, file_name)
                try:
                    shutil.copy(src_path, dst_path)
                    found = True
                    break
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
        
        if not found:
            print(f"Warning: Image not found: {file_name}")

def generate_statistics(train_json_path, test_json_path):
    """生成数据集统计信息"""
    # 处理训练集统计
    if os.path.exists(train_json_path):
        with open(train_json_path, 'r') as f:
            train_data = json.load(f)
        
        # 按类别统计实例数量
        train_category_stats = {}
        for category in train_data["categories"]:
            train_category_stats[category["id"]] = {
                "name": category["name"],
                "count": 0
            }
        
        for anno in train_data["annotations"]:
            if anno["category_id"] in train_category_stats:
                train_category_stats[anno["category_id"]]["count"] += 1
        
        print(f"\nStatistics for {os.path.basename(train_json_path)}:")
        print(f"Total images: {len(train_data['images'])}")
        print(f"Total annotations: {len(train_data['annotations'])}")
        print("Category distribution:")
        
        for cat_id, stats in sorted(train_category_stats.items()):
            print(f"  {stats['name']} (id={cat_id}): {stats['count']} instances")
    
    # 处理测试集统计
    if os.path.exists(test_json_path):
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
        
        # 按类别统计实例数量
        test_category_stats = {}
        for category in test_data["categories"]:
            test_category_stats[category["id"]] = {
                "name": category["name"],
                "count": 0
            }
        
        for anno in test_data["annotations"]:
            if anno["category_id"] in test_category_stats:
                test_category_stats[anno["category_id"]]["count"] += 1
        
        print(f"\nStatistics for {os.path.basename(test_json_path)}:")
        print(f"Total images: {len(test_data['images'])}")
        print(f"Total annotations: {len(test_data['annotations'])}")
        print("Category distribution:")
        
        for cat_id, stats in sorted(test_category_stats.items()):
            print(f"  {stats['name']} (id={cat_id}): {stats['count']} instances")

def convert_neu_det_to_coco(data_dir, output_dir, shot_nums=None):
    """将NEU-DET数据集转换为COCO格式，支持不同的few-shot设置"""
    if not shot_nums:
        shot_nums = [1, 5, 10]  # 默认处理1-shot, 5-shot, 10-shot设置
    
    print(f"Converting NEU-DET dataset to COCO format for shot settings: {shot_nums}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # 创建输出目录结构
    annotations_dir = create_directory_structure(output_dir, shot_nums)
    
    # 处理每个shot设置
    for shot_num in shot_nums:
        process_shot_setting(shot_num, data_dir, output_dir, annotations_dir)
    
    print("\nConversion completed!")
    print(f"COCO format dataset saved to: {output_dir}")
    print("Directory structure:")
    print(f"{output_dir}/")
    print("├── annotations/")
    for shot_num in shot_nums:
        print(f"│   ├── instances_train_{shot_num}shot.json")
        print(f"│   ├── instances_test_{shot_num}shot.json")
    for shot_num in shot_nums:
        print(f"├── train_{shot_num}shot/")
        print(f"├── test_{shot_num}shot/")

def main():
    parser = argparse.ArgumentParser(description="Convert NEU-DET dataset to COCO format with different few-shot settings")
    parser.add_argument("--data_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NEU-DET",
                        help="Path to NEU-DET dataset directory")
    parser.add_argument("--output_dir", type=str, default="/home/add_disk2/qiuxingyu/mmdetection/data/NEU-DET_coco",
                        help="Output directory for COCO format dataset")
    parser.add_argument("--shot_nums", type=str, default="1,5,10",
                        help="Shot numbers to process, comma-separated (default: 1,5,10)")
    
    args = parser.parse_args()
    
    # 解析要处理的shot设置
    shot_nums = [int(x.strip()) for x in args.shot_nums.split(",")]
    
    convert_neu_det_to_coco(args.data_dir, args.output_dir, shot_nums)

if __name__ == "__main__":
    main() 