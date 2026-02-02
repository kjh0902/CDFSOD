#!/usr/bin/env python
import os
import json
import cv2
from PIL import Image
import argparse

def check_image_sizes(json_file, base_image_dir):
    """
    读取COCO格式JSON文件中引用的所有图像，并获取它们的实际尺寸
    
    Args:
        json_file: COCO格式的JSON文件路径
        base_image_dir: 图像所在的基础目录
    """
    # 读取JSON文件
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 准备结果表格
    results = []
    headers = ["图像ID", "文件名", "JSON中标注尺寸", "实际尺寸", "是否一致", "图像路径"]
    
    # 遍历每张图像
    print(f"检查 {json_file} 中引用的图像尺寸...")
    
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_file = img_info['file_name']
        
        # 尝试获取JSON中的高度和宽度
        json_height = img_info.get('height', 'N/A')
        json_width = img_info.get('width', 'N/A')
        json_size = f"{json_width} x {json_height}" if json_width != 'N/A' and json_height != 'N/A' else "N/A"
        
        # 根据文件扩展名判断图像可能在哪些子目录
        # 通常.jpg或原始图像在train/目录，生成的filled*.png在images/目录
        possible_dirs = [
            "",  # 直接使用base_image_dir
            "train",
            "images",
            "test",
            "val"
        ]
        
        # 尝试在各个可能的路径查找图像
        img_path = None
        for dir_name in possible_dirs:
            path = os.path.join(base_image_dir, dir_name, img_file)
            if os.path.exists(path):
                img_path = path
                break
        
        # 如果找不到图像，记录未找到
        if img_path is None:
            results.append([
                img_id, 
                img_file, 
                json_size, 
                "未找到图像", 
                "N/A",
                "未找到"
            ])
            continue
        
        # 尝试使用不同的库读取图像尺寸
        actual_size = "读取失败"
        matches = "N/A"
        
        try:
            # 先尝试PIL
            with Image.open(img_path) as img:
                width, height = img.size
                actual_size = f"{width} x {height}"
                
                # 检查尺寸是否匹配
                if json_width != 'N/A' and json_height != 'N/A':
                    matches = "是" if width == json_width and height == json_height else "否"
        except Exception as e:
            # 如果PIL失败，尝试OpenCV
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    actual_size = f"{width} x {height}"
                    
                    # 检查尺寸是否匹配
                    if json_width != 'N/A' and json_height != 'N/A':
                        matches = "是" if width == json_width and height == json_height else "否"
            except Exception as e2:
                actual_size = f"读取失败: {str(e2)}"
        
        # 记录结果
        results.append([
            img_id, 
            img_file, 
            json_size, 
            actual_size, 
            matches,
            img_path
        ])
    
    # 输出表格形式的结果
    try:
        from tabulate import tabulate
        print(tabulate(results, headers=headers, tablefmt="grid"))
    except ImportError:
        # 如果没有tabulate库，使用简单格式输出
        print("\n" + "-" * 100)
        print("{:<8} {:<30} {:<15} {:<15} {:<10} {:<30}".format(*headers))
        print("-" * 100)
        for row in results:
            # 限制文件名和路径的长度以适应显示
            row_display = row.copy()
            if len(row_display[1]) > 28:
                row_display[1] = row_display[1][:25] + "..."
            if len(row_display[5]) > 28:
                row_display[5] = row_display[5][:25] + "..."
            print("{:<8} {:<30} {:<15} {:<15} {:<10} {:<30}".format(*row_display))
        print("-" * 100)
    
    # 输出统计信息
    total_images = len(results)
    found_images = sum(1 for row in results if "未找到" not in row[3])
    matching_sizes = sum(1 for row in results if row[4] == "是")
    
    print(f"\n统计信息:")
    print(f"- 总图像数: {total_images}")
    print(f"- 找到的图像数: {found_images}")
    print(f"- 尺寸匹配图像数: {matching_sizes}")
    
    if found_images < total_images:
        print(f"\n警告: 有 {total_images - found_images} 张图像未找到。")
        print("这可能是因为图像路径配置不正确，或者图像文件名与JSON不匹配。")
        print("请检查图像目录是否正确设置，并确保图像文件存在。")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查COCO格式JSON文件中图像的实际尺寸')
    parser.add_argument('--json', type=str, default='data/FISH/annotations/1_shot.json', 
                        help='COCO格式的JSON文件路径')
    parser.add_argument('--image_dir', type=str, default='data/FISH', 
                        help='图像所在的基础目录')
    args = parser.parse_args()
    
    check_image_sizes(args.json, args.image_dir) 