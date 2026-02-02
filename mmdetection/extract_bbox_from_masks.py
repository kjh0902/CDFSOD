#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle
import argparse

def extract_bbox_from_mask(mask_path):
    """
    从mask图像中提取边界框坐标
    
    Args:
        mask_path: mask图像的路径
        
    Returns:
        bbox: [x, y, width, height] 格式的边界框坐标，如果提取失败则返回None
    """
    try:
        # 读取mask图像
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # 如果图像是彩色的，转换为灰度图
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 确保mask是二值图像
        if np.max(mask) > 1:
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 查找非零像素的坐标
        non_zero = cv2.findNonZero(mask)
        
        if non_zero is None or len(non_zero) == 0:
            print(f"警告: {mask_path} 中没有找到非零像素")
            return None
            
        # 计算边界框
        x, y, w, h = cv2.boundingRect(non_zero)
        
        return [x, y, w, h]
    except Exception as e:
        print(f"处理 {mask_path} 时出错: {e}")
        return None

def process_folder(folder_path, output_json=None, show_visualization=True):
    """
    处理文件夹中的所有mask图像并提取边界框
    
    Args:
        folder_path: 包含mask图像的文件夹路径
        output_json: 可选，输出JSON文件的路径
        show_visualization: 是否显示可视化结果
    """
    # 查找所有mask_{n}.png文件
    mask_pattern = "mask_*.png"
    mask_files = [f for f in os.listdir(folder_path) if f.startswith("mask_") and f.endswith(".png")]
    mask_files.sort()  # 按名称排序
    
    if not mask_files:
        print(f"在 {folder_path} 中没有找到mask文件")
        return
    
    results = {}
    
    # 处理每个mask文件
    for mask_file in mask_files:
        mask_path = os.path.join(folder_path, mask_file)
        mask_num = mask_file.split("_")[1].split(".")[0]  # 从mask_1.png提取1
        
        # 提取边界框
        bbox = extract_bbox_from_mask(mask_path)
        
        if bbox:
            print(f"Mask {mask_num}: {mask_file} 边界框 = {bbox}")
            results[mask_num] = {
                "mask_file": mask_file,
                "bbox": bbox,
                "x": bbox[0],
                "y": bbox[1],
                "width": bbox[2],
                "height": bbox[3]
            }
        else:
            print(f"Mask {mask_num}: {mask_file} 无法提取边界框")
    
    # 保存结果为JSON文件
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到 {output_json}")
    
    # 可视化结果
    if show_visualization and results:
        plt.figure(figsize=(15, 10))
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols
        
        for i, (mask_num, info) in enumerate(results.items(), 1):
            mask_path = os.path.join(folder_path, info["mask_file"])
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if len(mask.shape) > 2 and mask.shape[2] > 1:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            # 查找对应的生成图像
            generated_img_path = os.path.join(folder_path, f"generated_image_rank{mask_num}.png")
            if os.path.exists(generated_img_path):
                img = cv2.imread(generated_img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros_like(mask)
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
            
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            
            # 绘制边界框
            x, y, w, h = info["bbox"]
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
            plt.title(f"Mask {mask_num} - 边界框: {info['bbox']}")
            plt.axis('off')
        
        plt.tight_layout()
        
        # 保存可视化结果
        viz_path = os.path.join(folder_path, "mask_bboxes.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到 {viz_path}")
        
        if show_visualization:
            plt.show()
    
    # 输出边界框信息的COCO格式结构
    coco_annotations = []
    for mask_num, info in results.items():
        coco_annotations.append({
            "id": int(mask_num),
            "image_id": int(mask_num),
            "category_id": 1,  # 假设类别ID为1
            "bbox": info["bbox"],
            "area": info["width"] * info["height"],
            "segmentation": [],
            "iscrowd": 0
        })
    
    print("\nCOCO格式的标注信息:")
    print(json.dumps(coco_annotations, indent=2))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从mask图像中提取边界框坐标')
    parser.add_argument('--folder', type=str, 
                        default='/home/add_disk2/qiuxingyu/mmdetection/result/FISH/results_coco_0.8_target_1_cocotext_1_targettext_1.2_20250429_155533/9852_Acanthopagrus_palmaris_f000070',
                        help='包含mask图像的文件夹路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件的路径')
    parser.add_argument('--no_viz', action='store_true',
                        help='不显示可视化结果')
    args = parser.parse_args()
    
    process_folder(args.folder, args.output, not args.no_viz) 