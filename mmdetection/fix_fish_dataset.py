#!/usr/bin/env python
import json
import os
import shutil

def fix_fish_dataset():
    """
    修复FISH数据集中的JSON文件，确保annotations.json和1_shot.json中的图像ID和文件名正确对应
    只保留annotations.json中的标注，并正确映射到1_shot.json中的图像ID
    """
    # 文件路径
    fish_dir = "data/FISH"
    annotations_file = os.path.join(fish_dir, "annotations/annotations.json")
    one_shot_file = os.path.join(fish_dir, "annotations/1_shot.json")
    output_file = os.path.join(fish_dir, "annotations/fixed_1_shot.json")
    
    # 读取两个JSON文件
    with open(annotations_file, 'r') as f1:
        annotations_data = json.load(f1)
    
    with open(one_shot_file, 'r') as f2:
        one_shot_data = json.load(f2)
    
    # 打印原始状态进行确认
    print("原始状态:")
    print(f"annotations.json - 图像数: {len(annotations_data['images'])}, 标注数: {len(annotations_data['annotations'])}")
    print(f"1_shot.json - 图像数: {len(one_shot_data['images'])}, 标注数: {len(one_shot_data['annotations'])}")
    
    # 创建映射: 从original_file到1_shot.json中的image_id
    original_file_to_one_shot_id = {}
    filled_num_to_one_shot_id = {}
    
    # 在1_shot.json中，获取filled_*.png文件的编号到ID映射
    for img in one_shot_data["images"]:
        if img["file_name"].startswith("filled_"):
            num = img["file_name"].replace("filled_", "").replace(".png", "")
            filled_num_to_one_shot_id[num] = img["id"]
    
    # 创建annotations.json中original_file到ID的映射
    filled_num_to_anno_id = {}
    for img in annotations_data["images"]:
        if "original_file" in img and img["original_file"].startswith("filled_"):
            num = img["original_file"].replace("filled_", "").replace(".png", "")
            filled_num_to_anno_id[num] = img["id"]
            # 如果这个编号在1_shot.json中存在，创建映射
            if num in filled_num_to_one_shot_id:
                original_file_to_one_shot_id[img["original_file"]] = filled_num_to_one_shot_id[num]
    
    # 显示映射信息
    print("\nfilled_*.png文件在两个文件中的ID映射:")
    for num in sorted(filled_num_to_anno_id.keys()):
        if num in filled_num_to_one_shot_id:
            print(f"  filled_{num}.png: annotations.json ID={filled_num_to_anno_id[num]} -> 1_shot.json ID={filled_num_to_one_shot_id[num]}")
    
    # 创建修正后的数据结构
    fixed_data = one_shot_data.copy()
    # 清空原有标注
    fixed_data["annotations"] = []
    
    # 从annotations.json中获取标注并映射到1_shot.json的image_id
    next_anno_id = 1
    for anno in annotations_data["annotations"]:
        # 获取标注对应的图像
        anno_img_id = anno["image_id"]
        
        # 查找对应的original_file
        original_file = None
        for img in annotations_data["images"]:
            if img["id"] == anno_img_id and "original_file" in img:
                original_file = img["original_file"]
                break
        
        # 如果找到original_file并且有对应的1_shot.json ID映射
        if original_file and original_file in original_file_to_one_shot_id:
            # 创建新标注
            new_anno = anno.copy()
            new_anno["id"] = next_anno_id
            next_anno_id += 1
            # 更新image_id为1_shot.json中的ID
            new_anno["image_id"] = original_file_to_one_shot_id[original_file]
            # 添加到fixed_data
            fixed_data["annotations"].append(new_anno)
            print(f"添加标注: {original_file} -> image_id={new_anno['image_id']}, bbox={new_anno['bbox']}")
    
    # 验证所有标注都有对应的图像
    image_ids = {img["id"] for img in fixed_data["images"]}
    valid_annotations = []
    
    for anno in fixed_data["annotations"]:
        if anno["image_id"] in image_ids:
            valid_annotations.append(anno)
        else:
            print(f"警告: 标注ID={anno['id']}引用了不存在的图像ID={anno['image_id']}")
    
    # 只保留有效的标注
    fixed_data["annotations"] = valid_annotations
    
    # 保存修正后的文件
    with open(output_file, 'w') as f:
        json.dump(fixed_data, f, indent=2)
    
    print(f"\n保存修正后的文件: {output_file}")
    print(f"- 总图像数: {len(fixed_data['images'])}")
    print(f"- 有效标注数: {len(fixed_data['annotations'])}")
    
    print("\n使用指南:")
    print(f"1. 检查 {output_file} 确认标注是否正确")
    print(f"2. 如果正确，可以替换原始文件: cp {output_file} {one_shot_file}")

if __name__ == "__main__":
    fix_fish_dataset() 