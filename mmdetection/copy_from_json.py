#!/usr/bin/env python
import os
import json
import shutil
import argparse
from PIL import Image

def copy_images_from_json(json_file, target_dir, dataset_name=None):
    """
    从outpaint结果JSON文件中提取图片路径，并复制到目标目录。
    
    Args:
        json_file: outpaint结果JSON文件路径
        target_dir: 目标目录路径
        dataset_name: 数据集名称，用于重命名
    
    Returns:
        copied_count: 复制的文件数量
        replaced_count: 替换的文件数量
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 读取JSON文件
    print(f"读取JSON文件: {json_file}")
    try:
        with open(json_file, 'r') as f:
            outpaint_results = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return 0, 0
    
    # 统计计数
    copied_count = 0
    replaced_count = 0
    failed_count = 0
    
    # 验证JSON结构
    if "result_dirs" in outpaint_results:
        # 标准结构
        result_dirs = outpaint_results["result_dirs"]
        print(f"找到 {len(result_dirs)} 个结果目录")
        
        # 处理每个结果目录
        for result_dir_info in result_dirs:
            result_dir = result_dir_info.get("result_dir", "")
            
            # 处理每个样本
            for sample in result_dir_info.get("samples", []):
                sample_id = sample.get("sample_id", "unknown")
                category = sample.get("category", "unknown")
                
                print(f"处理样本 {sample_id} (类别: {category})")
                
                # 处理每个outpaint图像
                for outpainted_image in sample.get("outpainted_images", []):
                    outpainted_image_path = outpainted_image.get("outpainted_image_path", "")
                    
                    if not outpainted_image_path or not os.path.exists(outpainted_image_path):
                        print(f"警告: 图像路径不存在: {outpainted_image_path}")
                        failed_count += 1
                        continue
                    
                    # 提取文件名部分
                    original_filename = os.path.basename(outpainted_image_path)
                    
                    # 获取rank编号
                    rank_number = original_filename.split('_')[-1].split('.')[0]
                    
                    # 创建新文件名
                    if dataset_name:
                        new_filename = f"{dataset_name}_{sample_id}_outpaint_{rank_number}.png"
                    else:
                        new_filename = f"{sample_id}_outpaint_{rank_number}.png"
                    
                    # 目标文件路径
                    dest_path = os.path.join(target_dir, new_filename)
                    
                    # 复制文件
                    try:
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                            shutil.copy2(outpainted_image_path, dest_path)
                            replaced_count += 1
                            print(f"替换: {outpainted_image_path} -> {dest_path}")
                        else:
                            shutil.copy2(outpainted_image_path, dest_path)
                            copied_count += 1
                            print(f"复制: {outpainted_image_path} -> {dest_path}")
                            
                        # 尝试保存图像元数据
                        try:
                            with Image.open(dest_path) as img:
                                width, height = img.size
                                
                                # 元数据保存在同目录下的metadata.json文件中
                                metadata_path = os.path.join(target_dir, "image_metadata.json")
                                metadata = {}
                                
                                if os.path.exists(metadata_path):
                                    try:
                                        with open(metadata_path, 'r') as f:
                                            metadata = json.load(f)
                                    except:
                                        metadata = {}
                                
                                # 添加元数据
                                metadata[new_filename] = {
                                    'width': width, 
                                    'height': height,
                                    'category': category,
                                    'sample_id': sample_id
                                }
                                
                                # 保存元数据
                                with open(metadata_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                        except Exception as e:
                            print(f"无法处理图像元数据: {e}")
                    except Exception as e:
                        print(f"复制文件失败 {outpainted_image_path}: {e}")
                        failed_count += 1
    else:
        # 尝试在直接的JSON结构中查找图片
        print("未找到standard result_dirs结构，尝试检查替代格式...")
        
        # 检查是否有直接的outpainted_images字段
        if "outpainted_images" in outpaint_results:
            outpainted_images = outpaint_results["outpainted_images"]
            print(f"找到 {len(outpainted_images)} 个outpaint图像")
            
            for img_info in outpainted_images:
                outpainted_image_path = img_info.get("outpainted_image_path", "")
                sample_id = img_info.get("sample_id", "unknown")
                
                if not outpainted_image_path or not os.path.exists(outpainted_image_path):
                    print(f"警告: 图像路径不存在: {outpainted_image_path}")
                    failed_count += 1
                    continue
                
                # 提取文件名和rank编号
                original_filename = os.path.basename(outpainted_image_path)
                parts = original_filename.split('_')
                rank_number = parts[-1].split('.')[0] if parts and parts[-1] else "1"
                
                # 创建新文件名
                if dataset_name:
                    new_filename = f"{dataset_name}_{sample_id}_outpaint_{rank_number}.png"
                else:
                    new_filename = f"{sample_id}_outpaint_{rank_number}.png"
                
                # 目标文件路径
                dest_path = os.path.join(target_dir, new_filename)
                
                # 复制文件
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                        shutil.copy2(outpainted_image_path, dest_path)
                        replaced_count += 1
                        print(f"替换: {outpainted_image_path} -> {dest_path}")
                    else:
                        shutil.copy2(outpainted_image_path, dest_path)
                        copied_count += 1
                        print(f"复制: {outpainted_image_path} -> {dest_path}")
                except Exception as e:
                    print(f"复制文件失败 {outpainted_image_path}: {e}")
                    failed_count += 1
        else:
            # 尝试搜索所有可能包含图片路径的字段
            print("尝试搜索所有可能包含图片路径的字段...")
            
            # 递归函数，搜索包含image_path的字段
            def find_image_paths(obj, paths=None):
                if paths is None:
                    paths = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if (isinstance(key, str) and 
                            ('image_path' in key.lower() or 'outpainted' in key.lower()) and 
                            isinstance(value, str) and value.endswith(('.png', '.jpg', '.jpeg'))):
                            paths.append(value)
                        elif isinstance(value, (dict, list)):
                            find_image_paths(value, paths)
                elif isinstance(obj, list):
                    for item in obj:
                        find_image_paths(item, paths)
                
                return paths
            
            # 搜索所有图片路径
            all_image_paths = find_image_paths(outpaint_results)
            print(f"找到 {len(all_image_paths)} 个可能的图片路径")
            
            # 处理找到的所有图片路径
            for img_path in all_image_paths:
                if not os.path.exists(img_path):
                    print(f"警告: 图像路径不存在: {img_path}")
                    failed_count += 1
                    continue
                
                # 从路径中提取sample_id
                path_parts = img_path.split(os.path.sep)
                sample_id = "unknown"
                for part in path_parts:
                    if part.isalnum() and len(part) >= 5:
                        sample_id = part
                        break
                
                # 提取文件名和rank编号
                original_filename = os.path.basename(img_path)
                if "outpaint" in original_filename or "rank" in original_filename:
                    parts = original_filename.split('_')
                    rank_part = [p for p in parts if p.startswith('rank') or p.isdigit()]
                    rank_number = rank_part[-1].replace("rank", "") if rank_part else "1"
                else:
                    rank_number = "1"
                
                # 创建新文件名
                if dataset_name:
                    new_filename = f"{dataset_name}_{sample_id}_outpaint_{rank_number}.png"
                else:
                    new_filename = f"{sample_id}_outpaint_{rank_number}.png"
                
                # 目标文件路径
                dest_path = os.path.join(target_dir, new_filename)
                
                # 复制文件
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                        shutil.copy2(img_path, dest_path)
                        replaced_count += 1
                        print(f"替换: {img_path} -> {dest_path}")
                    else:
                        shutil.copy2(img_path, dest_path)
                        copied_count += 1
                        print(f"复制: {img_path} -> {dest_path}")
                except Exception as e:
                    print(f"复制文件失败 {img_path}: {e}")
                    failed_count += 1
    
    # 输出统计信息
    total_processed = copied_count + replaced_count
    print(f"\n处理完成:")
    print(f"成功处理: {total_processed}个文件 ({copied_count}个新复制, {replaced_count}个替换)")
    
    if failed_count > 0:
        print(f"处理失败: {failed_count}个文件")
    
    return copied_count, replaced_count

def main():
    parser = argparse.ArgumentParser(description='从outpaint结果JSON中提取并复制图片')
    parser.add_argument('--json', type=str, required=True, help='outpaint结果JSON文件路径')
    parser.add_argument('--target', type=str, required=True, help='目标目录路径')
    parser.add_argument('--dataset', type=str, help='数据集名称，用于重命名 (可选)')
    
    args = parser.parse_args()
    
    print(f"JSON文件: {args.json}")
    print(f"目标目录: {args.target}")
    if args.dataset:
        print(f"数据集名称: {args.dataset}")
    
    # 检查JSON文件是否存在
    if not os.path.exists(args.json):
        print(f"错误: JSON文件不存在: {args.json}")
        return
    
    # 执行复制操作
    copied, replaced = copy_images_from_json(
        args.json,
        args.target,
        dataset_name=args.dataset
    )
    
    # 输出结果统计
    print(f"\n总结:")
    print(f"- 从 {args.json} 提取图片到 {args.target}")
    print(f"- 复制了 {copied} 个新文件")
    print(f"- 替换了 {replaced} 个已存在的文件")
    print(f"- 总计处理: {copied + replaced} 个文件")
    
    # 提供下一步操作建议
    if copied + replaced > 0:
        print("\n下一步操作建议:")
        print(f"1. 使用convert_to_coco_format.py将图片转换为COCO格式")
        print(f"2. 使用merge_coco_annotations.py合并原始标注和新生成的标注")
        print(f"   python merge_coco_annotations.py --dataset {args.dataset or 'YOUR_DATASET'}")
    else:
        print("\n警告: 未成功处理任何图片。请检查JSON文件格式和图片路径是否正确。")
        print("检查事项:")
        print("1. JSON文件是否包含有效的图片路径")
        print("2. 图片路径是否存在")
        print("3. JSON结构是否符合预期格式")

if __name__ == "__main__":
    main() 