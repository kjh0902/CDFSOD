#!/usr/bin/env python
import os
import json
import argparse

def update_by_sample_id(input_json, coco_json, output_json=None):
    """
    根据COCO格式的JSON文件中的图片文件名(不带扩展名)作为sample_id，更新输入JSON文件的category和category_id
    
    Args:
        input_json: 输入的JSON文件路径 (outpaint_results_1shot.json)
        coco_json: COCO格式的JSON文件路径，包含正确的category映射
        output_json: 输出的更新后JSON文件路径，如果为None则使用默认路径
    """
    print(f"开始处理...")
    print(f"输入文件: {input_json}")
    print(f"COCO参考文件: {coco_json}")
    
    # 如果未指定输出文件，则在输入文件名后添加_updated后缀
    if output_json is None:
        base_name, ext = os.path.splitext(input_json)
        output_json = f"{base_name}_updated{ext}"
    
    print(f"输出文件: {output_json}")
    
    # 1. 读取COCO格式的参考文件
    try:
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取COCO参考文件: {e}")
        return False
    
    # 2. 创建category_id到category_name的映射
    category_id_to_name = {}
    for category in coco_data.get('categories', []):
        category_id_to_name[category['id']] = category['name']
    
    if not category_id_to_name:
        print("错误: COCO参考文件中未找到有效的category信息")
        return False
    
    print(f"从COCO参考文件中提取了 {len(category_id_to_name)} 个类别")
    
    # 3. 创建image_id到annotation的映射
    image_id_to_annotation = {}
    for annotation in coco_data.get('annotations', []):
        image_id_to_annotation[annotation['image_id']] = annotation
    
    # 4. 创建sample_id(图片文件名不带扩展名)到category的映射
    sample_id_to_category = {}
    
    for image in coco_data.get('images', []):
        image_id = image['id']
        # 从文件名中提取sample_id (不带扩展名)
        file_name = image['file_name']
        sample_id = os.path.splitext(file_name)[0]
        
        # 查找对应的annotation
        if image_id in image_id_to_annotation:
            annotation = image_id_to_annotation[image_id]
            category_id = annotation['category_id']
            
            # 查找对应的category_name
            if category_id in category_id_to_name:
                category_name = category_id_to_name[category_id]
                
                sample_id_to_category[sample_id] = {
                    'category_id': category_id,
                    'category_name': category_name
                }
    
    if not sample_id_to_category:
        print("错误: 未能创建有效的sample_id到category的映射关系!")
        return False
    
    print(f"\n创建了 {len(sample_id_to_category)} 个sample_id到category的映射关系")
    
    # 5. 读取待处理文件
    try:
        with open(input_json, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取输入文件: {e}")
        return False
    
    # 6. 遍历并更新category和category_id
    updated_count = 0
    sample_count = 0
    not_found_sample_ids = set()
    
    # 检查文件结构，这里假设outpaint_results_1shot.json的结构是有一个顶级的samples列表
    if 'samples' in input_data:
        samples = input_data['samples']
    else:
        print("警告: 输入文件中未找到'samples'字段，尝试在根级别处理数据")
        samples = [input_data]  # 如果没有samples字段，就把整个数据当作一个样本
    
    for sample in samples:
        sample_count += 1
        sample_id = sample.get('sample_id')
        
        if not sample_id:
            print(f"警告: 样本 #{sample_count} 没有sample_id字段，跳过")
            continue
        
        # 查找映射关系
        if sample_id in sample_id_to_category:
            mapping_info = sample_id_to_category[sample_id]
            
            # 记录原始信息
            original_category = sample.get('category', '')
            original_category_id = sample.get('category_id', None)
            
            # 更新category和category_id
            if 'category_name' in mapping_info and mapping_info['category_name']:
                sample['category'] = mapping_info['category_name']
                # 同时更新categories字段（如果存在）
                if 'categories' in sample:
                    sample['categories'] = [mapping_info['category_name']]
            
            if 'category_id' in mapping_info and mapping_info['category_id'] is not None:
                sample['category_id'] = mapping_info['category_id']
            
            print(f"样本 {sample_id}: 类别从 '{original_category}' 更新为 '{sample.get('category')}', "
                  f"ID从 {original_category_id} 更新为 {sample.get('category_id')}")
            
            # 同时更新outpainted_images中的数据
            if 'outpainted_images' in sample:
                for image in sample['outpainted_images']:
                    if 'params' in image:
                        if 'category_name' in mapping_info and mapping_info['category_name']:
                            # 更新categories字段
                            if 'categories' in image['params']:
                                image['params']['categories'] = [mapping_info['category_name']]
                        
                        if 'category_id' in mapping_info and mapping_info['category_id'] is not None:
                            image['params']['category_id'] = mapping_info['category_id']
            
            updated_count += 1
        else:
            print(f"警告: 样本ID '{sample_id}' 在映射文件中未找到")
            not_found_sample_ids.add(sample_id)
    
    # 7. 保存更新后的文件
    try:
        with open(output_json, 'w') as f:
            json.dump(input_data, f, indent=2)
        print(f"\n更新完成! 共处理了 {sample_count} 个样本，更新了 {updated_count} 个样本")
        
        if not_found_sample_ids:
            print(f"\n警告: 有 {len(not_found_sample_ids)} 个sample_id未在映射文件中找到:")
            for sample_id in not_found_sample_ids:
                print(f"  - {sample_id}")
        
        print(f"更新后的文件已保存到: {output_json}")
        return True
    except Exception as e:
        print(f"错误: 保存输出文件时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="根据COCO格式的JSON文件更新JSON文件中的category和category_id")
    parser.add_argument("--input", type=str, required=True,
                      help="输入的JSON文件路径 (outpaint_results_1shot.json)")
    parser.add_argument("--coco", type=str, required=True,
                      help="COCO格式的JSON文件路径，包含正确的category映射")
    parser.add_argument("--output", type=str, default=None,
                      help="输出的更新后JSON文件路径 (默认为<input>_updated.json)")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    if not os.path.exists(args.coco):
        print(f"错误: COCO参考文件不存在: {args.coco}")
        return 1
    
    # 执行更新
    success = update_by_sample_id(args.input, args.coco, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 