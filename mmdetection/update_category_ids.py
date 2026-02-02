#!/usr/bin/env python
import os
import json
import argparse

def update_category_ids(input_json, reference_json, output_json=None):
    """
    更新输入JSON文件中的类别ID，使其与参考JSON文件中的类别ID一致
    
    Args:
        input_json: 输入的outpaint结果JSON文件路径
        reference_json: 包含标准类别ID的参考JSON文件路径
        output_json: 输出的更新后JSON文件路径，如果为None则使用默认路径
    """
    print(f"开始处理...")
    print(f"输入文件: {input_json}")
    print(f"参考文件: {reference_json}")
    
    # 如果未指定输出文件，则在输入文件名后添加_updated后缀
    if output_json is None:
        base_name, ext = os.path.splitext(input_json)
        output_json = f"{base_name}_updated{ext}"
    
    print(f"输出文件: {output_json}")
    
    # 1. 读取参考文件中的类别信息
    try:
        with open(reference_json, 'r') as f:
            reference_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取参考文件: {e}")
        return False
    
    # 从参考文件中提取类别名称和ID的映射关系
    category_name_to_id = {}
    for category in reference_data.get('categories', []):
        category_name_to_id[category['name']] = category['id']
    
    if not category_name_to_id:
        print("错误: 参考文件中未找到类别信息!")
        return False
    
    print(f"\n找到 {len(category_name_to_id)} 个标准类别:")
    for name, id in category_name_to_id.items():
        print(f"  - {name}: {id}")
    
    # 2. 读取待处理文件
    try:
        with open(input_json, 'r') as f:
            outpaint_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取输入文件: {e}")
        return False
    
    # 3. 遍历并更新类别ID
    updated_count = 0
    sample_count = 0
    unknown_categories = set()
    
    if 'result_dirs' in outpaint_data:
        for result_dir_info in outpaint_data['result_dirs']:
            for sample in result_dir_info.get('samples', []):
                sample_count += 1
                # 获取当前的类别名称
                current_category = sample.get('category', '')
                
                # 查找正确的类别ID
                if current_category in category_name_to_id:
                    # 记录原始类别ID (如果存在)
                    original_category_id = sample.get('category_id', None)
                    
                    # 更新为标准类别ID
                    sample['category_id'] = category_name_to_id[current_category]
                    
                    print(f"样本 {sample.get('sample_id')}: 类别 '{current_category}' - "
                          f"ID从 {original_category_id} 更新为 {sample['category_id']}")
                    updated_count += 1
                else:
                    print(f"警告: 样本 {sample.get('sample_id')} 的类别 '{current_category}' 在标准类别中未找到")
                    unknown_categories.add(current_category)
                    
                # 同时更新所有outpainted_images中的params中的类别信息
                for image in sample.get('outpainted_images', []):
                    if 'params' in image and 'category' in image['params']:
                        cat_name = image['params']['category']
                        if cat_name in category_name_to_id:
                            image['params']['category_id'] = category_name_to_id[cat_name]
    else:
        print("警告: 输入文件中未找到'result_dirs'字段!")
    
    # 4. 保存更新后的文件
    try:
        with open(output_json, 'w') as f:
            json.dump(outpaint_data, f, indent=2)
        print(f"\n更新完成! 共处理了 {sample_count} 个样本，更新了 {updated_count} 个样本的类别ID")
        
        if unknown_categories:
            print(f"\n警告: 发现 {len(unknown_categories)} 个未知类别:")
            for cat in unknown_categories:
                print(f"  - {cat}")
            print("这些类别在参考文件中未找到匹配项，可能需要手动处理。")
        
        print(f"更新后的文件已保存到: {output_json}")
        return True
    except Exception as e:
        print(f"错误: 保存输出文件时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="更新outpaint结果JSON文件中的类别ID")
    parser.add_argument("--input", type=str, required=True,
                        help="输入的outpaint结果JSON文件路径")
    parser.add_argument("--reference", type=str, required=True,
                        help="包含标准类别ID的参考JSON文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出的更新后JSON文件路径 (默认为<input>_updated.json)")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    if not os.path.exists(args.reference):
        print(f"错误: 参考文件不存在: {args.reference}")
        return 1
    
    # 执行更新
    success = update_category_ids(args.input, args.reference, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 