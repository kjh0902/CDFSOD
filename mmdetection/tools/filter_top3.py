import json
import os

# 输入输出文件路径
input_path = 'data/Camouflage/annotations/1_shot.json'
output_path = 'data/Camouflage/annotations/1_shot_top3.json'

# 读取原始json
with open(input_path, 'r') as f:
    data = json.load(f)

# 过滤images
filtered_images = []
remove_suffixes = ('outpaint_4.png', 'outpaint_5.png')
for img in data['images']:
    if not any(img['file_name'].endswith(suf) for suf in remove_suffixes):
        filtered_images.append(img)

# 获取保留的image_id
keep_ids = set(img['id'] for img in filtered_images)

# 过滤annotations
filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in keep_ids]

# 构建新json
new_data = {
    'info': data.get('info', {}),
    'licenses': data.get('licenses', []),
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': data.get('categories', [])
}

# 保存新json
with open(output_path, 'w') as f:
    json.dump(new_data, f, indent=2)

print(f"过滤完成，保留 {len(filtered_images)} 张图片，{len(filtered_annotations)} 条标注。输出到 {output_path}") 