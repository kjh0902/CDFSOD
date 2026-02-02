#!/usr/bin/env python
import json
import os
import argparse
from collections import Counter

def fix_duplicate_annotation_ids(input_json, output_json=None):
    """
    Fix duplicate annotation IDs in a COCO format JSON file while preserving all annotations
    
    Args:
        input_json: Path to input COCO JSON file with duplicate annotation IDs
        output_json: Path to output fixed JSON file. If None, adds '_fixed' suffix
    """
    print(f"Loading dataset: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Create output path if not specified
    if output_json is None:
        base_name, ext = os.path.splitext(input_json)
        output_json = f"{base_name}_fixed{ext}"
    
    # Check for duplicate annotation IDs
    all_ann_ids = [ann["id"] for ann in data["annotations"]]
    id_counter = Counter(all_ann_ids)
    duplicate_ids = [id for id, count in id_counter.items() if count > 1]
    
    if not duplicate_ids:
        print("No duplicate annotation IDs found. No changes needed.")
        return
    
    print(f"Found {len(duplicate_ids)} IDs with duplicates")
    
    # Find the maximum annotation ID
    max_ann_id = max(all_ann_ids)
    new_id_counter = max_ann_id + 1
    
    # Track annotations IDs that have been seen
    processed_ids = set()
    
    # Replace duplicate IDs with new unique IDs
    for i, ann in enumerate(data["annotations"]):
        ann_id = ann["id"]
        
        # If this ID appears multiple times and we've seen it before, assign a new ID
        if ann_id in duplicate_ids and ann_id in processed_ids:
            ann["id"] = new_id_counter
            new_id_counter += 1
            print(f"Reassigning duplicate ID {ann_id} to {ann['id']}")
        else:
            # Mark this ID as seen
            processed_ids.add(ann_id)
    
    # Verify all IDs are now unique
    updated_ann_ids = [ann["id"] for ann in data["annotations"]]
    if len(updated_ann_ids) != len(set(updated_ann_ids)):
        print("Warning: There are still duplicate IDs after fixing!")
    else:
        print("All annotation IDs are now unique")
    
    # Check for duplicate image IDs as well (just to be safe)
    all_img_ids = [img["id"] for img in data["images"]]
    img_id_counter = Counter(all_img_ids)
    duplicate_img_ids = [id for id, count in img_id_counter.items() if count > 1]
    
    if duplicate_img_ids:
        print(f"Found {len(duplicate_img_ids)} image IDs with duplicates")
        
        # Find the maximum image ID
        max_img_id = max(all_img_ids)
        new_img_id_counter = max_img_id + 1
        
        # Track image IDs that have been seen
        processed_img_ids = set()
        
        # Create mapping for old to new image IDs
        img_id_mapping = {}
        
        # Replace duplicate image IDs with new unique IDs
        for img in data["images"]:
            img_id = img["id"]
            
            # If this ID appears multiple times and we've seen it before, assign a new ID
            if img_id in duplicate_img_ids and img_id in processed_img_ids:
                new_id = new_img_id_counter
                img_id_mapping[img_id] = new_id
                img["id"] = new_id
                new_img_id_counter += 1
                print(f"Reassigning duplicate image ID {img_id} to {new_id}")
            else:
                # Mark this ID as seen
                processed_img_ids.add(img_id)
                # Keep the mapping for annotations that refer to this image
                img_id_mapping[img_id] = img_id
        
        # Update annotation image_id references for the replaced IDs
        for ann in data["annotations"]:
            if ann["image_id"] in img_id_mapping:
                ann["image_id"] = img_id_mapping[ann["image_id"]]
    
    # Save fixed dataset
    print(f"Saving fixed dataset to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print statistics
    print("\nFix completed successfully!")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Total categories: {len(data['categories'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix duplicate annotation IDs in a COCO JSON file")
    parser.add_argument("--input", required=True, help="Path to input COCO JSON file")
    parser.add_argument("--output", help="Path to output fixed JSON file. If not specified, adds '_fixed' suffix")
    
    args = parser.parse_args()
    
    fix_duplicate_annotation_ids(args.input, args.output) 