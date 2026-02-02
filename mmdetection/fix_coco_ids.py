#!/usr/bin/env python
import json
import os

def fix_coco_image_ids(json_file, output_file=None):
    """
    Fix the image_id in annotations to properly match the id in images section
    
    Args:
        json_file: Path to the COCO format JSON file
        output_file: Path to save the fixed JSON file (if None, will overwrite the input file)
    """
    # If output_file is not specified, overwrite the input file
    if output_file is None:
        output_file = json_file
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get original image IDs and their file_name/original_file
    images = coco_data["images"]
    annotations = coco_data["annotations"]
    
    # Create mappings for different matching strategies
    id_to_image = {img["id"]: img for img in images}
    original_file_to_id = {}
    filename_to_id = {}
    
    for img in images:
        if "original_file" in img:
            original_file_to_id[img["original_file"]] = img["id"]
        filename_to_id[img["file_name"]] = img["id"]
    
    # Print original state for verification
    print(f"Original state:")
    print(f"- Total images: {len(images)}")
    print(f"- Total annotations: {len(annotations)}")
    
    # Check for mismatches
    missing_ids = []
    for anno in annotations:
        if anno["image_id"] not in id_to_image:
            missing_ids.append(anno["image_id"])
    
    if missing_ids:
        print(f"Found {len(missing_ids)} annotations with image_id not matching any image.")
        missing_ids_set = set(missing_ids)
        print(f"Missing image IDs: {missing_ids_set}")
    
    # Fix annotations by matching based on original_file if available
    fixed_count = 0
    for anno in annotations:
        image_id = anno["image_id"]
        
        # If the image_id already exists in images, no need to fix
        if image_id in id_to_image:
            continue
        
        # Try to fix by matching original_file
        if "original_file" in anno and anno["original_file"] in original_file_to_id:
            old_id = anno["image_id"]
            anno["image_id"] = original_file_to_id[anno["original_file"]]
            print(f"Fixed: annotation {anno['id']} - changed image_id from {old_id} to {anno['image_id']} based on original_file")
            fixed_count += 1
    
    # Verify all annotations now have a valid image_id
    orphaned_annos = []
    for anno in annotations:
        if anno["image_id"] not in id_to_image:
            orphaned_annos.append(anno)
    
    if orphaned_annos:
        print(f"\nWarning: Still found {len(orphaned_annos)} annotations without matching images:")
        for anno in orphaned_annos[:5]:  # Only show first 5
            print(f"  - Annotation ID {anno['id']}, references image_id {anno['image_id']}")
        if len(orphaned_annos) > 5:
            print(f"  ... and {len(orphaned_annos)-5} more")
    else:
        print(f"\nSuccess: All annotations now have a matching image_id")
    
    # Save the fixed JSON file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nFixed {fixed_count} annotations.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Path to the COCO format JSON file
    json_file = "data/FISH/annotations/annotations.json"
    output_file = "data/FISH/annotations/annotations_fixed.json"
    
    # Fix the image_id in annotations
    fix_coco_image_ids(json_file, output_file) 