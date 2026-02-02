#!/usr/bin/env python
import os
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def apply_selective_blur(input_folder, output_folder, annotations_json, kernel_size=43):
    """
    Apply Gaussian blur to areas outside of bounding boxes in images.
    
    Args:
        input_folder: Path to folder containing original images
        output_folder: Path to save blurred images
        annotations_json: Path to COCO annotations json file
        kernel_size: Size of Gaussian kernel (must be odd)
    """
    # Make sure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load annotations
    with open(annotations_json, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping of image_id to annotations
    image_id_to_annot = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annot:
            image_id_to_annot[image_id] = []
        image_id_to_annot[image_id].append(annotation)
    
    # Process each image
    for img_info in tqdm(coco_data['images']):
        filename = img_info['file_name']
        image_id = img_info['id']
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Skip if input file doesn't exist
        if not os.path.exists(input_path):
            print(f"Warning: File not found: {input_path}")
            continue
        
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Could not read image: {input_path}")
            continue
        
        # Apply Gaussian blur to the entire image
        blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Get annotations for this image
        annotations = image_id_to_annot.get(image_id, [])
        
        # Create a mask for the bounding boxes
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Draw bbox areas on the mask
        for annotation in annotations:
            bbox = annotation['bbox']
            x, y, width, height = [int(v) for v in bbox]
            mask[y:y+height, x:x+width] = 255
        
        # Create the final image: blur outside, original inside bbox
        final_img = blurred_img.copy()
        final_img[mask > 0] = img[mask > 0]
        
        # Save the result
        cv2.imwrite(output_path, final_img)
    
    print(f"Processing complete. Selectively blurred images saved to {output_folder}")

def process_all_datasets(datasets, kernel_size=43):
    """
    Process multiple datasets with the same kernel size
    
    Args:
        datasets: List of dataset names
        kernel_size: Size of Gaussian kernel
    """
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        input_folder = f"data/{dataset}/train_1shotnew"
        output_folder = f"data/{dataset}/train_1shotnew_blurred"
        annotations_json = f"data/{dataset}/annotations/coco_annotations.json"
        
        # Check if the dataset files exist
        if not os.path.exists(annotations_json):
            print(f"Warning: Annotations file not found for {dataset}: {annotations_json}")
            continue
            
        if not os.path.exists(input_folder):
            print(f"Warning: Input folder not found for {dataset}: {input_folder}")
            continue
            
        # Apply selective blur for this dataset
        apply_selective_blur(input_folder, output_folder, annotations_json, kernel_size)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply Gaussian blur to areas outside of bounding boxes in images')
    parser.add_argument('--datasets', nargs='+', default=['UODD', 'ArTaxOr', 'clipart1k', "FISH", "NEU-DET", "DIOR"], 
                        help='List of dataset names to process')
    parser.add_argument('--kernel_size', type=int, default=43,
                        help='Size of Gaussian kernel (must be odd)')
    args = parser.parse_args()
    
    # Process all specified datasets
    process_all_datasets(args.datasets, args.kernel_size)
    
    print("\nAll datasets processed successfully!") 