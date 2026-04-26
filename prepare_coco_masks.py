import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_panoptic_to_semantic(json_file, img_dir, out_dir):
    print(f"Processing {json_file}...")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_file, 'r') as f:
        coco_panoptic = json.load(f)
    
    # 1. Map COCO's sparse Category IDs (up to 200) to continuous IDs (0 to 132)
    categories = coco_panoptic['categories']
    cat_id_to_continuous = {cat['id']: i for i, cat in enumerate(categories)}
    
    # 2. Process every image mask
    for annotation in tqdm(coco_panoptic['annotations']):
        file_name = annotation['file_name']
        img_path = os.path.join(img_dir, file_name)
        
        # Read the raw COCO panoptic RGB image
        pan_format = np.array(Image.open(img_path), dtype=np.uint32)
        
        # Decode RGB to the unique Segment ID
        pan = pan_format[:, :, 0] + pan_format[:, :, 1] * 256 + pan_format[:, :, 2] * 256**2
        
        # Create an empty semantic mask (default to 255 = ignore index)
        semantic_mask = np.ones(pan.shape, dtype=np.uint8) * 255
        
        # Map each segment ID to its continuous semantic class ID
        for seg_info in annotation['segments_info']:
            segment_id = seg_info['id']
            cat_id = seg_info['category_id']
            continuous_id = cat_id_to_continuous[cat_id]
            
            # Apply to mask
            semantic_mask[pan == segment_id] = continuous_id
            
        # Save as a standard 1-channel grayscale PNG
        out_path = os.path.join(out_dir, file_name)
        Image.fromarray(semantic_mask).save(out_path)

if __name__ == "__main__":
    # Convert Train Masks
    convert_panoptic_to_semantic(
        json_file='./coco/annotations/panoptic_train2017.json', 
        img_dir='./coco/annotations/panoptic_train2017', 
        out_dir='./coco/annotations/semantic_train2017'
    )
    
    # Convert Val Masks
    convert_panoptic_to_semantic(
        json_file='./coco/annotations/panoptic_val2017.json', 
        img_dir='./coco/annotations/panoptic_val2017', 
        out_dir='./coco/annotations/semantic_val2017'
    )
    print("Done! Masks are ready for PyTorch.")