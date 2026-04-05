import os
import cv2
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Map COCO classes to continuous IDs (1 to 8). Background is 0.
CLASS_MAPPING = {
    0: 1,   # Person
    1: 2,   # Bicycle
    2: 3,   # Car
    3: 4,   # Motorcycle
    5: 5,   # Bus
    7: 6,   # Truck
    9: 7,   # Traffic Light
    11: 8   # Stop Sign
}

def process_kitti(base_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Mask2Former (Swin-Large COCO Panoptic) on {device}...")
    
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    model.to(device)
    model.eval()

    search_path = os.path.join(base_path, "**", "image_0[23]", "data", "*.png")
    image_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in tqdm(image_files):
            mask_path = img_path.replace("image_0", "mask2former_instance_mask_0")
            
            if os.path.exists(mask_path):
                continue

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)

            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            # 1. Forward Pass
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # 2. Post-process to get panoptic segmentation maps
            result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)], label_ids_to_fuse=set())[0]
            
            segmentation_map = result["segmentation"].cpu().numpy()
            segments_info = result["segments_info"]

            # 3. Create the 16-bit Integer Canvas
            final_mask = np.zeros((h, w), dtype=np.uint16)
            instance_counts = {cls_id: 0 for cls_id in range(1, 9)}
            
            # 4. Map the Mask2Former outputs to our Custom 16-bit format
            for segment in segments_info:
                coco_cls = segment["label_id"]
                
                # Check if it's one of our target classes
                if coco_cls not in CLASS_MAPPING:
                    continue
                    
                mapped_cls = CLASS_MAPPING[coco_cls]
                
                # Increment instance counter for this class
                instance_counts[mapped_cls] += 1
                instance_id = instance_counts[mapped_cls]
                
                # Format: Class ID * 1000 + Instance ID (e.g., Car 2 = 3002)
                pixel_value = (mapped_cls * 1000) + instance_id
                
                # Find all pixels belonging to this specific instance and paint them
                mask_pixels = (segmentation_map == segment["id"])
                final_mask[mask_pixels] = pixel_value

            # Save as 16-bit PNG
            cv2.imwrite(mask_path, final_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to KITTI raw data")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Path {args.data_path} does not exist.")
    else:
        process_kitti(args.data_path)