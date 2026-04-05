import os
import cv2
import glob
import argparse
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# COCO Class IDs
TARGET_CLASSES =[0, 1, 2, 3, 5, 7, 9, 11]

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
    print("Loading YOLO11-X Segmentation model...")
    try:
        model = YOLO('yolo11x-seg.pt') 
    except Exception as e:
        print("Could not load yolo11x, falling back to yolo11l...")
        model = YOLO('yolo11l-seg.pt')

    model.to('cuda') 

    # Note: Added 0123 back in to match your prompt
    search_path = os.path.join(base_path, "**", "image_0[23]", "data", "*.png")
    image_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(image_files)} images to process.")

    for img_path in tqdm(image_files):
        # Save to a new folder so we don't overwrite the old semantic ones
        mask_path = img_path.replace("image_0", "yolo_instance_mask_0")
        
        if os.path.exists(mask_path):
            continue

        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        results = model(img, verbose=False, classes=TARGET_CLASSES, conf=0.25, retina_masks=True)
        
        # --- THE CHANGE: 16-bit Integer Canvas ---
        # 8-bit (0-255) is too small for instance IDs like 3001. We must use 16-bit!
        final_mask = np.zeros((h, w), dtype=np.uint16)
        
        # Keep track of how many of each class we've seen in this image
        instance_counts = {cls_id: 0 for cls_id in range(1, 9)}
        
        if results[0].masks is not None and results[0].boxes is not None:
            for seg, box in zip(results[0].masks.xy, results[0].boxes):
                coco_cls = int(box.cls[0].item())
                if coco_cls not in CLASS_MAPPING:
                    continue
                    
                mapped_cls = CLASS_MAPPING[coco_cls]
                
                # Increment the counter for this specific class
                instance_counts[mapped_cls] += 1
                instance_id = instance_counts[mapped_cls]
                
                # Format: Class ID * 1000 + Instance ID (e.g., Car 2 = 3002)
                pixel_value = (mapped_cls * 1000) + instance_id
                
                if len(seg) > 0:
                    seg_points = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    # Draw the polygon using the Panoptic Instance ID
                    cv2.fillPoly(final_mask,[seg_points], pixel_value)

        # OpenCV natively supports saving uint16 arrays as 16-bit PNGs!
        cv2.imwrite(mask_path, final_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help="Path to KITTI raw data")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Path {args.data_path} does not exist.")
    else:
        process_kitti(args.data_path)