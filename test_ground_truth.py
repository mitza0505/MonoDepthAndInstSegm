from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import cv2
import torch
from ultralytics import YOLO
from collections import Counter

# ==========================================
# KITTI UTILS (Included for Standalone Use)
# ==========================================
def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass
    return data

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds =[item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

# ==========================================
# MAIN SCRIPT
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description='Ground Truth LiDAR Annotator')
    parser.add_argument('--image_path', type=str, help='path to the KITTI test image', required=True)
    parser.add_argument('--velo_path', type=str, help='path to the corresponding .bin velodyne file', required=True)
    parser.add_argument('--calib_dir', type=str, help='path to the directory containing calib_cam_to_cam.txt', required=True)
    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init YOLO
    print("-> Initializing YOLO...")
    try:
        yolo_model = YOLO('yolo11x-seg.pt')
    except:
        yolo_model = YOLO('yolov8s-seg.pt')
    yolo_model.to(device)
    target_classes =[0, 1, 2, 3, 5, 7, 9, 11]

    # 2. Load Image
    print(f"-> Loading image: {args.image_path}")
    input_image = pil.open(args.image_path).convert('RGB')
    img_np_orig = np.array(input_image)
    original_width, original_height = input_image.size

    # 3. Generate Ground Truth Depth from LiDAR
    print(f"-> Projecting LiDAR points from: {args.velo_path}")
    depth_gt = generate_depth_map(args.calib_dir, args.velo_path, cam=2)

    # 4. Run YOLO
    print("-> Running inference and annotating...")
    results = yolo_model(img_np_orig, verbose=False, classes=target_classes, conf=0.25)
    
    annotated_img = cv2.cvtColor(img_np_orig, cv2.COLOR_RGB2BGR)
    drawn_text_positions = []

    if results[0].masks is not None and results[0].boxes is not None:
        for box, seg in zip(results[0].boxes, results[0].masks.xy):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            obj_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            if len(seg) > 0:
                seg_points = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(obj_mask, [seg_points], 255)
            
            # --- THE KEY DIFFERENCE ---
            # Extract ONLY pixels that are inside the mask AND have a valid LiDAR point (> 0)
            valid_depths = depth_gt[(obj_mask == 255) & (depth_gt > 0)]
            
            if len(valid_depths) > 0:
                # Use median to ignore extreme outliers in the sparse LiDAR cloud
                avg_depth = np.median(valid_depths) 
                text = f"{avg_depth:.2f}m"
                box_color = (0, 255, 0)
            else:
                # LiDAR is sparse. Sometimes no lasers hit distant/small objects!
                text = "No LiDAR"
                box_color = (0, 0, 255) 
                
            # Draw Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
            
            # Text overlapping logic
            y_text = y1 - 8
            if y_text < 20:
                y_text = y1 + 25 
                
            for (prev_x, prev_y) in drawn_text_positions:
                if abs(x1 - prev_x) < 50 and abs(y_text - prev_y) < 20:
                    y_text += 25
            
            drawn_text_positions.append((x1, y_text))

            # Draw Text
            cv2.putText(annotated_img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(annotated_img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # 5. Save output
    output_dir = os.path.dirname(args.image_path)
    output_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_path = os.path.join(output_dir, f"{output_name}_GT_annotated.jpg")
    
    cv2.imwrite(output_path, annotated_img)
    print(f"-> Done! Saved ground truth baseline to: {output_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)