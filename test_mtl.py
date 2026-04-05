from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2

import networks
from layers import disp_to_depth

# Map our 1-8 IDs back to human-readable names for the text labels
CLASS_NAMES = [
    "Background", "Person", "Bicycle", "Car", "Motorcycle", 
    "Bus", "Truck", "Traffic Light", "Stop Sign"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Panoptic testing function for Lite-Mono (Depth + Semantic + Centers).')
    parser.add_argument('--image_path', type=str, help='path to a test image or folder', required=True)
    parser.add_argument('--load_weights_folder', type=str, help='path to trained MTL weights', required=True)
    parser.add_argument('--model', type=str, help='model name', default="lite-mono-8m") 
    parser.add_argument('--ext', type=str, help='image extension', default="png")
    parser.add_argument("--no_cuda", action='store_true')
    return parser.parse_args()

def test_mtl(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"-> Loading Panoptic MTL model from {args.load_weights_folder}")
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    depth_path = os.path.join(args.load_weights_folder, "depth.pth")
    seg_path = os.path.join(args.load_weights_folder, "segmentation.pth")
    centers_path = os.path.join(args.load_weights_folder, "centers.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict.get('height', 192)
    feed_width = encoder_dict.get('width', 640)

    # --- 1. LOAD ENCODER ---
    encoder = networks.LiteMono(model=args.model, height=feed_height, width=feed_width, in_channels=3) 
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device).eval()

    # --- 2. LOAD DECODERS ---
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3)) 
    depth_decoder.load_state_dict(torch.load(depth_path, map_location=device))
    depth_decoder.to(device).eval()

    seg_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=9, is_seg=True) 
    seg_decoder.load_state_dict(torch.load(seg_path, map_location=device))
    seg_decoder.to(device).eval()
    
    centers_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=1, is_seg=False) 
    centers_decoder.load_state_dict(torch.load(centers_path, map_location=device))
    centers_decoder.to(device).eval()

    if os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, f'*.{args.ext}'))
        output_directory = args.image_path
    elif os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)

    print(f"-> Predicting on {len(paths)} test images")

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if any(x in image_path for x in ["_disp", "_depth", "viz", "annotated", "seg_mask", "centers", "panoptic"]):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            img_np_orig = np.array(input_image)
            annotated_img = cv2.cvtColor(img_np_orig, cv2.COLOR_RGB2BGR)
            
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0).to(device)

            # --- INFERENCE ---
            features = encoder(input_tensor)
            
            disp = depth_decoder(features)[("disp", 0)]
            disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            _, depth_metric = disp_to_depth(disp_resized, 0.1, 80.0)
            depth_metric = depth_metric.squeeze().cpu().numpy()

            seg_logits = seg_decoder(features)[("disp", 0)]
            seg_resized = F.interpolate(seg_logits, (original_height, original_width), mode="bilinear", align_corners=False)
            seg_np = seg_resized.squeeze().cpu().numpy() 

            centers_out = centers_decoder(features)[("disp", 0)]
            centers_resized = F.interpolate(centers_out, (original_height, original_width), mode="bilinear", align_corners=False)
            center_heatmap = centers_resized.squeeze().cpu().numpy()

            # ==========================================
            # HYBRID CLASS-BY-CLASS PANOPTIC POST-PROCESSING
            # ==========================================
            
            seg_probs = F.softmax(torch.tensor(seg_np), dim=0).numpy()
            class_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)
            max_probs = np.max(seg_probs, axis=0)
            class_mask[max_probs < 0.5] = 0

            center_heatmap_smoothed = cv2.GaussianBlur(center_heatmap, (5, 5), 0)

            # 16-bit panoptic mask: pixel value = class_id * 1000 + instance_id
            # e.g. Car(3) instance 2 → 3002, Truck(6) instance 1 → 6001, background → 0
            # To decode: class_id = value // 1000,  instance_id = value % 1000
            panoptic_mask = np.zeros((original_height, original_width), dtype=np.uint16)

            # Per-class instance counter, reset for each image
            instance_counters = {class_id: 0 for class_id in range(1, 9)}

            drawn_text_positions = []

            for class_id in range(1, 9):
                class_bin = (class_mask == class_id).astype(np.uint8) * 255
                if cv2.countNonZero(class_bin) == 0: continue

                class_bin = cv2.morphologyEx(class_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                if cv2.countNonZero(class_bin) == 0: continue

                contours, _ = cv2.findContours(class_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)

                    contour_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

                    needs_split = False
                    peaks = np.zeros_like(contour_mask)

                    if class_id not in [5, 6]:
                        contour_centers = center_heatmap_smoothed * (contour_mask / 255.0)
                        local_max = cv2.dilate(contour_centers, np.ones((11, 11), np.uint8))
                        peaks = ((contour_centers == local_max) & (contour_centers > 0.1)).astype(np.uint8) * 255
                        peaks = cv2.dilate(peaks, np.ones((3, 3), np.uint8))

                        num_peaks, markers = cv2.connectedComponents(peaks)

                        if num_peaks > 2:
                            needs_split = True
                        elif w > 1.6 * h and cv2.contourArea(contour) > 200:
                            dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 3)
                            dt_local_max = cv2.dilate(dist_transform, np.ones((15, 15), np.uint8))
                            dt_peaks = ((dist_transform == dt_local_max) & (dist_transform > 0.4 * dist_transform.max())).astype(np.uint8) * 255
                            peaks = np.uint8(dt_peaks)
                            num_peaks, markers = cv2.connectedComponents(peaks)
                            if num_peaks > 2:
                                needs_split = True

                    if needs_split:
                        sure_bg = cv2.bitwise_not(contour_mask)
                        peaks_eroded = cv2.erode(peaks, np.ones((3, 3), np.uint8))
                        if cv2.countNonZero(peaks_eroded) > 0:
                            peaks = peaks_eroded
                        unknown = cv2.subtract(contour_mask, peaks)
                        markers = markers + 1
                        markers[unknown == 255] = 0
                        dummy_rgb = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)
                        markers = cv2.watershed(dummy_rgb, markers)

                        instances_to_draw = []
                        for i in range(2, num_peaks + 1):
                            inst_mask = (markers == i).astype(np.uint8) * 255
                            inst_contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            instances_to_draw.extend(inst_contours)
                    else:
                        instances_to_draw = [contour]

                    # --- DRAWING & PANOPTIC MASK WRITING ---
                    for inst_contour in instances_to_draw:
                        if cv2.contourArea(inst_contour) < 30:
                            continue

                        ix, iy, iw, ih = cv2.boundingRect(inst_contour)
                        pad = 3
                        x_draw, y_draw = max(0, ix - pad), max(0, iy - pad)
                        w_draw, h_draw = min(original_width - x_draw, iw + pad*2), min(original_height - y_draw, ih + pad*2)

                        inst_mask = np.zeros((original_height, original_width), dtype=np.uint8)
                        cv2.drawContours(inst_mask, [inst_contour], -1, 255, thickness=cv2.FILLED)

                        valid_depths = depth_metric[inst_mask == 255]

                        if len(valid_depths) > 0:
                            avg_depth = np.median(valid_depths)

                            # Write panoptic ID into the mask: class_id * 1000 + instance_id
                            instance_counters[class_id] += 1
                            panoptic_id = np.uint16(class_id * 1000 + instance_counters[class_id])
                            panoptic_mask[inst_mask == 255] = panoptic_id

                            # Decode label directly from the mask — class and instance in one read
                            label_class_id  = int(panoptic_id) // 1000   # e.g. 3002 → 3
                            label_instance_id = int(panoptic_id) % 1000  # e.g. 3002 → 2
                            text = f"{CLASS_NAMES[label_class_id]} {label_instance_id} {avg_depth:.1f}m"

                            cv2.rectangle(annotated_img, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (0, 255, 0), 2)

                            y_text = y_draw - 8 if y_draw - 8 > 20 else y_draw + 25
                            for (prev_x, prev_y) in drawn_text_positions:
                                if abs(x_draw - prev_x) < 80 and abs(y_text - prev_y) < 20:
                                    y_text += 25
                            drawn_text_positions.append((x_draw, y_text))

                            cv2.putText(annotated_img, text, (x_draw, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                            cv2.putText(annotated_img, text, (x_draw, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- SAVING OUTPUTS ---
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_directory, f"{output_name}_MTL_annotated.jpg"), annotated_img)

            # 16-bit PNG: each pixel encodes class_id * 1000 + instance_id
            # Decode with: class_id = mask // 1000,  instance_id = mask % 1000
            pil.fromarray(panoptic_mask).save(os.path.join(output_directory, f"{output_name}_panoptic.png"))

            heatmap_viz = (cm.inferno(center_heatmap)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_directory, f"{output_name}_centers_viz.png"), cv2.cvtColor(heatmap_viz, cv2.COLOR_RGB2BGR))

            vmax = 80.0
            normalizer = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            colormapped_im = (mapper.to_rgba(depth_metric)[:, :, :3] * 255).astype(np.uint8)
            pil.fromarray(colormapped_im).save(os.path.join(output_directory, f"{output_name}_depth_viz.png"))

            print(f"   Processed {idx + 1}/{len(paths)} - Saved outputs for {output_name}")

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_mtl(args)