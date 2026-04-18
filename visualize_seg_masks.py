#!/usr/bin/env python
"""
Visualization script for generating qualitative segmentation results.
Outputs: full segmentation masks, per-class masks, and overlays for article figures.

Usage:
    python visualize_seg_masks.py --image_path /path/to/image.jpg --load_weights_folder /path/to/weights
"""

from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2

import networks
from layers import disp_to_depth

# Color map for 9 KITTI classes - visually distinct colors (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Person - Red
    2: (0, 255, 0),      # Bicycle - Green
    3: (0, 0, 255),      # Car - Blue
    4: (255, 255, 0),    # Motorcycle - Cyan
    5: (255, 0, 255),    # Bus - Magenta
    6: (0, 255, 255),    # Truck - Yellow
    7: (128, 255, 0),    # Traffic Light - Light Green
    8: (255, 128, 0),    # Stop Sign - Orange
}

CLASS_NAMES = [
    "Background", "Person", "Bicycle", "Car", "Motorcycle",
    "Bus", "Truck", "Traffic Light", "Stop Sign"
]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate segmentation visualizations for paper')
    parser.add_argument('--image_path', type=str, required=True,
                        help='path to a test image or folder of images')
    parser.add_argument('--load_weights_folder', type=str, required=True,
                        help='path to trained MTL weights')
    parser.add_argument('--model', type=str, default="lite-mono-8m",
                        choices=['lite-mono', 'lite-mono-small', 'lite-mono-tiny', 'lite-mono-8m'])
    parser.add_argument('--ext', type=str, default="png",
                        help='image extension to look for')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='output directory (default: same as input)')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_overlay', action='store_true', default=True,
                        help='save overlay of mask on original image')
    parser.add_argument('--save_individual_classes', action='store_true', default=True,
                        help='save individual class masks')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='transparency for overlay (0-1)')
    return parser.parse_args()


def create_semantic_mask(seg_pred, original_size):
    """Create a colored semantic segmentation mask."""
    h, w = original_size
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        mask[seg_pred == class_id] = color

    return mask


def create_instance_overlay(annotated_img, panoptic_mask, depth_metric=None):
    """
    Create an overlay with instance bounding boxes and labels.
    Similar to test_mtl but saving the full annotated image.
    """
    # Draw contours for each instance
    unique_ids = np.unique(panoptic_mask)

    for uid in unique_ids:
        if uid < 1000:  # Skip background
            continue

        class_id = uid // 1000
        instance_id = uid % 1000

        # Create mask for this instance
        inst_mask = (panoptic_mask == uid).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            color = CLASS_COLORS.get(class_id, (0, 255, 0))
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)

            # Prepare label
            if depth_metric is not None:
                # Get median depth for this instance
                valid_depths = depth_metric[inst_mask == 255]
                if len(valid_depths) > 0:
                    avg_depth = np.median(valid_depths)
                    label = f"{CLASS_NAMES[class_id]} {instance_id} {avg_depth:.1f}m"
                else:
                    label = f"{CLASS_NAMES[class_id]} {instance_id}"
            else:
                label = f"{CLASS_NAMES[class_id]} {instance_id}"

            # Draw label with background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(annotated_img, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated_img


def visualize_segmentation(args):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load models
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    depth_path = os.path.join(args.load_weights_folder, "depth.pth")
    seg_path = os.path.join(args.load_weights_folder, "segmentation.pth")
    centers_path = os.path.join(args.load_weights_folder, "centers.pth")

    print(f"-> Loading model from {args.load_weights_folder}")
    encoder_dict = torch.load(encoder_path, map_location=device)

    feed_height = encoder_dict.get('height', 192)
    feed_width = encoder_dict.get('width', 640)

    # Initialize models
    encoder = networks.LiteMono(model=args.model, height=feed_height, width=feed_width, in_channels=3)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device).eval()

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_decoder.load_state_dict(torch.load(depth_path, map_location=device))
    depth_decoder.to(device).eval()

    seg_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=9, is_seg=True)
    seg_decoder.load_state_dict(torch.load(seg_path, map_location=device))
    seg_decoder.to(device).eval()

    centers_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0], num_output_channels=1, is_seg=False)
    centers_decoder.load_state_dict(torch.load(centers_path, map_location=device))
    centers_decoder.to(device).eval()

    # Get image list
    if os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, f'*.{args.ext}'))
        output_directory = args.output_dir if args.output_dir else args.image_path
    else:
        paths = [args.image_path]
        output_directory = args.output_dir if args.output_dir else os.path.dirname(args.image_path)

    os.makedirs(output_directory, exist_ok=True)
    print(f"-> Processing {len(paths)} images")
    print(f"-> Saving outputs to {output_directory}")

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            print(f"   Processing {idx + 1}/{len(paths)}: {os.path.basename(image_path)}")

            # Load image
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size

            # Store original for overlay
            img_np_orig = np.array(input_image)

            # Preprocess
            input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0).to(device)

            # Inference
            features = encoder(input_tensor)

            # Depth prediction
            disp = depth_decoder(features)[("disp", 0)]
            disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            _, depth_metric = disp_to_depth(disp_resized, 0.1, 80.0)
            depth_metric = depth_metric.squeeze().cpu().numpy()

            # Segmentation prediction
            seg_logits = seg_decoder(features)[("disp", 0)]
            seg_resized = F.interpolate(seg_logits, (original_height, original_width), mode="bilinear", align_corners=False)
            seg_np = seg_resized.squeeze().cpu().numpy()

            # Get predicted class
            seg_pred = np.argmax(seg_np, axis=0).astype(np.uint8)

            # Centers prediction (for panoptic)
            centers_out = centers_decoder(features)[("disp", 0)]
            centers_resized = F.interpolate(centers_out, (original_height, original_width), mode="bilinear", align_corners=False)
            center_heatmap = centers_resized.squeeze().cpu().numpy()

            output_name = os.path.splitext(os.path.basename(image_path))[0]

            # 1. SAVE FULL SEMANTIC MASK (THE MAIN REQUEST)
            semantic_mask = create_semantic_mask(seg_pred, (original_height, original_width))
            cv2.imwrite(
                os.path.join(output_directory, f"{output_name}_semantic_mask.png"),
                semantic_mask
            )

            # 2. SAVE OVERLAY (if requested)
            if args.save_overlay:
                overlay = img_np_orig.copy()
                mask_bool = (seg_pred > 0)  # Non-background

                # Create colored overlay
                colored_overlay = np.zeros_like(overlay)
                for class_id in range(1, 9):  # Skip background
                    color = CLASS_COLORS[class_id]
                    colored_overlay[seg_pred == class_id] = color

                # Blend overlay with original
                overlay = cv2.addWeighted(overlay, 1 - args.alpha, colored_overlay, args.alpha, 0)

                cv2.imwrite(
                    os.path.join(output_directory, f"{output_name}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

            # 3. SAVE INDIVIDUAL CLASS MASKS (if requested)
            if args.save_individual_classes:
                class_masks_dir = os.path.join(output_directory, f"{output_name}_class_masks")
                os.makedirs(class_masks_dir, exist_ok=True)

                for class_id in range(1, 9):  # Skip background (0)
                    class_mask = (seg_pred == class_id).astype(np.uint8) * 255
                    if np.any(class_mask > 0):  # Only save if class exists
                        cv2.imwrite(
                            os.path.join(class_masks_dir, f"{output_name}_{CLASS_NAMES[class_id]}.png"),
                            class_mask
                        )

            # 4. SAVE DEPTH VISUALIZATION
            vmax = 80.0
            normalizer = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            colormapped_im = (mapper.to_rgba(depth_metric)[:, :, :3] * 255).astype(np.uint8)
            pil.fromarray(colormapped_im).save(
                os.path.join(output_directory, f"{output_name}_depth_viz.png")
            )

            # 5. SAVE CENTER HEATMAP
            heatmap_viz = (cm.hot(center_heatmap)[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_directory, f"{output_name}_centers_viz.png"),
                cv2.cvtColor(heatmap_viz, cv2.COLOR_RGB2BGR)
            )

            # 6. SAVE COMBINED VISUALIZATION (for paper figure)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # Input image
            axes[0, 0].imshow(img_np_orig)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')

            # Semantic mask
            axes[0, 1].imshow(cv2.cvtColor(semantic_mask, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Semantic Segmentation Mask')
            axes[0, 1].axis('off')

            # Overlay
            if args.save_overlay:
                axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[0, 2].set_title('Overlay')
                axes[0, 2].axis('off')

            # Depth
            axes[1, 0].imshow(colormapped_im)
            axes[1, 0].set_title('Depth Estimation')
            axes[1, 0].axis('off')

            # Centers heatmap
            axes[1, 1].imshow(center_heatmap, cmap='hot')
            axes[1, 1].set_title('Instance Centers Heatmap')
            axes[1, 1].axis('off')

            # Legend
            axes[1, 2].axis('off')
            legend_text = "Class Colors:\n\n"
            for i, name in enumerate(CLASS_NAMES):
                color = CLASS_COLORS[i]
                legend_text += f"{name}: RGB{color}\n"
            axes[1, 2].text(0.1, 0.5, legend_text, transform=axes[1, 2].transAxes,
                          fontsize=10, verticalalignment='center', family='monospace')

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_directory, f"{output_name}_combined.png"),
                dpi=150, bbox_inches='tight'
            )
            plt.close()

            print(f"     Saved outputs for {output_name}")

    print('-> Done!')
    print(f"Output files saved to: {output_directory}")
    print("\nGenerated files per image:")
    print("  - {name}_semantic_mask.png    : Full semantic segmentation mask")
    print("  - {name}_overlay.png          : Mask overlay on original image")
    print("  - {name}_depth_viz.png        : Depth map visualization")
    print("  - {name}_centers_viz.png      : Instance center heatmap")
    print("  - {name}_combined.png         : All visualizations in one figure")
    print("  - {name}_class_masks/*.png    : Individual class masks" if args.save_individual_classes else "")


if __name__ == '__main__':
    args = parse_args()
    visualize_segmentation(args)
