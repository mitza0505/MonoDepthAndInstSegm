#!/usr/bin/env python
"""
Create side-by-side comparison figures for the paper.
Shows input, depth, semantic mask, overlay, and panoptic results in one figure.

Usage:
    python visualize_comparison.py --image_path /path/to/image --load_weights_folder /path/to/weights
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2

import networks
from layers import disp_to_depth

# Color map for visualization
CLASS_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # Person
    2: (0, 255, 0),      # Bicycle
    3: (0, 0, 255),      # Car
    4: (255, 255, 0),    # Motorcycle
    5: (255, 0, 255),    # Bus
    6: (0, 255, 255),    # Truck
    7: (128, 255, 0),    # Traffic Light
    8: (255, 128, 0),    # Stop Sign
}

CLASS_NAMES = [
    "Background", "Person", "Bicycle", "Car", "Motorcycle",
    "Bus", "Truck", "Traffic Light", "Stop Sign"
]


def parse_args():
    parser = argparse.ArgumentParser(description='Create paper-ready comparison figure')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--model', type=str, default="lite-mono-8m")
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI for publication quality')
    return parser.parse_args()


def create_paper_figure(args):
    """Create a publication-ready figure showing all outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load models
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    depth_path = os.path.join(args.load_weights_folder, "depth.pth")
    seg_path = os.path.join(args.load_weights_folder, "segmentation.pth")
    centers_path = os.path.join(args.load_weights_folder, "centers.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict.get('height', 192)
    feed_width = encoder_dict.get('width', 640)

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

    # Load and process image
    input_image = pil.open(args.image_path).convert('RGB')
    original_width, original_height = input_image.size
    img_np = np.array(input_image)

    input_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)

        # Depth
        disp = depth_decoder(features)[("disp", 0)]
        disp_resized = F.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
        _, depth = disp_to_depth(disp_resized, 0.1, 80.0)
        depth = depth.squeeze().cpu().numpy()

        # Segmentation
        seg_logits = seg_decoder(features)[("disp", 0)]
        seg_resized = F.interpolate(seg_logits, (original_height, original_width), mode="bilinear", align_corners=False)
        seg_pred = seg_resized.squeeze().cpu().numpy()
        seg_class = np.argmax(seg_pred, axis=0)

        # Centers heatmap
        centers = centers_decoder(features)[("disp", 0)]
        centers_resized = F.interpolate(centers, (original_height, original_width), mode="bilinear", align_corners=False)
        centers_heatmap = centers_resized.squeeze().cpu().numpy()

    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 5, figure=fig, wspace=0.3, hspace=0.3)

    # 1. Input Image
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(img_np)
    ax1.set_title('(a) Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Depth Map
    ax2 = fig.add_subplot(gs[:, 1])
    depth_viz = ax2.imshow(depth, cmap='plasma', vmin=0, vmax=80)
    ax2.set_title('(b) Depth Estimation', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar = plt.colorbar(depth_viz, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', fontsize=10)

    # 3. Semantic Mask
    ax3 = fig.add_subplot(gs[:, 2])
    semantic_mask = np.zeros((*seg_class.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        semantic_mask[seg_class == class_id] = color[::-1]  # BGR to RGB

    ax3.imshow(semantic_mask)
    ax3.set_title('(c) Semantic Segmentation', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Overlay
    ax4 = fig.add_subplot(gs[0, 3])
    overlay = img_np.copy()
    alpha = 0.5
    mask_bool = (seg_class > 0)
    colored_overlay = np.zeros_like(overlay)
    for class_id in range(1, 9):
        color = CLASS_COLORS[class_id]
        colored_overlay[seg_class == class_id] = color[::-1]  # BGR to RGB
    overlay = cv2.addWeighted(overlay, 1 - alpha, colored_overlay, alpha, 0)
    ax4.imshow(overlay)
    ax4.set_title('(d) Overlay', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # 5. Centers Heatmap
    ax5 = fig.add_subplot(gs[1, 3])
    centers_viz = ax5.imshow(centers_heatmap, cmap='hot')
    ax5.set_title('(e) Instance Centers', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(centers_viz, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Class Legend
    ax6 = fig.add_subplot(gs[:, 4])
    ax6.axis('off')

    legend_elements = []
    for class_id, name in enumerate(CLASS_NAMES):
        color = [c / 255.0 for c in CLASS_COLORS[class_id]]  # Normalize to 0-1
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=f'{class_id}: {name}')
        )

    ax6.legend(handles=legend_elements, loc='center', fontsize=10,
              title='Class Legend', title_fontsize=11, frameon=True)

    # Overall title
    fig.suptitle('Multi-Task Visual Perception Results', fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_path = args.output_path if args.output_path else 'comparison_figure.png'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"Paper figure saved to: {output_path}")
    print(f"Resolution: {plt.gcf().get_size_inches()[0] * args.dpi:.0f} x {plt.gcf().get_size_inches()[1] * args.dpi:.0f} pixels")


if __name__ == '__main__':
    args = parse_args()
    create_paper_figure(args)
