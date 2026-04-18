#!/usr/bin/env python
"""
Batch script to generate multiple qualitative results for article figures.
Scans a directory and creates publication-ready visualizations for a subset of images.

Usage:
    python generate_qualitative_results.py \
        --image_dir /path/to/kitti/images \
        --load_weights_folder /path/to/weights \
        --output_dir /path/to/output \
        --num_images 10
"""

from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import random
import numpy as np

import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multiple qualitative results')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='directory containing test images')
    parser.add_argument('--load_weights_folder', type=str, required=True,
                        help='path to trained MTL weights')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory for all results')
    parser.add_argument('--num_images', type=int, default=10,
                        help='number of images to process (0 for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for image selection')
    parser.add_argument('--model', type=str, default='lite-mono-8m')
    parser.add_argument('--ext', type=str, default='png')
    return parser.parse_args()


def main():
    args = parse_args()

    # Find all images
    image_pattern = os.path.join(args.image_dir, f'*.{args.ext}')
    all_images = glob.glob(image_pattern)

    if len(all_images) == 0:
        print(f"No images found in {args.image_dir} with extension .{args.ext}")
        return

    print(f"Found {len(all_images)} images")

    # Select subset
    if args.num_images > 0 and args.num_images < len(all_images):
        random.seed(args.seed)
        selected_images = random.sample(all_images, args.num_images)
        print(f"Selected {args.num_images} images for processing")
    else:
        selected_images = all_images
        print(f"Processing all {len(all_images)} images")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image
    for idx, image_path in enumerate(selected_images):
        print(f"\n[{idx + 1}/{len(selected_images)}] Processing: {os.path.basename(image_path)}")

        # Run visualization script
        cmd = [
            'python', 'visualize_seg_masks.py',
            '--image_path', image_path,
            '--load_weights_folder', args.load_weights_folder,
            '--model', args.model,
            '--output_dir', args.output_dir,
            '--save_overlay',
            '--save_individual_classes'
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   Error processing {image_path}: {e}")
            continue

        # Also create comparison figure
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        comparison_output = os.path.join(args.output_dir, f"{output_name}_paper_fig.png")

        cmd_comp = [
            'python', 'visualize_comparison.py',
            '--image_path', image_path,
            '--load_weights_folder', args.load_weights_folder,
            '--model', args.model,
            '--output_path', comparison_output,
            '--dpi', '300'
        ]

        try:
            subprocess.run(cmd_comp, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   Error creating comparison for {image_path}: {e}")

    print("\n" + "=" * 60)
    print("Qualitative results generation complete!")
    print("=" * 60)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("\nFile naming convention:")
    print("  {name}_semantic_mask.png    - Full colored semantic mask")
    print("  {name}_overlay.png          - Mask overlay on image")
    print("  {name}_depth_viz.png        - Depth visualization")
    print("  {name}_centers_viz.png      - Center heatmap")
    print("  {name}_combined.png         - 6-panel combined figure")
    print("  {name}_paper_fig.png        - Publication-ready 5-panel figure")
    print("  {name}_class_masks/         - Individual class masks")
    print("\nSuggested images for paper:")
    print("  - Select images showing multiple object classes")
    print("  - Include scenes with occlusions and varying scales")
    print("  - Choose images with good clear segmentation results")


if __name__ == '__main__':
    main()
