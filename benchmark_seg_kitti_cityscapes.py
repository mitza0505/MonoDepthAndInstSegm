#!/usr/bin/env python
"""
Alternative: Evaluate segmentation on KITTI using pre-trained Cityscapes model.
This is a common practice in autonomous driving - train on Cityscapes, evaluate on KITTI.
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

import networks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation using Cityscapes pre-trained model on KITTI'
    )
    parser.add_argument('--load_weights_folder', type=str, required=True)
    parser.add_argument('--model', type=str, default='lite-mono-8m')
    parser.add_argument('--kitti_data_path', type=str, default='./kitti_data')
    parser.add_argument('--cityscapes_model', type=str, default=None,
                        help='Path to Cityscapes pre-trained model (optional)')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--no_cuda', action='store_true')
    return parser.parse_args()


def evaluate_kitti_with_cityscapes_model(args):
    """
    Evaluate segmentation on KITTI using a model pre-trained on Cityscapes.
    This tests domain generalization from urban scenes (Cityscapes) to suburban/highway (KITTI).
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Evaluating KITTI segmentation using Cityscapes pre-trained model")
    print(f"Device: {device}")

    # Load model
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    seg_path = os.path.join(args.load_weights_folder, "segmentation.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)

    # Cityscapes has 19 classes (or 30 for panoptic)
    # We'll use 19 for semantic segmentation
    num_classes = 19

    encoder = networks.LiteMono(
        model=args.model,
        height=args.height,
        width=args.width,
        in_chans=3
    )
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device)
    encoder.eval()

    seg_decoder = networks.DepthDecoder(
        encoder.num_ch_enc,
        scales=[0],
        num_output_channels=num_classes,
        is_seg=True,
        use_aspp=True
    )
    seg_decoder.load_state_dict(torch.load(seg_path, map_location=device))
    seg_decoder.to(device)
    seg_decoder.eval()

    # Cityscapes class names (19 classes for semantic)
    CITYSCAPES_CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    print("\nCityscapes Classes:")
    for i, cls in enumerate(CITYSCAPES_CLASSES):
        print(f"  {i}: {cls}")

    print("\n" + "=" * 60)
    print("NOTE: To benchmark on KITTI with real metrics, you would need:")
    print("  1. KITTI semantic ground truth (not available in Eigen split)")
    print("  2. Or: Human annotations for a subset of KITTI")
    print("  3. Or: Use SemanticKITTI (LiDAR-based) for point-wise evaluation")
    print("=" * 60)

    print("\nAlternative benchmarking approaches:")
    print("  A) Train on Cityscapes (19 classes), evaluate on Cityscapes val")
    print("  B) Train on COCO (133 classes), evaluate on COCO val")
    print("  C) Train on KITTI with pseudo-labels, qualitatively evaluate")
    print("  D) Use online benchmarks (KITTI leaderboard) if available")


def main():
    args = parse_args()
    evaluate_kitti_with_cityscapes_model(args)


if __name__ == '__main__':
    main()
