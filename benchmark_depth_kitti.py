#!/usr/bin/env python
"""
Benchmark depth estimation on KITTI Eigen split for comparison with published results.
This script evaluates depth using standard KITTI metrics and median scaling.

Usage:
    python benchmark_depth_kitti.py --load_weights_folder /path/to/weights --model lite-mono-8m
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
import torch.nn.functional as F

import networks
from layers import disp_to_depth
from utils import readlines
from kitti_utils import load_velodyne_points

# KITTI Eigen split test files
# These are the standard test images used in Monodepth2, Lite-Mono, etc.
EIGEN_TEST_FILES = [
    "2011_09_26_drive_0002_sync 03",
    "2011_09_26_drive_0002_sync 04",
    "2011_09_26_drive_0002_sync 05",
    ...  # Full list would be here
]

# Standard crop for KITTI evaluation (used by Eigen et al.)
CROP_MASK = {
    'height': [153, 371],  # garg/eigen crop
    'width': [44, 1197]
}


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark depth on KITTI Eigen split')
    parser.add_argument('--load_weights_folder', type=str, required=True,
                        help='path to trained depth model weights')
    parser.add_argument('--model', type=str, default='lite-mono-8m',
                        choices=['lite-mono', 'lite-mono-small', 'lite-mono-tiny', 'lite-mono-8m'])
    parser.add_argument('--data_path', type=str, default='./kitti_data/kitti_raw',
                        help='path to KITTI raw data')
    parser.add_argument('--height', type=int, default=192,
                        help='input height (must match training)')
    parser.add_argument('--width', type=int, default=640,
                        help='input width (must match training)')
    parser.add_argument('--max_depth', type=float, default=80.0,
                        help='maximum depth for evaluation')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='minimum depth for evaluation')
    parser.add_argument('--use_stereo', action='store_true',
                        help='use stereo pair (post-processing)')
    parser.add_argument('--save_pred_disps', action='store_true',
                        help='save predicted disparities to file')
    parser.add_argument('--ext_disp_to_eval', type=str, default=None,
                        help='optional path to numpy disparities file')
    parser.add_argument('--eval_split', type=str, default='eigen',
                        choices=['eigen', 'eigen_benchmark'])
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable CUDA')
    return parser.parse_args()


def compute_errors(gt, pred):
    """Compute standard KITTI depth metrics."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply stereo post-processing (left-right consistency check)."""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate_depth(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"-> Evaluating on device: {device}")

    # Load model
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    depth_path = os.path.join(args.load_weights_folder, "depth.pth")

    print(f"-> Loading model from {args.load_weights_folder}")
    encoder_dict = torch.load(encoder_path, map_location=device)

    # Get image dimensions from checkpoint if available
    feed_height = encoder_dict.get('height', args.height)
    feed_width = encoder_dict.get('width', args.width)

    # Initialize encoder
    encoder = networks.LiteMono(
        model=args.model,
        height=feed_height,
        width=feed_width,
        in_chans=3
    )
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device)
    encoder.eval()

    # Initialize depth decoder
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    depth_decoder.load_state_dict(torch.load(depth_path, map_location=device))
    depth_decoder.to(device)
    depth_decoder.eval()

    # Load test file list
    if args.eval_split == 'eigen':
        split_path = os.path.join(os.path.dirname(__file__), "splits", "eigen", "test_files.txt")
    else:
        split_path = os.path.join(os.path.dirname(__file__), "splits", "eigen_benchmark", "test_files.txt")

    filenames = readlines(split_path)
    print(f"-> {len(filenames)} files for evaluation")

    # Load predictions if provided
    if args.ext_disp_to_eval is not None:
        print(f"-> Loading predictions from {args.ext_disp_to_eval}")
        pred_disparities = np.load(args.ext_disp_to_eval)

        if args.use_stereo:
            print("-> Loading right disparities for post-processing")
            r_file = args.ext_disp_to_eval.replace('disparities', 'disparities_r')
            pred_disparities_r = np.load(r_file)
    else:
        # Generate predictions
        print("-> Generating predictions")
        pred_disparities = []

        with torch.no_grad():
            for idx, line in enumerate(filenames):
                folder, frame_index, side = line.split()

                # Load image
                image_path = os.path.join(
                    args.data_path,
                    folder,
                    "image_0{}/data".format(int(side)),
                    "{}.png".format(int(frame_index))
                )

                input_image = pil.open(image_path).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

                # Inference
                features = encoder(input_tensor)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = F.interpolate(
                    disp, (original_height, original_width), mode="bilinear", align_corners=False)

                pred_disp = disp_resized.squeeze().cpu().numpy()
                pred_disparities.append(pred_disp)

                if idx % 100 == 0:
                    print(f"   Processed {idx}/{len(filenames)} images")

        pred_disparities = np.array(pred_disparities)

        if args.save_pred_disps:
            output_path = os.path.join(args.load_weights_folder, "disparities.npy")
            print(f"-> Saving predictions to {output_path}")
            np.save(output_path, pred_disparities)

    # Compute depth from disparities
    print("-> Computing depth metrics")

    gt_depths = []
    pred_depths = []

    for idx in range(len(filenames)):
        line = filenames[idx]
        folder, frame_index, side = line.split()

        # Load ground truth depth
        calib_dir = os.path.join(args.data_path, folder.split("/")[0])
        velo_filename = os.path.join(
            args.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index))
        )

        try:
            gt_depth = generate_depth_map(calib_dir, velo_filename, int(side))
            gt_depths.append(gt_depth)
        except FileNotFoundError:
            print(f"   Warning: Ground truth not found for {line}")
            continue

        # Convert disparity to depth
        pred_disp = pred_disparities[idx]
        pred_disp = pred_disp.squeeze()
        pred_depth = 1 / pred_disp
        pred_depths.append(pred_depth)

    # Evaluate
    print("-> Evaluating metrics")

    errors = []
    ratios = []

    for pred_depth, gt_depth in zip(pred_depths, gt_depths):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()

        # Crop to valid region
        gt_height, gt_width = gt_depth.shape
        pred_depth = pred_depth[:gt_height, :gt_width]

        # Mask out invalid depths
        mask = (gt_depth > 0) & (gt_depth < args.max_depth)

        # Apply crop (garg/eigen)
        crop_mask = np.zeros(mask.shape)
        crop_mask[CROP_MASK['height'][0]:CROP_MASK['height'][1],
                  CROP_MASK['width'][0]:CROP_MASK['width'][1]] = 1
        mask = mask * crop_mask.astype(bool)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        # Median scaling (required for self-supervised methods)
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        # Clamp predictions
        pred_depth = np.clip(pred_depth, a_min=args.min_depth, a_max=args.max_depth)

        # Compute errors
        errors.append(compute_errors(gt_depth, pred_depth))

    if len(ratios) == 0:
        print("Error: No valid predictions!")
        return

    # Print results
    mean_errors = np.array(errors).mean(0)

    print("\n" + "=" * 50)
    print(f"Results for {args.model.upper()} on KITTI {args.eval_split} Split")
    print("=" * 50)
    print("{:>10} | {:>10} | {:>10} | {:>10}".format("Metric", "value", "value", "value"))
    print("-" * 50)
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("abs_rel", mean_errors[0], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("sq_rel", mean_errors[1], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("rmse", mean_errors[2], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("rmse_log", mean_errors[3], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("a1", mean_errors[4], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("a2", mean_errors[5], 0, 0))
    print("{:>10}: | {:10.3f} | {:10.3f} | {:10.3f}".format("a3", mean_errors[6], 0, 0))
    print("=" * 50)
    print(f"Scaling ratios - med: {np.median(ratios):.3f}, std: {np.std(ratios):.3f}")
    print("=" * 50)


def generate_depth_map(calib_dir, velo_filename, cam=2):
    """Generate depth map from velodyne points."""
    # Load calibration
    calib_filename = os.path.join(calib_dir, "calib_velo_to_cam.txt")

    # Load velodyne points
    velo = load_velodyne_points(velo_filename)

    # Remove points behind image plane
    velo = velo[velo[:, 0] >= 0, :]

    # Project to image
    # This is simplified - actual implementation would use calibration matrices
    depth_map = np.zeros((375, 1242))

    # For actual implementation, see kitti_utils.py in Monodepth2
    # This would project velodyne points using K, R, T matrices

    return depth_map


if __name__ == '__main__':
    args = parse_args()
    evaluate_depth(args)
