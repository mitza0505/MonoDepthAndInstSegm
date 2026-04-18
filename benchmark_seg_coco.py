#!/usr/bin/env python
"""
Benchmark segmentation on COCO or Cityscapes for comparison with published results.
Evaluates PQ (Panoptic Quality), mIoU, and AP metrics.

Usage:
    python benchmark_seg_coco.py --load_weights_folder /path/to/weights --dataset coco
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import networks


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark segmentation on COCO/Cityscapes')
    parser.add_argument('--load_weights_folder', type=str, required=True,
                        help='path to trained segmentation model')
    parser.add_argument('--model', type=str, default='lite-mono-8m',
                        choices=['lite-mono', 'lite-mono-small', 'lite-mono-tiny', 'lite-mono-8m'])
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco', 'cityscapes', 'kitti'])
    parser.add_argument('--data_path', type=str, default='./coco',
                        help='path to dataset')
    parser.add_argument('--split', type=str, default='val',
                        help='dataset split to evaluate on')
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--num_classes', type=int, default=133,
                        help='number of segmentation classes')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_results', action='store_true',
                        help='save prediction visualization')
    return parser.parse_args()


class COCOSemanticEvaluator:
    """Evaluate semantic segmentation on COCO."""

    def __init__(self, num_classes=133):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, label):
        """Update confusion matrix."""
        mask = (label >= 0) & (label < self.num_classes)

        # Flatten
        pred = pred[mask]
        label = label[mask]

        # Update confusion matrix
        indices = label * self.num_classes + pred
        bincount = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def get_iou(self):
        """Compute IoU for each class."""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) -
                 intersection)

        # Avoid division by zero
        iou = intersection / (union + 1e-10)
        return iou

    def get_miou(self):
        """Compute mean IoU."""
        iou = self.get_iou()
        # Exclude background class if needed
        return np.nanmean(iou)

    def get_pixel_accuracy(self):
        """Compute pixel accuracy."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total

    def get_results(self):
        """Get all metrics."""
        iou = self.get_iou()
        miou = self.get_miou()
        acc = self.get_pixel_accuracy()

        return {
            'mIoU': miou,
            'pixel_acc': acc,
            'per_class_iou': iou.tolist()
        }


class PanopticEvaluator:
    """Evaluate panoptic segmentation (PQ metric)."""

    def __init__(self, num_classes=133):
        self.num_classes = num_classes
        self.pq_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou': 0.0})

    def pq_score(self, pred_panoptic, gt_panoptic):
        """
        Compute PQ for a single image.

        pred_panoptic and gt_panoptic are encoded as: class_id * 1000 + instance_id
        """
        # Get unique instances
        pred_ids = np.unique(pred_panoptic)
        gt_ids = np.unique(gt_panoptic)

        # Filter out background
        pred_ids = pred_ids[pred_ids != 0]
        gt_ids = gt_ids[gt_ids != 0]

        # Build mask for each instance
        pred_masks = {pid: (pred_panoptic == pid) for pid in pred_ids}
        gt_masks = {gid: (gt_panoptic == gid) for gid in gt_ids}

        # Match predictions to ground truth
        matched = set()

        for gt_id in gt_ids:
            gt_mask = gt_masks[gt_id]
            gt_class = gt_id // 1000

            best_iou = 0.5  # Threshold for match
            best_pred = None

            for pred_id in pred_ids:
                if pred_id in matched:
                    continue

                pred_mask = pred_masks[pred_id]
                pred_class = pred_id // 1000

                # Must be same class
                if pred_class != gt_class:
                    continue

                # Compute IoU
                intersection = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                iou = intersection / (union + 1e-10)

                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id

            if best_pred is not None:
                # True positive
                self.pq_stats[gt_class]['tp'] += 1
                self.pq_stats[gt_class]['iou'] += best_iou
                matched.add(best_pred)
            else:
                # False negative
                self.pq_stats[gt_class]['fn'] += 1

        # False positives
        for pred_id in pred_ids:
            if pred_id not in matched:
                pred_class = pred_id // 1000
                self.pq_stats[pred_class]['fp'] += 1

    def get_pq(self):
        """Compute Panoptic Quality."""
        pq_scores = {}

        for cls in range(1, self.num_classes):  # Skip background
            stats = self.pq_stats[cls]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            iou = stats['iou']

            if tp == 0:
                pq_scores[cls] = 0.0
            else:
                sq = iou / tp  # Segmentation Quality
                rq = tp / (tp + 0.5 * fp + 0.5 * fn)  # Recognition Quality
                pq_scores[cls] = sq * rq

        # Average PQ
        avg_pq = np.mean([v for v in pq_scores.values() if v > 0])

        # PQ for things (countable objects) and stuff (background regions)
        # This requires knowing which classes are things vs stuff

        return {
            'PQ': avg_pq,
            'per_class_PQ': pq_scores
        }


def evaluate_coco(args):
    """Evaluate on COCO dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Evaluating on {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    # Load model
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    seg_path = os.path.join(args.load_weights_folder, "segmentation.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)

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
        num_output_channels=args.num_classes,
        is_seg=True,
        use_aspp=True
    )
    seg_decoder.load_state_dict(torch.load(seg_path, map_location=device))
    seg_decoder.to(device)
    seg_decoder.eval()

    # Initialize evaluators
    semantic_eval = COCOSemanticEvaluator(num_classes=args.num_classes)
    panoptic_eval = PanopticEvaluator(num_classes=args.num_classes)

    # TODO: Load COCO validation dataset
    # For now, this is a template structure

    print("Loading COCO validation set...")
    # val_dataset = COCOSegmentationDataset(args.data_path, split=args.split)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Running evaluation...")

    with torch.no_grad():
        # for idx, sample in enumerate(val_loader):
        #     image = sample['image'].to(device)
        #     label = sample['label'].numpy()
        #
        #     # Forward pass
        #     features = encoder(image)
        #     seg_out = seg_decoder(features)
        #
        #     logits = seg_out[("disp", 0)]
        #     pred = logits.argmax(dim=1).cpu().numpy()
        #
        #     # Update metrics
        #     semantic_eval.update(pred[0], label[0])
        #
        #     if (idx + 1) % 100 == 0:
        #         print(f"Processed {idx + 1}/{len(val_loader)} images")

        pass  # Placeholder

    # Compute and print results
    print("\n" + "=" * 60)
    print("Segmentation Results on COCO Validation Set")
    print("=" * 60)

    # results = semantic_eval.get_results()
    # print(f"mIoU: {results['mIoU']:.4f}")
    # print(f"Pixel Accuracy: {results['pixel_acc']:.4f}")

    print("\nComparison with Published Methods:")
    print("-" * 60)
    print("Method              | Backbone    | mIoU (%) | PQ (%)")
    print("-" * 60)
    print("Panoptic-DeepLab    | R101-DC5    |   42.3   |  46.5")
    print("Mask2Former         | R101        |   46.7   |  52.7")
    print("Mask2Former         | Swin-L      |   51.1   |  57.0")
    print("-" * 60)
    print("Ours (Lite-Mono-8M) | Lite-Mono   |    --    |   -- ")
    print("=" * 60)

    print("\nNote: To get actual results, you need to:")
    print("1. Download COCO dataset with panoptic annotations")
    print("2. Implement COCOSegmentationDataset class")
    print("3. Train seg_baseline_coco.py on COCO")
    print("4. Run this evaluation script")


def main():
    args = parse_args()

    if args.dataset == 'coco':
        evaluate_coco(args)
    elif args.dataset == 'cityscapes':
        print("Cityscapes evaluation not yet implemented")
    elif args.dataset == 'kitti':
        print("KITTI semantic evaluation not yet implemented")


if __name__ == '__main__':
    main()
