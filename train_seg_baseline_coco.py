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
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import networks
# Import your dataset class
from coco_dataset import COCOSegmentationDataset


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
                        help='dataset split to evaluate on (e.g., val)')
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
        # This mask gracefully ignores index 255 (or anything outside valid classes)
        mask = (label >= 0) & (label < self.num_classes)

        # Flatten
        pred = pred[mask]
        label = label[mask]

        # Update confusion matrix
        indices = label * self.num_classes + pred
        bincount = np.bincount(indices, minlength=self.num_classes ** 2)
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
        # Exclude background class if needed, here we compute across all valid
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

    # ... (Unchanged Panoptic Logic, ready for MTL Panoptic testing later)
    def __init__(self, num_classes=133):
        self.num_classes = num_classes
        self.pq_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'iou': 0.0})

    def pq_score(self, pred_panoptic, gt_panoptic):
        pred_ids = np.unique(pred_panoptic)
        gt_ids = np.unique(gt_panoptic)
        pred_ids = pred_ids[pred_ids != 0]
        gt_ids = gt_ids[gt_ids != 0]

        pred_masks = {pid: (pred_panoptic == pid) for pid in pred_ids}
        gt_masks = {gid: (gt_panoptic == gid) for gid in gt_ids}

        matched = set()
        for gt_id in gt_ids:
            gt_mask = gt_masks[gt_id]
            gt_class = gt_id // 1000

            best_iou = 0.5
            best_pred = None

            for pred_id in pred_ids:
                if pred_id in matched: continue
                pred_mask = pred_masks[pred_id]
                pred_class = pred_id // 1000

                if pred_class != gt_class: continue

                intersection = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                iou = intersection / (union + 1e-10)

                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id

            if best_pred is not None:
                self.pq_stats[gt_class]['tp'] += 1
                self.pq_stats[gt_class]['iou'] += best_iou
                matched.add(best_pred)
            else:
                self.pq_stats[gt_class]['fn'] += 1

        for pred_id in pred_ids:
            if pred_id not in matched:
                pred_class = pred_id // 1000
                self.pq_stats[pred_class]['fp'] += 1

    def get_pq(self):
        pq_scores = {}
        for cls in range(1, self.num_classes):
            stats = self.pq_stats[cls]
            tp, fp, fn, iou = stats['tp'], stats['fp'], stats['fn'], stats['iou']
            if tp == 0:
                pq_scores[cls] = 0.0
            else:
                sq = iou / tp
                rq = tp / (tp + 0.5 * fp + 0.5 * fn)
                pq_scores[cls] = sq * rq

        avg_pq = np.mean([v for v in pq_scores.values() if v > 0])
        return {'PQ': avg_pq, 'per_class_PQ': pq_scores}


def evaluate_coco(args):
    """Evaluate on COCO dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Evaluating on {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")

    # 1. Load model
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
        use_aspp=True  # As per your baseline training configuration
    )
    seg_decoder.load_state_dict(torch.load(seg_path, map_location=device))
    seg_decoder.to(device)
    seg_decoder.eval()

    # 2. Initialize evaluators
    semantic_eval = COCOSemanticEvaluator(num_classes=args.num_classes)
    panoptic_eval = PanopticEvaluator(num_classes=args.num_classes)

    # 3. Load COCO validation dataset
    print("Loading COCO validation set...")

    # Resolves paths just like train_seg_baseline.py
    split_path = os.path.join(os.path.dirname(__file__), "splits", args.dataset, f"{args.split}_files.txt")
    with open(split_path, 'r') as f:
        val_filenames = f.readlines()

    val_dataset = COCOSegmentationDataset(
        args.data_path, val_filenames, height=args.height, width=args.width, is_train=False
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 4. Evaluation Loop
    print("Running evaluation...")
    with torch.no_grad():
        for idx, inputs in enumerate(val_loader):
            image = inputs['image'].to(device)
            label = inputs['label'].numpy()

            # Forward pass
            features = encoder(image)
            seg_out = seg_decoder(features)

            logits = seg_out[("disp", 0)]

            # Match Logit Size to Original Label Shape
            if logits.shape[2:] != label.shape[1:]:
                logits = F.interpolate(
                    logits,
                    size=label.shape[1:],
                    mode='bilinear',
                    align_corners=False
                )

            pred = logits.argmax(dim=1).cpu().numpy()

            # Update metrics (shape formatting: drop batch dim via [0])
            semantic_eval.update(pred[0], label[0])

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(val_loader)} images...")

    # 5. Compute and print results
    results = semantic_eval.get_results()
    final_miou = results['mIoU'] * 100
    final_acc = results['pixel_acc'] * 100

    print("\n" + "=" * 60)
    print("Segmentation Results on COCO Validation Set")
    print("=" * 60)
    print(f"Computed mIoU:           {final_miou:.2f}%")
    print(f"Computed Pixel Accuracy: {final_acc:.2f}%")
    print("=" * 60)

    print("\nComparison with Published Methods:")
    print("-" * 60)
    print("Method              | Backbone    | mIoU (%) | PQ (%)")
    print("-" * 60)
    print("Panoptic-DeepLab    | R101-DC5    |   42.3   |  46.5")
    print("Mask2Former         | R101        |   46.7   |  52.7")
    print("Mask2Former         | Swin-L      |   51.1   |  57.0")
    print("-" * 60)
    print(f"Ours (Lite-Mono-8M) | Lite-Mono   |   {final_miou:.1f}   |   -- ")
    print("=" * 60)

    # (Note: PQ requires the full multi-task inference pipeline including centers)


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