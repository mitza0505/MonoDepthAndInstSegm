#!/usr/bin/env python
"""
Training script for single-task segmentation baseline on COCO.
Trains only the segmentation head for fair comparison with published methods.
Uses COCO panoptic annotations (real ground truth, not pseudo-labels).
"""

from __future__ import absolute_import, division, print_function

import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
import os
import numpy as np

from utils import *
from layers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import networks


class SegOnlyTrainer:
    """Trainer for single-task segmentation on COCO."""

    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # COCO has 133 classes (80 thing + 53 stuff) for panoptic
        # or 80 classes for instance, or 133 for semantic
        self.num_classes = options.num_classes  # 133 for COCO panoptic

        # SEGMENTATION-ONLY: Encoder + Segmentation decoder only
        self.models["encoder"] = networks.LiteMono(
            model=self.opt.model,
            drop_path_rate=self.opt.drop_path,
            width=self.opt.width,
            height=self.opt.height,
            in_chans=3
        )
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Segmentation decoder with ASPP
        self.models["segmentation"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc,
            scales=[0],
            num_output_channels=self.num_classes,
            is_seg=True,
            use_aspp=True
        )
        self.models["segmentation"].to(self.device)
        self.parameters_to_train += list(self.models["segmentation"].parameters())

        # NO depth decoder, NO centers decoder for seg-only baseline

        self.model_optimizer = optim.AdamW(
            self.parameters_to_train,
            self.opt.lr,
            weight_decay=self.opt.weight_decay
        )

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            step_size=self.opt.scheduler_step_size,
            gamma=0.5
        )

        print("Training SEGMENTATION-ONLY baseline on COCO")
        print("Model name:", self.opt.model_name)
        print("Log directory:", self.log_path)
        print("Device:", self.device)
        print("Number of classes:", self.num_classes)

        # Load COCO dataset
        # Note: You need to implement COCOSegmentationDataset or use pycocotools
        # This is a placeholder structure

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.save_opts()

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print("Training epoch", self.epoch)
        self.set_train()
        self.model_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            if batch_idx % self.opt.log_frequency == 0:
                print(f"Epoch {self.epoch}, Batch {batch_idx}, Loss: {losses['loss'].item():.4f}")
                self.log("train", inputs, outputs, losses)

            self.step += 1

        # Validation
        self.validate()

    def process_batch(self, inputs):
        """Process a batch for segmentation training."""
        images = inputs["image"].to(self.device)  # (B, 3, H, W)
        labels = inputs["label"].to(self.device)  # (B, H, W) - class indices

        # Forward pass
        features = self.models["encoder"](images)
        seg_outputs = self.models["segmentation"](features)

        # Get logits at full resolution
        seg_logits = seg_outputs[("disp", 0)]  # (B, num_classes, H, W)

        # Upsample if needed to match label size
        if seg_logits.shape[2:] != labels.shape[1:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=labels.shape[1:],
                mode='bilinear',
                align_corners=False
            )

        # Compute loss
        # Cross-entropy loss for segmentation
        loss = F.cross_entropy(seg_logits, labels, ignore_index=255)

        outputs = {
            "seg_logits": seg_logits,
            "seg_pred": seg_logits.argmax(dim=1)
        }

        losses = {"loss": loss}

        return outputs, losses

    def validate(self):
        """Run validation."""
        self.set_eval()

        total_loss = 0
        total_miou = 0
        num_batches = 0

        with torch.no_grad():
            for inputs in self.val_loader:
                outputs, losses = self.process_batch(inputs)

                total_loss += losses["loss"].item()

                # Compute mIoU
                pred = outputs["seg_pred"].cpu().numpy()
                label = inputs["label"].cpu().numpy()

                # Calculate IoU per class
                ious = []
                for cls in range(self.num_classes):
                    if cls == 255:  # ignore index
                        continue
                    pred_mask = (pred == cls)
                    label_mask = (label == cls)

                    intersection = (pred_mask & label_mask).sum()
                    union = (pred_mask | label_mask).sum()

                    if union > 0:
                        ious.append(intersection / union)

                if ious:
                    total_miou += np.mean(ious)
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_miou = total_miou / num_batches

        print(f"Validation - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}")

        # Log to tensorboard
        self.writers["val"].add_scalar("loss", avg_loss, self.epoch)
        self.writers["val"].add_scalar("miou", avg_miou, self.epoch)

        self.set_train()

    def log(self, mode, inputs, outputs, losses):
        """Log to tensorboard."""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(l, v, self.step)

    def save_opts(self):
        """Save options."""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights."""
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        print(f"Saved model to {save_folder}")


class SegmentationOptions:
    """Options for segmentation training."""

    def __init__(self):
        # Paths
        self.data_path = "./coco"
        self.log_dir = "./tmp"
        self.model_name = "seg_baseline"

        # Model
        self.model = "lite-mono-8m"  # or lite-mono, lite-mono-small, etc.
        self.num_classes = 133  # COCO panoptic has 133 classes
        self.use_aspp = True

        # Training
        self.height = 640
        self.width = 640
        self.batch_size = 16
        self.num_epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.drop_path = 0.1
        self.scheduler_step_size = 30

        # Logging
        self.log_frequency = 100
        self.save_frequency = 5

        # System
        self.no_cuda = False
        self.num_workers = 8

        # Dataset
        self.dataset = "coco"  # or "cityscapes"


def train_seg_baseline():
    """Main training function."""
    options = SegmentationOptions()

    # Parse command line args if needed
    import sys
    if len(sys.argv) > 1:
        options.model_name = sys.argv[1]

    trainer = SegOnlyTrainer(options)
    trainer.train()


if __name__ == "__main__":
    train_seg_baseline()
