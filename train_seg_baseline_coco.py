#!/usr/bin/env python
"""
Training script for single-task segmentation baseline on COCO.
Trains only the segmentation head for fair comparison with published methods.
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

# Import the new dataset
from coco_dataset import COCOSegmentationDataset


def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class SegOnlyTrainer:
    """Trainer for single-task segmentation on COCO."""

    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train =[]

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_classes = options.num_classes

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

        self.models["segmentation"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc,
            scales=[0],
            num_output_channels=self.num_classes,
            is_seg=True,
            use_aspp=self.opt.use_aspp
        )
        self.models["segmentation"].to(self.device)
        self.parameters_to_train += list(self.models["segmentation"].parameters())

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

        # ----------------------------------------------------
        # DATA LOADING (Mirrors train_depth_baseline.py)
        # ----------------------------------------------------
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = COCOSegmentationDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, is_train=True)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = COCOSegmentationDataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width, is_train=False)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        self.val_iter = iter(self.val_loader)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

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
            
            # Full validation run at the end of every epoch for real mIoU
            self.val_epoch()

    def run_epoch(self):
        print("Training epoch", self.epoch)
        self.set_train()
        self.model_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # Log frequently matching the depth baseline frequency
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val() # Quick single-batch validation to populate TensorBoard

            self.step += 1

    def process_batch(self, inputs):
        """Process a batch for segmentation training."""
        images = inputs["image"].to(self.device)  # (B, 3, H, W)
        labels = inputs["label"].to(self.device)  # (B, H, W)

        features = self.models["encoder"](images)
        seg_outputs = self.models["segmentation"](features)

        seg_logits = seg_outputs[("disp", 0)]  # (B, num_classes, H, W)

        if seg_logits.shape[2:] != labels.shape[1:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=labels.shape[1:],
                mode='bilinear',
                align_corners=False
            )

        loss = F.cross_entropy(seg_logits, labels, ignore_index=255)

        outputs = {
            "seg_logits": seg_logits,
            "seg_pred": seg_logits.argmax(dim=1)
        }

        losses = {"loss": loss}

        return outputs, losses

    def val(self):
        """Single batch validation logging logic mimicking train_depth_baseline.py"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def val_epoch(self):
        """Run full epoch validation for mIoU calculation."""
        self.set_eval()

        total_loss = 0
        total_miou = 0
        num_batches = 0

        print(f"Running full validation for epoch {self.epoch}...")
        with torch.no_grad():
            for inputs in self.val_loader:
                outputs, losses = self.process_batch(inputs)
                total_loss += losses["loss"].item()

                pred = outputs["seg_pred"].cpu().numpy()
                label = inputs["label"].cpu().numpy()

                ious =[]
                for cls in range(self.num_classes):
                    if cls == 255: continue
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

        print(f"Epoch {self.epoch} Validation - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}")

        # Log Epoch metrics to tensorboard
        self.writers["val"].add_scalar("epoch/loss", avg_loss, self.epoch)
        self.writers["val"].add_scalar("epoch/miou", avg_miou, self.epoch)

        self.set_train()

    def log(self, mode, inputs, outputs, losses):
        """Log scalars and images to tensorboard."""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(l, v, self.step)

        # Log a few images per batch
        for j in range(min(4, self.opt.batch_size)):
            # Original RGB Image
            writer.add_image(f"image/{j}", inputs["image"][j].data, self.step)

            # Ground Truth Mask
            gt_tensor = inputs["label"][j].unsqueeze(0).float()
            # Ignore 255 indexing by masking it for visualization
            gt_tensor[gt_tensor == 255] = 0 
            # Scale to [0, 1] based on number of classes for bright visual feedback
            gt_visual = gt_tensor / float(self.num_classes)
            writer.add_image(f"gt_mask/{j}", gt_visual, self.step)

            # Predicted Mask
            pred_tensor = outputs["seg_pred"][j].unsqueeze(0).float()
            pred_visual = pred_tensor / float(self.num_classes)
            writer.add_image(f"pred_mask/{j}", pred_visual, self.step)

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        print_string = "epoch {:>3} | lr {:.6f} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(
            self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
            batch_idx, samples_per_sec, loss,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
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
    def __init__(self):
        self.data_path = "./coco"
        self.log_dir = "./tmp"
        self.model_name = "seg_baseline"
        
        # Ensures that your code reads `splits/coco/train_files.txt`
        self.split = "coco"
        self.dataset = "coco"

        self.model = "lite-mono-8m"
        self.num_classes = 133 
        self.use_aspp = True

        self.height = 640
        self.width = 640
        self.batch_size = 16
        self.num_epochs = 100
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.drop_path = 0.1
        self.scheduler_step_size = 30

        self.log_frequency = 100
        self.save_frequency = 5
        self.no_cuda = False
        self.num_workers = 8

def train_seg_baseline():
    options = SegmentationOptions()
    import sys
    if len(sys.argv) > 1:
        options.model_name = sys.argv[1]

    trainer = SegOnlyTrainer(options)
    trainer.train()

if __name__ == "__main__":
    train_seg_baseline()