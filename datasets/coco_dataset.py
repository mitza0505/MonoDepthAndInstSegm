from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


class COCOSegmentationDataset(data.Dataset):
    """Dataset class for COCO Segmentation
    
    Expects filenames in the split txt file to be formatted as:
    <image_path> <mask_path>
    Example: images/train2017/000000000139.jpg annotations/semantic_train2017/000000000139.png
    """
    def __init__(self, data_path, filenames, height, width, is_train=False):
        super(COCOSegmentationDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.interp = Image.BILINEAR
        self.mask_interp = Image.NEAREST

        # Consistent Color Jitter parameters matching your MonoDataset
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.mask_resize = transforms.Resize((self.height, self.width), interpolation=self.mask_interp)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].strip().split()
        
        # Load paths relative to data_path
        img_path = os.path.join(self.data_path, line[0])
        mask_path = os.path.join(self.data_path, line[1])

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # Mask remains in its native mode (L or P)

        # 1. Flip Augmentation (Applied to both Image and Mask)
        do_flip = self.is_train and random.random() > 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Resize
        img = self.resize(img)
        mask = self.mask_resize(mask)

        # 3. Color Augmentation (Applied ONLY to Image)
        if self.is_train and random.random() > 0.5:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue)
            img = color_aug(img)

        # Convert to Tensors
        inputs["image"] = self.to_tensor(img)
        # Masks become 2D integer tensors where each pixel represents a class (e.g. 0-132, 255 for background)
        inputs["label"] = torch.from_numpy(np.array(mask, dtype=np.int64))

        return inputs