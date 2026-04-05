from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch
import random
from torchvision import transforms

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
            line = self.filenames[0].split()
            scene_name = line[0]
            
            if '\ufeff' in scene_name:
                scene_name = scene_name.replace('\ufeff', '')
                
            frame_index = int(line[1])

            velo_filename = os.path.join(
                self.data_path,
                scene_name,
                "velodyne_points", "data", "{:010d}.bin".format(int(frame_index)))

            return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}".format(self.side_map[side]), "data", f_str)
        return image_path

    def get_mask_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        # Assuming your masks are stored in 'yolo_instance_mask_0x' folder structure
        mask_path = os.path.join(
            self.data_path, folder, "yolo_instance_mask_0{}".format(self.side_map[side]), "data", f_str)
        return mask_path

    def get_mask(self, folder, frame_index, side, do_flip):
        path = self.get_mask_path(folder, frame_index, side)
        
        if os.path.exists(path):
            mask = pil.open(path).convert('I')
        else:
            # Return blank mask if not found
            mask = pil.Image.new('I', (1242, 375), 0)

        if do_flip:
            mask = mask.transpose(pil.FLIP_LEFT_RIGHT)

        return mask

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split(os.path.sep)[0])
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points", "data", "{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].split()
        folder = line[0].strip()

        folder = folder.replace('\ufeff', '')
        folder = folder.replace("/", os.path.sep)

        if len(line) == 3:
            frame_index = int(line[1])
            self.side = line[2]
        else:
            frame_index = 0
            self.side = None

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        for i in self.frame_idxs:
            if i == "s":
                other_side = self.side_map[self.side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, self.side, do_flip)

        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue)
        else:
            color_aug = (lambda x: x)

        # This generates ("color_aug", i, scale) which are 3-channel RGB
        self.preprocess(inputs, color_aug)

        mask_img = self.get_mask(folder, frame_index, self.side, do_flip)
        mask_resized = mask_img.resize((self.width, self.height), pil.NEAREST)
        
        mask_np = np.array(mask_resized, dtype=np.int64) 
        
        inputs["seg_gt"] = torch.from_numpy(mask_np) # Shape: (H, W)

        for i in self.frame_idxs:
            if ("color", i, -1) in inputs:
                del inputs[("color", i, -1)]
            if ("color_aug", i, -1) in inputs:
                del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, self.side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if self.side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, "sequences", "{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]), f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}".format(self.side_map[side]), "data", f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path, folder, "proj_depth", "groundtruth",
            "image_0{}".format(self.side_map[side]), f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.LANCZOS)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt