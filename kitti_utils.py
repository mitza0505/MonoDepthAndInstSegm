from __future__ import absolute_import, division, print_function
import os
import numpy as np

def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format"""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def read_calib_file(path):
    """Read KITTI calibration file"""
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass
    return data

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    Args:
        calib_dir: Directory containing calib files
        velo_filename: Path to .bin file
        cam: Camera index (2 for left color)
        vel_depth: If True, use Euclidean distance. If False, use Z-axis depth (Standard).
    """
    # 1. Load Calibration
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    
    # Reshape matrices
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # Get image shape (H, W)
    # S_rect_02 is [width, height], so we flip it for numpy shape [height, width]
    im_shape = cam2cam["S_rect_02"][::-1].astype(int)

    # Compute projection matrix: Velodyne -> Image Plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # 2. Load Velodyne Points
    velo = load_velodyne_points(velo_filename)
    
    # Remove points behind the capture point
    velo = velo[velo[:, 0] >= 0, :]

    # 3. Project to Image
    # Project 3D points to 2D image plane
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    
    # Normalization: x = X/Z, y = Y/Z
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        # Use Euclidean distance from sensor
        velo_pts_im[:, 2] = velo[:, 0]

    # 4. Filter Out-of-Bounds Points
    # Round to integer pixel coordinates
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]
    
    inds = np.argsort(velo_pts_im[:, 2])[::-1]
    velo_pts_im = velo_pts_im[inds]

    # Create depth map
    depth = np.zeros((im_shape[0], im_shape[1]), dtype=np.float32)
    
    # Assign depth values. 
    # Because of the sort, the closest point is assigned last.
    depth[velo_pts_im[:, 1].astype(int), velo_pts_im[:, 0].astype(int)] = velo_pts_im[:, 2]

    return depth