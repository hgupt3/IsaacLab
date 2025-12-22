"""GPU-accelerated depth image augmentation using NVIDIA Warp.

This module provides realistic depth sensor noise simulation for sim2real transfer.
All augmentations run on GPU in parallel using Warp kernels.

Augmentation types:
    - Correlated noise: Simulates stereo depth sensor noise patterns
    - Pixel dropout: Random pixel dropouts to simulate sensor failures
    - Random uniform: Random depth blobs (false readings)
    - Sticks: Random line artifacts (sensor glitches)

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907054
"""

import torch
import warp as wp

@wp.kernel
def add_pixel_dropout_and_randu_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_dropout: wp.array(dtype=float, ndim=3),
    rand_u: wp.array(dtype=float, ndim=3),
    rand_u_values: wp.array(dtype=float, ndim=3),
    p_dropout: float,
    p_randu: float,
    d_min: float,
    d_max: float,
    height: int,
    width: int,
    kernel_size: int,
    seed: int):

    batch_index, pixel_row, pixel_column = wp.tid()

    # Perform dropout
    if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
        depths[batch_index, pixel_row, pixel_column] = 0.

    # Insert random uniform value
    if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
        rand_depth =\
            rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        depths[batch_index, pixel_row, pixel_column] = rand_depth

        for i in range(kernel_size):
            for j in range(kernel_size):
                # BOUNDS CHECK: ensure we don't write outside the image
                new_row = pixel_row + i
                new_col = pixel_column + j
                if new_row < height and new_col < width:
                    state =\
                        wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                    if wp.randf(state) < 0.25:
                        depths[batch_index, new_row, new_col] = rand_depth

@wp.kernel
def add_sticks_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sticks: wp.array(dtype=float, ndim=3),
    rand_sticks_depths: wp.array(dtype=float, ndim=3),
    p_stick: float,
    max_stick_len: float,
    max_stick_width: float,
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    seed: int):

    batch_index, pixel_row, pixel_column = wp.tid()

    stick_width = float(0.)
    stick_len = float(0.)
    stick_rot = float(0.)
        
    rand_depth =\
        rand_sticks_depths[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
    
    if rand_sticks[batch_index, pixel_row, pixel_column] <= p_stick:
        for i in range(3):
            state =\
                wp.rand_init(seed, batch_index + pixel_row + pixel_column + i)

            if i == 0:
                stick_width = wp.randf(state) * max_stick_width
            if i == 1:
                stick_len = wp.randf(state) * max_stick_len + 1.
            if i == 2:
                stick_rot = wp.randf(state) * (3.14 * 2.)

        for i in range(int(wp.rint(stick_len))):
            hor_coord = float(pixel_column + i)
            vert_coord = wp.floor(float(i) * wp.sin(stick_rot)) + float(pixel_row)

            # BOUNDS CHECK: clamp to valid range [0, dim-1]
            hor_idx = wp.clamp(int(hor_coord), 0, width - 1)
            vert_idx = wp.clamp(int(vert_coord), 0, height - 1)

            depths[batch_index, vert_idx, hor_idx] = rand_depth

            # Draw stick width
            for j in range(1, int(max_stick_width)):
                if stick_rot > (3.14 / 4.) and stick_rot < (3. * 3.14 / 4.):
                    # Vertical stick - add pixels below
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth
                elif stick_rot > (5. * 3.14 / 4.) and stick_rot < (7. * 3.14 / 4.):
                    # Vertical stick - add pixels below
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth
                else:
                    # Horizontal stick - add pixels to the right
                    new_hor = hor_idx + j
                    if new_hor < width:
                        depths[batch_index, vert_idx, new_hor] = rand_depth

@wp.kernel
def add_correlated_noise_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_s_x: wp.array(dtype=float, ndim=3),
    rand_sigma_s_y: wp.array(dtype=float, ndim=3),
    rand_sigma_d: wp.array(dtype=float, ndim=3),
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    # outputs
    noisy_depths: wp.array(dtype=float, ndim=3)):
    """Apply correlated spatial + depth noise for normalized [0,1] depth.
    
    This kernel applies two types of correlated noise:
    1. Spatial noise: Bilinear interpolation from nearby pixels (simulates stereo matching errors)
    2. Depth noise: Additive noise scaled by depth squared (farther = more noise, like real sensors)
    
    Works with normalized depth in [0, 1] range (not real-world mm units).
    """

    batch_index, pixel_row, pixel_column = wp.tid()

    # Draw float pixel coordinates (spatial noise)
    nx = rand_sigma_s_x[batch_index, pixel_row, pixel_column]
    ny = rand_sigma_s_y[batch_index, pixel_row, pixel_column]

    u = nx + float(pixel_column)
    v = ny + float(pixel_row)

    u0 = int(u)
    v0 = int(v)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = u - float(u0)
    fv = v - float(v0)

    # Ensure bounds
    u0 = wp.max(0, wp.min(u0, width - 1))
    u1 = wp.max(0, wp.min(u1, width - 1))
    v0 = wp.max(0, wp.min(v0, height - 1))
    v1 = wp.max(0, wp.min(v1, height - 1))

    # Linear interp weights
    w_00 = (1. - fu) * (1. - fv)
    w_01 = (1. - fu) * fv
    w_10 = fu * (1. - fv)
    w_11 = fu * fv

    # Interpolated depth (spatial noise via bilinear sampling)
    interp_depth =\
        depths[batch_index, v0, u0] * w_00 +\
        depths[batch_index, v0, u1] * w_01 +\
        depths[batch_index, v1, u0] * w_10 +\
        depths[batch_index, v1, u1] * w_11
    
    # Depth-dependent noise: farther objects have more noise (quadratic scaling)
    # This mimics real depth sensor behavior where uncertainty grows with distance
    # For normalized depth d in [0,1], noise scale = d^2 * sigma_d
    depth_noise = rand_sigma_d[batch_index, pixel_row, pixel_column] * interp_depth * interp_depth
    
    # Apply depth noise and clamp to valid range
    noisy_depth = interp_depth + depth_noise
    noisy_depths[batch_index, pixel_row, pixel_column] = wp.clamp(noisy_depth, d_min, d_max)

@wp.kernel
def add_normal_noise_kernel(
    # inputs
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_theta: wp.array(dtype=float, ndim=3),
    cam_matrix: wp.mat44f,
    height: int,
    width: int,
    d_min: float,
    d_max: float):

    batch_index, pixel_row, pixel_column = wp.tid()
    
    if pixel_row == (height - 1):
        return
    if pixel_column == (width - 1):
        return

    # Get 3D point at current pixel
    uv = wp.vec4f(float(pixel_column), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column]
    point_3d = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    # Get 3D point of one pixel to the right
    uv = wp.vec4f(float(pixel_column + 1), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column + 1]
    point_3d_01 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)
    
    # Get 3D point of one pixel up
    uv = wp.vec4f(float(pixel_column), float(pixel_row + 1), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row + 1, pixel_column]
    point_3d_10 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    # Find normal of three points

    x_axis = wp.normalize(point_3d_01 - point_3d)
    y_axis = wp.normalize(point_3d_10 - point_3d)

    normal = wp.cross(x_axis, y_axis)
    
    #if pixel_row == 90 and pixel_column == 100:
    #    print(point_3d)

    # Now perturb 3D coordinate by noise amount along normal
    point_3d =\
        point_3d + 1000. * rand_sigma_theta[batch_index, pixel_row, pixel_column] * normal
    #if pixel_row == 90 and pixel_column == 100:
    #    print(point_3d)
    #    print('----')

    # Take the z value as depth in meters
    depths[batch_index, pixel_row, pixel_column] = -point_3d[2] / 1000.

class DepthAug():
    """GPU-accelerated depth image augmentation using NVIDIA Warp.
    
    Augmentations (all run on GPU in parallel):
    - Correlated noise: Simulates stereo depth sensor noise
    - Pixel dropout: Random pixel dropouts to 0
    - Random uniform: Random depth blobs
    - Sticks: Random line artifacts (sensor glitches)
    
    Default config suitable for 64x64 depth images in [0, 0.5] meter range.
    """
    
    # Default augmentation config
    DEFAULT_CONFIG = {
        "correlated_noise": {
            "sigma_s": 0.5,   # Spatial noise
            "sigma_d": 0.15,  # Depth noise
        },
        "pixel_dropout_and_randu": {
            "p_dropout": 0.003,  # Dropout probability
            "p_randu": 0.003,    # Random uniform probability
        },
        "sticks": {
            "p_stick": 0.00025,     # Stick probability
            "max_stick_len": 12.,   # Max stick length (pixels)
            "max_stick_width": 2.,  # Max stick width (pixels)
        },
    }
    
    def __init__(self, device, config: dict = None):
        """Initialize depth augmentation.
        
        Args:
            device: CUDA device string (e.g., 'cuda:0')
            config: Optional dict to override DEFAULT_CONFIG
        """
        self.device = device
        self.seed = 42
        self.kernel_size = 2
        
        # Merge config
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            for key in config:
                if key in self.config:
                    self.config[key].update(config[key])
    
    def augment(self, depth: torch.Tensor, d_min: float = 0.0, d_max: float = 1.0) -> torch.Tensor:
        """Apply all depth augmentations.
        
        Args:
            depth: Depth tensor of shape (B, H, W) - values in [d_min, d_max]
                   Our depth is normalized to [0, 1] based on camera clipping range.
            d_min: Minimum valid depth value (0.0 for normalized)
            d_max: Maximum valid depth value (1.0 for normalized)
            
        Returns:
            Augmented depth tensor (B, H, W)
        """
        # Clone to avoid modifying input
        depths = depth.clone()
        noisy_depths = depths.clone()
        
        # 1. Correlated noise (simulates stereo matching noise)
        cfg = self.config["correlated_noise"]
        self.add_correlated_noise(
            depth, noisy_depths, 
            cfg["sigma_s"], cfg["sigma_d"], 
            d_min, d_max
        )
        depths = noisy_depths
        
        # 2. Pixel dropout and random uniform blobs
        cfg = self.config["pixel_dropout_and_randu"]
        self.add_pixel_dropout_and_randu(
            depths,
            cfg["p_dropout"], cfg["p_randu"],
            d_min, d_max
        )
        
        # 3. Random sticks (sensor artifacts)
        cfg = self.config["sticks"]
        self.add_sticks(
            depths,
            cfg["p_stick"], cfg["max_stick_len"], cfg["max_stick_width"],
            d_min, d_max
        )
        
        # Clamp to valid range (don't zero out - that corrupts far depth!)
        # Only set truly invalid values (negative) to 0
        depths = depths.clamp(d_min, d_max)
        
        return depths

    def add_pixel_dropout_and_randu(self, depths, p_dropout, p_randu, d_min, d_max):
        
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]

        rand_dropout = torch.rand(batch_size, height, width, device=self.device)
        rand_u = torch.rand(batch_size, height, width, device=self.device)
        rand_u_values = torch.rand(batch_size, height, width, device=self.device)
        
        wp.launch(kernel=add_pixel_dropout_and_randu_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_dropout),
                      wp.torch.from_torch(rand_u),
                      wp.torch.from_torch(rand_u_values),
                      p_dropout,
                      p_randu,
                      d_min,
                      d_max,
                      height,
                      width,
                      self.kernel_size,
                      self.seed],
                  device=self.device)

    def add_sticks(self, depths, p_stick, max_stick_len, max_stick_width, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]

        rand_stick = torch.rand(batch_size, height, width, device=self.device)
        rand_stick_depths = torch.rand(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_sticks_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_stick),
                      wp.torch.from_torch(rand_stick_depths),
                      p_stick,
                      max_stick_len,
                      max_stick_width,
                      height,
                      width,
                      d_min,
                      d_max,
                      self.seed],
                  device=self.device)

    # NOTE: taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907054
    def add_correlated_noise(self, depths, noisy_depths, sigma_s, sigma_d, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]
        
        rand_sigma_s_x = sigma_s * torch.randn(batch_size, height, width, device=self.device)
        rand_sigma_s_y = sigma_s * torch.randn(batch_size, height, width, device=self.device)
        rand_sigma_d = sigma_d * torch.randn(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_correlated_noise_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_sigma_s_x),
                      wp.torch.from_torch(rand_sigma_s_y),
                      wp.torch.from_torch(rand_sigma_d),
                      height,
                      width,
                      d_min,
                      d_max],
                  outputs=[
                      wp.torch.from_torch(noisy_depths)],
                  device=self.device)

    def add_normal_noise(self, depths, sigma_theta, cam_matrix, d_min, d_max):
        batch_size = depths.shape[0]
        height = depths.shape[1]
        width = depths.shape[2]
        
        rand_sigma_theta =\
            sigma_theta * torch.randn(batch_size, height, width, device=self.device)

        wp.launch(kernel=add_normal_noise_kernel,
                  dim=[batch_size, height, width],
                  inputs=[
                      wp.torch.from_torch(depths),
                      wp.torch.from_torch(rand_sigma_theta),
                      cam_matrix,
                      height,
                      width,
                      d_min,
                      d_max],
                  device=self.device)
        
