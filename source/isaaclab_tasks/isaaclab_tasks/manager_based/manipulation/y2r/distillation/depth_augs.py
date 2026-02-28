"""GPU-accelerated depth image augmentation using NVIDIA Warp.

This module provides realistic depth sensor noise simulation for sim2real transfer.
All augmentations run on GPU in parallel using Warp kernels.

Supports both fp32 and fp16 depth tensors natively (fp16 kernels read/write half,
compute in fp32 internally for numerical stability).

Augmentation types:
    - Correlated noise: Simulates stereo depth sensor noise patterns
    - Pixel dropout: Random pixel dropouts to simulate sensor failures
    - Random uniform: Random depth blobs (false readings)
    - Sticks: Random line artifacts (sensor glitches)

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907054
"""

import torch
import warp as wp

# =============================================================================
# FP32 kernels
# =============================================================================

@wp.kernel
def add_pixel_dropout_and_randu_kernel(
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

    if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
        depths[batch_index, pixel_row, pixel_column] = 0.

    if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
        rand_depth =\
            rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        depths[batch_index, pixel_row, pixel_column] = rand_depth

        for i in range(kernel_size):
            for j in range(kernel_size):
                new_row = pixel_row + i
                new_col = pixel_column + j
                if new_row < height and new_col < width:
                    state =\
                        wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                    if wp.randf(state) < 0.25:
                        depths[batch_index, new_row, new_col] = rand_depth

@wp.kernel
def add_sticks_kernel(
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

            hor_idx = wp.clamp(int(hor_coord), 0, width - 1)
            vert_idx = wp.clamp(int(vert_coord), 0, height - 1)

            depths[batch_index, vert_idx, hor_idx] = rand_depth

            for j in range(1, int(max_stick_width)):
                if stick_rot > (3.14 / 4.) and stick_rot < (3. * 3.14 / 4.):
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth
                elif stick_rot > (5. * 3.14 / 4.) and stick_rot < (7. * 3.14 / 4.):
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth
                else:
                    new_hor = hor_idx + j
                    if new_hor < width:
                        depths[batch_index, vert_idx, new_hor] = rand_depth

@wp.kernel
def add_correlated_noise_kernel(
    depths: wp.array(dtype=float, ndim=3),
    rand_sigma_s_x: wp.array(dtype=float, ndim=3),
    rand_sigma_s_y: wp.array(dtype=float, ndim=3),
    rand_sigma_d: wp.array(dtype=float, ndim=3),
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    noisy_depths: wp.array(dtype=float, ndim=3)):
    """Correlated spatial + depth noise (fp32 variant)."""

    batch_index, pixel_row, pixel_column = wp.tid()

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

    u0 = wp.max(0, wp.min(u0, width - 1))
    u1 = wp.max(0, wp.min(u1, width - 1))
    v0 = wp.max(0, wp.min(v0, height - 1))
    v1 = wp.max(0, wp.min(v1, height - 1))

    w_00 = (1. - fu) * (1. - fv)
    w_01 = (1. - fu) * fv
    w_10 = fu * (1. - fv)
    w_11 = fu * fv

    interp_depth =\
        depths[batch_index, v0, u0] * w_00 +\
        depths[batch_index, v0, u1] * w_01 +\
        depths[batch_index, v1, u0] * w_10 +\
        depths[batch_index, v1, u1] * w_11

    depth_noise = rand_sigma_d[batch_index, pixel_row, pixel_column] * interp_depth * interp_depth
    noisy_depth = interp_depth + depth_noise
    noisy_depths[batch_index, pixel_row, pixel_column] = wp.clamp(noisy_depth, d_min, d_max)

# =============================================================================
# FP16 kernels — read/write half, compute in fp32 internally
# =============================================================================

@wp.kernel
def add_pixel_dropout_and_randu_kernel_f16(
    depths: wp.array(dtype=wp.float16, ndim=3),
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

    if rand_dropout[batch_index, pixel_row, pixel_column] <= p_dropout:
        depths[batch_index, pixel_row, pixel_column] = wp.float16(0.)

    if rand_u[batch_index, pixel_row, pixel_column] <= p_randu:
        rand_depth =\
            rand_u_values[batch_index, pixel_row, pixel_column] * (d_max - d_min) + d_min
        rand_depth_h = wp.float16(rand_depth)
        depths[batch_index, pixel_row, pixel_column] = rand_depth_h

        for i in range(kernel_size):
            for j in range(kernel_size):
                new_row = pixel_row + i
                new_col = pixel_column + j
                if new_row < height and new_col < width:
                    state =\
                        wp.rand_init(seed, batch_index + pixel_row + pixel_column + i + j)
                    if wp.randf(state) < 0.25:
                        depths[batch_index, new_row, new_col] = rand_depth_h

@wp.kernel
def add_sticks_kernel_f16(
    depths: wp.array(dtype=wp.float16, ndim=3),
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

        rand_depth_h = wp.float16(rand_depth)
        for i in range(int(wp.rint(stick_len))):
            hor_coord = float(pixel_column + i)
            vert_coord = wp.floor(float(i) * wp.sin(stick_rot)) + float(pixel_row)

            hor_idx = wp.clamp(int(hor_coord), 0, width - 1)
            vert_idx = wp.clamp(int(vert_coord), 0, height - 1)

            depths[batch_index, vert_idx, hor_idx] = rand_depth_h

            for j in range(1, int(max_stick_width)):
                if stick_rot > (3.14 / 4.) and stick_rot < (3. * 3.14 / 4.):
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth_h
                elif stick_rot > (5. * 3.14 / 4.) and stick_rot < (7. * 3.14 / 4.):
                    new_vert = vert_idx + j
                    if new_vert < height:
                        depths[batch_index, new_vert, hor_idx] = rand_depth_h
                else:
                    new_hor = hor_idx + j
                    if new_hor < width:
                        depths[batch_index, vert_idx, new_hor] = rand_depth_h

@wp.kernel
def add_correlated_noise_kernel_f16(
    depths: wp.array(dtype=wp.float16, ndim=3),
    rand_sigma_s_x: wp.array(dtype=float, ndim=3),
    rand_sigma_s_y: wp.array(dtype=float, ndim=3),
    rand_sigma_d: wp.array(dtype=float, ndim=3),
    height: int,
    width: int,
    d_min: float,
    d_max: float,
    noisy_depths: wp.array(dtype=wp.float16, ndim=3)):
    """Correlated spatial + depth noise (fp16 variant).

    Reads fp16 depth, computes bilinear interp + noise in fp32, writes fp16.
    """

    batch_index, pixel_row, pixel_column = wp.tid()

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

    u0 = wp.max(0, wp.min(u0, width - 1))
    u1 = wp.max(0, wp.min(u1, width - 1))
    v0 = wp.max(0, wp.min(v0, height - 1))
    v1 = wp.max(0, wp.min(v1, height - 1))

    w_00 = (1. - fu) * (1. - fv)
    w_01 = (1. - fu) * fv
    w_10 = fu * (1. - fv)
    w_11 = fu * fv

    # Read fp16 → fp32 for interpolation
    interp_depth =\
        float(depths[batch_index, v0, u0]) * w_00 +\
        float(depths[batch_index, v0, u1]) * w_01 +\
        float(depths[batch_index, v1, u0]) * w_10 +\
        float(depths[batch_index, v1, u1]) * w_11

    depth_noise = rand_sigma_d[batch_index, pixel_row, pixel_column] * interp_depth * interp_depth
    noisy_depth = interp_depth + depth_noise
    noisy_depths[batch_index, pixel_row, pixel_column] = wp.float16(wp.clamp(noisy_depth, d_min, d_max))

# =============================================================================
# Unused in augment() pipeline but kept for external callers
# =============================================================================

@wp.kernel
def add_normal_noise_kernel(
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

    uv = wp.vec4f(float(pixel_column), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column]
    point_3d = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    uv = wp.vec4f(float(pixel_column + 1), float(pixel_row), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row, pixel_column + 1]
    point_3d_01 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    uv = wp.vec4f(float(pixel_column), float(pixel_row + 1), 1., 1.)
    xyzw = wp.inverse(cam_matrix) * uv

    x_hat = xyzw[0] / xyzw[3]
    y_hat = xyzw[1] / xyzw[3]
    z_hat = xyzw[2] / xyzw[3]

    d = depths[batch_index, pixel_row + 1, pixel_column]
    point_3d_10 = wp.vec3f(d * x_hat, d * y_hat, d * z_hat)

    x_axis = wp.normalize(point_3d_01 - point_3d)
    y_axis = wp.normalize(point_3d_10 - point_3d)

    normal = wp.cross(x_axis, y_axis)

    point_3d =\
        point_3d + 1000. * rand_sigma_theta[batch_index, pixel_row, pixel_column] * normal

    depths[batch_index, pixel_row, pixel_column] = -point_3d[2] / 1000.

# =============================================================================
# DepthAug class
# =============================================================================

class DepthAug():
    """GPU-accelerated depth image augmentation using NVIDIA Warp.

    Supports fp16 and fp32 depth tensors natively. FP16 kernels read/write half
    precision but compute in fp32 internally for numerical stability.

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

        # Pre-allocated random buffers (lazily sized on first augment call)
        self._buf_shape = None  # (B, H, W) — invalidated if shape changes
        self._rand_bufs = {}    # name → pre-allocated tensor

        # Merge config
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            for key in config:
                if key in self.config:
                    self.config[key].update(config[key])

    def _get_buf(self, name: str, shape: tuple, normal: bool = False) -> torch.Tensor:
        """Return a pre-allocated fp32 random buffer, re-filling it in-place.

        Buffers are allocated once per shape and reused across steps.
        Random values are always fp32 regardless of depth dtype.
        """
        if self._buf_shape != shape:
            # Shape changed (first call or env count changed) — reallocate all
            self._rand_bufs.clear()
            self._buf_shape = shape

        buf = self._rand_bufs.get(name)
        if buf is None:
            buf = torch.empty(shape, device=self.device, dtype=torch.float32)
            self._rand_bufs[name] = buf

        # Fill in-place (no allocation)
        if normal:
            buf.normal_()
        else:
            buf.uniform_()
        return buf

    def augment(self, depth: torch.Tensor, d_min: float = 0.0, d_max: float = 1.0) -> torch.Tensor:
        """Apply all depth augmentations.

        Natively supports fp16 and fp32 depth tensors. FP16 uses dedicated
        Warp kernels that read/write half but compute in fp32 internally.

        Args:
            depth: Depth tensor of shape (B, H, W) — fp16 or fp32.
            d_min: Minimum valid depth value (0.0 for normalized)
            d_max: Maximum valid depth value (1.0 for normalized)

        Returns:
            Augmented depth tensor (B, H, W), same dtype as input.
        """
        # Allocate output buffer (same dtype as input — no cast)
        depths = torch.empty_like(depth)

        # 1. Correlated noise (simulates stereo matching noise)
        cfg = self.config["correlated_noise"]
        self.add_correlated_noise(
            depth, depths,
            cfg["sigma_s"], cfg["sigma_d"],
            d_min, d_max
        )

        # 2. Pixel dropout and random uniform blobs (in-place on depths)
        cfg = self.config["pixel_dropout_and_randu"]
        self.add_pixel_dropout_and_randu(
            depths,
            cfg["p_dropout"], cfg["p_randu"],
            d_min, d_max
        )

        # 3. Random sticks (in-place on depths)
        cfg = self.config["sticks"]
        self.add_sticks(
            depths,
            cfg["p_stick"], cfg["max_stick_len"], cfg["max_stick_width"],
            d_min, d_max
        )

        # Clamp to valid range
        depths.clamp_(d_min, d_max)

        return depths

    def add_pixel_dropout_and_randu(self, depths, p_dropout, p_randu, d_min, d_max):

        shape = depths.shape
        batch_size, height, width = shape

        rand_dropout = self._get_buf("dropout", shape)
        rand_u = self._get_buf("randu_mask", shape)
        rand_u_values = self._get_buf("randu_vals", shape)

        kernel = add_pixel_dropout_and_randu_kernel_f16 if depths.dtype == torch.float16 \
            else add_pixel_dropout_and_randu_kernel

        wp.launch(kernel=kernel,
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
        shape = depths.shape
        batch_size, height, width = shape

        rand_stick = self._get_buf("stick_mask", shape)
        rand_stick_depths = self._get_buf("stick_depths", shape)

        kernel = add_sticks_kernel_f16 if depths.dtype == torch.float16 \
            else add_sticks_kernel

        wp.launch(kernel=kernel,
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
        shape = depths.shape
        batch_size, height, width = shape

        rand_sigma_s_x = self._get_buf("cn_sx", shape, normal=True).mul_(sigma_s)
        rand_sigma_s_y = self._get_buf("cn_sy", shape, normal=True).mul_(sigma_s)
        rand_sigma_d = self._get_buf("cn_d", shape, normal=True).mul_(sigma_d)

        kernel = add_correlated_noise_kernel_f16 if depths.dtype == torch.float16 \
            else add_correlated_noise_kernel

        wp.launch(kernel=kernel,
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
