"""
Pinhole Camera Projection Module

Converts 2D image pixels with depth values to 3D point cloud coordinates
using the pinhole camera model equations.

Pinhole Camera Model:
    For a pixel (u, v) with depth Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth_value

    Where:
        (fx, fy) = focal lengths in pixels
        (cx, cy) = principal point (typically image center)
"""

import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for pinhole model."""

    fx: float  # Focal length in x (pixels)
    fy: float  # Focal length in y (pixels)
    cx: float  # Principal point x
    cy: float  # Principal point y

    @classmethod
    def from_image_size(
        cls,
        width: int,
        height: int,
        fov_degrees: float = 60.0
    ) -> "CameraIntrinsics":
        """
        Create intrinsics from image size with assumed field of view.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            fov_degrees: Horizontal field of view in degrees

        Returns:
            CameraIntrinsics with computed values
        """
        # Compute focal length from FOV
        # fov = 2 * atan(width / (2 * fx))
        # fx = width / (2 * tan(fov / 2))
        fov_rad = np.radians(fov_degrees)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # Assume square pixels

        # Principal point at image center
        cx = width / 2.0
        cy = height / 2.0

        return cls(fx=fx, fy=fy, cx=cx, cy=cy)

    @classmethod
    def default_for_image(cls, width: int, height: int) -> "CameraIntrinsics":
        """
        Create default intrinsics (focal length = image width).

        This is a common approximation when true intrinsics are unknown.

        Args:
            width: Image width
            height: Image height

        Returns:
            CameraIntrinsics with default values
        """
        return cls(
            fx=float(width),
            fy=float(width),  # Assume square pixels
            cx=width / 2.0,
            cy=height / 2.0
        )


def project_to_3d(
    depth_map: np.ndarray,
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth: float = 0.1,
    max_depth: float = 100.0
) -> Tuple[np.ndarray, float]:
    """
    Project 2D pixels to 3D point cloud using pinhole camera model.

    Uses vectorized numpy operations for efficient batch projection.

    Args:
        depth_map: Depth values (H, W) float32
        image: BGR image (H, W, 3) uint8
        intrinsics: Camera intrinsic parameters
        min_depth: Minimum depth threshold (filter noise)
        max_depth: Maximum depth threshold

    Returns:
        Tuple of:
            - Point cloud (N, 6) float32 where columns are [X, Y, Z, R, G, B]
            - Projection time in milliseconds
    """
    start_time = time.perf_counter()

    height, width = depth_map.shape

    # Create pixel coordinate grids
    # u = column index (x), v = row index (y)
    u_coords = np.arange(width, dtype=np.float32)
    v_coords = np.arange(height, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    # Flatten for vectorized operations
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    z_flat = depth_map.flatten()

    # Create depth mask for valid points
    valid_mask = (z_flat >= min_depth) & (z_flat <= max_depth) & np.isfinite(z_flat)

    # Apply mask
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    # Project to 3D using pinhole model
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    x_3d = (u_valid - intrinsics.cx) * z_valid / intrinsics.fx
    y_3d = (v_valid - intrinsics.cy) * z_valid / intrinsics.fy
    z_3d = z_valid

    # Get colors from image (convert BGR to RGB)
    image_rgb = image[:, :, ::-1]  # BGR to RGB
    colors_flat = image_rgb.reshape(-1, 3)
    colors_valid = colors_flat[valid_mask].astype(np.float32)

    # Stack into point cloud [X, Y, Z, R, G, B]
    point_cloud = np.column_stack([
        x_3d, y_3d, z_3d,
        colors_valid[:, 0],  # R
        colors_valid[:, 1],  # G
        colors_valid[:, 2]   # B
    ]).astype(np.float32)

    projection_time_ms = (time.perf_counter() - start_time) * 1000

    return point_cloud, projection_time_ms


def project_to_3d_with_mask(
    depth_map: np.ndarray,
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 100.0
) -> Tuple[np.ndarray, float]:
    """
    Project to 3D with optional mask for region selection.

    Args:
        depth_map: Depth values (H, W) float32
        image: BGR image (H, W, 3) uint8
        intrinsics: Camera intrinsic parameters
        mask: Optional binary mask (H, W) - True for pixels to include
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold

    Returns:
        Tuple of (point cloud (N, 6) float32, projection time in ms)
    """
    start_time = time.perf_counter()

    height, width = depth_map.shape

    # Create coordinate grids
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    # Create validity mask
    valid_mask = (
        (depth_map >= min_depth) &
        (depth_map <= max_depth) &
        np.isfinite(depth_map)
    )

    if mask is not None:
        valid_mask = valid_mask & mask

    # Extract valid coordinates and depths
    u_valid = u_grid[valid_mask]
    v_valid = v_grid[valid_mask]
    z_valid = depth_map[valid_mask]

    # Project to 3D
    x_3d = (u_valid - intrinsics.cx) * z_valid / intrinsics.fx
    y_3d = (v_valid - intrinsics.cy) * z_valid / intrinsics.fy

    # Get colors (BGR to RGB)
    colors = image[:, :, ::-1][valid_mask].astype(np.float32)

    # Build point cloud
    point_cloud = np.column_stack([
        x_3d, y_3d, z_valid,
        colors[:, 0], colors[:, 1], colors[:, 2]
    ]).astype(np.float32)

    projection_time_ms = (time.perf_counter() - start_time) * 1000

    return point_cloud, projection_time_ms


def subsample_point_cloud(
    point_cloud: np.ndarray,
    target_points: int
) -> np.ndarray:
    """
    Randomly subsample point cloud to target size.

    Args:
        point_cloud: Input point cloud (N, 6)
        target_points: Desired number of points

    Returns:
        Subsampled point cloud
    """
    n_points = point_cloud.shape[0]

    if n_points <= target_points:
        return point_cloud

    indices = np.random.choice(n_points, target_points, replace=False)
    return point_cloud[indices]


def point_cloud_to_bytes(point_cloud: np.ndarray) -> bytes:
    """
    Serialize point cloud to bytes for gRPC transmission.

    Args:
        point_cloud: Point cloud (N, 6) float32

    Returns:
        Raw bytes of the numpy array
    """
    return point_cloud.astype(np.float32).tobytes()


def bytes_to_point_cloud(data: bytes, num_points: int) -> np.ndarray:
    """
    Deserialize point cloud from bytes.

    Args:
        data: Raw bytes
        num_points: Expected number of points

    Returns:
        Point cloud (N, 6) float32
    """
    array = np.frombuffer(data, dtype=np.float32)
    return array.reshape((num_points, 6))
