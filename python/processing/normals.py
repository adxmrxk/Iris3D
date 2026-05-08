"""
Normal Estimation

Compute surface normals for point clouds using local neighborhood analysis.
Essential for mesh reconstruction and rendering.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time
import numpy as np


@dataclass
class NormalEstimationResult:
    """Result from normal estimation."""

    # Point cloud with normals (N, 9) [X, Y, Z, R, G, B, Nx, Ny, Nz]
    points_with_normals: np.ndarray

    # Just the normals (N, 3)
    normals: np.ndarray

    # Processing time in milliseconds
    processing_time_ms: float


class NormalEstimator:
    """
    Estimate surface normals for a point cloud.

    Uses PCA on local neighborhoods to compute normal vectors.
    Can optionally orient normals towards a camera position.

    Example:
        estimator = NormalEstimator(radius=0.1, max_nn=30)
        result = estimator.estimate(points)
    """

    def __init__(
        self,
        radius: float = 0.1,
        max_nn: int = 30,
        orient_to_camera: bool = True,
        camera_position: Optional[np.ndarray] = None
    ):
        """
        Initialize normal estimator.

        Args:
            radius: Search radius for neighborhood
            max_nn: Maximum number of neighbors
            orient_to_camera: Orient normals towards camera
            camera_position: Camera position for orientation (default: origin)
        """
        self.radius = radius
        self.max_nn = max_nn
        self.orient_to_camera = orient_to_camera
        self.camera_position = camera_position if camera_position is not None else np.array([0, 0, 0])

    def estimate(self, points: np.ndarray) -> NormalEstimationResult:
        """
        Estimate normals for point cloud.

        Args:
            points: Point cloud (N, 6) [X, Y, Z, R, G, B]

        Returns:
            NormalEstimationResult with normals
        """
        start = time.perf_counter()

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            if points.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.radius,
                    max_nn=self.max_nn
                )
            )

            # Orient normals
            if self.orient_to_camera:
                pcd.orient_normals_towards_camera_location(
                    camera_location=self.camera_position
                )

            normals = np.asarray(pcd.normals).astype(np.float32)

        except ImportError:
            normals = self._estimate_numpy(points[:, :3])

        processing_time = (time.perf_counter() - start) * 1000

        # Combine points with normals
        if points.shape[1] >= 6:
            points_with_normals = np.column_stack([points[:, :6], normals])
        else:
            # Pad with zero colors
            colors = np.zeros((len(points), 3), dtype=np.float32)
            points_with_normals = np.column_stack([points[:, :3], colors, normals])

        return NormalEstimationResult(
            points_with_normals=points_with_normals.astype(np.float32),
            normals=normals,
            processing_time_ms=processing_time
        )

    def _estimate_numpy(self, xyz: np.ndarray) -> np.ndarray:
        """Fallback numpy implementation using PCA."""
        from scipy.spatial import KDTree

        tree = KDTree(xyz)
        normals = np.zeros_like(xyz)

        for i, point in enumerate(xyz):
            # Find neighbors
            indices = tree.query_ball_point(point, r=self.radius)
            if len(indices) < 3:
                # Need at least 3 points for PCA
                indices = tree.query(point, k=min(self.max_nn, len(xyz)))[1]

            neighbors = xyz[indices]

            # PCA
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.dot(centered.T, centered) / len(neighbors)

            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Normal is eigenvector with smallest eigenvalue
            normal = eigenvectors[:, 0]

            # Orient towards camera
            if self.orient_to_camera:
                view_dir = self.camera_position - point
                if np.dot(normal, view_dir) < 0:
                    normal = -normal

            normals[i] = normal

        return normals.astype(np.float32)


def flip_normals_towards_viewpoint(
    points: np.ndarray,
    normals: np.ndarray,
    viewpoint: np.ndarray = None
) -> np.ndarray:
    """
    Flip normals to face towards a viewpoint.

    Args:
        points: Point positions (N, 3)
        normals: Normal vectors (N, 3)
        viewpoint: Viewpoint position (default: origin)

    Returns:
        Corrected normals
    """
    if viewpoint is None:
        viewpoint = np.array([0, 0, 0], dtype=np.float32)

    corrected = normals.copy()

    # View direction from each point to viewpoint
    view_dirs = viewpoint - points

    # Check if normal points towards viewpoint
    dot_products = np.sum(normals * view_dirs, axis=1)

    # Flip if pointing away
    flip_mask = dot_products < 0
    corrected[flip_mask] = -corrected[flip_mask]

    return corrected
