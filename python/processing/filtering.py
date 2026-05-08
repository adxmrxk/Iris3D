"""
Point Cloud Filtering

Statistical and radius-based outlier removal for cleaning noisy point clouds.
Uses Open3D for efficient spatial operations.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time
import numpy as np


@dataclass
class FilteringResult:
    """Result from point cloud filtering."""

    # Filtered point cloud (N, 6) [X, Y, Z, R, G, B]
    points: np.ndarray

    # Indices of points that were kept
    inlier_indices: np.ndarray

    # Number of points removed
    num_removed: int

    # Processing time in milliseconds
    processing_time_ms: float


class StatisticalOutlierRemoval:
    """
    Statistical outlier removal filter.

    Removes points that are further away from their neighbors than expected.
    For each point, computes the mean distance to its k nearest neighbors.
    Points with mean distance outside the global mean ± std_ratio * std are removed.

    Example:
        filter = StatisticalOutlierRemoval(nb_neighbors=20, std_ratio=2.0)
        result = filter.filter(points)
    """

    def __init__(
        self,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ):
        """
        Initialize statistical outlier removal.

        Args:
            nb_neighbors: Number of neighbors to analyze
            std_ratio: Standard deviation multiplier threshold
        """
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def filter(self, points: np.ndarray) -> FilteringResult:
        """
        Filter outliers from point cloud.

        Args:
            points: Point cloud (N, 6) [X, Y, Z, R, G, B]

        Returns:
            FilteringResult with filtered points
        """
        start = time.perf_counter()

        try:
            import open3d as o3d

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            if points.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

            # Apply statistical outlier removal
            _, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors,
                std_ratio=self.std_ratio
            )

            inlier_indices = np.asarray(inlier_indices)
            filtered_points = points[inlier_indices]

        except ImportError:
            # Fallback to numpy-based implementation
            filtered_points, inlier_indices = self._filter_numpy(points)

        processing_time = (time.perf_counter() - start) * 1000
        num_removed = len(points) - len(filtered_points)

        return FilteringResult(
            points=filtered_points,
            inlier_indices=inlier_indices,
            num_removed=num_removed,
            processing_time_ms=processing_time
        )

    def _filter_numpy(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback numpy implementation."""
        from scipy.spatial import KDTree

        xyz = points[:, :3]
        tree = KDTree(xyz)

        # Query k+1 neighbors (includes self)
        distances, _ = tree.query(xyz, k=self.nb_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self

        global_mean = mean_distances.mean()
        global_std = mean_distances.std()

        threshold = global_mean + self.std_ratio * global_std
        inlier_mask = mean_distances < threshold

        inlier_indices = np.where(inlier_mask)[0]
        return points[inlier_mask], inlier_indices


class RadiusOutlierRemoval:
    """
    Radius-based outlier removal.

    Removes points that have fewer than a threshold number of neighbors
    within a given radius.

    Example:
        filter = RadiusOutlierRemoval(radius=0.05, min_neighbors=5)
        result = filter.filter(points)
    """

    def __init__(
        self,
        radius: float = 0.05,
        min_neighbors: int = 5
    ):
        """
        Initialize radius outlier removal.

        Args:
            radius: Search radius for neighbors
            min_neighbors: Minimum number of neighbors required
        """
        self.radius = radius
        self.min_neighbors = min_neighbors

    def filter(self, points: np.ndarray) -> FilteringResult:
        """
        Filter outliers from point cloud.

        Args:
            points: Point cloud (N, 6) [X, Y, Z, R, G, B]

        Returns:
            FilteringResult with filtered points
        """
        start = time.perf_counter()

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            if points.shape[1] >= 6:
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

            _, inlier_indices = pcd.remove_radius_outlier(
                nb_points=self.min_neighbors,
                radius=self.radius
            )

            inlier_indices = np.asarray(inlier_indices)
            filtered_points = points[inlier_indices]

        except ImportError:
            # Fallback
            filtered_points, inlier_indices = self._filter_numpy(points)

        processing_time = (time.perf_counter() - start) * 1000
        num_removed = len(points) - len(filtered_points)

        return FilteringResult(
            points=filtered_points,
            inlier_indices=inlier_indices,
            num_removed=num_removed,
            processing_time_ms=processing_time
        )

    def _filter_numpy(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback numpy implementation."""
        from scipy.spatial import KDTree

        xyz = points[:, :3]
        tree = KDTree(xyz)

        # Count neighbors within radius
        neighbor_counts = tree.query_ball_point(xyz, r=self.radius, return_length=True)
        neighbor_counts = np.array(neighbor_counts)

        # Need min_neighbors + 1 because query includes self
        inlier_mask = neighbor_counts > self.min_neighbors

        inlier_indices = np.where(inlier_mask)[0]
        return points[inlier_mask], inlier_indices


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.01,
    use_centroid: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Voxel grid downsampling.

    Args:
        points: Point cloud (N, 6) [X, Y, Z, R, G, B]
        voxel_size: Voxel cell size
        use_centroid: Use centroid (True) or first point (False)

    Returns:
        Tuple of (downsampled points, processing time in ms)
    """
    start = time.perf_counter()

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        if points.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

        result_xyz = np.asarray(downsampled.points)
        result_colors = np.asarray(downsampled.colors) * 255.0 if downsampled.has_colors() else np.zeros((len(result_xyz), 3))

        result = np.column_stack([result_xyz, result_colors]).astype(np.float32)

    except ImportError:
        # Fallback implementation
        result = _voxel_downsample_numpy(points, voxel_size, use_centroid)

    processing_time = (time.perf_counter() - start) * 1000
    return result, processing_time


def _voxel_downsample_numpy(
    points: np.ndarray,
    voxel_size: float,
    use_centroid: bool
) -> np.ndarray:
    """Numpy-based voxel downsampling."""
    xyz = points[:, :3]
    inv_voxel_size = 1.0 / voxel_size

    # Compute voxel indices
    voxel_indices = np.floor(xyz * inv_voxel_size).astype(np.int64)

    # Create unique keys
    # Use a large multiplier to create unique hash
    max_idx = voxel_indices.max() - voxel_indices.min() + 1
    keys = (
        (voxel_indices[:, 0] - voxel_indices[:, 0].min()) * max_idx * max_idx +
        (voxel_indices[:, 1] - voxel_indices[:, 1].min()) * max_idx +
        (voxel_indices[:, 2] - voxel_indices[:, 2].min())
    )

    # Get unique voxels
    unique_keys, inverse_indices, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )

    if use_centroid:
        # Compute centroids
        result = np.zeros((len(unique_keys), points.shape[1]), dtype=np.float32)
        np.add.at(result, inverse_indices, points)
        result /= counts[:, np.newaxis]
    else:
        # Take first point per voxel
        first_indices = np.zeros(len(unique_keys), dtype=np.int64)
        seen = np.zeros(len(unique_keys), dtype=bool)
        for i, key_idx in enumerate(inverse_indices):
            if not seen[key_idx]:
                first_indices[key_idx] = i
                seen[key_idx] = True
        result = points[first_indices]

    return result
