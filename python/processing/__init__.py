"""
Iris3D Point Cloud Processing Pipeline

Provides advanced point cloud processing using Open3D:
- Statistical outlier removal
- Normal estimation
- Poisson surface reconstruction (meshing)
- Point cloud registration/alignment
"""

from .projection import CameraIntrinsics, project_to_3d, project_to_3d_vectorized
from .filtering import StatisticalOutlierRemoval, RadiusOutlierRemoval
from .normals import NormalEstimator
from .meshing import PoissonMesher, BallPivotMesher

__all__ = [
    "CameraIntrinsics",
    "project_to_3d",
    "project_to_3d_vectorized",
    "StatisticalOutlierRemoval",
    "RadiusOutlierRemoval",
    "NormalEstimator",
    "PoissonMesher",
    "BallPivotMesher",
]
