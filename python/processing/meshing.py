"""
Mesh Generation

Convert point clouds to triangle meshes using:
- Poisson surface reconstruction
- Ball pivoting algorithm
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time
import numpy as np


@dataclass
class MeshResult:
    """Result from mesh generation."""

    # Vertex positions (V, 3)
    vertices: np.ndarray

    # Triangle indices (F, 3)
    triangles: np.ndarray

    # Vertex colors (V, 3) optional
    vertex_colors: Optional[np.ndarray]

    # Vertex normals (V, 3) optional
    vertex_normals: Optional[np.ndarray]

    # Number of vertices
    num_vertices: int

    # Number of triangles
    num_triangles: int

    # Processing time in milliseconds
    processing_time_ms: float


class PoissonMesher:
    """
    Poisson surface reconstruction.

    Creates watertight mesh from point cloud with normals.
    Requires normals to be estimated first.

    Example:
        mesher = PoissonMesher(depth=9)
        mesh = mesher.reconstruct(points_with_normals)
    """

    def __init__(
        self,
        depth: int = 9,
        scale: float = 1.1,
        linear_fit: bool = False,
        density_threshold: float = 0.01
    ):
        """
        Initialize Poisson mesher.

        Args:
            depth: Octree depth (higher = more detail, 8-12 typical)
            scale: Scale factor for reconstruction
            linear_fit: Use linear interpolation
            density_threshold: Minimum vertex density (0-1) for trimming
        """
        self.depth = depth
        self.scale = scale
        self.linear_fit = linear_fit
        self.density_threshold = density_threshold

    def reconstruct(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> MeshResult:
        """
        Reconstruct mesh from point cloud.

        Args:
            points: Point cloud (N, 6+) with optional normals at indices 6-8
            normals: Optional separate normals array (N, 3)

        Returns:
            MeshResult with mesh data
        """
        start = time.perf_counter()

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            # Colors
            if points.shape[1] >= 6:
                colors = points[:, 3:6]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # Normals
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            elif points.shape[1] >= 9:
                pcd.normals = o3d.utility.Vector3dVector(points[:, 6:9])
            else:
                # Estimate normals if not provided
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                pcd.orient_normals_towards_camera_location()

            # Poisson reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=self.depth,
                scale=self.scale,
                linear_fit=self.linear_fit
            )

            # Trim low-density vertices
            if self.density_threshold > 0:
                densities = np.asarray(densities)
                density_threshold = np.quantile(densities, self.density_threshold)
                vertices_to_remove = densities < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

            vertices = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles).astype(np.int32)

            vertex_colors = None
            if mesh.has_vertex_colors():
                vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

            vertex_normals = None
            if mesh.has_vertex_normals():
                vertex_normals = np.asarray(mesh.vertex_normals).astype(np.float32)

        except ImportError:
            raise ImportError(
                "Open3D required for mesh generation. "
                "Install with: pip install open3d"
            )

        processing_time = (time.perf_counter() - start) * 1000

        return MeshResult(
            vertices=vertices,
            triangles=triangles,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals,
            num_vertices=len(vertices),
            num_triangles=len(triangles),
            processing_time_ms=processing_time
        )


class BallPivotMesher:
    """
    Ball pivoting algorithm for mesh reconstruction.

    Creates mesh by rolling a ball over the point cloud surface.
    Works well for clean, uniformly sampled point clouds.

    Example:
        mesher = BallPivotMesher(radii=[0.005, 0.01, 0.02])
        mesh = mesher.reconstruct(points_with_normals)
    """

    def __init__(
        self,
        radii: Optional[list] = None
    ):
        """
        Initialize ball pivot mesher.

        Args:
            radii: List of ball radii to try (auto-computed if None)
        """
        self.radii = radii

    def reconstruct(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> MeshResult:
        """
        Reconstruct mesh using ball pivoting.

        Args:
            points: Point cloud (N, 6+) with optional normals
            normals: Optional separate normals array (N, 3)

        Returns:
            MeshResult with mesh data
        """
        start = time.perf_counter()

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            if points.shape[1] >= 6:
                colors = points[:, 3:6]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)

            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)
            elif points.shape[1] >= 9:
                pcd.normals = o3d.utility.Vector3dVector(points[:, 6:9])
            else:
                pcd.estimate_normals()
                pcd.orient_normals_towards_camera_location()

            # Compute radii if not provided
            radii = self.radii
            if radii is None:
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 4.0]

            # Ball pivoting
            radii_vector = o3d.utility.DoubleVector(radii)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii_vector
            )

            vertices = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles).astype(np.int32)

            vertex_colors = None
            if mesh.has_vertex_colors():
                vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)

            vertex_normals = None
            if mesh.has_vertex_normals():
                vertex_normals = np.asarray(mesh.vertex_normals).astype(np.float32)

        except ImportError:
            raise ImportError(
                "Open3D required for mesh generation. "
                "Install with: pip install open3d"
            )

        processing_time = (time.perf_counter() - start) * 1000

        return MeshResult(
            vertices=vertices,
            triangles=triangles,
            vertex_colors=vertex_colors,
            vertex_normals=vertex_normals,
            num_vertices=len(vertices),
            num_triangles=len(triangles),
            processing_time_ms=processing_time
        )


def export_mesh_ply(
    mesh_result: MeshResult,
    output_path: str,
    binary: bool = True
) -> None:
    """
    Export mesh to PLY format.

    Args:
        mesh_result: MeshResult to export
        output_path: Output file path
        binary: Use binary format (smaller)
    """
    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_result.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_result.triangles)

        if mesh_result.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                mesh_result.vertex_colors / 255.0
            )

        if mesh_result.vertex_normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(
                mesh_result.vertex_normals
            )

        o3d.io.write_triangle_mesh(
            output_path,
            mesh,
            write_ascii=not binary
        )

    except ImportError:
        raise ImportError("Open3D required for mesh export")


def export_mesh_obj(
    mesh_result: MeshResult,
    output_path: str
) -> None:
    """
    Export mesh to OBJ format.

    Args:
        mesh_result: MeshResult to export
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        f.write("# Iris3D mesh export\n")

        # Vertices
        for i, v in enumerate(mesh_result.vertices):
            if mesh_result.vertex_colors is not None:
                c = mesh_result.vertex_colors[i] / 255.0
                f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Normals
        if mesh_result.vertex_normals is not None:
            for n in mesh_result.vertex_normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # Faces (OBJ uses 1-indexed)
        for tri in mesh_result.triangles:
            if mesh_result.vertex_normals is not None:
                f.write(f"f {tri[0]+1}//{tri[0]+1} {tri[1]+1}//{tri[1]+1} {tri[2]+1}//{tri[2]+1}\n")
            else:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
