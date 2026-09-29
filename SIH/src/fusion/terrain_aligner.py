from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
import trimesh


class TerrainElevationModel:
    def __init__(
        self,
        mesh: Optional[trimesh.Trimesh] = None,
        points: Optional[np.ndarray] = None,
        default_ground_z: float = 0.0,
    ):
        self.default_ground_z = default_ground_z
        self.tree: Optional[cKDTree] = None
        self.points_3d: Optional[np.ndarray] = None

        pts = None
        if mesh is not None and hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            pts = np.asarray(mesh.vertices, dtype=np.float64)
        elif points is not None and len(points) > 0:
            pts = np.asarray(points, dtype=np.float64)

        if pts is not None and len(pts) > 0:
            self.points_3d = pts
            self.tree = cKDTree(pts[:, :2])

    def query_elevation_and_normal(
        self,
        x: float,
        y: float,
        search_radius: float = 3.0,
        k_nearest: int = 24,
    ) -> Tuple[float, np.ndarray]:
        up_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if self.tree is None or self.points_3d is None:
            return self.default_ground_z, up_normal

        indices = self.tree.query_ball_point([x, y], r=search_radius)
        if len(indices) < 3:
            dists, idxs = self.tree.query([x, y], k=min(k_nearest, len(self.points_3d)))
            if np.isscalar(idxs):
                idxs = [idxs]
            indices = [i for i in idxs if i < len(self.points_3d)]

        if len(indices) < 3:
            return self.default_ground_z, up_normal

        local_pts = self.points_3d[indices]
        z_elev = float(np.median(local_pts[:, 2]))

        try:
            centroid = np.mean(local_pts, axis=0)
            shifted = local_pts - centroid
            cov = np.dot(shifted.T, shifted)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
            if normal[2] < 0:
                normal = -normal
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                normal = normal / norm_val
            else:
                normal = up_normal
        except Exception:
            normal = up_normal

        return z_elev, normal


class TerrainAligner:
    def __init__(
        self,
        terrain_model: Optional[TerrainElevationModel] = None,
        max_tilt_deg: float = 25.0,
    ):
        self.terrain_model = terrain_model or TerrainElevationModel()
        self.max_tilt_deg = max_tilt_deg

    def compute_alignment_matrix(
        self,
        yaw_deg: float,
        terrain_normal: np.ndarray,
        is_tree: bool = False,
    ) -> np.ndarray:
        yaw_rad = np.radians(yaw_deg)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        r_yaw = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0],
        ], dtype=np.float64)

        if is_tree:
            return r_yaw

        normal = np.asarray(terrain_normal, dtype=np.float64).flatten()
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-6:
            return r_yaw
        normal = normal / norm_val
        if normal[2] < 0:
            normal = -normal

        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cos_tilt = float(np.clip(np.dot(normal, up), -1.0, 1.0))
        tilt_angle = float(np.arccos(cos_tilt))

        max_tilt_rad = np.radians(self.max_tilt_deg)
        if tilt_angle > max_tilt_rad:
            horiz = normal[:2]
            h_norm = np.linalg.norm(horiz)
            if h_norm > 1e-6:
                sin_max = np.sin(max_tilt_rad)
                cos_max = np.cos(max_tilt_rad)
                normal = np.array([
                    horiz[0] / h_norm * sin_max,
                    horiz[1] / h_norm * sin_max,
                    cos_max,
                ], dtype=np.float64)
                tilt_angle = max_tilt_rad
            else:
                return r_yaw

        if tilt_angle < 1e-4:
            return r_yaw

        axis = np.cross(up, normal)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return r_yaw
        axis = axis / axis_norm

        kx, ky, kz = axis
        k_skew = np.array([
            [0.0, -kz,  ky],
            [kz,  0.0, -kx],
            [-ky,  kx,  0.0],
        ], dtype=np.float64)

        r_slope = (
            np.eye(3, dtype=np.float64)
            + np.sin(tilt_angle) * k_skew
            + (1.0 - np.cos(tilt_angle)) * np.dot(k_skew, k_skew)
        )

        return np.dot(r_slope, r_yaw)

    def compute_asset_scale(
        self,
        target_dimensions: dict,
        asset_dimensions: dict,
        class_name: str = "car",
    ) -> np.ndarray:
        t_l = float(target_dimensions.get("length", 4.4))
        t_w = float(target_dimensions.get("width", 1.8))
        t_h = float(target_dimensions.get("height", 1.4))

        a_l = max(0.01, float(asset_dimensions.get("length", t_l)))
        a_w = max(0.01, float(asset_dimensions.get("width", t_w)))
        a_h = max(0.01, float(asset_dimensions.get("height", t_h)))

        if class_name.lower() == "tree":
            scale_z = t_h / a_h
            canopy_s = (t_l + t_w) / (a_l + a_w)
            return np.array([canopy_s, canopy_s, scale_z], dtype=np.float64)

        scale_x = t_l / a_l
        scale_y = t_w / a_w
        scale_z = t_h / a_h
        return np.array([scale_x, scale_y, scale_z], dtype=np.float64)
