from typing import Any, Dict, List, Tuple
import numpy as np
from .placement_record import PlacedAsset


def compute_oriented_box_corners(
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: np.ndarray,
    base_dimensions: Dict[str, float],
) -> np.ndarray:
    l_val = float(base_dimensions.get("length", 4.4)) * scale[0]
    w_val = float(base_dimensions.get("width", 1.8)) * scale[1]
    h_val = float(base_dimensions.get("height", 1.4)) * scale[2]

    hx, hy = l_val / 2.0, w_val / 2.0
    local_corners = np.array([
        [-hx, -hy, 0.0],
        [ hx, -hy, 0.0],
        [ hx,  hy, 0.0],
        [-hx,  hy, 0.0],
        [-hx, -hy, h_val],
        [ hx, -hy, h_val],
        [ hx,  hy, h_val],
        [-hx,  hy, h_val],
    ], dtype=np.float64)

    world_corners = np.dot(local_corners, rotation_matrix.T) + position
    return world_corners


def compute_aabb(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    return min_corner, max_corner


def compute_3d_aabb_iou(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
) -> float:
    intersect_min = np.maximum(min1, min2)
    intersect_max = np.minimum(max1, max2)
    intersect_dims = np.maximum(0.0, intersect_max - intersect_min)
    intersect_vol = float(np.prod(intersect_dims))
    if intersect_vol <= 0.0:
        return 0.0

    vol1 = float(np.prod(np.maximum(0.0, max1 - min1)))
    vol2 = float(np.prod(np.maximum(0.0, max2 - min2)))
    union_vol = vol1 + vol2 - intersect_vol
    if union_vol <= 0.0:
        return 0.0
    return intersect_vol / union_vol


class CollisionChecker:
    def __init__(
        self,
        float_tolerance_m: float = 0.25,
        penetration_tolerance_m: float = 0.15,
        collision_iou_threshold: float = 0.1,
        snap_to_ground: bool = True,
    ):
        self.float_tolerance_m = float_tolerance_m
        self.penetration_tolerance_m = penetration_tolerance_m
        self.collision_iou_threshold = collision_iou_threshold
        self.snap_to_ground = snap_to_ground

    def check_ground_contact(
        self,
        asset: PlacedAsset,
        terrain_z: float,
    ) -> None:
        current_z = asset.transform.position[2]
        delta_z = current_z - terrain_z

        if delta_z > self.float_tolerance_m:
            asset.validation.is_floating = True
            asset.validation.float_distance_m = float(delta_z)
            if self.snap_to_ground:
                asset.transform.position[2] = float(terrain_z)
                t_mat = np.array(asset.transform.matrix, dtype=np.float64)
                t_mat[2, 3] = float(terrain_z)
                asset.transform.matrix = [[float(v) for v in row] for row in t_mat]
                asset.ground_contact_3d[2] = float(terrain_z)
                asset.validation.was_adjusted = True
                asset.validation.adjustment_details += f"Snapped floating object down by {delta_z:.3f}m. "
        elif delta_z < -self.penetration_tolerance_m:
            asset.validation.terrain_penetration = True
            asset.validation.penetration_depth_m = float(-delta_z)
            if self.snap_to_ground:
                asset.transform.position[2] = float(terrain_z)
                t_mat = np.array(asset.transform.matrix, dtype=np.float64)
                t_mat[2, 3] = float(terrain_z)
                asset.transform.matrix = [[float(v) for v in row] for row in t_mat]
                asset.ground_contact_3d[2] = float(terrain_z)
                asset.validation.was_adjusted = True
                asset.validation.adjustment_details += f"Raised sunken object up by {-delta_z:.3f}m. "

    def check_and_resolve_collisions(
        self,
        assets: List[PlacedAsset],
    ) -> Tuple[List[PlacedAsset], int]:
        n = len(assets)
        if n <= 1:
            return assets, 0

        aabbs = []
        for a in assets:
            pos = np.array(a.transform.position, dtype=np.float64)
            rot = np.array(a.transform.matrix, dtype=np.float64)[:3, :3]
            s = np.array(a.transform.scale, dtype=np.float64)
            dims = a.bounding_box_3d.get("dimensions", {"length": 4.4, "width": 1.8, "height": 1.4})
            corners = compute_oriented_box_corners(pos, rot, s, dims)
            min_c, max_c = compute_aabb(corners)
            aabbs.append((min_c, max_c))

        resolved_count = 0
        suppressed_indices = set()

        for i in range(n):
            for j in range(i + 1, n):
                iou = compute_3d_aabb_iou(aabbs[i][0], aabbs[i][1], aabbs[j][0], aabbs[j][1])
                if iou >= self.collision_iou_threshold:
                    assets[i].validation.has_collision = True
                    assets[i].validation.colliding_tracks.append(assets[j].track_id)
                    assets[j].validation.has_collision = True
                    assets[j].validation.colliding_tracks.append(assets[i].track_id)

                    if assets[i].confidence >= assets[j].confidence:
                        suppressed_indices.add(j)
                        assets[j].validation.adjustment_details += f"Suppressed due to collision with track {assets[i].track_id} (IoU={iou:.3f}). "
                    else:
                        suppressed_indices.add(i)
                        assets[i].validation.adjustment_details += f"Suppressed due to collision with track {assets[j].track_id} (IoU={iou:.3f}). "
                    resolved_count += 1

        filtered_assets = [a for idx, a in enumerate(assets) if idx not in suppressed_indices]
        return filtered_assets, resolved_count
