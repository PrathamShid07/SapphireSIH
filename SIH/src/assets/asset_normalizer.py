import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Tuple


def _axis_to_vector(axis_str: str) -> np.ndarray:
    mapping = {
        "+X": np.array([1, 0, 0], dtype=float),
        "-X": np.array([-1, 0, 0], dtype=float),
        "+Y": np.array([0, 1, 0], dtype=float),
        "-Y": np.array([0, -1, 0], dtype=float),
        "+Z": np.array([0, 0, 1], dtype=float),
        "-Z": np.array([0, 0, -1], dtype=float),
    }
    return mapping[axis_str]


def _rotation_from_two_axes(
    src_up: np.ndarray, tgt_up: np.ndarray,
    src_fwd: np.ndarray, tgt_fwd: np.ndarray,
) -> np.ndarray:
    def safe_cross(a, b):
        c = np.cross(a, b)
        n = np.linalg.norm(c)
        return c / n if n > 1e-12 else np.zeros(3)

    def rotation_between(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cross = np.cross(a, b)
        dot = np.dot(a, b)
        if np.linalg.norm(cross) < 1e-12:
            if dot > 0:
                return np.eye(3)
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis /= np.linalg.norm(axis)
            return trimesh.transformations.rotation_matrix(np.pi, axis)[:3, :3]
        angle = np.arctan2(np.linalg.norm(cross), dot)
        axis = cross / np.linalg.norm(cross)
        mat4 = trimesh.transformations.rotation_matrix(angle, axis)
        return mat4[:3, :3]

    R1 = rotation_between(src_up, tgt_up)
    rotated_fwd = R1 @ src_fwd
    rotated_fwd = rotated_fwd - np.dot(rotated_fwd, tgt_up) * tgt_up
    norm_rotated_fwd = np.linalg.norm(rotated_fwd)
    if norm_rotated_fwd > 1e-12:
        rotated_fwd /= norm_rotated_fwd
        tgt_fwd_proj = tgt_fwd - np.dot(tgt_fwd, tgt_up) * tgt_up
        norm_tgt = np.linalg.norm(tgt_fwd_proj)
        if norm_tgt > 1e-12:
            tgt_fwd_proj /= norm_tgt
            R2 = rotation_between(rotated_fwd, tgt_fwd_proj)
        else:
            R2 = np.eye(3)
    else:
        R2 = np.eye(3)

    return R2 @ R1


def normalize_mesh(
    mesh: trimesh.Trimesh,
    target_length: float,
    target_width: float,
    target_height: float,
    source_up: str = "+Z",
    source_forward: str = "+X",
) -> Tuple[trimesh.Trimesh, dict]:
    target_up_vec = np.array([0, 0, 1], dtype=float)
    target_fwd_vec = np.array([1, 0, 0], dtype=float)
    src_up_vec = _axis_to_vector(source_up)
    src_fwd_vec = _axis_to_vector(source_forward)

    R = _rotation_from_two_axes(src_up_vec, target_up_vec, src_fwd_vec, target_fwd_vec)
    mesh = mesh.copy()
    mesh.apply_transform(np.vstack([
        np.hstack([R, np.zeros((3, 1))]),
        [0, 0, 0, 1],
    ]))

    bounds = mesh.bounds
    mesh_l = bounds[1][0] - bounds[0][0]
    mesh_w = bounds[1][1] - bounds[0][1]
    mesh_h = bounds[1][2] - bounds[0][2]

    if mesh_l < 1e-6 or mesh_w < 1e-6 or mesh_h < 1e-6:
        scale_factors = {"sx": 1.0, "sy": 1.0, "sz": 1.0}
    else:
        sx = target_length / mesh_l
        sy = target_width / mesh_w
        sz = target_height / mesh_h
        scale_factors = {"sx": float(sx), "sy": float(sy), "sz": float(sz)}

    scale_mat = np.diag([scale_factors["sx"], scale_factors["sy"], scale_factors["sz"], 1.0])
    mesh.apply_transform(scale_mat)

    bounds = mesh.bounds
    cx = (bounds[0][0] + bounds[1][0]) / 2.0
    cy = (bounds[0][1] + bounds[1][1]) / 2.0
    cz_min = bounds[0][2]

    translation = np.eye(4)
    translation[:3, 3] = [-cx, -cy, -cz_min]
    mesh.apply_transform(translation)

    final_bounds = mesh.bounds
    diagnostics = {
        "normalized_length": float(final_bounds[1][0] - final_bounds[0][0]),
        "normalized_width": float(final_bounds[1][1] - final_bounds[0][1]),
        "normalized_height": float(final_bounds[1][2] - final_bounds[0][2]),
        "ground_z_min": float(final_bounds[0][2]),
        "scale_factors": scale_factors,
    }

    return mesh, diagnostics


def export_normalized_mesh(
    mesh: trimesh.Trimesh,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext == ".glb":
        export_data = mesh.export(file_type="glb")
        with open(output_path, "wb") as f:
            f.write(export_data)
    elif ext == ".obj":
        export_data = mesh.export(file_type="obj")
        if isinstance(export_data, bytes):
            with open(output_path, "wb") as f:
                f.write(export_data)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(export_data)
    else:
        export_data = mesh.export(file_type="glb")
        with open(output_path, "wb") as f:
            f.write(export_data)


def normalize_and_export(
    input_path: Optional[Path],
    output_glb: Path,
    output_obj: Path,
    target_length: float,
    target_width: float,
    target_height: float,
    source_up: str = "+Z",
    source_forward: str = "+X",
    mesh: Optional[trimesh.Trimesh] = None,
) -> dict:
    if mesh is None:
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {input_path}")
        loaded = trimesh.load(str(input_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                raise ValueError(f"No meshes found in: {input_path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = loaded

    normalized, diag = normalize_mesh(
        mesh, target_length, target_width, target_height, source_up, source_forward
    )
    export_normalized_mesh(normalized, output_glb)
    export_normalized_mesh(normalized, output_obj)
    return diag
