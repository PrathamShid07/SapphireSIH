import numpy as np
from pathlib import Path
from scipy.spatial import Delaunay
from plyfile import PlyData
import trimesh

def triangulate_point_cloud(input_ply_path, output_mesh_path, max_edge_length=5.0, max_slope_ratio=None):
    ply = PlyData.read(str(input_ply_path))
    v = ply["vertex"]
    points = np.vstack([v["x"], v["y"], v["z"]]).T

    colors = None
    if "red" in v and "green" in v and "blue" in v:
        colors = np.vstack([v["red"], v["green"], v["blue"]]).T

    # This is a 2D (XY) Delaunay triangulation, i.e. a heightfield/2.5D mesher:
    # it assumes exactly one Z per (X, Y) and cannot represent overhangs or
    # vertical structures (tree canopies, walls). It is only a last-resort
    # fallback when COLMAP's own 3D-aware meshers are unavailable/fail - see
    # stage5_pipeline.run_stage5.
    tri = Delaunay(points[:, :2])
    faces = tri.simplices

    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(np.maximum(e1, e2), e3)
    valid = max_e <= max_edge_length

    if max_slope_ratio is not None:
        # A blind XY triangulation will happily connect an outlier sitting
        # meters below/above the real surface to its nearest XY neighbors,
        # producing thin near-vertical spikes whose 3D edge length can still
        # be under max_edge_length. Reject triangles whose vertical rise is
        # large relative to their horizontal footprint - real terrain/canopy
        # patches are not needle-shaped.
        z = points[:, 2]
        z_range = np.maximum(np.maximum(z[faces[:, 0]], z[faces[:, 1]]), z[faces[:, 2]]) - \
            np.minimum(np.minimum(z[faces[:, 0]], z[faces[:, 1]]), z[faces[:, 2]])
        xy = points[:, :2]
        xy0, xy1, xy2 = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        horiz_extent = np.maximum(
            np.maximum(np.linalg.norm(xy1 - xy0, axis=1), np.linalg.norm(xy2 - xy1, axis=1)),
            np.linalg.norm(xy0 - xy2, axis=1),
        )
        slope_ratio = z_range / np.maximum(horiz_extent, 1e-6)
        valid = valid & (slope_ratio <= max_slope_ratio)

    faces = faces[valid]

    mesh = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=colors, process=True)
    out_p = Path(output_mesh_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_p))
    return mesh

def process_mesh_and_export(input_mesh_path, output_dir, export_basename="dense_mesh"):
    in_mesh_path = Path(input_mesh_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_mesh_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {in_mesh_path}")

    mesh = trimesh.load(str(in_mesh_path), process=True)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) > 0:
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        else:
            mesh = trimesh.Trimesh()

    if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0:
        mesh.remove_infinite_values()
        mesh.fill_holes()

    ply_path = out_dir / f"{export_basename}.ply"
    obj_path = out_dir / f"{export_basename}.obj"
    glb_path = out_dir / f"{export_basename}.glb"

    mesh.export(str(ply_path), file_type="ply")
    mesh.export(str(obj_path), file_type="obj")
    mesh.export(str(glb_path), file_type="glb")

    return {
        "vertex_count": len(mesh.vertices) if isinstance(mesh, trimesh.Trimesh) else 0,
        "face_count": len(mesh.faces) if isinstance(mesh, trimesh.Trimesh) else 0,
        "exported_files": {
            "ply": str(ply_path),
            "obj": str(obj_path),
            "glb": str(glb_path),
        }
    }
