from typing import Any, Dict, Tuple
import numpy as np
import trimesh


class UVProcessor:
    def __init__(self, method: str = "box"):
        self.method = method

    def compute_uv(
        self,
        mesh: trimesh.Trimesh,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        mesh = mesh.copy()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        n_verts = len(verts)

        if n_verts == 0:
            return mesh, {"uv_count": 0, "method": self.method}

        # TextureVisuals (assigned below when there is no existing UV) has no
        # vertex_colors of its own, so the mesh's original per-vertex colors
        # would otherwise be silently lost. Stash them so TextureOptimizer can
        # still bake real colors instead of falling back to a flat fill.
        if (
            isinstance(mesh.visual, trimesh.visual.color.ColorVisuals)
            and mesh.visual.vertex_colors is not None
            and len(mesh.visual.vertex_colors) == n_verts
        ):
            mesh.vertex_attributes["orig_vertex_colors"] = np.asarray(mesh.visual.vertex_colors)

        existing_uv = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            if len(mesh.visual.uv) == n_verts:
                existing_uv = np.asarray(mesh.visual.uv, dtype=np.float64)

        if existing_uv is not None:
            uv = np.clip(existing_uv, 0.0, 1.0)
            mesh.visual.uv = uv
            return mesh, {
                "uv_count": len(uv),
                "method": "preserved_existing",
                "uv_bounds": [float(np.min(uv)), float(np.max(uv))],
            }

        bounds = mesh.bounds
        min_b, max_b = bounds[0], bounds[1]
        extent = max_b - min_b
        extent[extent < 1e-6] = 1.0

        if self.method == "planar":
            u = (verts[:, 0] - min_b[0]) / extent[0]
            v = (verts[:, 1] - min_b[1]) / extent[1]
            uv = np.column_stack([u, v])
        else:
            normals = mesh.vertex_normals
            abs_n = np.abs(normals)
            dominant_axis = np.argmax(abs_n, axis=1)

            uv = np.zeros((n_verts, 2), dtype=np.float64)

            mask_z = (dominant_axis == 2)
            if np.any(mask_z):
                u_z = (verts[mask_z, 0] - min_b[0]) / extent[0] * 0.48 + 0.01
                v_z = (verts[mask_z, 1] - min_b[1]) / extent[1] * 0.48 + 0.01
                uv[mask_z] = np.column_stack([u_z, v_z])

            mask_x = (dominant_axis == 0)
            if np.any(mask_x):
                u_x = (verts[mask_x, 1] - min_b[1]) / extent[1] * 0.48 + 0.51
                v_x = (verts[mask_x, 2] - min_b[2]) / extent[2] * 0.48 + 0.01
                uv[mask_x] = np.column_stack([u_x, v_x])

            mask_y = (dominant_axis == 1)
            if np.any(mask_y):
                u_y = (verts[mask_y, 0] - min_b[0]) / extent[0] * 0.48 + 0.01
                v_y = (verts[mask_y, 2] - min_b[2]) / extent[2] * 0.48 + 0.51
                uv[mask_y] = np.column_stack([u_y, v_y])

        uv = np.clip(uv, 0.0, 1.0)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)

        return mesh, {
            "uv_count": len(uv),
            "method": self.method,
            "uv_bounds": [float(np.min(uv)), float(np.max(uv))],
        }
