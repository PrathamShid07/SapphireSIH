import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import trimesh

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


class MeshCleaner:
    def __init__(
        self,
        hole_max_size: int = 50,
        min_component_faces: int = 10,
        smooth_normals: bool = True,
    ):
        self.hole_max_size = hole_max_size
        self.min_component_faces = min_component_faces
        self.smooth_normals = smooth_normals

    def clean(
        self,
        mesh: trimesh.Trimesh,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        orig_verts = len(mesh.vertices)
        orig_faces = len(mesh.faces)

        mesh = mesh.copy()
        mesh.remove_infinite_values()

        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())

        mesh.remove_unreferenced_vertices()

        if self.min_component_faces > 0 and len(mesh.faces) > 0:
            components = mesh.split(only_watertight=False)
            if isinstance(components, list) and len(components) > 1:
                valid_comps = [c for c in components if len(c.faces) >= self.min_component_faces]
                if valid_comps:
                    mesh = trimesh.util.concatenate(valid_comps)
                    mesh.remove_unreferenced_vertices()

        if HAS_PYMESHLAB and temp_dir is not None and len(mesh.faces) > 0:
            t_dir = Path(temp_dir).resolve()
            t_dir.mkdir(parents=True, exist_ok=True)
            tmp_in = t_dir / "_temp_clean_in.obj"
            tmp_out = t_dir / "_temp_clean_out.obj"
            orig_cwd = os.getcwd()
            try:
                mesh.export(str(tmp_in), file_type="obj")
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(tmp_in))
                ms.meshing_remove_duplicate_faces()
                ms.meshing_remove_duplicate_vertices()
                if self.hole_max_size > 0:
                    ms.meshing_close_holes(maxholesize=int(self.hole_max_size))
                if self.smooth_normals:
                    ms.compute_normal_per_vertex()
                    ms.compute_normal_per_face()
                    ms.apply_normal_normalization_per_vertex()
                ms.save_current_mesh(str(tmp_out))
                reloaded = trimesh.load(str(tmp_out), force="mesh")
                if isinstance(reloaded, trimesh.Trimesh) and len(reloaded.faces) > 0:
                    mesh = reloaded
            except Exception:
                mesh.fill_holes()
                mesh.fix_normals()
            finally:
                os.chdir(orig_cwd)
                if tmp_in.exists():
                    tmp_in.unlink()
                if tmp_out.exists():
                    tmp_out.unlink()
        else:
            mesh.fill_holes()
            mesh.fix_normals()

        stats = {
            "original_vertices": orig_verts,
            "cleaned_vertices": len(mesh.vertices),
            "original_faces": orig_faces,
            "cleaned_faces": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight),
            "euler_number": int(mesh.euler_number),
        }
        return mesh, stats
