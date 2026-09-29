from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import trimesh
from .placement_record import FusedSceneManifest, PlacedAsset


class SceneComposer:
    def __init__(self):
        self._asset_cache: Dict[str, trimesh.Trimesh] = {}

    def _load_asset_mesh(self, mesh_path: Union[str, Path]) -> Optional[trimesh.Trimesh]:
        p_str = str(mesh_path)
        if p_str in self._asset_cache:
            return self._asset_cache[p_str].copy()

        p = Path(mesh_path)
        if not p.exists():
            return None

        try:
            loaded = trimesh.load(str(p), force="mesh")
            if isinstance(loaded, trimesh.Trimesh):
                self._asset_cache[p_str] = loaded
                return loaded.copy()
            elif isinstance(loaded, trimesh.Scene):
                mesh = loaded.to_geometry()
                if isinstance(mesh, trimesh.Trimesh):
                    self._asset_cache[p_str] = mesh
                    return mesh.copy()
        except Exception:
            return None
        return None

    def compose_scene(
        self,
        placed_assets: List[PlacedAsset],
        base_mesh: Optional[trimesh.Trimesh] = None,
        base_mesh_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[trimesh.Scene, List[trimesh.Trimesh]]:
        scene = trimesh.Scene()
        transformed_meshes: List[trimesh.Trimesh] = []

        loaded_base = base_mesh
        if loaded_base is None and base_mesh_path is not None:
            p = Path(base_mesh_path)
            if p.exists():
                try:
                    loaded = trimesh.load(str(p), force="mesh")
                    if isinstance(loaded, trimesh.Trimesh):
                        loaded_base = loaded
                except Exception:
                    loaded_base = None

        if loaded_base is not None and hasattr(loaded_base, "vertices") and len(loaded_base.vertices) > 0:
            scene.add_geometry(loaded_base, node_name="reconstructed_environment")
            transformed_meshes.append(loaded_base)

        for asset in placed_assets:
            mesh = self._load_asset_mesh(asset.mesh_file)
            if mesh is None:
                continue

            t_mat = np.array(asset.transform.matrix, dtype=np.float64)
            mesh.apply_transform(t_mat)

            node_name = f"asset_{asset.class_name}_{asset.track_id}"
            scene.add_geometry(mesh, node_name=node_name)
            transformed_meshes.append(mesh)

        return scene, transformed_meshes

    def export_scene(
        self,
        scene: trimesh.Scene,
        transformed_meshes: List[trimesh.Trimesh],
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["glb", "obj", "ply"]

        exported_files: Dict[str, str] = {}

        for fmt in formats:
            fmt_clean = fmt.lower().lstrip(".")
            out_file = out_dir / f"fused_scene.{fmt_clean}"

            try:
                if fmt_clean == "glb":
                    glb_bytes = scene.export(file_type="glb")
                    if isinstance(glb_bytes, bytes) and len(glb_bytes) > 0:
                        with open(out_file, "wb") as f:
                            f.write(glb_bytes)
                        exported_files["glb"] = str(out_file)

                elif fmt_clean == "obj":
                    res = scene.export(file_type="obj")
                    if isinstance(res, str) and len(res) > 0:
                        with open(out_file, "w", encoding="utf-8") as f:
                            f.write(res)
                        exported_files["obj"] = str(out_file)
                    elif isinstance(res, bytes) and len(res) > 0:
                        with open(out_file, "wb") as f:
                            f.write(res)
                        exported_files["obj"] = str(out_file)
                    elif isinstance(res, dict):
                        for fname, fdata in res.items():
                            sub = out_dir / fname
                            mode = "wb" if isinstance(fdata, bytes) else "w"
                            enc = None if isinstance(fdata, bytes) else "utf-8"
                            with open(sub, mode, encoding=enc) as f:
                                f.write(fdata)
                        exported_files["obj"] = str(out_file)
                    else:
                        if transformed_meshes:
                            fused = trimesh.util.concatenate(transformed_meshes)
                            obj_data = fused.export(file_type="obj")
                            with open(out_file, "w", encoding="utf-8") as f:
                                f.write(obj_data)
                            exported_files["obj"] = str(out_file)

                elif fmt_clean == "ply":
                    res = scene.export(file_type="ply")
                    if isinstance(res, bytes) and len(res) > 0:
                        with open(out_file, "wb") as f:
                            f.write(res)
                        exported_files["ply"] = str(out_file)
                    elif isinstance(res, str) and len(res) > 0:
                        with open(out_file, "w", encoding="utf-8") as f:
                            f.write(res)
                        exported_files["ply"] = str(out_file)
                    else:
                        if transformed_meshes:
                            fused = trimesh.util.concatenate(transformed_meshes)
                            ply_data = fused.export(file_type="ply")
                            with open(out_file, "wb") as f:
                                f.write(ply_data)
                            exported_files["ply"] = str(out_file)

            except Exception:
                pass

        return exported_files
