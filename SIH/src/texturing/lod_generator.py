import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import trimesh

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False


class LODGenerator:
    def __init__(
        self,
        high_faces: int = 500000,
        medium_faces: int = 120000,
        low_faces: int = 35000,
    ):
        self.tiers = {
            "high": high_faces,
            "medium": medium_faces,
            "low": low_faces,
        }

    def decimate(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> trimesh.Trimesh:
        current_faces = len(mesh.faces)
        if current_faces <= target_faces or target_faces <= 0:
            return mesh.copy()

        if HAS_PYMESHLAB and temp_dir is not None:
            t_dir = Path(temp_dir).resolve()
            t_dir.mkdir(parents=True, exist_ok=True)
            tmp_in = t_dir / f"_decim_in_{target_faces}.obj"
            tmp_out = t_dir / f"_decim_out_{target_faces}.obj"
            orig_cwd = os.getcwd()
            try:
                mesh.export(str(tmp_in), file_type="obj")
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(tmp_in))
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=int(target_faces),
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True,
                )
                ms.save_current_mesh(str(tmp_out))
                reloaded = trimesh.load(str(tmp_out), force="mesh")
                if isinstance(reloaded, trimesh.Trimesh) and len(reloaded.faces) > 0:
                    return reloaded
            except Exception:
                pass
            finally:
                os.chdir(orig_cwd)
                if tmp_in.exists():
                    tmp_in.unlink()
                if tmp_out.exists():
                    tmp_out.unlink()

        return mesh.copy()

    def generate_tier(
        self,
        mesh: trimesh.Trimesh,
        tier: str,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        target_faces = self.tiers.get(tier.lower(), self.tiers["medium"])
        orig_faces = len(mesh.faces)
        decimated_mesh = self.decimate(mesh, target_faces, temp_dir=temp_dir)
        stats = {
            "tier": tier.lower(),
            "target_faces": target_faces,
            "input_faces": orig_faces,
            "output_faces": len(decimated_mesh.faces),
            "output_vertices": len(decimated_mesh.vertices),
            "decimation_ratio": round(len(decimated_mesh.faces) / max(1, orig_faces), 4),
        }
        return decimated_mesh, stats
