from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image
import trimesh

from .mesh_cleaner import MeshCleaner
from .uv_processor import UVProcessor
from .texture_optimizer import TextureOptimizer
from .lod_generator import LODGenerator


def export_mesh_tier(
    mesh: trimesh.Trimesh,
    texture: Image.Image,
    tier_dir: Path,
    tier_name: str,
) -> Dict[str, str]:
    tier_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    tex_path = tier_dir / f"{tier_name}_texture.png"
    texture.save(tex_path, format="PNG", optimize=True)
    exported["texture"] = str(tex_path)

    uv_coords = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uv_coords = mesh.visual.uv
    elif hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
        uv_proc = UVProcessor(method="planar")
        mesh, _ = uv_proc.compute_uv(mesh)
        uv_coords = mesh.visual.uv

    if uv_coords is not None:
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture)

    glb_path = tier_dir / f"{tier_name}_scene.glb"
    try:
        glb_data = mesh.export(file_type="glb")
        if isinstance(glb_data, bytes) and len(glb_data) > 0:
            with open(glb_path, "wb") as f:
                f.write(glb_data)
            exported["glb"] = str(glb_path)
    except Exception:
        pass

    obj_path = tier_dir / f"{tier_name}_scene.obj"
    try:
        res = mesh.export(file_type="obj")
        if isinstance(res, str) and len(res) > 0:
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write(res)
            exported["obj"] = str(obj_path)
        elif isinstance(res, bytes) and len(res) > 0:
            with open(obj_path, "wb") as f:
                f.write(res)
            exported["obj"] = str(obj_path)
        elif isinstance(res, dict):
            for fname, fdata in res.items():
                sub = tier_dir / fname
                mode = "wb" if isinstance(fdata, bytes) else "w"
                enc = None if isinstance(fdata, bytes) else "utf-8"
                with open(sub, mode, encoding=enc) as f:
                    f.write(fdata)
            exported["obj"] = str(obj_path)
    except Exception:
        pass

    ply_path = tier_dir / f"{tier_name}_scene.ply"
    try:
        ply_data = mesh.export(file_type="ply")
        if isinstance(ply_data, (bytes, str)) and len(ply_data) > 0:
            mode = "wb" if isinstance(ply_data, bytes) else "w"
            with open(ply_path, mode) as f:
                f.write(ply_data)
            exported["ply"] = str(ply_path)
    except Exception:
        pass

    return exported


def run_stage12(
    input_mesh_path: Union[str, Path] = "outputs/sample_drone/fusion/fused_scene.obj",
    output_dir: Union[str, Path] = "outputs/sample_drone/textured",
    quality_tier: str = "all",
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = (out_dir / "_temp").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    orig_cwd = os.getcwd()

    cfg: Dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "get"):
            cfg = config.get("texturing_cleanup", {}) or {}
        elif isinstance(config, dict):
            cfg = config.get("texturing_cleanup", {}) or {}

    high_faces = int(cfg.get("high_quality", {}).get("max_faces", 500000))
    high_tex_size = int(cfg.get("high_quality", {}).get("texture_size", 4096))
    med_faces = int(cfg.get("medium_quality", {}).get("max_faces", 120000))
    med_tex_size = int(cfg.get("medium_quality", {}).get("texture_size", 2048))
    low_faces = int(cfg.get("low_quality", {}).get("max_faces", 35000))
    low_tex_size = int(cfg.get("low_quality", {}).get("texture_size", 1024))

    hole_max_size = int(cfg.get("medium_quality", {}).get("hole_max_size", 50))
    smooth_normals = bool(cfg.get("smooth_normals", True))

    in_path = Path(input_mesh_path).resolve()
    if not in_path.exists():
        fallback_path = Path("outputs/sample_drone/dense/dense_mesh.obj").resolve()
        if fallback_path.exists():
            in_path = fallback_path
        else:
            raise FileNotFoundError(f"Input mesh not found: {in_path}")

    try:
        loaded = trimesh.load(str(in_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) > 0:
                mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
            else:
                mesh = trimesh.Trimesh()
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError(f"Could not load valid mesh from {in_path}")

        cleaner = MeshCleaner(hole_max_size=hole_max_size, smooth_normals=smooth_normals)
        cleaned_mesh, clean_stats = cleaner.clean(mesh, temp_dir=temp_dir)

        uv_proc = UVProcessor(method="box")
        uv_mesh, uv_stats = uv_proc.compute_uv(cleaned_mesh)

        tex_opt = TextureOptimizer()
        base_tex = tex_opt.bake_vertex_colors(uv_mesh, texture_size=med_tex_size)

        lod_gen = LODGenerator(high_faces=high_faces, medium_faces=med_faces, low_faces=low_faces)

        tier_specs = {
            "high": (high_tex_size, "master"),
            "medium": (med_tex_size, "medium"),
            "low": (low_tex_size, "low"),
        }

        selected_tiers = list(tier_specs.keys()) if quality_tier.lower() == "all" else [quality_tier.lower()]

        tier_results: Dict[str, Any] = {}
        master_exports: Dict[str, str] = {}
        curr_mesh = uv_mesh

        for tier in selected_tiers:
            if tier not in tier_specs:
                continue
            tex_sz, dir_sub = tier_specs[tier]
            tier_mesh, lod_stats = lod_gen.generate_tier(curr_mesh, tier, temp_dir=temp_dir)
            if quality_tier.lower() == "all":
                curr_mesh = tier_mesh

            if tier_mesh is not uv_mesh:
                uv_mesh_tier, _ = uv_proc.compute_uv(tier_mesh)
            else:
                uv_mesh_tier = tier_mesh

            tex_resized = tex_opt.resize_texture(base_tex, target_size=tex_sz)
            tier_out_dir = out_dir / dir_sub
            exported = export_mesh_tier(uv_mesh_tier, tex_resized, tier_out_dir, tier)

            if tier == "high" or (tier == selected_tiers[0] and not master_exports):
                master_exports = exported

            tier_results[tier] = {
                "lod_stats": lod_stats,
                "texture_resolution": [tex_sz, tex_sz],
                "exported_files": exported,
            }
            import gc
            gc.collect()

    finally:
        os.chdir(orig_cwd)
        try:
            for f in temp_dir.glob("**/*"):
                if f.is_file():
                    f.unlink()
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    elapsed = time.time() - t0
    report = {
        "status": "COMPLETED",
        "elapsed_seconds": round(elapsed, 4),
        "input_mesh": str(in_path),
        "clean_stats": clean_stats,
        "uv_stats": uv_stats,
        "tiers": tier_results,
        "master_exports": master_exports,
        "output_dir": str(out_dir),
    }

    report_path = out_dir / "stage12_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
